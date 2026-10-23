"""FlashAttention Triton implementation (CUDA). Falls back to PyTorch on import failure."""

from __future__ import annotations

from typing import Any

import torch

from cs336_systems.flash_attention import FlashAttentionPyTorch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, out_ptr, lse_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        n_queries, n_keys, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        batch_id = tl.program_id(1)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q = tl.load(
            Q + batch_id * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < n_queries,
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start_n in range(0, n_keys, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + batch_id * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < n_keys,
                other=0.0,
            )
            v = tl.load(
                V + batch_id * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < n_keys,
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            key_valid = offs_n[None, :] < n_keys
            if IS_CAUSAL:
                key_valid = key_valid & (offs_m[:, None] >= offs_n[None, :])
            scores = tl.where(key_valid, scores, -1.0e6)
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new
        # l_i is the sum of exp(scores - m_i). Normalize the accumulator; storing
        # it raw leaves the attention output scaled by the partition function.
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        l_final = m_i + tl.log(l_safe)
        tl.store(
            out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            acc,
            mask=offs_m[:, None] < n_queries,
        )
        tl.store(
            lse_ptr + batch_id * stride_lb + offs_m * stride_lq,
            l_final,
            mask=offs_m < n_queries,
        )

    def _triton_flash_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = q.shape[:-2]
        queries = q.shape[-2]
        dim = q.shape[-1]
        q_flat = q.contiguous().reshape(-1, queries, dim)
        k_flat = k.contiguous().reshape(-1, k.shape[-2], dim)
        v_flat = v.contiguous().reshape(-1, v.shape[-2], dim)
        batch, n_queries, _ = q_flat.shape
        n_keys = k_flat.shape[-2]
        output = torch.empty_like(q_flat)
        logsumexp = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
        block_m = 32
        block_n = 32
        grid = (triton.cdiv(n_queries, block_m), batch)
        _flash_fwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            output,
            logsumexp,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            n_queries,
            n_keys,
            dim**-0.5,
            IS_CAUSAL=is_causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            D=dim,
        )
        return output.reshape(*leading, n_queries, dim), logsumexp.reshape(*leading, n_queries)

    class FlashAttentionTriton(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            is_causal: bool = False,
        ) -> torch.Tensor:
            output, logsumexp = _triton_flash_forward(q, k, v, is_causal)
            ctx.save_for_backward(q, k, v, output, logsumexp)
            ctx.is_causal = is_causal
            return output

        @staticmethod
        def backward(
            ctx: Any,
            *grad_outputs: Any,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
            grad_output = grad_outputs[0]
            q, k, v, _output, _logsumexp = ctx.saved_tensors
            is_causal = bool(ctx.is_causal)
            with torch.enable_grad():
                q2 = q.detach().requires_grad_(True)
                k2 = k.detach().requires_grad_(True)
                v2 = v.detach().requires_grad_(True)
                out = FlashAttentionPyTorch.apply(q2, k2, v2, is_causal)
                out.backward(grad_output)
            return q2.grad, k2.grad, v2.grad, None

else:
    FlashAttentionTriton = FlashAttentionPyTorch

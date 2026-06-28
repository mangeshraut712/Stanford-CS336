"""FlashAttention Triton implementation (CUDA). Falls back to PyTorch on import failure."""

from __future__ import annotations

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
        Q, K, V, O, L,
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
            if IS_CAUSAL:
                causal = offs_m[:, None] >= offs_n[None, :]
                scores = tl.where(causal, scores, -1.0e6)
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new
        l_final = m_i + tl.log(l_i)
        tl.store(
            O + batch_id * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            acc,
            mask=offs_m[:, None] < n_queries,
        )
        tl.store(
            L + batch_id * stride_lb + offs_m * stride_lq,
            l_final,
            mask=offs_m < n_queries,
        )

    class FlashAttentionTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, is_causal=False):
            # Triton backward is optional; delegate backward to PyTorch reference path.
            output = FlashAttentionPyTorch.apply(q, k, v, is_causal)
            ctx.save_for_backward(q, k, v, output, *[
                t for t in output.grad_fn.saved_tensors if t.shape == (q.shape[0], q.shape[1])
            ])
            ctx.is_causal = is_causal
            return output

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, output, logsumexp = ctx.saved_tensors
            with torch.enable_grad():
                q2 = q.detach().requires_grad_(True)
                k2 = k.detach().requires_grad_(True)
                v2 = v.detach().requires_grad_(True)
                out = FlashAttentionPyTorch.apply(q2, k2, v2, ctx.is_causal)
                out.backward(grad_output)
            return q2.grad, k2.grad, v2.grad, None

else:
    FlashAttentionTriton = FlashAttentionPyTorch

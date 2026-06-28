"""FlashAttention-2 style autograd using PyTorch ops (no Triton)."""

from __future__ import annotations

import math

import torch


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            causal = torch.arange(n_queries, device=q.device)[None, :, None] >= torch.arange(
                n_keys, device=k.device
            )[None, None, :]
            scores = torch.where(causal, scores, torch.full_like(scores, -1e6))
        logsumexp = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.save_for_backward(q, k, v, output, logsumexp)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, logsumexp = ctx.saved_tensors
        scale = ctx.scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if ctx.is_causal:
            n_queries = q.shape[-2]
            n_keys = k.shape[-2]
            causal = torch.arange(n_queries, device=q.device)[None, :, None] >= torch.arange(
                n_keys, device=k.device
            )[None, None, :]
            scores = torch.where(causal, scores, torch.full_like(scores, -1e6))
        probs = torch.exp(scores - logsumexp.unsqueeze(-1))
        delta = (grad_output * output).sum(dim=-1, keepdim=True)
        grad_probs = torch.matmul(grad_output, v.transpose(-2, -1))
        grad_scores = probs * (grad_probs - delta)
        grad_q = torch.matmul(grad_scores, k) * scale
        grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale
        grad_v = torch.matmul(probs.transpose(-2, -1), grad_output)
        return grad_q, grad_k, grad_v, None


def flash_attention_pytorch(q, k, v, is_causal=False):
    return FlashAttentionPyTorch.apply(q, k, v, is_causal)

from __future__ import annotations

import math

import einx
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

from cs336_basics.nn_utils import softmax


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    return F.linear(in_features, weights)


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    return F.embedding(token_ids, weights)


def rmsnorm(weights: Tensor, in_features: Tensor, eps: float) -> Tensor:
    in_dtype = in_features.dtype
    x = in_features.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (weights * (x * rms)).to(in_dtype)


def silu(in_features: Tensor) -> Tensor:
    return F.silu(in_features)


def swiglu(
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    return linear(w2_weight, silu(linear(w1_weight, in_features)) * linear(w3_weight, in_features))


def _rope_cache(max_seq_len: int, dim: int, theta: float, device: torch.device) -> Tensor:
    d = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    freqs = torch.tensor(theta, device=device, dtype=torch.float32) ** (-d)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = einsum(t, freqs, "t, f -> t f")
    return torch.stack((torch.cos(angles), torch.sin(angles)))


def apply_rope(
    in_query_or_key: Tensor,
    token_positions: Tensor | None,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    dim = in_query_or_key.shape[-1]
    cache = _rope_cache(max_seq_len, dim, theta, in_query_or_key.device)
    x1, x2 = rearrange(in_query_or_key, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)
    if token_positions is not None:
        cos, sin = einx.get_at(
            "cos_sin [pos] half_dim, ... -> cos_sin ... half_dim",
            cache,
            token_positions,
        )
    else:
        seq_len = in_query_or_key.shape[-2]
        cos, sin = cache[:, :seq_len, :].unbind(0)
    x1_rot = cos * x1 - sin * x2
    x2_rot = sin * x1 + cos * x2
    return einx.id("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = k.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask, scores, torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype))
    weights = softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    max_seq_len: int | None = None,
    theta: float | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    q = linear(q_proj_weight, in_features)
    k = linear(k_proj_weight, in_features)
    v = linear(v_proj_weight, in_features)

    q = rearrange(q, "... seq (heads d) -> ... heads seq d", heads=num_heads)
    k = rearrange(k, "... seq (heads d) -> ... heads seq d", heads=num_heads)
    v = rearrange(v, "... seq (heads d) -> ... heads seq d", heads=num_heads)

    if theta is not None and max_seq_len is not None:
        if token_positions is None:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=in_features.device)
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        q = apply_rope(q, token_positions, theta, max_seq_len)
        k = apply_rope(k, token_positions, theta, max_seq_len)

    sequence_length = in_features.shape[-2]
    iota = torch.arange(sequence_length, device=in_features.device)
    causal_mask = rearrange(iota, "query -> query 1") >= rearrange(iota, "key -> 1 key")
    leading_dims = in_features.ndim - 2
    causal_mask = causal_mask[(None,) * leading_dims + (None, Ellipsis)]

    attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    attn_output = rearrange(attn_output, "... heads seq d -> ... seq (heads d)")
    return linear(o_proj_weight, attn_output)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    x = in_features
    normed = rmsnorm(weights["ln1.weight"], x, eps=1e-5)
    x = x + multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normed,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    normed = rmsnorm(weights["ln2.weight"], x, eps=1e-5)
    x = x + swiglu(
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        normed,
    )
    return x


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    x = embedding(weights["token_embeddings.weight"], in_indices)
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {
            key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)
        }
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )
    x = rmsnorm(weights["ln_final.weight"], x, eps=1e-5)
    return linear(weights["lm_head.weight"], x)

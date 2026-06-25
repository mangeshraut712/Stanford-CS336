from __future__ import annotations

import json
import logging
import math
import os
import warnings

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.nn_utils import softmax

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std),
            requires_grad=True,
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std),
            requires_grad=True,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x * rms).to(in_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta),
            persistent=False,
        )

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0
        d = torch.arange(0, dim, 2) / dim
        freqs = torch.tensor(theta) ** -d
        t = torch.arange(context_length)
        freqs = einsum(t, freqs, "t, f -> t f")
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None
    ) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)
        if pos_ids is not None:
            cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        return einx.id("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, positional_encoder: RotaryEmbedding | None = None):
        super().__init__()
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)
        self.positional_encoder = positional_encoder

    def forward(
        self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q, K, V = (
            rearrange(tensor, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for tensor in (Q, K, V)
        )
        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)
        iota = torch.arange(sequence_length, device=x.device)
        causal_mask = rearrange(iota, "query -> query 1") >= rearrange(iota, "key -> 1 key")
        causal_mask = causal_mask[(None,) * len(batch_dims) + (...,)]
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        return self.output_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | None,
        pre_norm: bool = True,
        use_rmsnorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, positional_encoder=positional_encoder
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff) if ffn_type == "swiglu" else SiLUFeedForward(d_model, d_ff)
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_cls(d_model)
        self.ln2 = norm_cls(d_model)

    def forward(self, x: torch.Tensor):
        if self.pre_norm:
            x_attn = self.attn(self.ln1(x))
            attn_sublayer_output = x + x_attn
            return attn_sublayer_output + self.ffn(self.ln2(attn_sublayer_output))
        x_attn = self.attn(x)
        attn_sublayer_output = self.ln1(x + x_attn)
        x_ffn = self.ffn(attn_sublayer_output)
        return self.ln2(attn_sublayer_output + x_ffn)


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10_000.0,
        use_rope: bool = True,
        pre_norm: bool = True,
        use_rmsnorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, rope_theta) if use_rope and rope_theta is not None else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    pre_norm=pre_norm,
                    use_rmsnorm=use_rmsnorm,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)
        logger.info("non-embedding parameters: %.2fM", self.get_num_params() / 1e6)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        hidden = self.token_embeddings(x)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1] / temperature
            if top_k:
                topk_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                threshold = topk_values[:, -1]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, float("-inf"))
            next_token_probabilities = softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        return x[:, original_sequence_length:]

    def save_pretrained(self, output_dir: str | os.PathLike) -> None:
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path, weights_only=True)
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model

"""Correctness checks that the staff fixtures do not cover."""

from __future__ import annotations

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_basics.model import RotaryEmbedding
from cs336_systems.ddp import DDP
from cs336_systems.fsdp import FSDP, gather_full_params
from cs336_systems.sharded_optimizer import ShardedOptimizer

from .common import ToyModel, _cleanup_process_group, _setup_process_group
from .test_attention import _attention_and_lse


def test_rope_restores_rotated_pairs_in_place() -> None:
    layer = RotaryEmbedding(context_length=8, dim=4, theta=10_000.0)
    x = torch.randn(2, 3, 4)
    y = layer(x, None)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = layer._freq_cis_cache[0, :3]
    sin = layer._freq_cis_cache[1, :3]
    expected_even = cos * x1 - sin * x2
    expected_odd = sin * x1 + cos * x2
    expected = torch.stack((expected_even, expected_odd), dim=-1).flatten(-2)
    torch.testing.assert_close(y, expected)
    concatenated = torch.cat((expected_even, expected_odd), dim=-1)
    assert not torch.allclose(y, concatenated)


def test_flash_attention_pytorch_causal_matches_reference() -> None:
    from cs336_systems.flash_attention import FlashAttentionPyTorch

    torch.manual_seed(0)
    q = torch.randn(2, 16, 8, requires_grad=True)
    k = torch.randn(2, 16, 8, requires_grad=True)
    v = torch.randn(2, 16, 8, requires_grad=True)
    out = FlashAttentionPyTorch.apply(q, k, v, True)
    reference, _ = _attention_and_lse(q, k, v, True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


def test_ddp_finish_does_not_average_twice() -> None:
    world_size = 2
    mp.spawn(_ddp_finish_once, args=(world_size,), nprocs=world_size, join=True)


def _ddp_finish_once(rank: int, world_size: int) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(0)
    model = DDP(ToyModel().to(device))
    model(torch.ones(4, 10, device=device)).sum().backward()
    model.finish_gradient_synchronization()
    before = [param.grad.detach().clone() for param in model.parameters() if param.grad is not None]
    model.finish_gradient_synchronization()
    after = [param.grad for param in model.parameters() if param.grad is not None]
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right)
    _cleanup_process_group()


def test_sharded_optimizer_zero_grad_clears_unowned_parameters() -> None:
    world_size = 2
    mp.spawn(_sharded_zero_grad, args=(world_size,), nprocs=world_size, join=True)


def _sharded_zero_grad(rank: int, world_size: int) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(0)
    model = ToyModel().to(device)
    optimizer = ShardedOptimizer(model.parameters(), torch.optim.SGD, lr=0.1)
    model(torch.ones(4, 10, device=device)).sum().backward()
    assert any(param.grad is not None for param in model.parameters())
    optimizer.zero_grad()
    assert all(param.grad is None for param in model.parameters())
    _cleanup_process_group()


class _UnevenShardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from cs336_basics.model import Embedding, Linear, RMSNorm

        self.embedding = Embedding(5, 4)
        self.norm = RMSNorm(4)
        self.linear = Linear(4, 5)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.embedding(token_ids))
        return self.linear(hidden)


def test_fsdp_keeps_rows_when_width_is_not_divisible_by_world_size() -> None:
    world_size = 2
    mp.spawn(_fsdp_uneven, args=(world_size,), nprocs=world_size, join=True)


def _fsdp_uneven(rank: int, world_size: int) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(0)
    base = _UnevenShardModel().to(device)
    reference = _UnevenShardModel().to(device)
    reference.load_state_dict(base.state_dict())
    fsdp_model = FSDP(base)
    torch.manual_seed(1)
    token_ids = torch.randint(0, 5, (4, 3), device=device)
    local = token_ids[rank * 2 : (rank + 1) * 2]
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    reference(token_ids).mean().backward()
    fsdp_model(local).mean().backward()
    fsdp_model.finish_gradient_synchronization()
    reference_optimizer.step()
    fsdp_optimizer.step()
    full_params = gather_full_params(fsdp_model)
    for name, param in reference.named_parameters():
        assert full_params[name].shape == param.shape
        torch.testing.assert_close(full_params[name], param.detach(), atol=1e-5, rtol=1e-4)
    _cleanup_process_group()

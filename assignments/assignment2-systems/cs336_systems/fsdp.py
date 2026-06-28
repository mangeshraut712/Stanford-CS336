"""Fully Sharded Data Parallel for cs336_basics Linear and Embedding layers."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.model import Embedding, Linear
from einops import einsum


def _gather_full_weight(weight_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    shards = [torch.empty_like(weight_shard) for _ in range(world_size)]
    dist.all_gather(shards, weight_shard)
    return torch.cat(shards, dim=0)


def _reduce_to_shard(grad_full: torch.Tensor, world_size: int) -> torch.Tensor:
    grad_full = grad_full.contiguous()
    dist.all_reduce(grad_full, op=dist.ReduceOp.SUM)
    grad_full.div_(world_size)
    return grad_full.chunk(world_size, dim=0)[dist.get_rank()].contiguous()


def _gathered_weight_with_hook(
    weight_shard: nn.Parameter,
    world_size: int,
    compute_dtype: torch.dtype | None,
) -> torch.Tensor:
    with torch.no_grad():
        full_weight = _gather_full_weight(weight_shard, world_size)
    full_weight = full_weight.detach().requires_grad_(True)

    def _shard_grad_hook(grad_full: torch.Tensor) -> None:
        grad_shard = _reduce_to_shard(grad_full, world_size)
        if compute_dtype is not None:
            grad_shard = grad_shard.to(torch.float32)
        if weight_shard.grad is None:
            weight_shard.grad = grad_shard
        else:
            weight_shard.grad.add_(grad_shard)

    full_weight.register_hook(_shard_grad_hook)
    if compute_dtype is not None:
        return full_weight.to(compute_dtype)
    return full_weight


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._sharded_modules: dict[str, nn.Module] = {}
        self._wrap_sharded_layers()

    def _wrap_sharded_layers(self) -> None:
        for name, module in self.module.named_modules():
            if isinstance(module, Linear):
                self._shard_module(name, module, linear=True)
            elif isinstance(module, Embedding):
                self._shard_module(name, module, linear=False)

    def _shard_module(self, name: str, module: nn.Module, *, linear: bool) -> None:
        full_weight = module.weight
        shard_rows = full_weight.shape[0] // self._world_size
        start = self._rank * shard_rows
        module.weight = nn.Parameter(full_weight.data[start : start + shard_rows].clone())
        self._sharded_modules[name] = module

        if linear:

            def linear_forward(x, mod=module):
                weight = _gathered_weight_with_hook(mod.weight, self._world_size, self.compute_dtype)
                return einsum(x, weight, "... d_in, d_out d_in -> ... d_out")

            module.forward = linear_forward
        else:

            def embedding_forward(token_ids, mod=module):
                weight = _gathered_weight_with_hook(mod.weight, self._world_size, self.compute_dtype)
                return weight[token_ids, :]

            module.forward = embedding_forward

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for name, param in self.module.named_parameters():
            module_name = name.rsplit(".", 1)[0]
            if module_name in self._sharded_modules:
                continue
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self._world_size)


def gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, param in fsdp_model.module.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        if module_name not in fsdp_model._sharded_modules:
            result[name] = param.data.clone()
            continue
        shard_shape = param.shape
        shards = [torch.empty(shard_shape, device=param.device, dtype=param.dtype) for _ in range(fsdp_model._world_size)]
        dist.all_gather(shards, param.data)
        result[name] = torch.cat(shards, dim=0)
    return result

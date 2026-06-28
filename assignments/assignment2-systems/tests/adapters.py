from __future__ import annotations

import torch

from cs336_systems.ddp import DDP
from cs336_systems.flash_attention import FlashAttentionPyTorch
from cs336_systems.flash_attention_triton import FlashAttentionTriton
from cs336_systems.fsdp import FSDP, gather_full_params
from cs336_systems.sharded_optimizer import ShardedOptimizer


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return gather_full_params(fsdp_model)


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)

"""Distributed Data Parallel with per-parameter async all-reduce."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._pending_handles: list[dist.Work] = []
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        if self._world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0, async_op=False)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: nn.Parameter) -> None:
        if param.grad is None:
            return
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._pending_handles:
            handle.wait()
        self._pending_handles.clear()
        if self._world_size > 1:
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.div_(self._world_size)

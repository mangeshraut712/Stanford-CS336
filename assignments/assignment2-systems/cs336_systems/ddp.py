"""Distributed Data Parallel with per-parameter async all-reduce."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._pending_handles: list[tuple[nn.Parameter, dist.Work]] = []
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        if self._world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0, async_op=False)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: nn.Parameter) -> None:
        if param.grad is None or self._world_size == 1:
            return
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        if not isinstance(handle, dist.Work):
            raise RuntimeError("asynchronous all_reduce did not return a work handle")
        self._pending_handles.append((param, handle))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        pending = self._pending_handles
        self._pending_handles = []
        for param, handle in pending:
            handle.wait()
            if param.grad is not None:
                param.grad.div_(self._world_size)

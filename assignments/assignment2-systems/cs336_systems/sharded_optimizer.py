"""ZeRO-style optimizer state sharding with post-step parameter broadcast."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn


class ShardedOptimizer:
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ):
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        param_list = list(params)
        self._param_to_owner: dict[nn.Parameter, int] = {}
        self._owner_to_params: dict[int, list[nn.Parameter]] = defaultdict(list)

        for index, param in enumerate(param_list):
            owner = index % self._world_size
            self._param_to_owner[param] = owner
            self._owner_to_params[owner].append(param)

        local_params = self._owner_to_params[self._rank]
        self._local_optimizer = optimizer_cls(local_params, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._local_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._local_optimizer.step(closure)
        for owner, params in self._owner_to_params.items():
            for param in params:
                dist.broadcast(param.data, src=owner)
        return loss

    def add_param_group(self, param_group: dict) -> None:
        params = list(param_group["params"])
        new_group = {k: v for k, v in param_group.items() if k != "params"}
        new_group["params"] = []
        for param in params:
            owner = len(self._param_to_owner) % self._world_size
            self._param_to_owner[param] = owner
            self._owner_to_params[owner].append(param)
            if owner == self._rank:
                new_group["params"].append(param)
        if new_group["params"]:
            self._local_optimizer.add_param_group(new_group)

    def state_dict(self):
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self._local_optimizer.load_state_dict(state_dict)

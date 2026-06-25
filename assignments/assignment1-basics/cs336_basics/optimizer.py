from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.data
                state = self.state[parameter]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)

                alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
                parameter.data -= alpha * group["weight_decay"] * parameter.data

                prev_m = state.get("m", torch.zeros_like(grad))
                prev_v = state.get("v", torch.zeros_like(grad))
                m = beta_1 * prev_m + (1 - beta_1) * grad
                v = beta_2 * prev_v + (1 - beta_2) * torch.square(grad)
                parameter.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss

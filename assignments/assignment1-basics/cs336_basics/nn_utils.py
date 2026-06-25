from __future__ import annotations

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def log_softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    return shifted - torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    negative_log_probs = -log_softmax(inputs, dim=-1)
    return torch.mean(torch.gather(negative_log_probs, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm: float) -> None:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return
    norm = torch.sqrt(sum(torch.sum(grad * grad) for grad in grads))
    clip_coef = min(1.0, max_norm / (norm + 1e-6))
    for grad in grads:
        grad.mul_(clip_coef)

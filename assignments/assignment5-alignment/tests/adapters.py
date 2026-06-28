from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.grpo import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_per_instance_dpo_loss,
    compute_policy_gradient_loss,
    compute_rollout_rewards,
    get_packed_sft_dataset as build_packed_sft_dataset,
    get_response_log_probs,
    grpo_train_step,
    iterate_batches,
    parse_gsm8k_response,
    parse_mmlu_response,
    tokenize_prompt_and_output,
)


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    return get_response_log_probs(model, input_ids, labels, return_token_entropy)


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    return compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    return compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return compute_policy_gradient_loss(
        raw_rewards_or_advantages,
        policy_log_probs,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        response_mask=response_mask,
    )


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    return aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss,
        mask,
        loss_normalization=loss_normalization,
        normalization_constant=normalization_constant,
    )


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    return grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        reward_fn=reward_fn,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        loss_normalization=loss_normalization,
        normalization_constant=normalization_constant,
    )


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return build_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return iterate_batches(dataset, batch_size, shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    return parse_mmlu_response(mmlu_example, model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    return parse_gsm8k_response(model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    return compute_per_instance_dpo_loss(
        lm,
        lm_ref,
        tokenizer,
        beta,
        prompt,
        response_chosen,
        response_rejected,
    )

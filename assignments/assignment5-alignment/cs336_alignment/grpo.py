from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_ALPACA_SFT_PROMPT_PATH = Path(__file__).parent / "prompts_safety" / "alpaca_sft.prompt"


def _load_alpaca_sft_template() -> str:
    return _ALPACA_SFT_PROMPT_PATH.read_text()


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    pad_id = tokenizer.pad_token_id

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        prompt_lens.append(len(prompt_ids))
        sequences.append(prompt_ids + output_ids)

    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]
    positions = torch.arange(labels.shape[1]).unsqueeze(0)
    prompt_len_tensor = torch.tensor(prompt_lens, dtype=torch.long).unsqueeze(1)
    response_mask = (positions >= prompt_len_tensor - 1) & (labels != pad_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
    attention_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    model_kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    logits = model(**model_kwargs).logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    output: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        probs = log_probs.exp()
        output["token_entropy"] = -(probs * log_probs).sum(dim=-1)
    return output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    reward_components = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [component["reward"] for component in reward_components],
        dtype=torch.float32,
    )
    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_format_reward": float(
            sum(component["format_reward"] for component in reward_components)
            / len(reward_components)
        ),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    grouped = raw_rewards.view(-1, group_size)
    advantages = raw_rewards.clone()

    if baseline == "mean":
        group_means = grouped.mean(dim=1, keepdim=True)
        advantages = (grouped - group_means).reshape_as(raw_rewards)
    elif baseline == "none":
        advantages = raw_rewards.clone()

    if advantage_normalizer == "std":
        group_stds = grouped.std(dim=1, unbiased=True, keepdim=True).clamp(min=advantage_eps)
        advantages = (advantages.view(-1, group_size) / group_stds).reshape_as(raw_rewards)
    elif advantage_normalizer == "mean":
        group_means = grouped.mean(dim=1, keepdim=True).clamp(min=advantage_eps)
        advantages = (advantages.view(-1, group_size) / group_means).reshape_as(raw_rewards)

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "std_reward": float(raw_rewards.std(unbiased=False).item()),
        "max_reward": float(raw_rewards.max().item()),
        "min_reward": float(raw_rewards.min().item()),
    }
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    advantages = raw_rewards_or_advantages
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)

    metadata: dict[str, Tensor] = {}

    if importance_reweighting_method == "none":
        per_token_loss = -advantages * policy_log_probs
        return per_token_loss, metadata

    assert old_log_probs is not None
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    if importance_reweighting_method == "noclip":
        per_token_loss = -advantages * ratio
        return per_token_loss, metadata

    assert cliprange is not None

    if importance_reweighting_method == "grpo":
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        per_token_loss = -torch.min(advantages * ratio, advantages * clipped_ratio)
        metadata["clip_fraction"] = (ratio != clipped_ratio).float().mean()
        return per_token_loss, metadata

    assert importance_reweighting_method == "gspo"
    assert response_mask is not None
    mask = response_mask.float()
    seq_log_ratio = (log_ratio * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    seq_ratio = torch.exp(seq_log_ratio).expand_as(policy_log_probs)
    clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - cliprange, 1.0 + cliprange)
    per_token_loss = -torch.min(advantages * seq_ratio, advantages * clipped_seq_ratio)
    metadata["clip_fraction"] = (seq_ratio != clipped_seq_ratio).float().mean()
    return per_token_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    mask = mask.float()
    if loss_normalization == "sequence":
        per_sequence_loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return per_sequence_loss.mean()

    assert normalization_constant is not None
    return (per_token_policy_gradient_loss * mask).sum() / normalization_constant


def grpo_train_step(
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
) -> tuple[Tensor, dict[str, Tensor | float]]:
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )
    batch_size = len(repeated_prompts)
    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    microbatch_losses: list[Tensor] = []
    step_metadata: dict[str, Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
    }

    for microbatch_idx in range(gradient_accumulation_steps):
        start = microbatch_idx * microbatch_size
        end = start + microbatch_size
        micro_input_ids = tokenized["input_ids"][start:end]
        micro_labels = tokenized["labels"][start:end]
        micro_response_mask = tokenized["response_mask"][start:end]
        micro_advantages = advantages[start:end]
        micro_old_log_probs = None if old_log_probs is None else old_log_probs[start:end]

        log_prob_output = get_response_log_probs(
            model=model,
            input_ids=micro_input_ids,
            labels=micro_labels,
            return_token_entropy=False,
        )
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=micro_advantages,
            policy_log_probs=log_prob_output["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=micro_old_log_probs,
            cliprange=cliprange,
            response_mask=micro_response_mask,
        )
        micro_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=micro_response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )
        if loss_normalization == "constant":
            micro_loss.backward()
        else:
            (micro_loss / gradient_accumulation_steps).backward()
        microbatch_losses.append(micro_loss.detach())

        for key, value in loss_metadata.items():
            if isinstance(value, Tensor):
                step_metadata[key] = value.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    )
    if max_grad_norm is not None:
        step_metadata["grad_norm"] = float(grad_norm.item())

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if loss_normalization == "constant":
        reported_loss = torch.stack(microbatch_losses).sum()
    else:
        reported_loss = torch.stack(microbatch_losses).mean()
    return reported_loss, step_metadata


SFT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        input_ids: list[list[int]],
        labels: list[list[int]],
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    examples: list[dict[str, str]] = []
    with open(dataset_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(0)
        permutation = torch.randperm(len(examples), generator=generator).tolist()
        examples = [examples[index] for index in permutation]

    all_tokens: list[int] = []
    for example in examples:
        text = SFT_TEMPLATE.format(prompt=example["prompt"], response=example["response"])
        encoded = tokenizer.encode(tokenizer.bos_token + text + tokenizer.eos_token, add_special_tokens=False)
        all_tokens.extend(encoded)

    num_chunks = len(all_tokens) // seq_length

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * seq_length
        chunk_input = all_tokens[start : start + seq_length]
        chunk_labels = all_tokens[start + 1 : start + seq_length + 1]
        input_ids.append(chunk_input)
        labels.append(chunk_labels)

    return PackedSFTDataset(input_ids=input_ids, labels=labels)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(
    mmlu_example: dict,
    model_output: str,
) -> str | None:
    del mmlu_example
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"(?:^|\s)([A-D])(?:[\s\.:,]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1]


def _sequence_log_prob_sum(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
) -> Tensor:
    template = _load_alpaca_sft_template()
    text = template.format(instruction=instruction, response=response)
    token_ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([token_ids[:-1]], device=next(model.parameters()).device)
    labels = torch.tensor([token_ids[1:]], device=input_ids.device)
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    was_training = lm.training
    lm.eval()
    lm_ref.eval()

    chosen_log_prob = _sequence_log_prob_sum(lm, tokenizer, prompt, response_chosen)
    rejected_log_prob = _sequence_log_prob_sum(lm, tokenizer, prompt, response_rejected)

    with torch.no_grad():
        ref_chosen_log_prob = _sequence_log_prob_sum(lm_ref, tokenizer, prompt, response_chosen)
        ref_rejected_log_prob = _sequence_log_prob_sum(lm_ref, tokenizer, prompt, response_rejected)

    if was_training:
        lm.train()

    pi_logratio = chosen_log_prob - rejected_log_prob
    ref_logratio = ref_chosen_log_prob.to(pi_logratio.device) - ref_rejected_log_prob.to(
        pi_logratio.device
    )
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio))

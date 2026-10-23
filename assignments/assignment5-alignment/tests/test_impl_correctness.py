"""Correctness checks that the staff fixtures do not cover."""

from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from cs336_alignment.drgrpo_grader import _is_frac, _str_to_int, question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import (
    _microbatch_slices,
    compute_group_normalized_rewards,
    compute_rollout_rewards,
    get_packed_sft_dataset,
    parse_gsm8k_response,
    parse_mmlu_response,
)


def test_microbatch_slices_cover_a_remainder() -> None:
    assert _microbatch_slices(5, 2) == [(0, 3), (3, 5)]
    assert _microbatch_slices(4, 2) == [(0, 2), (2, 4)]
    with pytest.raises(ValueError):
        _microbatch_slices(1, 2)


def test_group_rewards_reject_a_partial_group_and_zero_std() -> None:
    with pytest.raises(ValueError):
        compute_group_normalized_rewards(torch.tensor([1.0, 0.0, 1.0]), group_size=2)
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor([1.0, 1.0]),
        group_size=2,
        baseline="mean",
        advantage_normalizer="std",
    )
    torch.testing.assert_close(advantages, torch.zeros(2))


def test_empty_rollout_rewards_are_empty() -> None:
    rewards, metadata = compute_rollout_rewards(lambda _response, _truth: {"reward": 1.0}, [], [])
    assert rewards.numel() == 0
    assert metadata["mean_reward"] == 0.0


def test_packed_sft_labels_match_input_length_when_token_count_divides_sequence(tmp_path) -> None:
    vocab = {"<s>": 0, "</s>": 1, "a": 2, "b": 3}
    word_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<s>"))
    word_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<s>",
    )
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"prompt": "a", "response": "b"}) + "\n")
    dataset = get_packed_sft_dataset(tokenizer, path, seq_length=8, shuffle=False)
    assert len(dataset) > 0
    for row in dataset:
        assert row["input_ids"].shape == row["labels"].shape
        assert row["input_ids"].shape[-1] == 8


def test_parsers_cover_signed_numbers_and_parenthesized_choices() -> None:
    assert parse_gsm8k_response("After step 2 the change is -3.5.") == "-3.5"
    example = {"question": "q", "options": ["a", "b", "c", "d"], "answer": "C"}
    assert parse_mmlu_response(example, "I choose (C).") == "C"


def test_integer_normalization_rounds_and_fraction_check_rejects_junk() -> None:
    assert _str_to_int("1.999999999") == 2
    assert _str_to_int("1,000") == 1000
    assert _is_frac("10/2")
    assert not _is_frac("10x/2")


def test_reward_functions_do_not_crash_on_an_unexpected_ground_truth_type() -> None:
    formatted = "</think> <answer>42</answer>"
    assert r1_zero_reward_fn(formatted, {"answer": "42"})["reward"] == 0.0
    assert question_only_reward_fn("no box", None)["reward"] == 0.0
    assert question_only_reward_fn(r"\boxed{42}", {"answer": "42"})["reward"] == 0.0


def test_question_only_reward_accepts_a_boxed_integer() -> None:
    response = r"The result is \boxed{42}."
    assert question_only_reward_fn(response, "42")["reward"] == 1.0
    assert r1_zero_reward_fn("</think> <answer>" + response + "</answer>", "42")["reward"] == 1.0

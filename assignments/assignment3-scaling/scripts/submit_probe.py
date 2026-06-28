#!/usr/bin/env python3
"""Submit a small probe experiment to the hosted API (uses ~1 min GPU budget)."""

from __future__ import annotations

import os
import sys

from cs336_scaling.client import get_budget, get_experiment, submit_experiment
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig


def probe_config(model_seed: int = 0) -> TrainingConfig:
    """Tiny model, short run — good first API test."""
    return TrainingConfig(
        architecture_config=BasicTransformerConfig(
            attention_bias=False,
            head_dim=32,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10_000,
            tie_word_embeddings=False,
            dtype="bfloat16",
            vocab_size=32000,
        ),
        optimizer_config=AdamWConfig(),
        train_batch_size=4,
        val_batch_size=2,
        n_evals=2,
        total_train_tokens=4096,
        max_runtime_seconds=60,
        model_seed=model_seed,
    )


def main() -> None:
    if len(os.getenv("A3_API_KEY", "")) != 8:
        print("Set A3_API_KEY (8-digit) before submitting.")
        sys.exit(1)

    before = get_budget()
    print(f"Budget before: {before.remaining_seconds / 3600:.2f} h remaining")

    config = probe_config(model_seed=int(os.getenv("PROBE_SEED", "0")))
    result = submit_experiment(config)
    print(f"Submitted experiment_id={result.experiment_id}")
    print(f"Budget after submit: {result.budget_summary.remaining_seconds / 3600:.2f} h remaining")

    exp = get_experiment(result.experiment_id)
    print(f"Status: {exp.status.status_type if hasattr(exp.status, 'status_type') else exp.status}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local training smoke test on processed sample data (CPU/MPS, tiny model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cs336_basics.train_config import Config, ModelConfig, PathsConfig, TrainingConfig
from cs336_data.common import get_shared_assets_path

sys.path.insert(0, str(Path(__file__).parent))
from train import train_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--valid-bin", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    valid_bin = args.valid_bin or (
        get_shared_assets_path() / "tokenized_paloma_c4_100_domains_validation.bin"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg = Config(
        paths=PathsConfig(
            train_bin=args.train_bin,
            valid_bin=valid_bin,
            model_output=Path("artifacts/experiments/smoke_train"),
        ),
        model=ModelConfig(
            d_model=256,
            d_ff=512,
            num_layers=2,
            num_heads=4,
            context_length=128,
        ),
        training=TrainingConfig(
            device=device,
            dtype="float32",
            train_batch_size=4,
            eval_batch_size=4,
            train_steps=args.steps,
            eval_iterations=5,
            eval_interval=args.steps + 1,
            compile=False,
            log_interval=2,
            lr=3e-4,
        ),
    )
    cfg.paths.model_output.mkdir(parents=True, exist_ok=True)
    print(f"Smoke training on {device} for {args.steps} steps")
    train_from_config(cfg)


if __name__ == "__main__":
    main()

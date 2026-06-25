#!/usr/bin/env python3
"""Learning-rate sweep for §7.2."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 6e-4, 1e-3])
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for lr in args.learning_rates:
        out = args.artifacts_dir / "experiments" / f"lr_sweep_{lr:g}"
        cmd = [
            "uv",
            "run",
            "python",
            "train.py",
            "--train-tokens",
            str(args.artifacts_dir / "tokens/tinystories_train.bin"),
            "--val-tokens",
            str(args.artifacts_dir / "tokens/tinystories_valid.bin"),
            "--tokenizer-dir",
            str(args.artifacts_dir / "bpe/tinystories_10k"),
            "--output-dir",
            str(out),
            "--run-name",
            f"lr_sweep_{lr:g}",
            "--max-lr",
            str(lr),
            "--min-lr",
            str(lr / 10),
            "--max-steps",
            str(args.max_steps),
            "--eval-every",
            "200",
            "--checkpoint-every",
            "100000",
        ]
        subprocess.run(cmd, check=True)
        summary = json.loads((out / "run_summary.json").read_text())
        results.append({"lr": lr, **summary})
    out_path = args.artifacts_dir / "experiments/lr_sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

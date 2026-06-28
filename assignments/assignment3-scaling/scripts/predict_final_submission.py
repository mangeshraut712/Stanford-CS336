#!/usr/bin/env python3
"""Predict final validation loss; optionally submit to hyperturing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_scaling.analysis.scaling import (
    count_training_config_params,
    estimate_training_flops,
    fit_log_parabolas,
    load_isoflops,
    predict_loss,
)
from cs336_scaling.training.run import default_training_config

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load_parabolas() -> dict:
    path = ART / "isoflops_parabola_fits.json"
    if path.exists():
        return json.loads(path.read_text())
    return fit_log_parabolas(load_isoflops())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    config = default_training_config()
    parabolas = load_parabolas()
    n = count_training_config_params(config)
    c = estimate_training_flops(n, config.total_train_tokens)
    loss = predict_loss(n, c, parabolas)
    result = {"estimated_params": n, "estimated_compute_flops": c, "predicted_final_loss": round(loss, 4)}
    print(json.dumps(result, indent=2))

    if args.submit:
        from cs336_scaling.client import save_final_submission

        print(save_final_submission(config, result["predicted_final_loss"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit simple power-law scaling models to reference isoflops data.

Models (compute-optimal frontier):
  N_opt(C) = a * C^b          (Chinchilla expects b ≈ 0.5)
  L_opt(C) = c * C^d          (loss vs compute)

Also fits per-budget parabola in log(N): log L ≈ p log N + q log N + r (U-shape proxy).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "isoflops_curves.json"
OPTIMAL = ROOT / "artifacts" / "isoflops_compute_optimal.json"
OUT = ROOT / "artifacts" / "scaling_law_fits.json"


def _log_log_slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    cov = sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx)))
    var = sum((x - mx) ** 2 for x in lx)
    return cov / var if var else float("nan")


def main() -> None:
    with open(OPTIMAL) as f:
        optimal = json.load(f)

    c_vals = [r["compute_budget"] for r in optimal]
    n_vals = [r["parameters"] for r in optimal]
    l_vals = [r["final_loss"] for r in optimal]

    b = _log_log_slope(c_vals, n_vals)
    d = _log_log_slope(c_vals, l_vals)
    a = math.exp(
        sum(math.log(n) - b * math.log(c) for n, c in zip(n_vals, c_vals)) / len(optimal)
    )
    c_coef = math.exp(
        sum(math.log(l) - d * math.log(c) for l, c in zip(l_vals, c_vals)) / len(optimal)
    )

    fits = {
        "compute_optimal_N_vs_C": {
            "formula": "N_opt ≈ a * C^b",
            "a": a,
            "b": b,
            "chinchilla_expected_b": 0.5,
        },
        "compute_optimal_L_vs_C": {
            "formula": "L_opt ≈ c * C^d",
            "c": c_coef,
            "d": d,
        },
        "notes": [
            "Fitted on staff reference isoflops compute-optimal points.",
            "Use your own API runs to refit before final submission prediction.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fits, indent=2))

    print("Scaling-law fits (compute-optimal frontier)")
    print(f"  N_opt ≈ {a:.4e} * C^{b:.4f}   (Chinchilla b≈0.5)")
    print(f"  L_opt ≈ {c_coef:.4e} * C^{d:.4f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

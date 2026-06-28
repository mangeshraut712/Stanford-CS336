#!/usr/bin/env python3
"""Analyze reference isoflops curves from data/isoflops_curves.json.

Isoflops: at fixed compute budget C, plot final loss vs model parameters N.
The compute-optimal point per budget is argmin_N L(N, C).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "isoflops_curves.json"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def load_isoflops() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_optimal_per_budget(rows: list[dict]) -> list[dict]:
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["compute_budget"]].append(row)

    optimal = []
    for budget in sorted(by_budget):
        points = by_budget[budget]
        best = min(points, key=lambda r: r["final_loss"])
        optimal.append(
            {
                "compute_budget": budget,
                "parameters": best["parameters"],
                "final_loss": best["final_loss"],
            }
        )
    return optimal


def main() -> None:
    rows = load_isoflops()
    optimal = compute_optimal_per_budget(rows)

    print(f"Loaded {len(rows)} isoflops points from {DATA_PATH.name}")
    print(f"Compute budgets: {len(optimal)}")
    print()
    print("Compute-optimal (min loss at each fixed compute budget):")
    print(f"{'Compute':>12}  {'Params':>14}  {'Loss':>8}")
    print("-" * 40)
    for row in optimal:
        print(
            f"{row['compute_budget']:12.2e}  "
            f"{row['parameters']:14,.0f}  "
            f"{row['final_loss']:8.4f}"
        )

    out_json = ARTIFACTS / "isoflops_compute_optimal.json"
    out_json.write_text(json.dumps(optimal, indent=2))
    print(f"\nWrote {out_json}")

    # Simple scaling-law sanity: log-log slope of optimal N vs C
    import math

    if len(optimal) >= 2:
        log_c = [math.log(r["compute_budget"]) for r in optimal]
        log_n = [math.log(r["parameters"]) for r in optimal]
        log_l = [math.log(r["final_loss"]) for r in optimal]
        n = len(optimal)
        mean_lc = sum(log_c) / n
        mean_ln = sum(log_n) / n
        mean_ll = sum(log_l) / n
        cov_cn = sum((log_c[i] - mean_lc) * (log_n[i] - mean_ln) for i in range(n))
        var_c = sum((x - mean_lc) ** 2 for x in log_c)
        cov_cl = sum((log_c[i] - mean_lc) * (log_l[i] - mean_ll) for i in range(n))
        slope_n_c = cov_cn / var_c if var_c else float("nan")
        slope_l_c = cov_cl / var_c if var_c else float("nan")
        print()
        print("Log-log slopes along compute-optimal frontier (rough Chinchilla check):")
        print(f"  d log N / d log C ≈ {slope_n_c:.3f}  (Chinchilla ~0.5)")
        print(f"  d log L / d log C ≈ {slope_l_c:.3f}")


if __name__ == "__main__":
    main()

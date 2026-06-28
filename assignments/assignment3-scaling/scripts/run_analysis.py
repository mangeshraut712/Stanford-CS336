#!/usr/bin/env python3
"""Generate all Assignment 3 analysis artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from cs336_scaling.analysis.scaling import (
    compute_optimal_per_budget,
    count_training_config_params,
    estimate_training_flops,
    fit_log_parabolas,
    fit_power_laws,
    load_isoflops,
    predict_loss,
)
from cs336_scaling.training.run import default_training_config

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def write_isoflops_svg(rows: list[dict], optimal: list[dict], path: Path) -> None:
    w, h, margin = 720, 480, 60
    by_b: dict[float, list] = defaultdict(list)
    for r in rows:
        by_b[r["compute_budget"]].append(r)

    all_n = [r["parameters"] for r in rows]
    all_l = [r["final_loss"] for r in rows]
    log_n_min, log_n_max = math.log(min(all_n)), math.log(max(all_n))
    log_l_min, log_l_max = math.log(min(all_l)), math.log(max(all_l))

    def x_px(log_n: float) -> float:
        return margin + (log_n - log_n_min) / (log_n_max - log_n_min) * (w - 2 * margin)

    def y_px(log_l: float) -> float:
        return h - margin - (log_l - log_l_min) / (log_l_max - log_l_min) * (h - 2 * margin)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c", "#65a30d"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="#fafafa"/>',
        f'<text x="{w//2}" y="24" text-anchor="middle" font-size="14" font-family="sans-serif">IsoFLOPs: final loss vs parameters</text>',
    ]
    for i, (budget, pts) in enumerate(sorted(by_b.items())):
        col = colors[i % len(colors)]
        pts = sorted(pts, key=lambda p: p["parameters"])
        d = "M " + " L ".join(
            f"{x_px(math.log(p['parameters'])):.1f},{y_px(math.log(p['final_loss'])):.1f}"
            for p in pts
        )
        lines.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.5"/>')
    for o in optimal:
        lines.append(
            f'<circle cx="{x_px(math.log(o["parameters"])):.1f}" cy="{y_px(math.log(o["final_loss"])):.1f}" r="4" fill="#000"/>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def main() -> None:
    ART.mkdir(exist_ok=True)
    rows = load_isoflops()
    optimal = compute_optimal_per_budget(rows)
    power = fit_power_laws(optimal)
    parabolas = fit_log_parabolas(rows)

    (ART / "isoflops_compute_optimal.json").write_text(json.dumps(optimal, indent=2))
    (ART / "scaling_law_fits.json").write_text(json.dumps(power, indent=2))
    (ART / "isoflops_parabola_fits.json").write_text(json.dumps(parabolas, indent=2))
    write_isoflops_svg(rows, optimal, ART / "isoflops_curves.svg")

    cfg = default_training_config()
    n = count_training_config_params(cfg)
    c = estimate_training_flops(n, cfg.total_train_tokens)
    pred = predict_loss(n, c, parabolas)

    submission = {
        "training_config_summary": {
            "hidden_size": cfg.architecture_config.hidden_size,
            "num_hidden_layers": cfg.architecture_config.num_hidden_layers,
            "total_train_tokens": cfg.total_train_tokens,
            "model_seed": cfg.model_seed,
        },
        "estimated_params": n,
        "estimated_compute_flops": c,
        "predicted_final_loss": round(pred, 3),
        "note": "Self-study prediction from reference isoflops; refit after API runs if enrolled.",
    }
    (ART / "final_submission_prediction.json").write_text(json.dumps(submission, indent=2))
    print(f"Done. Example config: N={n:,}, predicted L≈{pred:.3f}")


if __name__ == "__main__":
    main()

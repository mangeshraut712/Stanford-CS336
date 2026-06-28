"""Scaling-law utilities for Assignment 3 (isoflops analysis)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISO_FLOPS_PATH = ROOT / "data" / "isoflops_curves.json"


def load_isoflops(path: Path | None = None) -> list[dict]:
    with open(path or ISO_FLOPS_PATH) as f:
        return json.load(f)


def compute_optimal_per_budget(rows: list[dict]) -> list[dict]:
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["compute_budget"]].append(row)
    return [
        {
            "compute_budget": budget,
            "parameters": min(pts, key=lambda r: r["final_loss"])["parameters"],
            "final_loss": min(pts, key=lambda r: r["final_loss"])["final_loss"],
        }
        for budget in sorted(by_budget)
        for pts in [by_budget[budget]]
    ]


def log_log_slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    cov = sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx)))
    var = sum((x - mx) ** 2 for x in lx)
    return cov / var if var else float("nan")


def fit_power_laws(optimal: list[dict]) -> dict:
    c_vals = [r["compute_budget"] for r in optimal]
    n_vals = [r["parameters"] for r in optimal]
    l_vals = [r["final_loss"] for r in optimal]
    b = log_log_slope(c_vals, n_vals)
    d = log_log_slope(c_vals, l_vals)
    a = math.exp(sum(math.log(n) - b * math.log(c) for n, c in zip(n_vals, c_vals)) / len(optimal))
    c_coef = math.exp(sum(math.log(l) - d * math.log(c) for l, c in zip(l_vals, c_vals)) / len(optimal))
    return {
        "N_opt_formula": "N_opt ≈ a * C^b",
        "a": a,
        "b": b,
        "chinchilla_expected_b": 0.5,
        "L_opt_formula": "L_opt ≈ c * C^d",
        "c": c_coef,
        "d": d,
    }


def fit_log_parabolas(rows: list[dict]) -> dict[str, dict]:
    """Per compute budget: log L ≈ α (log N)² + β log N + γ."""
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["compute_budget"]].append(row)

    fits: dict[str, dict] = {}
    for budget, pts in sorted(by_budget.items()):
        ln = [math.log(p["parameters"]) for p in pts]
        ll = [math.log(p["final_loss"]) for p in pts]
        n = len(pts)
        s0 = float(n)
        s1 = sum(ln)
        s2 = sum(x * x for x in ln)
        s3 = sum(x * x * x for x in ln)
        s4 = sum(x**4 for x in ln)
        t0 = sum(ll)
        t1 = sum(x * y for x, y in zip(ln, ll))
        t2 = sum(x * x * y for x, y in zip(ln, ll))
        mat = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
        rhs = [t2, t1, t0]

        def det3(m: list[list[float]]) -> float:
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        d_main = det3(mat)
        if abs(d_main) < 1e-12:
            continue
        alpha = det3([[rhs[0], mat[0][1], mat[0][2]], [rhs[1], mat[1][1], mat[1][2]], [rhs[2], mat[2][1], mat[2][2]]]) / d_main
        beta = det3([[mat[0][0], rhs[0], mat[0][2]], [mat[1][0], rhs[1], mat[1][2]], [mat[2][0], rhs[2], mat[2][2]]]) / d_main
        gamma = det3([[mat[0][0], mat[0][1], rhs[0]], [mat[1][0], mat[1][1], rhs[1]], [mat[2][0], mat[2][1], rhs[2]]]) / d_main
        key = f"{budget:.6e}"
        fits[key] = {
            "compute_budget": budget,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_min": min(p["parameters"] for p in pts),
            "n_max": max(p["parameters"] for p in pts),
        }
    return fits


def predict_loss_at_budget(n: float, parabola: dict) -> float:
    ln = math.log(n)
    log_l = parabola["alpha"] * ln**2 + parabola["beta"] * ln + parabola["gamma"]
    return math.exp(log_l)


def nearest_budget_key(compute: float, parabolas: dict[str, dict]) -> str:
    budgets = [(v["compute_budget"], k) for k, v in parabolas.items()]
    return min(budgets, key=lambda x: abs(math.log(x[0]) - math.log(compute)))[1]


def predict_loss(n: float, compute: float, parabolas: dict[str, dict]) -> float:
    key = nearest_budget_key(compute, parabolas)
    p = parabolas[key]
    n_clamped = max(p["n_min"], min(n, p["n_max"]))
    return predict_loss_at_budget(n_clamped, p)


def estimate_training_flops(n_params: int, train_tokens: int) -> float:
    return 6.0 * n_params * train_tokens


def count_training_config_params(config) -> int:
    import jax

    from cs336_scaling.training.model.basic_model import BasicCausalLM
    from cs336_scaling.training.model.jax_utils import count_params

    model = BasicCausalLM(config.architecture_config, key=jax.random.key(0))
    return count_params(model)

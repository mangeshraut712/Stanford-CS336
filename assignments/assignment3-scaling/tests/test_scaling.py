"""Unit tests for scaling-law helpers and the FSDP loss reduction."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from cs336_scaling.analysis.scaling import (
    compute_optimal_per_budget,
    estimate_training_flops,
    fit_power_laws,
    log_log_slope,
    nearest_budget_key,
    predict_loss,
    predict_loss_at_budget,
)
from cs336_scaling.training.loop import fsdp_mean


def test_log_log_slope_of_a_square_root() -> None:
    xs = [1.0, 4.0, 16.0, 64.0]
    ys = [math.sqrt(x) for x in xs]
    assert log_log_slope(xs, ys) == pytest.approx(0.5)


def test_optimal_point_is_the_minimum_loss_at_each_budget() -> None:
    rows = [
        {"compute_budget": 1e6, "parameters": 10, "final_loss": 3.0},
        {"compute_budget": 1e6, "parameters": 20, "final_loss": 2.0},
        {"compute_budget": 1e8, "parameters": 30, "final_loss": 1.5},
        {"compute_budget": 1e8, "parameters": 40, "final_loss": 1.7},
    ]
    optimal = compute_optimal_per_budget(rows)
    assert [point["parameters"] for point in optimal] == [20, 30]
    assert [point["final_loss"] for point in optimal] == [2.0, 1.5]


def test_power_law_fit_recovers_known_exponents() -> None:
    optimal = []
    for compute in (1e6, 1e8, 1e10, 1e12):
        optimal.append(
            {
                "compute_budget": compute,
                "parameters": 2.0 * compute**0.5,
                "final_loss": 3.0 * compute**-0.25,
            }
        )
    fit = fit_power_laws(optimal)
    assert fit["b"] == pytest.approx(0.5)
    assert fit["d"] == pytest.approx(-0.25)
    assert fit["a"] == pytest.approx(2.0)
    assert fit["c"] == pytest.approx(3.0)


def test_predict_loss_clamps_to_the_fitted_parameter_range() -> None:
    parabola = {
        "alpha": 0.0,
        "beta": 1.0,
        "gamma": 0.0,
        "n_min": 10.0,
        "n_max": 100.0,
        "compute_budget": 1e6,
    }
    parabolas = {"1.000000e+06": parabola}
    assert nearest_budget_key(1e6, parabolas) == "1.000000e+06"
    assert predict_loss_at_budget(math.e, parabola) == pytest.approx(math.e)
    # n=1 is below n_min, so the prediction uses n_min.
    assert predict_loss(1.0, 1e6, parabolas) == pytest.approx(10.0)


def test_training_flops_use_the_six_nd_approximation() -> None:
    assert estimate_training_flops(100, 50) == 30_000


def test_fsdp_mean_divides_by_the_global_example_count() -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("need two CPU devices to see the cross-device reduction")
    mesh = jax.make_mesh((2,), ("fsdp",))
    losses = jax.device_put(
        jnp.arange(1, 5, dtype=jnp.float32),
        NamedSharding(mesh, P("fsdp")),
    )

    def reduce(values: jax.Array) -> jax.Array:
        return fsdp_mean(values)

    sharded = jax.shard_map(reduce, mesh=mesh, in_specs=P("fsdp"), out_specs=P())
    assert float(sharded(losses)) == pytest.approx(2.5)

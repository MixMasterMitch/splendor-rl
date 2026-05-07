"""Basic integration test for the benchmark script."""

from __future__ import annotations

import torch

from agent.scripts.benchmark import run_benchmark

EXPECTED_KEYS = {
    "selfplay_games_per_s",
    "selfplay_wall_s",
    "learner_steps_per_s",
    "learner_wall_s",
    "eval_games_per_s",
    "eval_wall_s",
}


def test_run_benchmark_cpu_returns_valid_structure() -> None:
    """run_benchmark on CPU with small workloads returns the expected dict
    structure and all positive float values."""
    results = run_benchmark(
        num_games=16,
        num_sims=2,
        learner_steps=5,
        eval_games=16,
        hidden=32,
        arch="flat",
        devices=["cpu"],
    )

    assert "cpu" in results
    cpu_metrics = results["cpu"]
    assert set(cpu_metrics.keys()) == EXPECTED_KEYS

    for key, val in cpu_metrics.items():
        assert isinstance(val, float), f"{key} should be a float, got {type(val)}"
        assert val > 0, f"{key} should be positive, got {val}"

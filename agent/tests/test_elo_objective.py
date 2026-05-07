"""Tests for compute_elo_objective in agent/scripts/tune.py."""

from __future__ import annotations

from unittest.mock import patch

import torch

from agent.net.model import SplendorNet
from agent.scripts.tune import compute_elo_objective


def test_compute_elo_objective_returns_finite_float():
    """Smoke test: compute_elo_objective returns a finite float Elo rating."""
    net = SplendorNet(hidden=32, arch="flat")
    net.eval()

    # Use very small num_games and num_sims for speed
    elo = compute_elo_objective(
        net,
        num_players=2,
        num_games=4,
        num_sims=1,
        device="cpu",
        seed=123,
    )
    assert isinstance(elo, float)
    assert not torch.tensor(elo).isnan().item()
    assert not torch.tensor(elo).isinf().item()


def test_compute_elo_objective_uses_correct_anchors():
    """Verify that fit_anchored_ratings is called with the correct anchors."""
    net = SplendorNet(hidden=32, arch="flat")
    net.eval()

    captured_calls: list[dict] = []
    original_fit = None

    from agent.train import ranking as RK

    original_fit = RK.fit_anchored_ratings

    def spy_fit(results, anchors=None, **kwargs):
        captured_calls.append({"results": results, "anchors": anchors})
        return original_fit(results, anchors=anchors, **kwargs)

    with patch("agent.train.ranking.fit_anchored_ratings", side_effect=spy_fit):
        elo = compute_elo_objective(
            net,
            num_players=2,
            num_games=4,
            num_sims=1,
            device="cpu",
            seed=42,
        )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["anchors"] == {"random": 1000, "heuristic": 2500}
    assert isinstance(elo, float)


def test_compute_elo_objective_includes_all_standard_opponents():
    """Verify match results include random, heuristic, and heuristic_opus."""
    net = SplendorNet(hidden=32, arch="flat")
    net.eval()

    captured_calls: list[dict] = []

    from agent.train import ranking as RK

    original_fit = RK.fit_anchored_ratings

    def spy_fit(results, anchors=None, **kwargs):
        captured_calls.append({"results": list(results), "anchors": anchors})
        return original_fit(results, anchors=anchors, **kwargs)

    with patch("agent.train.ranking.fit_anchored_ratings", side_effect=spy_fit):
        compute_elo_objective(
            net,
            num_players=2,
            num_games=4,
            num_sims=1,
            device="cpu",
            seed=42,
        )

    results = captured_calls[0]["results"]
    # Collect all opponent names from match results
    opponent_names = set()
    for r in results:
        for name in (r["a"], r["b"]):
            if name != "trial_agent":
                opponent_names.add(name)

    assert "random" in opponent_names
    assert "heuristic" in opponent_names
    assert "heuristic_opus" in opponent_names


def test_compute_elo_objective_rating_between_anchors_or_beyond():
    """The Elo rating should be a reasonable number (not zero or NaN)."""
    net = SplendorNet(hidden=32, arch="flat")
    net.eval()

    elo = compute_elo_objective(
        net,
        num_players=2,
        num_games=8,
        num_sims=1,
        device="cpu",
        seed=99,
    )
    # A random untrained net should get some Elo rating; it won't be exactly 0
    # unless something went wrong with the fitting.
    assert isinstance(elo, float)
    # The rating should be in a reasonable range (not degenerate)
    assert -5000 < elo < 10000

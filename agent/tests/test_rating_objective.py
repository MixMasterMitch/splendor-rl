"""Tests for compute_rating_objective in agent/scripts/tune.py.

Uses a module-scoped fixture to run the expensive game simulation once and
share the results across all assertions.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from agent.net.model import SplendorNet
from agent.scripts.tune import compute_rating_objective


@pytest.fixture(scope="module")
def rating_result() -> dict:
    """Run compute_rating_objective once with a spy on fit_anchored_ratings.

    Returns a dict with 'rating', 'anchors', and 'results' (match results).
    """
    net = SplendorNet(hidden=32, arch="flat")
    net.eval()

    captured_calls: list[dict] = []

    from agent.train import ranking as RK

    original_fit = RK.fit_anchored_ratings

    def spy_fit(results, anchors=None, **kwargs):
        captured_calls.append({"results": list(results), "anchors": anchors})
        return original_fit(results, anchors=anchors, **kwargs)

    with patch("agent.train.ranking.fit_anchored_ratings", side_effect=spy_fit):
        rating = compute_rating_objective(
            net,
            num_players=2,
            num_games=4,
            num_sims=1,
            device="cpu",
            seed=42,
        )

    assert len(captured_calls) == 1
    return {
        "rating": rating,
        "anchors": captured_calls[0]["anchors"],
        "results": captured_calls[0]["results"],
    }


def test_compute_rating_objective_returns_finite_float(rating_result: dict) -> None:
    """Smoke test: compute_rating_objective returns a finite float rating."""
    rating = rating_result["rating"]
    assert isinstance(rating, float)
    assert not torch.tensor(rating).isnan().item()
    assert not torch.tensor(rating).isinf().item()


def test_compute_rating_objective_uses_correct_anchors(rating_result: dict) -> None:
    """Verify that fit_anchored_ratings is called with the correct anchors."""
    assert rating_result["anchors"] == {"random": 1000}
    assert isinstance(rating_result["rating"], float)


def test_compute_rating_objective_includes_all_standard_opponents(rating_result: dict) -> None:
    """Verify match results include random, heuristic, and heuristic_opus."""
    results = rating_result["results"]
    opponent_names = set()
    for r in results:
        for name in (r["a"], r["b"]):
            if name != "trial_agent":
                opponent_names.add(name)

    assert "random" in opponent_names
    assert "heuristic" in opponent_names
    assert "heuristic_opus" in opponent_names


def test_compute_rating_objective_rating_between_anchors_or_beyond(rating_result: dict) -> None:
    """The rating should be a reasonable number (not zero or NaN)."""
    rating = rating_result["rating"]
    assert isinstance(rating, float)
    # A random untrained net should get some rating; it won't be exactly 0
    # unless something went wrong with the fitting.
    assert -5000 < rating < 10000

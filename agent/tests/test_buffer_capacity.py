"""Tests for _validate_buffer_capacity in agent/train/loop.py."""

from __future__ import annotations

import json
import tempfile

from agent.obs.run import Run
from agent.train.loop import LoopConfig, _validate_buffer_capacity


def _make_run(tmp_path: str) -> Run:
    """Create a Run in a temporary directory."""
    return Run("test_buffer", runs_root=tmp_path)


def _read_events(run: Run) -> list[dict]:
    """Read all events from the run's events log."""
    if not run.events_path.exists():
        return []
    events = []
    with open(run.events_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class TestValidateBufferCapacity:
    """Tests for the _validate_buffer_capacity advisory warning."""

    def test_warns_when_capacity_too_small(self, tmp_path) -> None:
        """Should log WARN when replay_capacity < selfplay_games * 200."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=4096, replay_capacity=100_000)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert len(warn_events) == 1
        assert warn_events[0]["lvl"] == "WARN"
        assert warn_events[0]["fields"]["replay_capacity"] == 100_000
        assert warn_events[0]["fields"]["selfplay_games"] == 4096
        assert warn_events[0]["fields"]["estimated_max_samples"] == 4096 * 200
        run.close()

    def test_no_warning_when_capacity_sufficient(self, tmp_path) -> None:
        """Should not log warning when replay_capacity >= selfplay_games * 200."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=4096, replay_capacity=4096 * 200)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert len(warn_events) == 0
        run.close()

    def test_no_warning_when_capacity_exceeds_threshold(self, tmp_path) -> None:
        """Should not log warning when replay_capacity > selfplay_games * 200."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=512, replay_capacity=600_000)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert len(warn_events) == 0
        run.close()

    def test_boundary_exact_threshold_no_warning(self, tmp_path) -> None:
        """Exactly at the threshold (replay_capacity == selfplay_games * 200) should not warn."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=100, replay_capacity=20_000)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert len(warn_events) == 0
        run.close()

    def test_boundary_one_below_threshold_warns(self, tmp_path) -> None:
        """One below the threshold should warn."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=100, replay_capacity=19_999)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert len(warn_events) == 1
        run.close()

    def test_does_not_raise(self, tmp_path) -> None:
        """Advisory only — should never raise even with very small capacity."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=4096, replay_capacity=1)
        # Should not raise
        _validate_buffer_capacity(cfg, run)
        run.close()

    def test_recommendation_message(self, tmp_path) -> None:
        """Warning should include a recommendation string."""
        run = _make_run(str(tmp_path))
        cfg = LoopConfig(selfplay_games=4096, replay_capacity=100_000)
        _validate_buffer_capacity(cfg, run)

        events = _read_events(run)
        warn_events = [e for e in events if e["event"] == "buffer_capacity_warning"]
        assert "Consider replay_capacity >= 819200" in warn_events[0]["fields"]["recommendation"]
        run.close()

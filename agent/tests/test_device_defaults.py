"""Tests for apply_device_defaults in agent/train/loop.py."""

from __future__ import annotations

import dataclasses

from agent.train.loop import LoopConfig, apply_device_defaults


class TestApplyDeviceDefaultsCPU:
    """When device is 'cpu', all fields should remain unchanged."""

    def test_cpu_returns_unchanged_defaults(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cpu")
        assert result == LoopConfig()

    def test_cpu_does_not_mutate_input(self) -> None:
        cfg = LoopConfig()
        original = dataclasses.replace(cfg)
        apply_device_defaults(cfg, "cpu")
        assert cfg == original

    def test_cpu_returns_new_instance(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cpu")
        assert result is not cfg

    def test_cpu_preserves_custom_values(self) -> None:
        cfg = LoopConfig(selfplay_games=1024, learner_batch=512)
        result = apply_device_defaults(cfg, "cpu")
        assert result.selfplay_games == 1024
        assert result.learner_batch == 512


class TestApplyDeviceDefaultsCUDA:
    """When device starts with 'cuda', GPU defaults should be applied."""

    def test_cuda_applies_gpu_defaults_to_fresh_config(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.selfplay_games == 4096
        assert result.selfplay_sims == 32
        assert result.learner_batch == 4096
        assert result.replay_capacity == 820_000
        assert result.learner_steps_per_iter == 64

    def test_cuda_bare_device_string(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cuda")
        assert result.selfplay_games == 4096

    def test_cuda1_device_string(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cuda:1")
        assert result.selfplay_games == 4096

    def test_cuda_does_not_mutate_input(self) -> None:
        cfg = LoopConfig()
        original = dataclasses.replace(cfg)
        apply_device_defaults(cfg, "cuda:0")
        assert cfg == original

    def test_cuda_returns_new_instance(self) -> None:
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cuda:0")
        assert result is not cfg

    def test_cuda_preserves_non_overridden_fields(self) -> None:
        """Fields not in the GPU defaults map should remain at their values."""
        cfg = LoopConfig()
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.num_players == cfg.num_players
        assert result.checkpoint_every == cfg.checkpoint_every
        assert result.lr == cfg.lr
        assert result.max_iters == cfg.max_iters


class TestCLIOverridesSurvive:
    """CLI-provided values (differing from defaults) must be preserved."""

    def test_custom_selfplay_games_preserved(self) -> None:
        cfg = LoopConfig(selfplay_games=2048)
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.selfplay_games == 2048
        # Other GPU defaults should still apply
        assert result.learner_batch == 4096

    def test_custom_learner_batch_preserved(self) -> None:
        cfg = LoopConfig(learner_batch=8192)
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.learner_batch == 8192

    def test_custom_replay_capacity_preserved(self) -> None:
        cfg = LoopConfig(replay_capacity=1_000_000)
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.replay_capacity == 1_000_000

    def test_all_fields_overridden(self) -> None:
        """When every GPU-default field is overridden, none should change."""
        cfg = LoopConfig(
            selfplay_games=1024,
            selfplay_sims=16,
            learner_batch=512,
            replay_capacity=100_000,
            learner_steps_per_iter=10,
        )
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.selfplay_games == 1024
        assert result.selfplay_sims == 16
        assert result.learner_batch == 512
        assert result.replay_capacity == 100_000
        assert result.learner_steps_per_iter == 10

    def test_custom_learner_steps_preserved(self) -> None:
        cfg = LoopConfig(learner_steps_per_iter=100)
        result = apply_device_defaults(cfg, "cuda:0")
        assert result.learner_steps_per_iter == 100

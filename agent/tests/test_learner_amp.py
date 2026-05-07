"""Tests for learner step with and without AMP (GradScaler) support."""

from __future__ import annotations

import pytest
import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.net import encoder as ENC
from agent.net import model as M
from agent.train.learner import make_optimizer, step as learner_step
from agent.train.replay_buffer import ReplayBuffer

HAS_CUDA = torch.cuda.is_available()

EXPECTED_METRIC_KEYS = {"loss", "policy_loss", "value_loss", "entropy", "grad_norm"}


def _fill_buffer(buffer: ReplayBuffer, n: int = 32) -> None:
    """Add *n* synthetic samples to *buffer*."""
    global_feat = torch.randn(n, ENC.D_GLOBAL)
    card_feat = torch.randn(n, ENC.N_CARDS, ENC.D_CARD)
    # Mark several actions as legal so the policy has support.
    legal_mask = torch.zeros((n, A.NUM_ACTIONS), dtype=torch.bool)
    legal_mask[:, :5] = True
    # Policy concentrated on legal actions only.
    policy = torch.zeros(n, A.NUM_ACTIONS)
    policy[:, :5] = torch.rand(n, 5).clamp_min(0.01)
    value = torch.randn(n, BE.MAX_PLAYERS)
    buffer.add_batch(global_feat, card_feat, legal_mask, policy, value)


# ---------------------------------------------------------------------------
# CPU tests (always run)
# ---------------------------------------------------------------------------


def test_learner_step_cpu_no_scaler_returns_expected_keys() -> None:
    """A learner step on CPU with grad_scaler=None returns the standard metric keys."""
    torch.manual_seed(0)
    net = M.SplendorNet(hidden=32, arch="flat")
    optim = make_optimizer(net, lr=1e-3)
    buffer = ReplayBuffer(capacity=64, device="cpu")
    _fill_buffer(buffer)

    metrics = learner_step(
        net,
        buffer,
        optim,
        batch_size=8,
        device="cpu",
        grad_scaler=None,
    )

    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    for key in EXPECTED_METRIC_KEYS:
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key])), f"{key} is not finite"


# ---------------------------------------------------------------------------
# CUDA tests (skipped when no GPU is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_learner_step_cuda_with_grad_scaler_returns_expected_keys() -> None:
    """A learner step on CUDA with a GradScaler returns the same metric keys and finite loss."""
    torch.manual_seed(0)
    net = M.SplendorNet(hidden=32, arch="flat").to("cuda")
    optim = make_optimizer(net, lr=1e-3)
    buffer = ReplayBuffer(capacity=64, device="cuda")
    _fill_buffer(buffer)

    grad_scaler = torch.amp.GradScaler("cuda")
    metrics = learner_step(
        net,
        buffer,
        optim,
        batch_size=8,
        device="cuda",
        grad_scaler=grad_scaler,
    )

    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    for key in EXPECTED_METRIC_KEYS:
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key])), f"{key} is not finite"

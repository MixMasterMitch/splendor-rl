"""Tests for cross-device checkpoint portability and CUDA RNG state handling."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest
import torch

from agent.net import model as M
from agent.train import checkpointing as CK
from agent.train.replay_buffer import ReplayBuffer

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_net(hidden: int = 32, arch: str = "flat") -> M.SplendorNet:
    torch.manual_seed(42)
    return M.SplendorNet(hidden=hidden, arch=arch)


def _save_on_device(
    tmp_path: pathlib.Path,
    device: str,
    *,
    hidden: int = 32,
    arch: str = "flat",
) -> pathlib.Path:
    """Create a net on *device*, save a checkpoint, and return the path."""
    net = _make_net(hidden=hidden, arch=arch).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    path = tmp_path / f"ckpt_{device.replace(':', '_')}.pt"
    CK.save_checkpoint(
        path,
        net,
        optim=optim,
        iteration=1,
        config={"num_players": 2, "device": device},
    )
    return path


# ---------------------------------------------------------------------------
# CPU-only tests (always run)
# ---------------------------------------------------------------------------


def test_cpu_checkpoint_has_no_cuda_rng_key(tmp_path: pathlib.Path) -> None:
    """A checkpoint saved from a CPU net must NOT contain 'cuda_rng'."""
    path = _save_on_device(tmp_path, "cpu")
    payload = CK.load_checkpoint_payload(path, map_location="cpu")
    assert "cuda_rng" not in payload


def test_cpu_to_cpu_load_preserves_parameters(tmp_path: pathlib.Path) -> None:
    """Save on CPU, load on CPU — all parameters must match."""
    net_orig = _make_net()
    optim = torch.optim.AdamW(net_orig.parameters(), lr=1e-3)
    path = tmp_path / "cpu_ckpt.pt"
    CK.save_checkpoint(
        path, net_orig, optim=optim, iteration=5, config={"num_players": 2}
    )

    net_loaded = _make_net()
    # Randomise weights so we can confirm they get overwritten.
    for p in net_loaded.parameters():
        p.data.uniform_(-1, 1)

    CK.load_checkpoint(path, net_loaded, map_location="cpu")

    for key in net_orig.state_dict():
        assert torch.equal(
            net_orig.state_dict()[key], net_loaded.state_dict()[key]
        ), f"mismatch on {key}"


def test_load_cpu_checkpoint_with_cuda_rng_on_cpu_skips_gracefully(
    tmp_path: pathlib.Path,
) -> None:
    """If a payload contains 'cuda_rng' but we load on CPU, no error is raised."""
    net = _make_net()
    path = tmp_path / "fake_cuda_ckpt.pt"
    # Manually craft a payload with a cuda_rng key.
    payload = {
        "net": net.state_dict(),
        "optim": None,
        "iteration": 1,
        "config": {"hidden": 32, "arch": "flat", "num_players": 2},
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.ByteTensor([0] * 16),  # dummy
    }
    torch.save(payload, path)

    net_loaded = _make_net()
    # Should not raise even though cuda_rng is present.
    CK.load_checkpoint(path, net_loaded, map_location="cpu")

    for key in net.state_dict():
        assert torch.equal(
            net.state_dict()[key], net_loaded.state_dict()[key]
        ), f"mismatch on {key}"


# ---------------------------------------------------------------------------
# CUDA tests (skipped when no GPU is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_checkpoint_contains_cuda_rng_key(tmp_path: pathlib.Path) -> None:
    """A checkpoint saved from a CUDA net must contain 'cuda_rng'."""
    path = _save_on_device(tmp_path, "cuda")
    payload = CK.load_checkpoint_payload(path, map_location="cpu")
    assert "cuda_rng" in payload
    assert isinstance(payload["cuda_rng"], torch.Tensor)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cpu_to_cuda_load_moves_all_parameters(tmp_path: pathlib.Path) -> None:
    """Save on CPU, load on CUDA — every parameter must reside on CUDA."""
    path = _save_on_device(tmp_path, "cpu")
    net_loaded = _make_net().to("cuda")
    CK.load_checkpoint(path, net_loaded, map_location="cuda")

    for name, param in net_loaded.named_parameters():
        assert param.device.type == "cuda", f"{name} not on CUDA: {param.device}"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_to_cpu_load_moves_all_parameters(tmp_path: pathlib.Path) -> None:
    """Save on CUDA, load on CPU — every parameter must reside on CPU."""
    path = _save_on_device(tmp_path, "cuda")
    net_loaded = _make_net()
    CK.load_checkpoint(path, net_loaded, map_location="cpu")

    for name, param in net_loaded.named_parameters():
        assert param.device.type == "cpu", f"{name} not on CPU: {param.device}"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_to_cuda_restores_rng_state(tmp_path: pathlib.Path) -> None:
    """Save on CUDA with RNG state, load on CUDA — RNG state must be restored."""
    net = _make_net().to("cuda")
    # Advance the CUDA RNG so the state is non-trivial.
    torch.randn(100, device="cuda")
    rng_before_save = torch.cuda.get_rng_state()

    path = tmp_path / "cuda_rng_ckpt.pt"
    CK.save_checkpoint(
        path, net, optim=None, iteration=1, config={"num_players": 2}
    )

    # Advance RNG further so it differs from the saved state.
    torch.randn(500, device="cuda")
    rng_after_advance = torch.cuda.get_rng_state()
    assert not torch.equal(rng_before_save, rng_after_advance)

    net_loaded = _make_net().to("cuda")
    CK.load_checkpoint(path, net_loaded, map_location="cuda")

    rng_after_load = torch.cuda.get_rng_state()
    assert torch.equal(rng_before_save, rng_after_load)

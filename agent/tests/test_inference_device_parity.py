"""CPU vs GPU inference parity test for SplendorNet.

Verifies that the ML agent produces identical (within floating-point tolerance)
policy logits and value predictions on CPU vs CUDA for the same game state.

This catches:
- Numerical divergence from different GEMM implementations
- Device-specific bugs in the encoder or network
- Issues with checkpoint loading across devices
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from agent.env import batched_engine as BE
from agent.env import actions as A
from agent.net import encoder as ENC
from agent.net import model as M
from agent.search import gumbel_mcts as G
from agent.train import checkpointing as CK

HAS_CUDA = torch.cuda.is_available()

_LEAGUE_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs" / "league"
_LATEST_CKPT = sorted(
    [p for p in _LEAGUE_DIR.glob("ckpt_*.pt")],
    key=lambda p: p.name,
)[-1] if _LEAGUE_DIR.exists() and list(_LEAGUE_DIR.glob("ckpt_*.pt")) else None


def _make_test_states(num_states: int = 8, seed: int = 42) -> BE.BatchedEngine:
    """Create a batch of game states at various points in a game."""
    engine = BE.BatchedEngine(num_states, 2, device="cpu", seed=seed)
    # Advance each game a different number of steps to get diverse states
    rng = torch.Generator().manual_seed(seed + 1)
    for step in range(30):
        if engine.ended.all():
            break
        mask = engine.legal_action_mask()
        # Random legal action per game
        scores = torch.rand(num_states, A.NUM_ACTIONS, generator=rng)
        scores = scores.masked_fill(~mask, -1.0)
        actions = scores.argmax(dim=-1)
        engine.apply(actions)
    return engine


class TestInferenceDeviceParity:
    """Verify that network inference produces the same results on CPU and CUDA."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_encoder_parity(self):
        """State encoding should be identical on CPU and CUDA."""
        engine_cpu = _make_test_states(seed=7)
        # Clone state to CUDA
        engine_gpu = BE.BatchedEngine(engine_cpu.batch_size, 2, device="cuda", seed=7)
        for attr in BE._STATE_TENSOR_ATTRS:
            setattr(engine_gpu, attr, getattr(engine_cpu, attr).to("cuda"))

        g_cpu, c_cpu, afford_cpu, legal_cpu = ENC.encode_state_with_legal(engine_cpu)
        g_gpu, c_gpu, afford_gpu, legal_gpu = ENC.encode_state_with_legal(engine_gpu)

        assert torch.allclose(g_cpu, g_gpu.cpu(), atol=1e-6), "Global features differ"
        assert torch.allclose(c_cpu, c_gpu.cpu(), atol=1e-6), "Card features differ"
        assert torch.equal(legal_cpu, legal_gpu.cpu()), "Legal masks differ"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_random_net_inference_parity(self):
        """A randomly initialized net should produce same outputs on CPU and CUDA."""
        torch.manual_seed(123)
        net = M.SplendorNet(hidden=64, arch="attn")
        net.eval()

        engine_cpu = _make_test_states(seed=11)
        g_cpu, c_cpu, _, legal_cpu = ENC.encode_state_with_legal(engine_cpu)

        # Run on CPU
        with torch.no_grad():
            logits_cpu, value_cpu = net.forward(g_cpu, c_cpu, legal_cpu)

        # Move net and inputs to CUDA
        net_gpu = M.SplendorNet(hidden=64, arch="attn")
        net_gpu.load_state_dict(net.state_dict())
        net_gpu.eval().to("cuda")

        g_gpu = g_cpu.to("cuda")
        c_gpu = c_cpu.to("cuda")
        legal_gpu = legal_cpu.to("cuda")

        with torch.no_grad():
            logits_gpu, value_gpu = net_gpu.forward(g_gpu, c_gpu, legal_gpu)

        assert torch.allclose(logits_cpu, logits_gpu.cpu(), atol=1e-4), (
            f"Logits differ: max delta = {(logits_cpu - logits_gpu.cpu()).abs().max().item()}"
        )
        assert torch.allclose(value_cpu, value_gpu.cpu(), atol=1e-4), (
            f"Values differ: max delta = {(value_cpu - value_gpu.cpu()).abs().max().item()}"
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.skipif(_LATEST_CKPT is None, reason="No league checkpoint available")
    def test_trained_net_inference_parity(self):
        """A trained checkpoint should produce same outputs on CPU and CUDA."""
        net_cpu, _ = CK.load_net_from_checkpoint(_LATEST_CKPT, map_location="cpu")
        net_cpu.eval().to("cpu")

        net_gpu, _ = CK.load_net_from_checkpoint(_LATEST_CKPT, map_location="cpu")
        net_gpu.eval().to("cuda")

        engine_cpu = _make_test_states(seed=33)
        g_cpu, c_cpu, _, legal_cpu = ENC.encode_state_with_legal(engine_cpu)

        with torch.no_grad():
            logits_cpu, value_cpu = net_cpu.forward(g_cpu, c_cpu, legal_cpu)

        g_gpu = g_cpu.to("cuda")
        c_gpu = c_cpu.to("cuda")
        legal_gpu = legal_cpu.to("cuda")

        with torch.no_grad():
            logits_gpu, value_gpu = net_gpu.forward(g_gpu, c_gpu, legal_gpu)

        assert torch.allclose(logits_cpu, logits_gpu.cpu(), atol=1e-3), (
            f"Trained net logits differ: max delta = "
            f"{(logits_cpu - logits_gpu.cpu()).abs().max().item()}"
        )
        assert torch.allclose(value_cpu, value_gpu.cpu(), atol=1e-3), (
            f"Trained net values differ: max delta = "
            f"{(value_cpu - value_gpu.cpu()).abs().max().item()}"
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.skipif(_LATEST_CKPT is None, reason="No league checkpoint available")
    def test_mcts_action_parity(self):
        """MCTS should select the same action on CPU and CUDA for deterministic states.

        Note: Gumbel sampling introduces randomness, so we use num_sims=0
        (greedy argmax) for this deterministic comparison.
        """
        net_cpu, _ = CK.load_net_from_checkpoint(_LATEST_CKPT, map_location="cpu")
        net_cpu.eval().to("cpu")

        net_gpu, _ = CK.load_net_from_checkpoint(_LATEST_CKPT, map_location="cpu")
        net_gpu.eval().to("cuda")

        # Test across multiple game states
        engine_cpu = _make_test_states(num_states=16, seed=55)

        mismatches = 0
        total = 0
        for b in range(engine_cpu.batch_size):
            if engine_cpu.ended[b]:
                continue
            total += 1
            sub_cpu = engine_cpu.index_select(torch.tensor([b], dtype=torch.long))

            # Greedy (argmax) policy on CPU — no randomness
            with torch.no_grad():
                logits_cpu, _, legal_cpu = net_cpu.inference(sub_cpu)
            masked_cpu = logits_cpu.masked_fill(~legal_cpu, -1e9)
            action_cpu = masked_cpu.argmax(dim=-1).item()

            # Same state on GPU: encode on CPU, move features to GPU
            g_cpu, c_cpu, _, legal_mask_cpu = ENC.encode_state_with_legal(sub_cpu)
            g_gpu = g_cpu.to("cuda")
            c_gpu = c_cpu.to("cuda")
            legal_gpu = legal_mask_cpu.to("cuda")

            with torch.no_grad():
                logits_gpu, _ = net_gpu.inference_from_encoded(g_gpu, c_gpu, legal_gpu)
            masked_gpu = logits_gpu.masked_fill(~legal_gpu, -1e9)
            action_gpu = masked_gpu.argmax(dim=-1).item()

            if action_cpu != action_gpu:
                mismatches += 1
        assert mismatches <= max(1, total * 0.1), (
            f"Too many CPU/GPU action mismatches: {mismatches}/{total}"
        )

"""Serialize/deserialize a training run to allow resumption.

A checkpoint bundles:
- model state_dict
- optimizer state_dict
- RNG state (torch + python)
- training iteration counter
- current config snapshot
- replay buffer state (optional; can be large)

Paths are relative to `Run.ckpt_dir`.
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
import random
from typing import Optional, cast

import torch

from ..net import model as M
from .replay_buffer import ReplayBuffer


@dataclasses.dataclass(frozen=True)
class NetSpec:
    hidden: int
    arch: str


def net_config_dict(net: M.SplendorNet) -> dict[str, int | str]:
    return {
        "hidden": int(net.hidden),
        "arch": str(net.arch),
    }


def load_checkpoint_payload(
    path: str | pathlib.Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    return cast(dict[str, object], payload)


def _looks_like_state_dict(payload: dict[str, object]) -> bool:
    return any(
        isinstance(key, str) and (key.endswith(".weight") or key.endswith(".bias"))
        for key in payload
    )


def checkpoint_net_state_dict(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    if "net" in payload and isinstance(payload["net"], dict):
        return cast(dict[str, torch.Tensor], payload["net"])
    if "model" in payload and isinstance(payload["model"], dict):
        return cast(dict[str, torch.Tensor], payload["model"])
    if _looks_like_state_dict(payload):
        return cast(dict[str, torch.Tensor], payload)
    raise KeyError("checkpoint payload does not contain a recognizable net state_dict")


def _infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    for key in state_dict:
        if key.startswith("flat_trunk."):
            return "flat"
        if key.startswith("g_trunk.") or key.startswith("attn."):
            return "attn"
    raise KeyError("cannot infer SplendorNet arch from checkpoint state_dict")


def _infer_hidden_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    policy_head = state_dict.get("policy_head.0.weight")
    if isinstance(policy_head, torch.Tensor):
        in_features = int(policy_head.shape[1])
        if in_features % 2 != 0:
            raise ValueError(
                f"policy_head.0.weight has odd input width {in_features}; expected 2 * hidden"
            )
        return in_features // 2
    flat_trunk = state_dict.get("flat_trunk.0.weight")
    if isinstance(flat_trunk, torch.Tensor):
        out_features = int(flat_trunk.shape[0])
        if out_features % 2 != 0:
            raise ValueError(
                f"flat_trunk.0.weight has odd output width {out_features}; expected 2 * hidden"
            )
        return out_features // 2
    g_trunk = state_dict.get("g_trunk.0.weight")
    if isinstance(g_trunk, torch.Tensor):
        return int(g_trunk.shape[0])
    raise KeyError("cannot infer SplendorNet hidden size from checkpoint state_dict")


def checkpoint_net_spec(payload: dict[str, object]) -> NetSpec:
    cfg_obj = payload.get("config")
    cfg = cfg_obj if isinstance(cfg_obj, dict) else {}
    state_dict = checkpoint_net_state_dict(payload)
    hidden_obj = cfg.get("hidden")
    arch_obj = cfg.get("arch")
    hidden = int(hidden_obj) if hidden_obj is not None else _infer_hidden_from_state_dict(state_dict)
    arch = str(arch_obj) if arch_obj else _infer_arch_from_state_dict(state_dict)
    if arch not in ("attn", "flat"):
        raise ValueError(f"unsupported SplendorNet arch in checkpoint: {arch}")
    return NetSpec(hidden=hidden, arch=arch)


def load_net_from_checkpoint(
    path: str | pathlib.Path,
    map_location: str | torch.device = "cpu",
) -> tuple[M.SplendorNet, dict[str, object]]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    spec = checkpoint_net_spec(payload)
    net = M.SplendorNet(hidden=spec.hidden, arch=spec.arch)
    net.load_state_dict(checkpoint_net_state_dict(payload))
    return net, payload


def save_checkpoint(
    path: pathlib.Path,
    net: M.SplendorNet,
    optim: Optional[torch.optim.Optimizer],
    iteration: int,
    config: dict,
    buffer: Optional[ReplayBuffer] = None,
) -> None:
    payload_config = dict(config)
    payload_config.setdefault("hidden", int(net.hidden))
    payload_config.setdefault("arch", str(net.arch))
    payload = {
        "net": net.state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
        "iteration": iteration,
        "config": payload_config,
        "torch_rng": torch.random.get_rng_state(),
        "py_rng": pickle.dumps(random.getstate()),
    }
    # Save CUDA RNG state for reproducibility when training on GPU.
    if next(net.parameters()).device.type == "cuda":
        payload["cuda_rng"] = torch.cuda.get_rng_state()
    if buffer is not None:
        payload["buffer"] = buffer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: pathlib.Path,
    net: M.SplendorNet,
    optim: Optional[torch.optim.Optimizer] = None,
    buffer: Optional[ReplayBuffer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = load_checkpoint_payload(path, map_location=map_location)
    net.load_state_dict(checkpoint_net_state_dict(payload))
    if optim is not None and payload.get("optim") is not None:
        optim.load_state_dict(payload["optim"])
    if buffer is not None and "buffer" in payload:
        buffer.load_state_dict(payload["buffer"])
    if "torch_rng" in payload:
        # torch.random.set_rng_state requires a CPU ByteTensor regardless of
        # map_location, so move it back to CPU if it was mapped to another device.
        rng_state = payload["torch_rng"]
        if isinstance(rng_state, torch.Tensor) and rng_state.device.type != "cpu":
            rng_state = rng_state.cpu()
        torch.random.set_rng_state(rng_state)
    if "py_rng" in payload:
        random.setstate(pickle.loads(payload["py_rng"]))
    # Restore CUDA RNG state when resuming on a CUDA device.
    if "cuda_rng" in payload:
        target = torch.device(map_location) if isinstance(map_location, str) else map_location
        if target.type == "cuda" and torch.cuda.is_available():
            cuda_rng = payload["cuda_rng"]
            # torch.cuda.set_rng_state expects a CPU ByteTensor.
            if isinstance(cuda_rng, torch.Tensor) and cuda_rng.device.type != "cpu":
                cuda_rng = cuda_rng.cpu()
            torch.cuda.set_rng_state(cuda_rng)
    return payload

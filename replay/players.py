"""Player policy classes for driving ``BatchedEngine`` in play and training tools.

All policies implement the PlayerPolicy protocol:
    choose(engine) -> torch.Tensor  # shape (B,) int64

This matches the interface used by bots in agent/eval/bots.py and by the
MCTS search in agent/search/gumbel_mcts.py.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Protocol

import torch

from agent.env import batched_engine as BE
from agent.eval import bots as B
from agent.eval import heuristic_opus as HO
from agent.net import model as M
from agent.search import gumbel_mcts as G
from agent.train import checkpointing as CK


class PlayerPolicy(Protocol):
    """Callable that returns one action per game in the batch."""

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        """Returns (engine.batch_size,) int64 action tensor."""
        ...

    def info(self) -> dict[str, Any]:
        """Returns a JSON-serializable dict describing this policy."""
        ...


def _load_policy_net(checkpoint_path: str | pathlib.Path, device: str) -> M.SplendorNet:
    net, _ = CK.load_net_from_checkpoint(pathlib.Path(checkpoint_path), map_location=device)
    net.eval()
    net.to(device)
    return net


class RandomPolicy:
    """Uniform random over legal actions."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._bot = B.RandomBot(seed=seed)

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        return self._bot.choose(engine)

    def info(self) -> dict[str, Any]:
        return {"kind": "random", "seed": self._seed}


class HeuristicPolicy:
    """Greedy scoring bot (buy > reserve > take tokens)."""

    def __init__(self) -> None:
        self._bot = B.HeuristicBot()

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        return self._bot.choose(engine)

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic"}


class HeuristicOpusPolicy:
    """The heuristic-opus V15 agent.

    V15 dispatches by player count:
    * 2p: V13 path planner (race-to-15 with 2-card lookahead)
    * 3p: V13 path planner (where tighter races reward multi-card lookahead)
    * 4p: V10 single-card target with selective denial (where chaos breaks
      multi-card plans)

    V15 ships as production after V17-V20 tried 1-ply and 2-ply opponent
    lookahead variants and a 192-game V15-vs-V20 head-to-head settled
    at 49.3% / 48.4% (statistically tied). The added compute cost of
    lookahead is not justified by measurable strength gains. Further
    gains beyond ~2645 require learned evaluators (trained-net + MCTS),
    not more elaborate heuristics. Anchored aggregate rating ~2645 vs
    heuristic (2500) and random (1000).
    """

    def __init__(self) -> None:
        self._bot = HO.HeuristicOpusV15()

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        return self._bot.choose(engine)

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 15}


class NetPolicy:
    """Trained SplendorNet with Gumbel-MCTS search."""

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        num_sims: int = 8,
        device: str = "cpu",
    ) -> None:
        self._ckpt = str(checkpoint_path)
        self._num_sims = num_sims
        self._device = device
        self._G = G
        self._net = _load_policy_net(checkpoint_path, device)

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self._G.gumbel_root_act(engine, self._net, num_sims=self._num_sims)
        return action

    def info(self) -> dict[str, Any]:
        return {
            "kind": "net",
            "checkpoint": self._ckpt,
            "num_sims": self._num_sims,
        }


class GreedyNetPolicy:
    """Trained SplendorNet with argmax on policy logits (no MCTS)."""

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        device: str = "cpu",
    ) -> None:
        self._ckpt = str(checkpoint_path)
        self._device = device
        self._net = _load_policy_net(checkpoint_path, device)

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        with torch.no_grad():
            logits, _, legal = self._net.inference(engine)
        masked = logits.masked_fill(~legal, -1e9)
        return masked.argmax(dim=-1)

    def info(self) -> dict[str, Any]:
        return {
            "kind": "net",
            "checkpoint": self._ckpt,
            "num_sims": 0,
        }


def parse_player_spec(spec: str, device: str = "cpu") -> PlayerPolicy:
    """Parse a player spec string into the appropriate PlayerPolicy."""
    if spec == "random":
        return RandomPolicy(seed=0)

    if spec.startswith("random:"):
        tail = spec[len("random:") :]
        try:
            seed = int(tail)
        except ValueError as e:
            raise ValueError(f"invalid random seed in spec {spec!r}; expected 'random:<int>'") from e
        return RandomPolicy(seed=seed)

    if spec == "heuristic":
        return HeuristicPolicy()

    if spec == "heuristic_opus":
        return HeuristicOpusPolicy()

    if spec.startswith("net:"):
        rest = spec[len("net:") :]
        num_sims = 8
        if ":sims=" in rest:
            idx = rest.rindex(":sims=")
            sims_str = rest[idx + len(":sims=") :]
            try:
                num_sims = int(sims_str)
            except ValueError as e:
                raise ValueError(f"invalid sims value in spec {spec!r}") from e
            rest = rest[:idx]
        ckpt = rest
        if not ckpt:
            raise ValueError(f"net spec requires a checkpoint path: {spec!r}")
        ckpt_path = pathlib.Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = pathlib.Path.cwd() / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path!r}")
        if num_sims == 0:
            return GreedyNetPolicy(ckpt_path, device=device)
        return NetPolicy(ckpt_path, num_sims=num_sims, device=device)

    raise ValueError(
        f"unrecognized player spec {spec!r}. "
        "Expected: 'random', 'random:<seed>', 'heuristic', 'heuristic_opus', "
        "'net:<ckpt>', or 'net:<ckpt>:sims=<N>'"
    )

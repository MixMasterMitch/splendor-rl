"""CLI to run a candidate tournament for the heuristic-opus agent.

Plays every selected candidate plus the existing reference bots
(``random`` and ``heuristic``) against each other, fits anchored
ratings (random=1000, heuristic=2500), and writes the result to a JSON
report. The output directory is independent from the training league JSON,
so candidate tournament games never enter the main rating system.

Usage::

    bazel run --config=mlinfra_v7 \\
        //experimental/mloeppky/splendor/agent/scripts:heuristic_opus_tournament -- \\
        --candidates v1,v2 \\
        --num-games 48 \\
        --output runs_heuristic_opus/round1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

from agent.eval import bots as B
from agent.eval import heuristic_opus as HO
from agent.eval import tournament as T
from play import models as PM
from replay import players as RP


class _RandomBotWithInfo:
    """Random bot adapter with an explicit ``info()`` for cache keys."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._bot = B.RandomBot(seed=seed)

    def choose(self, engine: Any) -> Any:
        return self._bot.choose(engine)

    def info(self) -> dict[str, Any]:
        return {"kind": "random", "seed": self._seed}


class _HeuristicBotWithInfo:
    """Heuristic bot adapter with an ``info()`` for cache keys."""

    def __init__(self) -> None:
        self._bot = B.HeuristicBot()

    def choose(self, engine: Any) -> Any:
        return self._bot.choose(engine)

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic"}


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class _NetBotWithInfo:
    """Trained-net policy adapter with an ``info()`` for cache keys.

    Wraps :class:`replay.players.GreedyNetPolicy` (``num_sims=0``) or
    :class:`replay.players.NetPolicy` (``num_sims>0``) so the trained
    checkpoint can compete in the candidate tournament. The cache key
    includes a SHA-256 of the checkpoint file so swapping checkpoints
    invalidates affected cached matchups automatically.
    """

    def __init__(self, checkpoint: pathlib.Path, num_sims: int, device: str) -> None:
        self._ckpt = str(checkpoint)
        self._num_sims = int(num_sims)
        self._device = device
        self._ckpt_sha = _file_sha256(checkpoint)
        if num_sims <= 0:
            self._policy: Any = RP.GreedyNetPolicy(checkpoint, device=device)
        else:
            self._policy = RP.NetPolicy(
                checkpoint, num_sims=num_sims, device=device
            )

    def choose(self, engine: Any) -> Any:
        return self._policy.choose(engine)

    def info(self) -> dict[str, Any]:
        return {
            "kind": "net",
            "checkpoint_sha256": self._ckpt_sha,
            "num_sims": self._num_sims,
            "device": self._device,
        }


def _discover_best_net(
    workspace_root: pathlib.Path,
) -> dict[str, Any] | None:
    """Pick the highest-rated net checkpoint discovered across runs.

    Returns the model dict (with ``id``, ``rating``, ``ckpt``) or ``None``
    if no rated net entry is available. Only considers entries whose
    ``rating`` field is set -- pre-rating-fit checkpoints are skipped.
    """
    models = PM.discover_models(workspace_root)
    rated_nets = [
        m for m in models if m.get("kind") == "net" and m.get("rating") is not None
    ]
    if not rated_nets:
        return None
    rated_nets.sort(key=lambda m: float(m["rating"]), reverse=True)
    return rated_nets[0]


def _parse_candidates(spec: str) -> list[str]:
    if not spec.strip():
        return []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[str] = []
    for part in parts:
        if part in HO.list_candidate_names():
            out.append(part)
            continue
        # Allow short form "v1", "v2", ... -> heuristic_opus_v1, ...
        if part.startswith("v") and ("heuristic_opus_" + part) in HO.list_candidate_names():
            out.append("heuristic_opus_" + part)
            continue
        raise ValueError(
            f"unknown candidate {part!r}; valid: {HO.list_candidate_names()}"
        )
    return out


def _build_factories(
    candidate_names: list[str],
    *,
    include_random: bool,
    include_heuristic: bool,
    random_seed: int,
    net_entity: tuple[str, pathlib.Path, int, str] | None = None,
) -> dict[str, Any]:
    factories: dict[str, Any] = {}
    if include_random:
        factories["random"] = lambda seed=random_seed: _RandomBotWithInfo(seed)
    if include_heuristic:
        factories["heuristic"] = lambda: _HeuristicBotWithInfo()
    for name in candidate_names:
        factories[name] = lambda n=name: HO.make_candidate(n)
    if net_entity is not None:
        ent_name, ckpt_path, num_sims, device = net_entity
        factories[ent_name] = (
            lambda c=ckpt_path, s=num_sims, d=device: _NetBotWithInfo(c, s, d)
        )
    return factories


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=",".join(HO.list_candidate_names()),
        help="Comma-separated heuristic-opus candidate names (or short v1/v2/...).",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=24,
        help=(
            "Per-matchup game count; split evenly across 2*num_players seat "
            "rotations (so 24 = 12/4/3 games per rotation at 2/3/4 players)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="agent/runs_heuristic_opus/cache",
        help=(
            "Directory for the matchup result cache. Pass empty string to "
            "disable. The key includes a SHA-256 of heuristic_opus.py and "
            "bots.py, so editing either invalidates affected entries."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick smoke mode: 12 games per matchup, max-turns 200, only "
            "2-player. Useful for early iteration."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Number of parallel matchup workers (thread pool). Each worker "
            "runs a full play_matchup independently; torch threads per "
            "worker are scaled down so total CPU stays bounded."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="Player count when --player-counts is not used.",
    )
    parser.add_argument(
        "--player-counts",
        default=None,
        help=(
            "Comma-separated list of player counts to evaluate "
            "(e.g. '2,3,4'). When set, the tournament fits one anchored "
            "rating across all player counts plus per-count breakdowns. "
            "Overrides --num-players."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--analyze",
        default=None,
        help=(
            "Comma-separated candidate names whose per-player-count game-log "
            "analysis should be printed (e.g. 'heuristic_opus_v1,v2'). "
            "Implies --record-logs."
        ),
    )
    parser.add_argument(
        "--record-logs",
        action="store_true",
        help=(
            "Capture per-game logs (action-class counts + final scores) "
            "for the analyzer. Adds modest overhead but enables --analyze."
        ),
    )
    parser.add_argument(
        "--timeout-as-tie",
        action="store_true",
        help=(
            "Treat timed-out games as ties (the default behaviour is to "
            "award timeouts to the highest-scoring seat).  Use this only "
            "if you want strict 'must reach 15' rating semantics."
        ),
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Skip the random anchor in the tournament (still anchored in rating).",
    )
    parser.add_argument(
        "--no-heuristic",
        action="store_true",
        help="Skip the heuristic anchor in the tournament (still anchored in rating).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the tournament report JSON (relative to workspace).",
    )
    parser.add_argument(
        "--include-net",
        action="store_true",
        default=True,
        help=(
            "Include the top trained ML checkpoint as a tournament "
            "participant (2-player matchups only, since training only "
            "covers 2 players). Use --no-include-net to disable."
        ),
    )
    parser.add_argument(
        "--no-include-net",
        dest="include_net",
        action="store_false",
        help="Disable the trained-net participant.",
    )
    parser.add_argument(
        "--net-checkpoint",
        default=None,
        help=(
            "Path to a trained checkpoint to enter into the tournament. "
            "When omitted, the highest-rated entry across all "
            "agent/runs/<run>/checkpoints/league/league.json files is "
            "auto-selected."
        ),
    )
    parser.add_argument(
        "--net-sims",
        type=int,
        default=0,
        help=(
            "Gumbel-MCTS sims for the trained net. 0 (default) is "
            "GreedyNetPolicy: argmax over masked logits. Higher counts "
            "are stronger but slower per decision."
        ),
    )
    parser.add_argument(
        "--net-name",
        default=None,
        help=(
            "Override the net's tournament entity name. Defaults to the "
            "auto-discovered model id (e.g. 'net:real30_v9:873')."
        ),
    )
    args = parser.parse_args(argv)

    candidate_names = _parse_candidates(args.candidates)
    if not candidate_names:
        print("no candidates selected; nothing to do.", file=sys.stderr)
        sys.exit(2)

    workspace = _workspace_root()
    net_entity: tuple[str, pathlib.Path, int, str] | None = None
    net_entity_name: str | None = None
    if args.include_net:
        if args.net_checkpoint:
            ckpt = pathlib.Path(args.net_checkpoint)
            if not ckpt.is_absolute():
                ckpt = workspace / ckpt
            if not ckpt.exists():
                raise FileNotFoundError(f"net checkpoint not found: {ckpt!r}")
            ent = args.net_name or f"net:{ckpt.stem}"
            net_entity = (ent, ckpt, args.net_sims, "cpu")
            net_entity_name = ent
            print(f"net participant: {ent}  ckpt={ckpt}", file=sys.stderr)
        else:
            best = _discover_best_net(workspace)
            if best is None:
                print(
                    "warning: --include-net set but no rated net checkpoints "
                    "found; running without a net participant.",
                    file=sys.stderr,
                )
            else:
                ckpt = pathlib.Path(str(best["ckpt"]))
                ent = args.net_name or str(best["id"])
                net_entity = (ent, ckpt, args.net_sims, "cpu")
                net_entity_name = ent
                print(
                    f"net participant: {ent}  rating={float(best['rating']):.0f}  "
                    f"ckpt={ckpt}",
                    file=sys.stderr,
                )

    factories = _build_factories(
        candidate_names,
        include_random=not args.no_random,
        include_heuristic=not args.no_heuristic,
        random_seed=args.seed,
        net_entity=net_entity,
    )

    if args.player_counts:
        pcs = [int(x.strip()) for x in args.player_counts.split(",") if x.strip()]
    else:
        pcs = None

    if args.quick:
        if args.num_games == 24:
            args.num_games = 12
        if args.max_turns == 300:
            args.max_turns = 200
        if pcs is None:
            pcs = [2]

    record_logs = args.record_logs or bool(args.analyze)
    cache_dir: pathlib.Path | None
    if not args.cache_dir:
        cache_dir = None
    else:
        cache_dir = pathlib.Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = _workspace_root() / cache_dir
    participant_player_counts: dict[str, list[int]] | None = None
    if net_entity_name is not None:
        participant_player_counts = {net_entity_name: [2]}
    report = T.run_tournament(
        factories,
        num_games=args.num_games,
        num_players=args.num_players,
        max_turns=args.max_turns,
        seed=args.seed,
        device="cpu",
        player_counts=pcs,
        record_logs=record_logs,
        timeout_winner_uses_points=not args.timeout_as_tie,
        cache_dir=cache_dir,
        workers=args.workers,
        participant_player_counts=participant_player_counts,
    )
    print(report.summary)

    if args.analyze:
        targets = [t.strip() for t in args.analyze.split(",") if t.strip()]
        normalized: list[str] = []
        for t in targets:
            if t in HO.list_candidate_names() or t in factories:
                normalized.append(t)
                continue
            if t.startswith("v") and ("heuristic_opus_" + t) in HO.list_candidate_names():
                normalized.append("heuristic_opus_" + t)
                continue
            raise ValueError(f"unknown analyze target: {t!r}")
        for t in normalized:
            print()
            print(T.analyze_logs_for_candidate(report.matches, t))

    if args.output:
        ws = _workspace_root()
        out_path = pathlib.Path(args.output)
        if not out_path.is_absolute():
            out_path = ws / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "ratings": report.ratings,
            "ratings_per_pc": {
                str(pc): r for pc, r in report.ratings_per_pc.items()
            },
            "games_per_entity": report.games_per_entity,
            "games_per_entity_per_pc": {
                str(pc): g for pc, g in report.games_per_entity_per_pc.items()
            },
            "wall_seconds": report.wall_seconds,
            "matches": [
                {
                    "name_a": m.name_a,
                    "name_b": m.name_b,
                    "games_played": m.games_played,
                    "games_finished": m.games_finished,
                    "wins_a": m.wins_a,
                    "wins_b": m.wins_b,
                    "ties": m.ties,
                    "avg_finished_turns": m.avg_finished_turns,
                }
                for m in report.matches
            ],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote report to {out_path}")


if __name__ == "__main__":
    main()

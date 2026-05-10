"""Per-opponent 3p diagnostic for a checkpoint.

Rules out two competing hypotheses for the persistent 3p-below-random win
rate on ``attn256_v4``:

  (A) The checkpoint is genuinely broken at 3p — it loses to random / weak
      bots, not only to league peers. This would indicate a training-signal
      bug deeper than opponent diversity.

  (B) The checkpoint is fine in isolation but loses to the diverse eval
      pool, confirming the alignment bug in ``_league_trigger`` (3p self-play
      never saw league opponents).

The script plays N games at each requested player count against each
opponent in isolation and prints per-opponent win rates. If the checkpoint
beats ``random`` at 3p (typical cutoff >= 60%) but loses or ties against
league peers, hypothesis (B) is confirmed.

Usage:

    python -m agent.scripts.diagnose_3p \
        --ckpt agent/runs/attn256_v4/checkpoints/iter_001000.pt \
        --num-games 120 --num-sims 64 \
        --opponents random heuristic heuristic_opus \
        --league-dir agent/runs/league --league-count 3

Run on CPU (default). ``--num-games`` should be divisible by 2*num_players
so seat permutations balance; otherwise it is silently truncated.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random as stdlib_random
import sys
import time
from typing import Callable, List

import torch

from agent.env import batched_engine as BE
from agent.eval import bots as B
from agent.eval import heuristic_opus as HO
from agent.eval import tournament as T
from agent.net import model as M
from agent.search import gumbel_mcts as G
from agent.train import checkpointing as CK


def _load_league_opponents(
    league_dir: pathlib.Path, count: int, seed: int
) -> List[pathlib.Path]:
    """Pick `count` league .pt files at random (excluding the checkpoint
    under diagnosis; caller passes its path separately)."""
    candidates = sorted(league_dir.glob("ckpt_*.pt"))
    rng = stdlib_random.Random(seed)
    if count >= len(candidates):
        return candidates
    return rng.sample(candidates, count)


def _make_net_policy(net: M.SplendorNet, num_sims: int):
    def choose(engine: BE.BatchedEngine) -> torch.Tensor:
        with torch.no_grad():
            act, _ = G.gumbel_root_act(engine, net, num_sims=num_sims)
        return act
    return choose


def _build_opponents(
    names: List[str],
    seed: int,
    num_sims: int,
    league_paths: List[pathlib.Path],
) -> dict[str, Callable[[BE.BatchedEngine], torch.Tensor]]:
    out: dict[str, Callable[[BE.BatchedEngine], torch.Tensor]] = {}
    for n in names:
        if n == "random":
            out[n] = B.RandomBot(seed=seed).choose
        elif n == "heuristic":
            out[n] = B.HeuristicBot().choose
        elif n == "heuristic_opus":
            out[n] = HO.HeuristicOpusV15().choose
        else:
            raise ValueError(f"unknown non-net opponent: {n}")
    for path in league_paths:
        net, _ = CK.load_net_from_checkpoint(path, map_location="cpu")
        net.eval()
        out[f"league:{path.stem}"] = _make_net_policy(net, num_sims)
    return out


def _play_vs_one_opponent(
    eval_policy: Callable[[BE.BatchedEngine], torch.Tensor],
    opp_policy: Callable[[BE.BatchedEngine], torch.Tensor],
    num_players: int,
    num_games: int,
    num_sims: int,  # noqa: ARG001 - kept for future use
    max_turns: int,
    seed: int,
) -> dict:
    """Run one head-to-head: the eval net holds one seat, the opponent
    fills the rest.

    Seat rotation is handled by ``tournament.play_matchup`` via
    ``_balanced_rotations`` so role 0 (eval) occupies every seat an equal
    number of times. Opponent fills the remaining seats, all controlled by
    the same policy callable (not a bot swarm of different nets).
    """
    matchup = T.Matchup(
        name_a="eval",
        name_b="opp",
        num_games=num_games,
        num_players=num_players,
        max_turns=max_turns,
        seed=seed,
    )
    result = T.play_matchup(
        matchup,
        eval_policy,
        opp_policy,
        device="cpu",
        record_logs=False,
        timeout_winner_uses_points=True,
    )
    wa = result.wins_a
    wb = result.wins_b
    ties = result.ties
    played = max(1, result.games_played)
    return {
        "games_played": result.games_played,
        "games_finished": result.games_finished,
        "eval_win_rate": wa / played,
        "opp_win_rate": wb / played,
        "tie_rate": ties / played,
        "avg_finished_turns": result.avg_finished_turns,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-opponent 3p diagnostic")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument(
        "--player-counts",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Player counts to test (default: 2 3 4).",
    )
    p.add_argument(
        "--num-games",
        type=int,
        default=120,
        help="Games per (opponent, player-count). Will be truncated to a "
             "multiple of 2*num_players for balanced seat rotation.",
    )
    p.add_argument("--num-sims", type=int, default=64)
    p.add_argument("--max-turns-per-player", type=int, default=60)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "heuristic", "heuristic_opus"],
        help="Non-league opponents to test against.",
    )
    p.add_argument(
        "--league-dir",
        default="agent/runs/league",
        help="Directory containing league ckpt_*.pt files.",
    )
    p.add_argument(
        "--league-count",
        type=int,
        default=3,
        help="Number of league peers to sample. Set 0 to skip league.",
    )
    args = p.parse_args(argv)

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.exists():
        print(json.dumps({"error": f"checkpoint not found: {ckpt_path}"}),
              file=sys.stderr)
        return 1

    # Load the eval net
    eval_net, payload = CK.load_net_from_checkpoint(ckpt_path, map_location="cpu")
    eval_net.eval()
    spec = CK.checkpoint_net_spec(payload)
    eval_policy = _make_net_policy(eval_net, args.num_sims)

    # Gather league peers (exclude the checkpoint under diagnosis if present)
    league_paths: List[pathlib.Path] = []
    if args.league_count > 0:
        league_dir = pathlib.Path(args.league_dir)
        all_league = sorted(league_dir.glob("ckpt_*.pt"))
        # Drop any league file that is byte-identical-path to the ckpt under diagnosis
        all_league = [p for p in all_league if p.resolve() != ckpt_path.resolve()]
        rng = stdlib_random.Random(args.seed)
        if args.league_count < len(all_league):
            league_paths = rng.sample(all_league, args.league_count)
        else:
            league_paths = all_league

    opponents = _build_opponents(
        args.opponents, args.seed, args.num_sims, league_paths
    )

    t_start = time.monotonic()
    rows: list[dict] = []
    for pc in args.player_counts:
        max_turns = args.max_turns_per_player * pc
        # Ensure balanced rotation: need multiple of 2*pc games.
        per_opp = max(2 * pc, (args.num_games // (2 * pc)) * (2 * pc))
        for opp_name, opp_policy in opponents.items():
            # Reset random-bot state per (pc, opp) for reproducibility
            if opp_name == "random":
                opponents[opp_name] = B.RandomBot(seed=args.seed + pc * 100).choose
                opp_policy = opponents[opp_name]
            t0 = time.monotonic()
            stats = _play_vs_one_opponent(
                eval_policy,
                opp_policy,
                num_players=pc,
                num_games=per_opp,
                num_sims=args.num_sims,
                max_turns=max_turns,
                seed=args.seed + pc * 1000 + hash(opp_name) % 7919,
            )
            wall = time.monotonic() - t0
            row = {
                "player_count": pc,
                "opponent": opp_name,
                "wall_s": round(wall, 2),
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in stats.items()},
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    total_wall = time.monotonic() - t_start

    # Per-PC summary
    print("\n=== summary ===", flush=True)
    print(f"ckpt: {ckpt_path}")
    print(f"iteration: {payload.get('iteration')}")
    print(f"hidden: {spec.hidden} arch: {spec.arch}")
    print(f"total wall: {total_wall:.1f}s\n")
    print(f"{'PC':>3s}  {'opponent':<28s}  {'win%':>6s}  {'tie%':>6s}  "
          f"{'loss%':>6s}  {'fin%':>6s}  {'avgT':>6s}")
    for row in rows:
        loss = 1.0 - row["eval_win_rate"] - row["tie_rate"]
        fin = row["games_finished"] / max(1, row["games_played"])
        print(
            f"{row['player_count']:>3d}  {row['opponent']:<28s}  "
            f"{row['eval_win_rate']*100:>5.1f}%  "
            f"{row['tie_rate']*100:>5.1f}%  "
            f"{loss*100:>5.1f}%  "
            f"{fin*100:>5.1f}%  "
            f"{row['avg_finished_turns']:>6.1f}"
        )

    # Bottom-line interpretation for 3p:
    print("\n=== 3p interpretation ===")
    rows_3p = [r for r in rows if r["player_count"] == 3]
    if not rows_3p:
        print("(3p not tested)")
    else:
        # If we beat random at 3p (>50%) and still lose to league peers,
        # the problem is opponent diversity, not the head itself.
        random_wr = next((r["eval_win_rate"] for r in rows_3p
                          if r["opponent"] == "random"), None)
        heur_wr = next((r["eval_win_rate"] for r in rows_3p
                        if r["opponent"] == "heuristic"), None)
        league_wrs = [r["eval_win_rate"] for r in rows_3p
                      if r["opponent"].startswith("league:")]
        print(f"  vs random:    {random_wr if random_wr is None else f'{random_wr*100:.1f}%'}")
        print(f"  vs heuristic: {heur_wr if heur_wr is None else f'{heur_wr*100:.1f}%'}")
        if league_wrs:
            avg_league = sum(league_wrs) / len(league_wrs)
            print(f"  vs league:    mean={avg_league*100:.1f}%  "
                  f"range={min(league_wrs)*100:.1f}%-{max(league_wrs)*100:.1f}%")
        if random_wr is not None:
            if random_wr > 0.6:
                print("  -> net is NOT broken at 3p in isolation.")
            elif random_wr < 0.4:
                print("  -> net IS broken at 3p (loses/ties vs random).")
            else:
                print("  -> ambiguous; tighten num-games and rerun.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

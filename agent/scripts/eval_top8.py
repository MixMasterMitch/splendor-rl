"""Thorough evaluation of the top 8 league checkpoints.

Runs ~10,000 randomly-assigned games per player count (2p, 3p, 4p) across
an 11-entity pool: 8 ML models + random + heuristic + heuristic_opus.

Game assignments are generated randomly (uniform sampling of seats from the
entity pool, filtered to require at least one ML agent per game). This gives
diverse seating/opponent combinations and produces per-agent winrate estimates
with ~±2% confidence intervals.

GPU-optimized: all games for a player count run in one mega-batch. The game
loop dispatches MCTS for ML agents and heuristic logic for bots, all in one
pass per turn. This maximizes GPU utilization.

Usage:
    python -m agent.scripts.eval_top8 [--games-per-pc 10000] [--num-sims 16] [--device cuda]

Output: per-agent winrate table with 95% CIs, pairwise head-to-head matrix,
and JSON results file.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch

from agent.env import batched_engine as BE
from agent.eval import bots as B
from agent.eval import heuristic_opus as HO
from agent.net import encoder as ENC
from agent.net import model as M
from agent.search import gumbel_mcts as G
from agent.train import checkpointing as CK
from agent.train import league as L


LEAGUE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "runs" / "league"


def _to_league_entity(name: str, entries: list[dict]) -> str:
    """Convert an entity name like 'ckpt_2646_i2575' to league entity ID 'ckpt:2646'."""
    if name in ("random", "heuristic", "heuristic_opus"):
        return name
    # Parse ckpt_{idx}_{tag}
    parts = name.split("_", 2)
    if len(parts) >= 2 and parts[0] == "ckpt":
        return f"ckpt:{parts[1]}"
    return name


def _normalize_device(device: str) -> str:
    """Normalize device string so 'cuda' becomes 'cuda:0' etc."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", 0)
    return str(d)


def _load_league_top8() -> list[dict]:
    """Load league manifest and return all entries with files on disk."""
    league = L.League(LEAGUE_ROOT)
    entries = league.list_entries()
    available = [e for e in entries if (LEAGUE_ROOT / e["path"]).exists()]
    available.sort(key=lambda e: e.get("rating", 0), reverse=True)
    return available


def _load_net(path: pathlib.Path, device: str) -> M.SplendorNet:
    """Load a net from checkpoint and put it in eval mode."""
    net, _ = CK.load_net_from_checkpoint(path, map_location=device)
    net.to(device)
    net.eval()
    return net


def _wilson_ci(wins: float, n: float, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (center, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (p_hat, max(0.0, center - margin), min(1.0, center + margin))


def _generate_game_assignments(
    num_games: int,
    num_players: int,
    num_entities: int,
    num_ml_agents: int,
    seed: int,
) -> torch.Tensor:
    """Generate random game seat assignments.

    Returns: (num_games, num_players) int tensor of entity indices.
    Entity indices 0..num_ml_agents-1 are ML agents.
    Filters to require at least one ML agent per game.
    Seats within a game are sampled WITHOUT replacement (no entity plays
    against itself in the same game).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Over-generate to account for filtering
    oversample = int(num_games * 1.5) + 1000
    assignments = torch.zeros((0, num_players), dtype=torch.long)

    while assignments.shape[0] < num_games:
        # Sample seats without replacement within each game
        # Use torch.multinomial per-game or just random permutations
        batch = torch.zeros((oversample, num_players), dtype=torch.long)
        for seat in range(num_players):
            if seat == 0:
                batch[:, 0] = torch.randint(
                    0, num_entities, (oversample,), generator=gen
                )
            else:
                # Sample from remaining entities (avoid duplicates)
                # Simple rejection: sample and re-roll collisions
                batch[:, seat] = torch.randint(
                    0, num_entities, (oversample,), generator=gen
                )

        # Remove games with duplicate entities in same game
        # For small num_players (2-4) and 11 entities, collisions are rare
        valid = torch.ones(oversample, dtype=torch.bool)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                valid &= batch[:, i] != batch[:, j]

        # Filter: at least one ML agent (entity idx < num_ml_agents)
        has_ml = (batch < num_ml_agents).any(dim=1)
        valid &= has_ml

        good = batch[valid]
        assignments = torch.cat([assignments, good], dim=0)

    return assignments[:num_games]


def _run_games(
    assignments: torch.Tensor,
    num_players: int,
    ml_nets: List[M.SplendorNet],
    bot_policies: List[Callable[[BE.BatchedEngine], torch.Tensor]],
    num_ml: int,
    num_sims: int,
    device: str,
    max_turns: int,
    max_batch: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run all games and return per-game results.

    Args:
        assignments: (N, num_players) entity indices per seat
        ml_nets: list of ML nets (indices 0..num_ml-1)
        bot_policies: list of bot choose() callables (indices num_ml..num_entities-1)
        num_ml: number of ML agents
        num_sims: MCTS sims for ML agents
        max_batch: max games per GPU batch (memory limit)

    Returns:
        winner_entity: (N,) entity index of winner per game
        game_finished: (N,) bool whether game finished before max_turns
    """
    N = assignments.shape[0]
    all_winner_entity = torch.zeros(N, dtype=torch.long)
    all_finished = torch.zeros(N, dtype=torch.bool)

    # Process in chunks to fit GPU memory
    for chunk_start in range(0, N, max_batch):
        chunk_end = min(chunk_start + max_batch, N)
        chunk_size = chunk_end - chunk_start
        chunk_assign = assignments[chunk_start:chunk_end].to(device)

        engine = BE.BatchedEngine(
            chunk_size, num_players, device=device,
            seed=chunk_start * 7 + 42,
        )

        game_end_step = torch.full(
            (chunk_size,), max_turns, dtype=torch.int32, device=device
        )
        prev_ended = torch.zeros(chunk_size, dtype=torch.bool, device=device)
        b_range = torch.arange(chunk_size, device=device)

        turn = 0
        while turn < max_turns and not engine.ended.all():
            alive = ~engine.ended
            cp = engine.current_player.to(torch.long)

            # Which entity is the current player in each game?
            # chunk_assign[game, seat] = entity_idx
            current_entity = chunk_assign[b_range, cp]  # (chunk_size,)

            actions = torch.zeros(chunk_size, dtype=torch.int64, device=device)

            # --- ML agents: batch all ML-turn games together ---
            is_ml_turn = alive & (current_entity < num_ml)
            if is_ml_turn.any():
                ml_idx = is_ml_turn.nonzero(as_tuple=True)[0]
                ml_entities = current_entity[ml_idx]  # which ML net per game

                # Group by ML net for efficient batched inference
                for net_i in range(num_ml):
                    net_mask = ml_entities == net_i
                    if not net_mask.any():
                        continue
                    net_game_idx = ml_idx[net_mask]
                    net_engine = engine.index_select(net_game_idx)
                    with torch.no_grad():
                        net_actions, _ = G.gumbel_root_act(
                            net_engine, ml_nets[net_i], num_sims=num_sims
                        )
                    actions.index_copy_(0, net_game_idx, net_actions)

            # --- Bot agents: dispatch per bot type ---
            is_bot_turn = alive & (current_entity >= num_ml)
            if is_bot_turn.any():
                bot_idx = is_bot_turn.nonzero(as_tuple=True)[0]
                bot_entities = current_entity[bot_idx]

                for bot_i, bot_choose in enumerate(bot_policies):
                    entity_id = num_ml + bot_i
                    bot_mask = bot_entities == entity_id
                    if not bot_mask.any():
                        continue
                    bot_game_idx = bot_idx[bot_mask]
                    bot_engine = engine.index_select(bot_game_idx)
                    bot_actions = bot_choose(bot_engine)
                    actions.index_copy_(0, bot_game_idx, bot_actions)

            engine.apply(actions)

            cur_ended = engine.ended
            newly_ended = cur_ended & ~prev_ended
            if newly_ended.any():
                game_end_step[newly_ended] = turn + 1
            prev_ended = cur_ended
            turn += 1

        # Determine winners
        pts = engine.points.to(torch.int32)
        bonuses_total = engine.bonuses.sum(dim=-1).to(torch.int32)
        score = pts * 1000 - bonuses_total
        score = torch.where(
            engine.active_mask, score, torch.full_like(score, -(10**9))
        )
        winner_seat = score.argmax(dim=-1)  # (chunk_size,)
        winner_entity = chunk_assign[b_range, winner_seat]

        all_winner_entity[chunk_start:chunk_end] = winner_entity.cpu()
        all_finished[chunk_start:chunk_end] = engine.ended.cpu()

        # Free GPU memory
        del engine, chunk_assign
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return all_winner_entity, all_finished


def run_eval(
    games_per_pc: int = 10000,
    num_sims: int = 16,
    device: str = "cpu",
    max_turns: int = 200,
    max_batch: int = 4096,
    seed: int = 42,
) -> str:
    """Run the full evaluation and return a formatted report."""
    device = _normalize_device(device)
    top8 = _load_league_top8()
    if len(top8) < 2:
        return f"Need at least 2 checkpoints, found {len(top8)}."

    num_ml = len(top8)
    # Entity pool: ML agents + random + heuristic + opus
    entity_names = []
    for e in top8:
        entity_names.append(f"ckpt_{e['idx']}_{e['tag']}")
    entity_names.extend(["random", "heuristic", "heuristic_opus"])
    num_entities = len(entity_names)

    print(f"Entity pool ({num_entities} entities):")
    for i, name in enumerate(entity_names):
        ml_tag = " [ML]" if i < num_ml else " [bot]"
        rating = top8[i]["rating"] if i < num_ml else "—"
        print(f"  {i:>2}. {name:<28} {ml_tag}  league_rating={rating}")
    print()

    # Load ML nets
    print("Loading ML checkpoints...")
    ml_nets: List[M.SplendorNet] = []
    for e in top8:
        path = LEAGUE_ROOT / e["path"]
        net = _load_net(path, device)
        ml_nets.append(net)
        print(f"  Loaded {e['tag']}")
    print()

    # Bot policies
    random_bot = B.RandomBot(seed=seed)
    heuristic_bot = B.HeuristicBot()
    opus_bot = HO.HeuristicOpusV15()
    bot_policies = [random_bot.choose, heuristic_bot.choose, opus_bot.choose]

    # Run evaluation for each player count
    all_pc_results: Dict[int, Dict] = {}

    for pc in [2, 3, 4]:
        print(f"\n{'='*70}")
        print(f"  {pc}-PLAYER EVALUATION: {games_per_pc} games")
        print(f"{'='*70}")

        # Generate random game assignments
        t0 = time.monotonic()
        assignments = _generate_game_assignments(
            num_games=games_per_pc,
            num_players=pc,
            num_entities=num_entities,
            num_ml_agents=num_ml,
            seed=seed + pc * 99991,
        )
        print(f"  Generated {assignments.shape[0]} game assignments ({time.monotonic()-t0:.1f}s)")

        # Count appearances per entity
        appearances = torch.zeros(num_entities, dtype=torch.long)
        for i in range(num_entities):
            appearances[i] = (assignments == i).sum().item()
        print(f"  Appearances per entity: min={appearances.min().item()}, "
              f"max={appearances.max().item()}, "
              f"mean={appearances.float().mean().item():.0f}")

        # Run games
        t0 = time.monotonic()
        winner_entity, game_finished = _run_games(
            assignments=assignments,
            num_players=pc,
            ml_nets=ml_nets,
            bot_policies=bot_policies,
            num_ml=num_ml,
            num_sims=num_sims,
            device=device,
            max_turns=max_turns,
            max_batch=max_batch,
        )
        wall = time.monotonic() - t0
        print(f"  Games completed in {wall:.1f}s "
              f"({games_per_pc/wall:.0f} games/s, "
              f"{game_finished.sum().item()}/{games_per_pc} finished)")

        # Compute per-entity winrates
        entity_results: Dict[str, Dict] = {}
        for i, name in enumerate(entity_names):
            # Games where this entity participated
            participated = (assignments == i).any(dim=1)
            n_participated = int(participated.sum().item())

            # Games where this entity won
            won = participated & (winner_entity == i)
            n_won = int(won.sum().item())

            # Wilson CI
            wr, ci_lo, ci_hi = _wilson_ci(n_won, n_participated)

            entity_results[name] = {
                "games": n_participated,
                "wins": n_won,
                "winrate": wr,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }

        # Compute pairwise head-to-head (for ML agents)
        pairwise: Dict[str, Dict[str, Dict]] = {}
        for i in range(num_entities):
            name_i = entity_names[i]
            pairwise[name_i] = {}
            for j in range(num_entities):
                if i == j:
                    continue
                name_j = entity_names[j]
                # Games where both i and j participated
                both = (assignments == i).any(dim=1) & (assignments == j).any(dim=1)
                n_both = int(both.sum().item())
                if n_both == 0:
                    continue
                # Of those, how many did i win?
                i_won = both & (winner_entity == i)
                n_i_won = int(i_won.sum().item())
                wr_ij, _, _ = _wilson_ci(n_i_won, n_both)
                pairwise[name_i][name_j] = {
                    "games": n_both,
                    "wins": n_i_won,
                    "winrate": wr_ij,
                }

        all_pc_results[pc] = {
            "entity_results": entity_results,
            "pairwise": pairwise,
            "wall_s": round(wall, 1),
            "games_finished": int(game_finished.sum().item()),
            "games_total": games_per_pc,
        }

        # Print results table
        print(f"\n  {'Entity':<28} {'Games':>6} {'Wins':>6} {'WR':>7} {'95% CI':>14}")
        print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*7} {'-'*14}")
        sorted_entities = sorted(
            entity_results.items(), key=lambda x: -x[1]["winrate"]
        )
        for name, r in sorted_entities:
            ci_str = f"[{r['ci_lo']*100:.1f}%, {r['ci_hi']*100:.1f}%]"
            print(f"  {name:<28} {r['games']:>6} {r['wins']:>6} "
                  f"{r['winrate']*100:>6.1f}% {ci_str:>14}")

    # Final cross-player-count summary
    print(f"\n\n{'='*90}")
    print("SUMMARY: Per-agent winrates across player counts")
    print(f"{'='*90}")
    header = f"{'Entity':<28} {'2p WR':>10} {'3p WR':>10} {'4p WR':>10}"
    print(header)
    print("-" * len(header))

    # Sort by average winrate across player counts
    avg_wr: Dict[str, float] = {}
    for name in entity_names:
        wrs = []
        for pc in [2, 3, 4]:
            r = all_pc_results[pc]["entity_results"].get(name, {})
            if r:
                wrs.append(r["winrate"])
        avg_wr[name] = sum(wrs) / len(wrs) if wrs else 0

    for name in sorted(entity_names, key=lambda n: -avg_wr[n]):
        parts = [f"{name:<28}"]
        for pc in [2, 3, 4]:
            r = all_pc_results[pc]["entity_results"].get(name, {})
            if r:
                wr = r["winrate"]
                ci_lo = r["ci_lo"]
                ci_hi = r["ci_hi"]
                parts.append(f"{wr*100:5.1f}%±{(ci_hi-ci_lo)*50:.1f}")
            else:
                parts.append(f"{'—':>10}")
        print(" ".join(parts))

    # Pairwise matrix for ML agents (2p only, most interpretable)
    print(f"\n\nPairwise head-to-head winrates (2-player games):")
    ml_names = entity_names[:num_ml]
    all_names = entity_names
    pw = all_pc_results[2]["pairwise"]

    # Print compact matrix header
    short_names = [n.replace("ckpt_", "").replace("heuristic_opus", "opus")[:12]
                   for n in all_names]
    print(f"  {'':>12}", end="")
    for sn in short_names:
        print(f" {sn:>7}", end="")
    print()

    for i, name_i in enumerate(all_names):
        sn_i = short_names[i]
        print(f"  {sn_i:>12}", end="")
        for j, name_j in enumerate(all_names):
            if i == j:
                print(f"    {'—':>4}", end="")
            elif name_j in pw.get(name_i, {}):
                wr = pw[name_i][name_j]["winrate"]
                print(f" {wr*100:6.1f}%", end="")
            else:
                print(f"    {'?':>4}", end="")
        print()

    # Save JSON
    json_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "games_per_player_count": games_per_pc,
            "num_sims": num_sims,
            "device": device,
            "max_turns": max_turns,
            "seed": seed,
            "entities": entity_names,
            "num_ml_agents": num_ml,
        },
        "results_by_pc": {},
    }
    for pc in [2, 3, 4]:
        pcr = all_pc_results[pc]
        json_out["results_by_pc"][str(pc)] = {
            "entity_results": pcr["entity_results"],
            "pairwise": pcr["pairwise"],
            "wall_s": pcr["wall_s"],
            "games_finished": pcr["games_finished"],
            "games_total": pcr["games_total"],
        }

    out_path = LEAGUE_ROOT.parent / "top8_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Write results back to league.json using the new rating system
    print("\nUpdating league.json with new ratings...")
    league = L.League(LEAGUE_ROOT)
    # Clear old results and write fresh pairwise data
    league.manifest["results"] = []
    for pc in [2, 3, 4]:
        pcr = all_pc_results[pc]
        pairwise = pcr["pairwise"]
        for a in pairwise:
            for b in pairwise[a]:
                wins = pairwise[a][b]["wins"]
                games = pairwise[a][b]["games"]
                losses = games - wins
                if wins + losses > 0:
                    # Map entity names to league entity IDs
                    entity_a = _to_league_entity(a, top8)
                    entity_b = _to_league_entity(b, top8)
                    league.record_result(entity_a, entity_b, wins, losses,
                                         num_players=pc)
    ratings = league.recompute_ratings()
    print("League ratings updated:")
    for name, rating in sorted(ratings.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<28} {rating:>7.0f}")

    return json.dumps(json_out, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Thorough evaluation of top 8 league checkpoints"
    )
    p.add_argument("--games-per-pc", type=int, default=10000,
                   help="Games per player count (default: 10000)")
    p.add_argument("--num-sims", type=int, default=16,
                   help="MCTS simulations for ML agents (default: 16)")
    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu)")
    p.add_argument("--max-turns", type=int, default=200,
                   help="Max turns per game (default: 200)")
    p.add_argument("--max-batch", type=int, default=4096,
                   help="Max games per GPU batch (default: 4096)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_eval(
        games_per_pc=args.games_per_pc,
        num_sims=args.num_sims,
        device=args.device,
        max_turns=args.max_turns,
        max_batch=args.max_batch,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

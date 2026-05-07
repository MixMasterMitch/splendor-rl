"""Play rated games with an LLM Bedrock agent against various opponents.

Plays sequential 1v1, 3-player, and 4-player games with random matchups
against random, heuristic, heuristic_opus, and ML bot opponents. Results
are recorded into the league rating system so they feed into the cloud
ratings on the next deploy.

Usage:
    python -m play.scripts.llm_rating_games --games 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Any

import torch

from agent.env import batched_engine as BE
from agent.train import league as LG
from play.llm.policy import LLMBedrockPolicy
from replay import players as POL

# Suppress verbose LLM/boto logging — only show warnings+
import logging
logging.getLogger("play.llm").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LEAGUE_ROOT = WORKSPACE_ROOT / "agent" / "runs" / "league"

# Available opponents and their league entity IDs
OPPONENT_SPECS: dict[str, dict[str, Any]] = {
    "random": {"entity_id": "random", "factory": lambda: POL.RandomPolicy(seed=int(time.time()))},
    "heuristic": {"entity_id": "heuristic", "factory": lambda: POL.HeuristicPolicy()},
    "heuristic_opus": {"entity_id": "heuristic_opus", "factory": lambda: POL.HeuristicOpusPolicy()},
}


def _find_top_ml_bot(league: LG.League) -> dict[str, Any] | None:
    """Find the highest-rated ML bot in the league (min index 2489, file must exist)."""
    entries = league.list_entries()
    if not entries:
        return None
    # Only consider checkpoints from multi-player training onwards
    # and whose checkpoint file actually exists on disk
    eligible = [
        e for e in entries
        if int(e.get("idx", 0)) >= 2489
        and league._resolve_path(e["path"]).exists()
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda e: float(e.get("rating", 0)))


def _make_ml_policy(league: LG.League, entry: dict) -> POL.PlayerPolicy:
    """Create a NetPolicy from a league entry."""
    path = league._resolve_path(entry["path"])
    return POL.NetPolicy(path, num_sims=16, device="cpu")


def _pick_opponents(
    num_players: int,
    ml_entry: dict[str, Any] | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Pick (num_players - 1) random opponents from the pool."""
    pool = list(OPPONENT_SPECS.keys())
    if ml_entry is not None:
        pool.append("ml_bot")

    opponents = []
    for _ in range(num_players - 1):
        choice = rng.choice(pool)
        opponents.append(choice)
    return opponents


def _play_one_game(
    llm_policy: LLMBedrockPolicy,
    opponent_policies: list[POL.PlayerPolicy],
    num_players: int,
    llm_seat: int,
    seed: int,
    max_turns: int = 200,
    verbose: bool = False,
) -> dict[str, Any]:
    """Play a single game with the LLM at llm_seat, opponents at other seats.

    Returns dict with game results.
    """
    engine = BE.BatchedEngine(1, num_players, device="cpu", seed=seed)

    # Map seats to policies
    seat_policies: list[Any] = [None] * num_players
    seat_policies[llm_seat] = llm_policy
    opp_idx = 0
    for s in range(num_players):
        if s != llm_seat:
            seat_policies[s] = opponent_policies[opp_idx]
            opp_idx += 1

    turn = 0
    llm_moves = 0
    llm_time_total = 0.0
    fallbacks = 0

    if verbose and llm_moves == 0:
        from play.llm.prompts import SYSTEM_PROMPT_DEBUG
        print(f"\n{'='*60}")
        print(f"[LLM System Prompt]")
        print(f"{'='*60}")
        print(SYSTEM_PROMPT_DEBUG)
        print(f"{'='*60}\n")

    while turn < max_turns and not engine.ended.all():
        cp = int(engine.current_player[0])
        policy = seat_policies[cp]

        if cp == llm_seat:
            t0 = time.time()
            action_tensor = policy.choose(engine)
            elapsed = time.time() - t0
            llm_moves += 1
            llm_time_total += elapsed
            if verbose and int(engine.phase[0]) == 0:
                action_val = int(action_tensor[0])
                raw = policy.last_raw_response or "(no response)"
                reasoning = policy.last_reasoning
                print(f"\n{'='*60}")
                print(f"[LLM Agent Turn {llm_moves}] ({elapsed:.1f}s)")
                print(f"{'='*60}")
                print(f"[LLM Agent Prompt]")
                print(policy._last_user_prompt or "(not captured)")
                print(f"\n[LLM Agent Raw Response]")
                print(raw)
                print()
                if reasoning and not reasoning.startswith("[fallback:"):
                    print(f"[Parsed Response]")
                    print(f"  THINKING: {reasoning}")
                    print(f"  ACTION:   {action_val}")
                elif reasoning and reasoning.startswith("[fallback:"):
                    print(f"[Failed to parse LLM response]")
                    print(f"  Fallback: {reasoning}")
                    print(f"  ACTION (random): {action_val}")
                else:
                    print(f"[Parsed Response]")
                    print(f"  THINKING: (none)")
                    print(f"  ACTION:   {action_val}")
                print(f"{'='*60}\n")
        else:
            action_tensor = policy.choose(engine)

        engine.apply(action_tensor)
        turn += 1

    # Determine winner
    pts = engine.points[0].tolist()
    bonuses = engine.bonuses[0].sum(dim=-1).tolist()
    # Score: points * 1000 - bonus_count (fewer cards = tiebreak)
    scores = [pts[s] * 1000 - int(bonuses[s]) for s in range(num_players)]
    # Mask inactive seats
    for s in range(num_players):
        if not engine.active_mask[0, s]:
            scores[s] = -(10**9)
    winner_seat = scores.index(max(scores))
    # Check for ties
    max_score = max(scores)
    tied = sum(1 for s in scores if s == max_score) > 1
    finished = bool(engine.ended[0])

    return {
        "finished": finished,
        "tied": tied,
        "winner_seat": winner_seat,
        "llm_won": winner_seat == llm_seat and not tied,
        "llm_seat": llm_seat,
        "points": pts[:num_players],
        "turns": turn,
        "llm_moves": llm_moves,
        "llm_avg_time": llm_time_total / max(llm_moves, 1),
        "llm_total_time": llm_time_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play rated games with LLM agent against various opponents"
    )
    parser.add_argument(
        "--model",
        default="bedrock_claude_sonnet",
        choices=["bedrock_claude_sonnet"],
        help="LLM model to evaluate (default: bedrock_claude_sonnet)",
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=200,
        help="Max turns per game before declaring timeout (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: based on current time)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable LLM debug mode: ask for one-sentence justification per move",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFFFFFF
    rng = random.Random(seed)

    # Model config
    model_configs = {
        "bedrock_claude_sonnet": {
            "bedrock_model_id": "global.anthropic.claude-sonnet-4-6",
            "entity_id": "bedrock_claude_sonnet",
        },
    }
    model_cfg = model_configs[args.model]
    llm_entity = model_cfg["entity_id"]

    print(f"=== LLM Rating Games ===")
    print(f"Model: {args.model} (entity: {llm_entity})")
    print(f"Games: {args.games}")
    print(f"Seed: {seed}")
    print()

    # Load league
    league = LG.League(LEAGUE_ROOT)
    ml_entry = _find_top_ml_bot(league)
    if ml_entry:
        print(f"Top ML bot: idx={ml_entry['idx']}, rating={ml_entry['rating']:.1f}")
    else:
        print("No ML bot found in league")
    print()

    # Create LLM policy (always use debug mode for reasoning capture)
    llm_policy = LLMBedrockPolicy(
        model_id=args.model,
        bedrock_model_id=model_cfg["bedrock_model_id"],
        region="us-west-2",
        debug=True,
    )

    # Track results per opponent entity
    results_by_opponent: dict[str, dict[str, float]] = {}
    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_time = 0.0

    for game_num in range(args.games):
        # Random player count: 2, 3, or 4
        num_players = rng.choice([2, 3, 4])
        # Random LLM seat
        llm_seat = rng.randint(0, num_players - 1)
        # Pick opponents
        opponent_names = _pick_opponents(num_players, ml_entry, rng)

        # Build opponent policies
        opponent_policies: list[POL.PlayerPolicy] = []
        opponent_entities: list[str] = []
        for name in opponent_names:
            if name == "ml_bot":
                opponent_policies.append(_make_ml_policy(league, ml_entry))
                opponent_entities.append(league._entry_entity_id(int(ml_entry["idx"])))
            else:
                spec = OPPONENT_SPECS[name]
                opponent_policies.append(spec["factory"]())
                opponent_entities.append(spec["entity_id"])

        game_seed = rng.randint(0, 2**31)
        opp_desc = ", ".join(opponent_names)
        print(
            f"Game {game_num + 1}/{args.games}: "
            f"{num_players}p, LLM seat {llm_seat}, vs [{opp_desc}]",
            end="",
            flush=True,
        )
        if args.verbose:
            print()

        t0 = time.time()
        result = _play_one_game(
            llm_policy=llm_policy,
            opponent_policies=opponent_policies,
            num_players=num_players,
            llm_seat=llm_seat,
            seed=game_seed,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
        game_time = time.time() - t0
        total_time += game_time

        # Determine outcome
        if result["tied"]:
            outcome = "TIE"
            total_ties += 1
        elif result["llm_won"]:
            outcome = "WIN"
            total_wins += 1
        else:
            outcome = "LOSS"
            total_losses += 1

        if not args.verbose:
            print(
                f" → {outcome} "
                f"(pts={result['points']}, turns={result['turns']}, "
                f"llm_avg={result['llm_avg_time']:.1f}s, "
                f"game={game_time:.0f}s)"
            )
        else:
            print(
                f"  Result: {outcome}, points={result['points']}, "
                f"turns={result['turns']}, llm_avg={result['llm_avg_time']:.1f}s, "
                f"game_time={game_time:.0f}s"
            )

        # Record pairwise results into the league immediately after each game
        for opp_entity in opponent_entities:
            if opp_entity not in results_by_opponent:
                results_by_opponent[opp_entity] = {"wins": 0, "losses": 0, "ties": 0}

            if result["tied"]:
                results_by_opponent[opp_entity]["ties"] += 1
                league.record_result(llm_entity, opp_entity, 0.0, 0.0, 1.0)
            elif result["llm_won"]:
                results_by_opponent[opp_entity]["wins"] += 1
                league.record_result(llm_entity, opp_entity, 1.0, 0.0, 0.0)
            else:
                results_by_opponent[opp_entity]["losses"] += 1
                league.record_result(llm_entity, opp_entity, 0.0, 1.0, 0.0)

    # Recompute ratings
    print("\n--- Final results ---")
    for opp_entity, counts in results_by_opponent.items():
        print(f"  vs {opp_entity}: W={int(counts['wins'])} L={int(counts['losses'])} T={int(counts['ties'])}")

    ratings = league.recompute_ratings()
    llm_rating = ratings.get(llm_entity, 0.0)

    # Print summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Games played: {args.games}")
    print(f"Wins: {total_wins}, Losses: {total_losses}, Ties: {total_ties}")
    print(f"Win rate: {total_wins / max(args.games, 1):.1%}")
    print(f"Total time: {total_time:.0f}s ({total_time / max(args.games, 1):.0f}s/game avg)")
    print(f"\nLLM Rating ({args.model}): {llm_rating:.1f}")
    print(f"\nAll ratings after update:")
    for name, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>30s}: {rating:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

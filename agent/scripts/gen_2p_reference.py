"""Generate 2p reference matchups for the heuristic triangle.

Plays heuristic vs random, heuristic_opus vs random, and heuristic vs
heuristic_opus at 2p to establish a fixed anchor triangle for the 2p
rating scale.

Usage:
    python -m agent.scripts.gen_2p_reference [--num-games 2000] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from ..eval import bots as B
from ..eval import heuristic_opus as HO
from ..eval.tournament import Matchup, play_matchup


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2p reference triangle games")
    parser.add_argument("--num-games", type=int, default=2000,
                        help="Games per matchup (default: 2000)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    random_bot = B.RandomBot(seed=args.seed)
    heuristic_bot = B.HeuristicBot()
    opus_bot = HO.HeuristicOpusV15()

    pairs = [
        ("heuristic", heuristic_bot.choose, "random", random_bot.choose),
        ("heuristic_opus", opus_bot.choose, "random", random_bot.choose),
        ("heuristic", heuristic_bot.choose, "heuristic_opus", opus_bot.choose),
    ]

    results = []
    for name_a, policy_a, name_b, policy_b in pairs:
        matchup = Matchup(
            name_a=name_a,
            name_b=name_b,
            num_games=args.num_games,
            num_players=2,
            max_turns=200,
            seed=args.seed,
        )
        print(f"Playing {name_a} vs {name_b} ({args.num_games} games at 2p)...",
              flush=True)
        result = play_matchup(matchup, policy_a, policy_b, device=args.device)
        print(f"  wins_a={result.wins_a:.0f} wins_b={result.wins_b:.0f} "
              f"ties={result.ties:.0f} finished={result.games_finished}/{result.games_played}")
        results.append({
            "a": name_a,
            "b": name_b,
            "wins_a_2p": int(result.wins_a),
            "wins_b_2p": int(result.wins_b),
        })

    print("\n--- Results (paste into league.json or use for anchor computation) ---")
    print(json.dumps(results, indent=2))

    # Compute BT-MLE anchors from these results
    from ..train import ranking as R
    anchors_2p = R.fit_ratings_for_pc(results, pc=2, use_reference_anchors=False)
    print(f"\n--- Fitted 2p anchors ---")
    print(f"REFERENCE_ANCHORS_2P = {{")
    for k in sorted(anchors_2p.keys()):
        print(f'    "{k}": {anchors_2p[k]:.1f},')
    print(f"}}")


if __name__ == "__main__":
    main()

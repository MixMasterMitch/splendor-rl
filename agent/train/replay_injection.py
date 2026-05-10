"""Replay injection: convert saved human-vs-AI games into training samples.

Replays a game step-by-step through the BatchedEngine, encoding each state
and producing (state, policy_target, value_target) triples suitable for the
replay buffer.

Policy target: one-hot on the action actually taken (since we don't have MCTS
improved policy for human games, the actual move is the best signal we have).

Value target: the final game outcome (win=+1, loss=-1) discounted by
time_discount^(game_end - current_step), same as selfplay.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from .replay_buffer import ReplayBuffer


def _load_replay(path: pathlib.Path) -> dict:
    """Load a saved game replay from JSON."""
    with open(path) as f:
        return json.load(f)


def _final_values_from_ranks(ranks: list[int], num_players: int) -> torch.Tensor:
    """Convert rank list [0-indexed, 0=winner] to value targets in [-1, +1].

    rank 0 (winner) -> +1.0
    rank num_players-1 (last) -> -1.0
    Intermediate ranks linearly interpolated.
    """
    values = torch.zeros(BE.MAX_PLAYERS)
    for seat, rank in enumerate(ranks):
        if seat < num_players:
            # Linear scale: rank 0 -> +1, rank (n-1) -> -1
            values[seat] = 1.0 - 2.0 * rank / max(num_players - 1, 1)
    return values


def encode_replay_to_samples(
    replay: dict,
    time_discount: float = 1.0,
    device: str = "cpu",
) -> Optional[dict]:
    """Replay a game and produce training samples for all players.

    Returns a dict with tensors ready for buffer.add_batch():
        global_feat: (N, D_GLOBAL)
        card_feat: (N, N_CARDS, D_CARD)
        legal_mask: (N, NUM_ACTIONS)
        policy: (N, NUM_ACTIONS)
        value: (N, MAX_PLAYERS)

    Returns None if the replay can't be processed.
    """
    steps = replay.get("steps", [])
    num_players = int(replay.get("num_players", 2))
    seed = int(replay.get("seed", 0))

    if not steps:
        return None

    # Determine final outcome from the replay
    ranks = replay.get("ranks")
    if ranks is None:
        # Try to infer from final_scores
        final_scores = replay.get("final_scores")
        if final_scores:
            scores = [s.get("points", 0) for s in final_scores]
        else:
            # Try to get scores from the last step's state_after
            last_step = steps[-1] if steps else {}
            state_after = last_step.get("state_after", {})
            players = state_after.get("players", [])
            if players:
                scores = [p.get("points", 0) for p in players]
            else:
                # Last resort: check rating_update for winner
                rating_update = replay.get("rating_update") or replay.get("elo_update") or {}
                per_opp = rating_update.get("per_opponent", [])
                if per_opp:
                    human_seat = int(replay.get("human_seat", 0))
                    human_score = per_opp[0].get("score", 0)
                    # score=1 means human won, score=0 means AI won
                    ranks = [0] * num_players
                    if human_score > 0.5:
                        ranks[human_seat] = 0
                        for s in range(num_players):
                            if s != human_seat:
                                ranks[s] = 1
                    else:
                        ranks[human_seat] = 1
                        for s in range(num_players):
                            if s != human_seat:
                                ranks[s] = 0
                    scores = None
                else:
                    return None

        if ranks is None and scores is not None:
            # Rank by score descending (0 = best)
            sorted_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
            ranks = [0] * len(scores)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank

    final_values = _final_values_from_ranks(ranks, num_players)  # (MAX_PLAYERS,)

    # Replay the game through the engine
    engine = BE.BatchedEngine(1, num_players, device=device, seed=seed)

    rec_global: list[torch.Tensor] = []
    rec_card: list[torch.Tensor] = []
    rec_legal: list[torch.Tensor] = []
    rec_policy: list[torch.Tensor] = []
    rec_step: list[int] = []
    rec_cp: list[int] = []

    game_length = len([s for s in steps if s.get("phase", 0) == 0])
    step_counter = 0

    for step_data in steps:
        action_idx = step_data.get("action")
        if action_idx is None:
            continue

        # Encode current state
        g, c, _, legal = ENC.encode_state_with_legal(engine)

        # Policy target: one-hot on the action taken
        policy = torch.zeros(1, A.NUM_ACTIONS)
        policy[0, action_idx] = 1.0

        rec_global.append(g)
        rec_card.append(c)
        rec_legal.append(legal)
        rec_policy.append(policy)
        rec_cp.append(int(engine.current_player[0].item()))
        rec_step.append(step_counter)

        # Apply the action
        action_tensor = torch.tensor([action_idx], dtype=torch.int64, device=device)
        engine.apply(action_tensor)

        if step_data.get("phase", 0) == 0:
            step_counter += 1

    if not rec_global:
        return None

    all_g = torch.cat(rec_global, dim=0)  # (N, D_GLOBAL)
    all_c = torch.cat(rec_card, dim=0)  # (N, N_CARDS, D_CARD)
    all_l = torch.cat(rec_legal, dim=0)  # (N, NUM_ACTIONS)
    all_p = torch.cat(rec_policy, dim=0)  # (N, NUM_ACTIONS)

    N = all_g.shape[0]

    # Value targets: rotate final_values so seat 0 = current player for each sample
    all_values = torch.zeros(N, BE.MAX_PLAYERS)
    for i, cp in enumerate(rec_cp):
        # Rotate so current player is seat 0
        rotated = torch.zeros(BE.MAX_PLAYERS)
        for s in range(num_players):
            original_seat = (s + cp) % num_players
            rotated[s] = final_values[original_seat]
        all_values[i] = rotated

    # Apply time discount
    steps_tensor = torch.tensor(rec_step, dtype=torch.float32)
    ttg = (float(game_length) - steps_tensor).clamp_min(0)
    discount = torch.pow(torch.tensor(time_discount), ttg).unsqueeze(-1)
    all_values = all_values * discount

    return {
        "global_feat": all_g,
        "card_feat": all_c,
        "legal_mask": all_l,
        "policy": all_p,
        "value": all_values,
    }


def load_all_replays(
    replay_dir: str | pathlib.Path = "agent/training_replays",
    time_discount: float = 1.0,
    device: str = "cpu",
) -> Optional[dict]:
    """Load and encode all replay files in the directory.

    Returns combined tensors ready for buffer.add_batch(), or None if no
    valid replays found.
    """
    replay_dir = pathlib.Path(replay_dir)
    if not replay_dir.exists():
        return None

    all_samples: list[dict] = []
    for path in sorted(replay_dir.glob("*.json")):
        try:
            replay = _load_replay(path)
            samples = encode_replay_to_samples(replay, time_discount=time_discount, device=device)
            if samples is not None:
                all_samples.append(samples)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to encode replay %s: %s", path, exc)
            continue

    if not all_samples:
        return None

    return {
        "global_feat": torch.cat([s["global_feat"] for s in all_samples], dim=0),
        "card_feat": torch.cat([s["card_feat"] for s in all_samples], dim=0),
        "legal_mask": torch.cat([s["legal_mask"] for s in all_samples], dim=0),
        "policy": torch.cat([s["policy"] for s in all_samples], dim=0),
        "value": torch.cat([s["value"] for s in all_samples], dim=0),
    }


def inject_replays_into_buffer(
    buffer: ReplayBuffer,
    replay_dir: str | pathlib.Path = "agent/training_replays",
    time_discount: float = 1.0,
) -> int:
    """Load all replays and inject them into the buffer.

    Returns the number of samples injected.
    """
    samples = load_all_replays(
        replay_dir=replay_dir,
        time_discount=time_discount,
        device=str(buffer.device),
    )
    if samples is None:
        return 0

    n = samples["global_feat"].shape[0]
    buffer.add_batch(
        samples["global_feat"],
        samples["card_feat"],
        samples["legal_mask"],
        samples["policy"],
        samples["value"],
    )
    return n

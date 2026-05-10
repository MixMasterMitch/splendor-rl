"""Batched self-play: runs a population of games to completion with the current
network, collecting (state, policy, value) triples for the replay buffer.

For each game turn we record the encoded state, legal mask, the improved policy
from `gumbel_root_act`, and (only once the game ends) the per-seat final
placement target: +1 for a winning seat, -1 for a losing seat, 0 for ties,
rotated so seat 0 is the acting seat at recording time.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search import gumbel_mcts as G
from . import active_batching as AB
from .replay_buffer import ReplayBuffer


def _final_rank_values(engine: BE.BatchedEngine) -> torch.Tensor:
    """For each game, return (B, MAX_PLAYERS) where inactive seats are 0 and
    active seats carry zero-mean normalized rewards based on ranking.

    Normalization: win = +(n-1)/n, loss = -1/n, where n = number of active
    players. This ensures the expected value of a random policy is 0.0
    regardless of player count, giving balanced gradient signal across 2p/3p/4p.
    """
    B = engine.batch_size
    pts = engine.points.to(torch.int32)
    bonuses = engine.bonuses.sum(dim=-1).to(torch.int32)
    score = pts * 1000 - bonuses
    score = torch.where(
        engine.active_mask, score, torch.full_like(score, -(10**9))
    )
    best = score.max(dim=-1).values.unsqueeze(-1)
    winners = (score == best) & engine.active_mask

    # Number of active players per game (B,)
    n_players = engine.active_mask.sum(dim=-1, dtype=torch.float32).unsqueeze(-1)  # (B, 1)
    # Zero-mean rewards: win = +(n-1)/n, loss = -1/n
    win_reward = (n_players - 1.0) / n_players   # (B, 1)
    loss_reward = -1.0 / n_players                # (B, 1)

    values = torch.where(winners, win_reward, loss_reward)
    values = torch.where(
        engine.active_mask, values, torch.zeros_like(values, dtype=torch.float32)
    )
    return values


def _rotate_for_cp(values: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """Rotate seat values so index 0 corresponds to current player at record
    time. values (B, P), cp (B,) int."""
    B, P = values.shape
    idx = (
        torch.arange(P, device=values.device).unsqueeze(0) + cp.unsqueeze(-1)
    ) % P
    return values.gather(1, idx.to(torch.long))


def _default_temperature_schedule(step: int) -> float:
    """Warm -> cool schedule. Hot at the start (more exploration), cold once
    games should be winding down (commit to the best move).
    """
    if step < 30:
        return 1.0
    if step < 60:
        return 0.5
    return 0.25


def run_selfplay(
    net: M.SplendorNet,
    buffer: ReplayBuffer,
    num_players: int = 2,
    num_games: int = 128,
    device: str = "cpu",
    max_turns: int = 200,
    num_sims: int = 4,
    seed: int = 0,
    temperature_schedule: Optional[callable] = None,
    time_discount: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_mix: float = 0.25,
    q_scale: float = 10.0,
) -> dict:
    """Play `num_games` games in parallel to completion. Writes samples to the
    buffer. Returns simple metrics.

    Value target shaping: we apply a per-position discount
    `time_discount ** (game_end_step - sample_step)` to the terminal reward.
    Without this, the net learns that *any* win/loss carries the same signal
    regardless of how long it took, and has no pressure to finish the game.
    The symptom was agents that looped TAKE3 -> DISCARD -> TAKE3 -> ... forever.
    """
    if temperature_schedule is None:
        temperature_schedule = _default_temperature_schedule

    net.eval()
    net.to(device)

    engine = BE.BatchedEngine(num_games, num_players, device=device, seed=seed)
    storage_device = buffer.device

    t_start = time.monotonic()

    # Per-turn recordings (policy/value supervision)
    rec_global: list[torch.Tensor] = []
    rec_card: list[torch.Tensor] = []
    rec_legal: list[torch.Tensor] = []
    rec_policy: list[torch.Tensor] = []
    rec_cp: list[torch.Tensor] = []
    rec_game_idx: list[torch.Tensor] = []
    rec_step: list[torch.Tensor] = []

    # Track the step at which each game ended (for discounting).
    game_end_step = torch.full(
        (num_games,), fill_value=max_turns, dtype=torch.int32, device=storage_device
    )
    prev_ended = torch.zeros(num_games, dtype=torch.bool, device=storage_device)

    turn = 0
    while turn < max_turns and (~engine.ended).any():
        alive = ~engine.ended
        alive_idx = alive.nonzero(as_tuple=True)[0]
        temp = temperature_schedule(turn)
        actions = torch.full(
            (engine.batch_size,),
            A.PASS_ACTION,
            dtype=torch.int64,
            device=engine.device,
        )
        use_bucketed_subset = (
            getattr(net, "_compiled", False)
            and AB.should_bucket_compact(int(alive_idx.numel()), engine.batch_size)
        )
        use_single_subset = alive_idx.numel() < engine.batch_size and not getattr(net, "_compiled", False)
        if alive_idx.numel() == engine.batch_size or not (use_bucketed_subset or use_single_subset):
            g, c, _, legal = ENC.encode_state_with_legal(engine)
            with torch.no_grad():
                actions, improved = G.gumbel_root_act(
                    engine,
                    net,
                    num_sims=num_sims,
                    temperature=temp,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_mix=dirichlet_mix,
                    q_scale=q_scale,
                    precomputed=(g, c, legal),
                )
            if alive_idx.numel() > 0:
                rec_global.append(g.index_select(0, alive_idx).to(storage_device))
                rec_card.append(c.index_select(0, alive_idx).to(storage_device))
                rec_legal.append(legal.index_select(0, alive_idx).to(storage_device))
                rec_policy.append(improved.index_select(0, alive_idx).to(storage_device))
                rec_cp.append(engine.current_player.index_select(0, alive_idx).to(storage_device))
                rec_game_idx.append(alive_idx.to(storage_device))
                rec_step.append(
                    torch.full(
                        (alive_idx.numel(),),
                        turn,
                        dtype=torch.int32,
                        device=storage_device,
                    )
                )
        else:
            idx_groups = (
                AB.bucket_indices(alive_idx, max_bucket=engine.batch_size)
                if use_bucketed_subset
                else [alive_idx]
            )
            for idx in idx_groups:
                sub_engine = engine.index_select(idx)
                g, c, _, legal = ENC.encode_state_with_legal(sub_engine)
                with torch.no_grad():
                    sub_actions, improved = G.gumbel_root_act(
                        sub_engine,
                        net,
                        num_sims=num_sims,
                        temperature=temp,
                        dirichlet_alpha=dirichlet_alpha,
                        dirichlet_mix=dirichlet_mix,
                        q_scale=q_scale,
                        precomputed=(g, c, legal),
                    )
                actions.index_copy_(0, idx, sub_actions)
                rec_global.append(g.to(storage_device))
                rec_card.append(c.to(storage_device))
                rec_legal.append(legal.to(storage_device))
                rec_policy.append(improved.to(storage_device))
                rec_cp.append(sub_engine.current_player.to(storage_device))
                rec_game_idx.append(idx.to(storage_device))
                rec_step.append(
                    torch.full(
                        (idx.numel(),),
                        turn,
                        dtype=torch.int32,
                        device=storage_device,
                    )
                )
        engine.apply(actions)
        # Capture end-of-game step for games that transitioned this step.
        cur_ended = engine.ended.to(storage_device)
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

    final_values = _final_rank_values(engine).to(storage_device)  # (B, P)
    wall_s = time.monotonic() - t_start

    # For games that did NOT finish within max_turns, assign -1 to every
    # active seat: stalling out is treated as a group loss. Without this the
    # agent could learn that stalling a losing game carries no signal and is
    # therefore preferable to finishing. The value is set AFTER computing
    # final_values so it replaces the neutral-zero defaults for inactive
    # seats with stall penalties only for the active ones.
    ended_mask = engine.ended.to(storage_device)
    active_mask = engine.active_mask.to(storage_device)
    unfinished = ~ended_mask  # (B,)
    if unfinished.any():
        stall_penalty = torch.where(
            active_mask[unfinished],
            torch.full_like(final_values[unfinished], -1.0),
            torch.zeros_like(final_values[unfinished]),
        )
        final_values[unfinished] = stall_penalty

    if not rec_global:
        return {
            "steps": turn,
            "avg_finished_step": 0.0,
            "max_finished_step": 0.0,
            "finished": int(ended_mask.sum().item()),
            "samples_added": 0,
            "games_total": int(num_games),
            "wall_s": round(wall_s, 3),
            "games_per_s": 0.0,
            "steps_per_s": round(turn / wall_s, 2) if wall_s > 0 else 0.0,
        }

    all_g = torch.cat(rec_global, dim=0)
    all_c = torch.cat(rec_card, dim=0)
    all_l = torch.cat(rec_legal, dim=0)
    all_p = torch.cat(rec_policy, dim=0)
    all_cp = torch.cat(rec_cp, dim=0)
    all_gi = torch.cat(rec_game_idx, dim=0)
    all_step = torch.cat(rec_step, dim=0)

    per_sample_values = final_values[all_gi]  # (N, P)
    rotated = _rotate_for_cp(per_sample_values, all_cp.to(torch.long))  # (N, P)

    # Per-position time discount applies ONLY to finished games, so fast
    # wins get more signal. For unfinished games we keep the raw -1 penalty
    # so the agent is pressured out of stalling at every position.
    finished_mask = ended_mask[all_gi]
    ttg = (game_end_step[all_gi].to(torch.float32) - all_step.to(torch.float32)).clamp_min(0)
    discount = torch.pow(torch.tensor(time_discount, dtype=torch.float32), ttg)
    apply_discount = finished_mask.to(discount.dtype)
    effective_discount = apply_discount * discount + (1.0 - apply_discount)
    discounted = rotated * effective_discount.unsqueeze(-1)

    buffer.add_batch(all_g, all_c, all_l, all_p, discounted)

    finished_count = int(engine.ended.sum().item())
    avg_game_turns = turn / max(num_players, 1)
    # Average length (in global step count) of games that finished.
    if finished_count > 0:
        fin_lens = game_end_step[ended_mask].to(torch.float32)
        avg_finished_len = float(fin_lens.mean().item())
        max_finished_len = float(fin_lens.max().item())
    else:
        avg_finished_len = 0.0
        max_finished_len = 0.0
    return {
        "steps": turn,
        "avg_game_turns": round(avg_game_turns, 2),
        "avg_finished_step": round(avg_finished_len, 2),
        "max_finished_step": round(max_finished_len, 2),
        "finished": finished_count,
        "games_total": int(num_games),
        "samples_added": int(all_g.shape[0]),
        "samples_from_finished": (
            int(finished_mask.sum().item()) if finished_mask.numel() else 0
        ),
        "avg_points_winner": round(
            float(engine.points.max(dim=-1).values.float().mean().item()), 3
        ),
        "wall_s": round(wall_s, 3),
        "games_per_s": round(finished_count / wall_s, 2) if wall_s > 0 else 0.0,
        "steps_per_s": round(turn / wall_s, 2) if wall_s > 0 else 0.0,
    }

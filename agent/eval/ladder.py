"""Evaluation harness: measures the current net's win-rate against reference bots
and recent league checkpoints.

Each matchup plays `num_games` batched games where seats are filled by two
players drawn from (agent, opponent). The agent always sits at seat 0 or seat 1
alternately to neutralize first-player advantage; results are averaged.

Produces a dict of metrics suitable for logging to `metrics.jsonl`:
- winrate_vs_random
- winrate_vs_heuristic
- avg_game_length
- rating estimates (anchored Bradley-Terry; random=1000, heuristic=2500).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search import gumbel_mcts as G
from ..train import active_batching as AB
from . import bots as B
from . import heuristic_opus as HO


def _agent_act(
    net: M.SplendorNet,
    engine: BE.BatchedEngine,
    num_sims: int,
    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        act, _ = G.gumbel_root_act(
            engine, net, num_sims=num_sims, precomputed=precomputed
        )
    return act


def _apply_mixed(
    engine: BE.BatchedEngine,
    seat_is_agent: torch.Tensor,
    net: M.SplendorNet,
    bot_choose: Callable[[BE.BatchedEngine], torch.Tensor],
    num_sims: int,
) -> None:
    """Applies one action to each game using either agent or bot per current player."""
    alive = ~engine.ended
    if not alive.any():
        return
    cp = engine.current_player.to(torch.long)
    agent_games = alive & seat_is_agent[torch.arange(engine.batch_size, device=engine.device), cp]
    compiled_net = getattr(net, "_compiled", False)
    use_subset = not compiled_net

    actions = torch.full(
        (engine.batch_size,),
        A.PASS_ACTION,
        dtype=torch.int64,
        device=engine.device,
    )

    if agent_games.any():
        agent_idx = agent_games.nonzero(as_tuple=True)[0]
        use_bucketed_agent = compiled_net and AB.should_bucket_compact(
            int(agent_idx.numel()), engine.batch_size
        )
        if (use_subset and agent_idx.numel() < engine.batch_size) or use_bucketed_agent:
            idx_groups = (
                AB.bucket_indices(agent_idx, max_bucket=engine.batch_size)
                if use_bucketed_agent
                else [agent_idx]
            )
            for idx in idx_groups:
                agent_engine = engine.index_select(idx)
                g, c, _, legal = ENC.encode_state_with_legal(agent_engine)
                agent_actions = _agent_act(
                    net, agent_engine, num_sims, precomputed=(g, c, legal)
                )
                actions.index_copy_(0, idx, agent_actions)
        else:
            g, c, _, legal = ENC.encode_state_with_legal(engine)
            agent_actions = _agent_act(
                net, engine, num_sims, precomputed=(g, c, legal)
            )
            actions = torch.where(agent_games, agent_actions, actions)

    bot_needed = alive & ~agent_games
    if bot_needed.any():
        bot_idx = bot_needed.nonzero(as_tuple=True)[0]
        if bot_idx.numel() < engine.batch_size:
            bot_engine = engine.index_select(bot_idx)
            bot_actions = bot_choose(bot_engine)
            actions.index_copy_(0, bot_idx, bot_actions)
        else:
            bot_actions = bot_choose(engine)
            actions = torch.where(bot_needed, bot_actions, actions)

    engine.apply(actions)


def _play_match(
    engine: BE.BatchedEngine,
    seat_is_agent: torch.Tensor,
    net: M.SplendorNet,
    bot_choose: Callable[[BE.BatchedEngine], torch.Tensor],
    max_turns: int,
    num_sims: int,
) -> Dict[str, float]:
    turn = 0
    game_end_step = torch.full(
        (engine.batch_size,),
        fill_value=max_turns,
        dtype=torch.int32,
        device=engine.device,
    )
    prev_ended = torch.zeros(engine.batch_size, dtype=torch.bool, device=engine.device)
    # Over-limit tracking: count how often the agent enters discard phase
    agent_overlimit_count = 0
    agent_main_actions_count = 0
    while turn < max_turns and (~engine.ended).any():
        # Track pre-action state for over-limit detection
        alive = ~engine.ended
        cp = engine.current_player.to(torch.long)
        is_agent_turn = alive & seat_is_agent[torch.arange(engine.batch_size, device=engine.device), cp]
        phase_before = engine.phase.clone()
        is_main_phase = (phase_before == 0) & is_agent_turn
        agent_main_actions_count += int(is_main_phase.sum().item())

        _apply_mixed(engine, seat_is_agent, net, bot_choose, num_sims)

        # Detect transitions into discard phase (phase 0 → phase 1) for agent seats
        entered_discard = is_main_phase & (engine.phase == 1)
        agent_overlimit_count += int(entered_discard.sum().item())

        cur_ended = engine.ended
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

    B = engine.batch_size
    # Determine winner per game: highest (points, -bonus_count)
    pts = engine.points.to(torch.int32)
    bonuses_total = engine.bonuses.sum(dim=-1).to(torch.int32)
    # score = pts * 1000 - bonuses_total (higher is better)
    score = pts * 1000 - bonuses_total
    # Mask inactive seats with -inf
    score = torch.where(engine.active_mask, score, torch.full_like(score, -(10**9)))
    winner = score.argmax(dim=-1)  # (B,)
    # Agent win if winner seat is an agent seat in that game
    agent_won = seat_is_agent[torch.arange(B, device=engine.device), winner]
    finished = engine.ended
    finished_count = finished.sum().item()
    capped_count = B - finished_count
    agent_wins = (agent_won & finished).sum().item()
    # Ties: pure highest score; our tie-breaking via bonuses; remaining ties count as 0.5
    ties = (
        (score == score[torch.arange(B, device=engine.device), winner].unsqueeze(-1)).sum(dim=-1)
        > 1
    )
    tie_and_agent = (agent_won & finished & ties).sum().item()
    turns_sum = float(game_end_step.to(torch.float32).sum().item())
    if finished_count > 0:
        finished_turns_sum = float(game_end_step[finished].to(torch.float32).sum().item())
        max_finished_step = float(game_end_step[finished].max().item())
    else:
        finished_turns_sum = 0.0
        max_finished_step = 0.0
    return {
        "games_finished": float(finished_count),
        "games_capped": float(capped_count),
        "games_total": float(B),
        "agent_wins": float(agent_wins),
        "agent_ties": float(tie_and_agent),
        "turns_sum": turns_sum,
        "finished_turns_sum": finished_turns_sum,
        "max_finished_step": max_finished_step,
        "agent_overlimit_count": float(agent_overlimit_count),
        "agent_main_actions": float(agent_main_actions_count),
        "agent_overlimit_rate": (
            float(agent_overlimit_count) / max(agent_main_actions_count, 1)
        ),
    }


def evaluate(
    net: M.SplendorNet,
    num_players: int = 2,
    num_games: int = 64,
    opponents: Optional[Dict[str, Callable[[], Callable[[BE.BatchedEngine], torch.Tensor]]]] = None,
    device: str = "cpu",
    num_sims: int = 4,
    max_turns: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluates the agent vs. each opponent. Returns flat dict of metrics.

    opponents: maps name -> factory producing a callable bot.choose(engine).
    Defaults to random + heuristic.
    """
    if opponents is None:
        opponents = {
            "random": lambda: B.RandomBot(seed=seed).choose,
            "heuristic": lambda: B.HeuristicBot().choose,
            "heuristic_opus": lambda: HO.HeuristicOpusV15().choose,
        }

    results: Dict[str, float] = {}
    net.eval()
    net.to(device)

    t_eval_start = time.monotonic()
    total_games = 0
    total_finished = 0

    for name, factory in opponents.items():
        bot_choose = factory()
        # Rotate the agent through all num_players seats so first-player
        # advantage averages out. We run one batched engine per rotation so
        # that `agent_games` is "all alive" on the agent's turn and "empty"
        # on the bot's turn; this lets `_apply_mixed` skip MCTS on bot turns
        # entirely via the `if agent_games.any()` guard.
        per_seat = max(1, num_games // num_players)
        wins = 0.0
        ties = 0.0
        finished = 0
        capped = 0
        turns_sum = 0.0
        finished_turns_sum = 0.0
        max_finished_step = 0.0
        overlimit_count = 0
        main_actions_count = 0

        for seat_of_agent in range(num_players):
            eng = BE.BatchedEngine(
                per_seat, num_players, device=device, seed=seed + seat_of_agent
            )
            seat_is_agent = torch.zeros(
                (per_seat, BE.MAX_PLAYERS), dtype=torch.bool, device=eng.device
            )
            seat_is_agent[:, seat_of_agent] = True
            result = _play_match(
                eng, seat_is_agent, net, bot_choose, max_turns, num_sims
            )
            wins += result["agent_wins"]
            ties += result["agent_ties"]
            finished += int(result["games_finished"])
            capped += int(result["games_capped"])
            turns_sum += result["turns_sum"]
            finished_turns_sum += result["finished_turns_sum"]
            max_finished_step = max(max_finished_step, float(result["max_finished_step"]))
            overlimit_count += int(result["agent_overlimit_count"])
            main_actions_count += int(result["agent_main_actions"])

        total = num_players * per_seat
        results[f"winrate_vs_{name}"] = wins / max(total, 1)
        results[f"ties_vs_{name}"] = ties / max(total, 1)
        results[f"finished_vs_{name}"] = finished / max(total, 1)
        results[f"capped_vs_{name}"] = capped / max(total, 1)
        results[f"avg_turns_vs_{name}"] = turns_sum / max(total, 1)
        results[f"avg_finished_step_vs_{name}"] = (
            finished_turns_sum / max(finished, 1) if finished > 0 else 0.0
        )
        results[f"max_finished_step_vs_{name}"] = max_finished_step
        results[f"overlimit_rate_vs_{name}"] = (
            overlimit_count / max(main_actions_count, 1)
        )
        results[f"overlimit_count_vs_{name}"] = float(overlimit_count)
        total_games += total
        total_finished += finished

    wall_s = time.monotonic() - t_eval_start
    results["eval_wall_s"] = round(wall_s, 3)
    results["eval_games_total"] = float(total_games)
    results["eval_games_per_s"] = (
        round(total_finished / wall_s, 2) if wall_s > 0 else 0.0
    )
    return results

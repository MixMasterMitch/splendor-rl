"""Self-play with a mix of opponents sampled from the league.

Each self-play game randomly assigns seats to either the latest agent or a
sampled past checkpoint. Only states where the latest agent is the current
player are recorded to the replay buffer. This is a simpler but effective
variant of population-based training.

The exploiter track is a separate network that is trained *only on* games
against the latest main agent. Its goal is to find weaknesses in the main
agent. The exploiter's losing behavior generates strong adversarial data
that the main agent eventually learns from via interaction.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search import gumbel_mcts as G
from . import active_batching as AB
from .league import League
from .replay_buffer import ReplayBuffer
from .selfplay import _final_rank_values, _rotate_for_cp


def _build_seat_path_idx(
    seat_net_path: dict[tuple[int, int], Optional[str]],
    num_games: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    seat_path_idx = torch.full(
        (num_games, BE.MAX_PLAYERS),
        -1,
        dtype=torch.long,
        device=device,
    )
    path_to_idx: dict[str, int] = {}
    path_list: list[str] = []
    for (b, p), path in seat_net_path.items():
        if path is None:
            continue
        idx = path_to_idx.get(path)
        if idx is None:
            idx = len(path_list)
            path_to_idx[path] = idx
            path_list.append(path)
        seat_path_idx[b, p] = idx
    return seat_path_idx, path_list


def _choose_seat_actions(
    engine: BE.BatchedEngine,
    main_net: M.SplendorNet,
    league: League,
    seat_path_idx: torch.Tensor,
    path_list: list[str],
    num_sims: int,
    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device_t = engine.device
    g, c, legal = precomputed
    alive = ~engine.ended
    batch_idx = torch.arange(engine.batch_size, device=device_t)
    cp = engine.current_player.to(torch.long)
    active_path_idx = seat_path_idx[batch_idx, cp]
    is_main = (active_path_idx < 0) & alive

    actions = torch.full(
        (engine.batch_size,),
        A.PASS_ACTION,
        dtype=torch.int64,
        device=device_t,
    )

    main_idx = is_main.nonzero(as_tuple=True)[0]
    main_improved: torch.Tensor | None = None
    compiled_main = getattr(main_net, "_compiled", False)
    use_main_subset = not compiled_main and main_idx.numel() > 0 and main_idx.numel() < engine.batch_size
    use_main_bucketed = compiled_main and AB.should_bucket_compact(
        int(main_idx.numel()), engine.batch_size
    )
    if main_idx.numel() > 0:
        if use_main_subset or use_main_bucketed:
            idx_groups = (
                AB.bucket_indices(main_idx, max_bucket=engine.batch_size)
                if use_main_bucketed
                else [main_idx]
            )
            main_improved = torch.zeros(
                (main_idx.numel(), A.NUM_ACTIONS),
                dtype=torch.float32,
                device=device_t,
            )
            start = 0
            for idx in idx_groups:
                main_engine = engine.index_select(idx)
                main_g = g.index_select(0, idx)
                main_c = c.index_select(0, idx)
                main_legal = legal.index_select(0, idx)
                with torch.no_grad():
                    main_actions, improved = G.gumbel_root_act(
                        main_engine,
                        main_net,
                        num_sims=num_sims,
                        precomputed=(main_g, main_c, main_legal),
                    )
                actions.index_copy_(0, idx, main_actions)
                main_improved[start : start + idx.numel()] = improved
                start += idx.numel()
        else:
            with torch.no_grad():
                main_actions, improved = G.gumbel_root_act(
                    engine,
                    main_net,
                    num_sims=num_sims,
                    precomputed=(g, c, legal),
                )
            actions = torch.where(is_main, main_actions, actions)
            main_improved = improved.index_select(0, main_idx)

    opponent_games = alive & ~is_main
    if opponent_games.any():
        active_opponents = active_path_idx[opponent_games]
        active_opponents = active_opponents[active_opponents >= 0]
        for path_idx in torch.unique(active_opponents).tolist():
            opp_idx = ((active_path_idx == path_idx) & opponent_games).nonzero(as_tuple=True)[0]
            if opp_idx.numel() == 0:
                continue
            opp = league.load_cached_net(path_list[path_idx], device_t)
            opp_engine = engine.index_select(opp_idx)
            opp_g = g.index_select(0, opp_idx)
            opp_c = c.index_select(0, opp_idx)
            opp_legal = legal.index_select(0, opp_idx)
            with torch.no_grad():
                opp_actions, _ = G.gumbel_root_act(
                    opp_engine,
                    opp,
                    num_sims=num_sims,
                    precomputed=(opp_g, opp_c, opp_legal),
                )
            actions.index_copy_(0, opp_idx, opp_actions)

    return actions, main_idx, main_improved


def run_league_selfplay(
    main_net: M.SplendorNet,
    buffer: ReplayBuffer,
    league: League,
    num_players: int = 2,
    num_games: int = 128,
    device: str = "cpu",
    max_turns: int = 200,
    num_sims: int = 4,
    seed: int = 0,
    league_prob: float = 0.5,
    time_discount: float = 0.995,
) -> dict:
    """Runs games where each game randomly has 0, 1, or more seats filled by
    past-league opponents (with probability `league_prob` per seat). Only the
    main-agent seats contribute records to the buffer.
    """
    device_t = torch.device(device)
    main_net.eval().to(device_t)
    storage_device = buffer.device

    # Sample per-seat league opponents for each game
    rng = random.Random(seed)
    engine = BE.BatchedEngine(num_games, num_players, device=device, seed=seed)

    t_start = time.monotonic()

    # For each (game, seat), decide whether to use main_net (None) or a league entry
    seat_net_path: dict[tuple[int, int], Optional[str]] = {}
    league_entries = league.list_entries()
    for b in range(num_games):
        for p in range(num_players):
            if league_entries and rng.random() < league_prob:
                e = league.sample_opponent(rng)
                seat_net_path[(b, p)] = e["path"] if e else None
            else:
                seat_net_path[(b, p)] = None

    seat_path_idx, path_list = _build_seat_path_idx(seat_net_path, num_games, device_t)

    rec_global: list[torch.Tensor] = []
    rec_card: list[torch.Tensor] = []
    rec_legal: list[torch.Tensor] = []
    rec_policy: list[torch.Tensor] = []
    rec_cp: list[torch.Tensor] = []
    rec_game_idx: list[torch.Tensor] = []
    rec_step: list[torch.Tensor] = []

    game_end_step = torch.full(
        (num_games,), fill_value=max_turns, dtype=torch.int32, device=storage_device
    )
    prev_ended = torch.zeros(num_games, dtype=torch.bool, device=storage_device)

    turn = 0
    while turn < max_turns and (~engine.ended).any():
        g, c, _, legal = ENC.encode_state_with_legal(engine)
        cp = engine.current_player.to(torch.long)
        actions, main_idx, main_improved = _choose_seat_actions(
            engine,
            main_net,
            league,
            seat_path_idx,
            path_list,
            num_sims,
            precomputed=(g, c, legal),
        )
        if main_improved is not None and main_idx.numel() > 0:
            rec_global.append(g.index_select(0, main_idx).to(storage_device))
            rec_card.append(c.index_select(0, main_idx).to(storage_device))
            rec_legal.append(legal.index_select(0, main_idx).to(storage_device))
            rec_policy.append(main_improved.to(storage_device))
            rec_cp.append(cp.index_select(0, main_idx).to(storage_device))
            rec_game_idx.append(main_idx.to(storage_device))
            rec_step.append(
                torch.full(
                    (main_idx.numel(),),
                    turn,
                    dtype=torch.int32,
                    device=storage_device,
                )
            )
        engine.apply(actions)
        cur_ended = engine.ended.to(storage_device)
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

    final_values = _final_rank_values(engine).to(storage_device)
    ended_mask = engine.ended.to(storage_device)
    active_mask = engine.active_mask.to(storage_device)
    unfinished = ~ended_mask
    if unfinished.any():
        stall_penalty = torch.where(
            active_mask[unfinished],
            torch.full_like(final_values[unfinished], -1.0),
            torch.zeros_like(final_values[unfinished]),
        )
        final_values[unfinished] = stall_penalty

    samples_added = 0
    samples_from_finished = 0
    if rec_global:
        all_g = torch.cat(rec_global, dim=0)
        all_c = torch.cat(rec_card, dim=0)
        all_l = torch.cat(rec_legal, dim=0)
        all_p = torch.cat(rec_policy, dim=0)
        all_cp = torch.cat(rec_cp, dim=0)
        all_gi = torch.cat(rec_game_idx, dim=0)
        all_step = torch.cat(rec_step, dim=0)
        per_sample_values = final_values[all_gi]
        rotated = _rotate_for_cp(per_sample_values, all_cp.to(torch.long))
        finished_mask = ended_mask[all_gi]
        ttg = (game_end_step[all_gi].to(torch.float32) - all_step.to(torch.float32)).clamp_min(0)
        discount = torch.pow(torch.tensor(time_discount, dtype=torch.float32), ttg)
        apply_discount = finished_mask.to(discount.dtype)
        effective_discount = apply_discount * discount + (1.0 - apply_discount)
        discounted = rotated * effective_discount.unsqueeze(-1)
        buffer.add_batch(all_g, all_c, all_l, all_p, discounted)
        samples_added = int(all_g.shape[0])
        samples_from_finished = int(finished_mask.sum().item())

    wall_s = time.monotonic() - t_start
    finished = int(ended_mask.sum().item())
    if finished > 0:
        fin_lens = game_end_step[ended_mask].to(torch.float32)
        avg_finished_len = float(fin_lens.mean().item())
        max_finished_len = float(fin_lens.max().item())
    else:
        avg_finished_len = 0.0
        max_finished_len = 0.0
    return {
        "steps": turn,
        "avg_game_turns": round(turn / max(num_players, 1), 2),
        "avg_finished_step": round(avg_finished_len, 2),
        "max_finished_step": round(max_finished_len, 2),
        "finished": finished,
        "games_total": int(num_games),
        "samples_added": samples_added,
        "samples_from_finished": samples_from_finished,
        "avg_points_winner": round(
            float(engine.points.max(dim=-1).values.float().mean().item()), 3
        ),
        "league_opponents_used": len(path_list),
        "wall_s": round(wall_s, 3),
        "games_per_s": round(finished / wall_s, 2) if wall_s > 0 else 0.0,
        "steps_per_s": round(turn / wall_s, 2) if wall_s > 0 else 0.0,
    }

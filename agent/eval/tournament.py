"""Round-robin tournament harness for heuristic candidate Splendor bots.

The training league (``agent/train/league.py``) tracks ratings for trained
checkpoints; this harness is a deliberately separate tournament space for
candidate heuristic bots. It uses the *same* anchored Bradley-Terry rating
fit (``agent/train/ranking.py``) so candidate ratings are directly
comparable to ``random=1000`` and ``heuristic=2500`` anchors, but it does
not write to or read from any league JSON.

Public surface:

* ``run_tournament(...)``: play a configured set of matchups and return both
  the per-pair raw results and the fitted ratings.
* ``Matchup``: a 2-policy face-off with seat rotation across ``num_games``
  total games (split evenly across all seat permutations to neutralize first
  -player advantage).
* ``play_matchup(...)``: low-level driver used by ``run_tournament``.

Per-game outcomes are mapped to score(a) using the standard tie convention:
1.0 for a win, 0.5 for a tie, 0.0 for a loss. Capped (unfinished) games are
treated as ties so a deterministic but slow policy is not unduly punished;
the harness records the cap rate alongside ratings so it stays visible.

The harness expects each policy to expose a callable
``choose(engine: BatchedEngine) -> torch.Tensor`` of shape ``(B,)`` int64.
Both the existing reference bots (``RandomBot``, ``HeuristicBot``) and the
heuristic-opus candidates satisfy that contract directly.
"""

from __future__ import annotations

import concurrent.futures as _cf
import dataclasses
import pathlib
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..train import ranking as R
from . import match_cache as MC


def _balanced_rotations(num_players: int) -> list[tuple[int, ...]]:
    """Build a balanced set of role-assignments for a 2-policy matchup.

    Returns ``2 * num_players`` seat-to-policy assignments such that:

    * Role 0 (a) sits at every seat at least once.
    * Role 1 (b) sits at every seat at least once.
    * Across all rotations, both roles get the same total number of seats.

    For each ``r`` in ``range(num_players)``:
    - Rotation 2r places role 0 at seat ``r`` and role 1 elsewhere.
    - Rotation 2r+1 places role 1 at seat ``r`` and role 0 elsewhere.

    This is preferred over ``itertools.permutations`` because the latter
    yields ``num_players!`` rotations (24 for 4p), forcing very high
    ``num_games`` to keep each rotation statistically meaningful.
    """
    rotations: list[tuple[int, ...]] = []
    for r in range(num_players):
        a_at_r = tuple(0 if seat == r else 1 for seat in range(num_players))
        b_at_r = tuple(1 if seat == r else 0 for seat in range(num_players))
        rotations.append(a_at_r)
        rotations.append(b_at_r)
    return rotations


PolicyChoose = Callable[[BE.BatchedEngine], torch.Tensor]


def _coerce_policy(p: Any) -> tuple[PolicyChoose, dict[str, Any]]:
    """Accept either a (choose) callable or an object with both a ``choose``
    method and an ``info`` method.

    Returns ``(choose, info_dict)``. Bare callables get an info dict that is
    just ``{}`` -- the cache key will therefore rely on the policy's name to
    distinguish it, which is fine for stable bot constructors.
    """
    if callable(p) and not hasattr(p, "choose"):
        return (p, {})
    choose = p.choose
    info_fn = getattr(p, "info", None)
    if callable(info_fn):
        info = dict(info_fn())
    else:
        info = {}
    return (choose, info)


@dataclasses.dataclass(frozen=True)
class Matchup:
    """One pair of policies face-off, played for ``num_games`` total games.

    Seat assignments are rotated across ``itertools.permutations`` so each
    policy sits at every seat an equal number of times. Therefore
    ``num_games`` should be divisible by the number of seat permutations
    used (``num_players!`` for 2/3/4 players); any remainder is silently
    truncated to the largest divisible count.
    """

    name_a: str
    name_b: str
    num_games: int
    num_players: int = 2
    max_turns: int = 200
    seed: int = 1234


@dataclasses.dataclass
class GameLog:
    """Compact summary of one tournament game.

    Aggregated for analysis: per-seat policy, per-seat ending points/cards/
    nobles, the winning role, and counts of each main-action class taken
    during the game. This is intentionally NOT a full step-by-step replay
    -- the goal is to support 'why did role X lose?' analysis with cheap
    aggregate features rather than reconstruct the engine state.
    """

    matchup: str
    num_players: int
    seed: int
    seat_role: list[int]
    final_points: list[int]
    final_cards: list[int]
    final_nobles: list[int]
    winner_seat: int
    winner_role: int
    finished: bool
    ended_step: int
    action_counts_per_seat: list[dict[str, int]]


@dataclasses.dataclass
class MatchResult:
    name_a: str
    name_b: str
    games_played: int
    games_finished: int
    wins_a: float
    wins_b: float
    ties: float
    avg_finished_turns: float
    game_logs: list[GameLog] = dataclasses.field(default_factory=list)


def _per_game_actions(
    engine: BE.BatchedEngine,
    seat_to_policy: Sequence[int],  # length = MAX_PLAYERS, index = seat -> policy slot
    policies: Sequence[PolicyChoose],
) -> torch.Tensor:
    """Compute one action per game by routing each game's current_player to
    the right policy.

    We invoke each policy at most once per call by gathering the games whose
    ``current_player`` maps to that policy. Inactive (already-ended) games
    fall through with PASS_ACTION (they are masked off by the engine anyway).
    """
    B = engine.batch_size
    actions = torch.full(
        (B,), A.PASS_ACTION, dtype=torch.int64, device=engine.device
    )
    alive = ~engine.ended
    if not alive.any():
        return actions
    cp = engine.current_player.to(torch.long)
    seat_to_policy_t = torch.tensor(
        seat_to_policy, dtype=torch.int64, device=engine.device
    )
    policy_per_game = seat_to_policy_t.index_select(0, cp)
    for slot in range(len(policies)):
        mask = alive & (policy_per_game == slot)
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        sub_engine = engine.index_select(idx)
        sub_actions = policies[slot](sub_engine)
        actions.index_copy_(0, idx, sub_actions)
    return actions


def _winner_per_game(engine: BE.BatchedEngine) -> torch.Tensor:
    pts = engine.points.to(torch.int32)
    bonuses_total = engine.bonuses.sum(dim=-1).to(torch.int32)
    score = pts * 1000 - bonuses_total
    score = torch.where(
        engine.active_mask, score, torch.full_like(score, -(10**9))
    )
    return score.argmax(dim=-1)


def _payload_from_result(result: "MatchResult") -> dict[str, Any]:
    """Serialize a MatchResult to a JSON-safe dict for the cache."""
    return {
        "name_a": result.name_a,
        "name_b": result.name_b,
        "games_played": result.games_played,
        "games_finished": result.games_finished,
        "wins_a": result.wins_a,
        "wins_b": result.wins_b,
        "ties": result.ties,
        "avg_finished_turns": result.avg_finished_turns,
        "game_logs": [
            {
                "matchup": g.matchup,
                "num_players": g.num_players,
                "seed": g.seed,
                "seat_role": g.seat_role,
                "final_points": g.final_points,
                "final_cards": g.final_cards,
                "final_nobles": g.final_nobles,
                "winner_seat": g.winner_seat,
                "winner_role": g.winner_role,
                "finished": g.finished,
                "ended_step": g.ended_step,
                "action_counts_per_seat": g.action_counts_per_seat,
            }
            for g in result.game_logs
        ],
    }


def _result_from_payload(
    payload: Mapping[str, Any], name_a: str, name_b: str
) -> "MatchResult":
    """Inverse of _payload_from_result; cached results are loaded as-is."""
    logs_in = payload.get("game_logs") or []
    logs: list[GameLog] = []
    for g in logs_in:
        logs.append(
            GameLog(
                matchup=str(g.get("matchup", f"{name_a}-vs-{name_b}")),
                num_players=int(g.get("num_players", 2)),
                seed=int(g.get("seed", 0)),
                seat_role=list(g.get("seat_role", [])),
                final_points=list(g.get("final_points", [])),
                final_cards=list(g.get("final_cards", [])),
                final_nobles=list(g.get("final_nobles", [])),
                winner_seat=int(g.get("winner_seat", 0)),
                winner_role=int(g.get("winner_role", 0)),
                finished=bool(g.get("finished", False)),
                ended_step=int(g.get("ended_step", 0)),
                action_counts_per_seat=[
                    dict(d) for d in g.get("action_counts_per_seat", [])
                ],
            )
        )
    return MatchResult(
        name_a=str(payload.get("name_a", name_a)),
        name_b=str(payload.get("name_b", name_b)),
        games_played=int(payload.get("games_played", 0)),
        games_finished=int(payload.get("games_finished", 0)),
        wins_a=float(payload.get("wins_a", 0.0)),
        wins_b=float(payload.get("wins_b", 0.0)),
        ties=float(payload.get("ties", 0.0)),
        avg_finished_turns=float(payload.get("avg_finished_turns", 0.0)),
        game_logs=logs,
    )


def _action_class(action: int) -> str:
    """Bucket an action index into a small set of class names for analysis."""
    if A.TAKE3_BASE <= action < A.TAKE3_BASE + A.TAKE3_COUNT:
        return "take3"
    if A.TAKE2_BASE <= action < A.TAKE2_BASE + A.TAKE2_COUNT:
        return "take2"
    if A.RESERVE_GRID_BASE <= action < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
        return "reserve_grid"
    if A.RESERVE_BLIND_BASE <= action < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT:
        return "reserve_blind"
    if A.BUY_GRID_BASE <= action < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
        return "buy_grid"
    if A.BUY_RESERVED_BASE <= action < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
        return "buy_reserved"
    if action == A.PASS_ACTION:
        return "pass"
    if A.DISCARD_BASE <= action < A.DISCARD_BASE + A.DISCARD_COUNT:
        return "discard"
    if A.PICK_NOBLE_BASE <= action < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT:
        return "pick_noble"
    return "other"


def play_matchup(
    matchup: Matchup,
    policy_a: PolicyChoose,
    policy_b: PolicyChoose,
    device: str = "cpu",
    record_logs: bool = False,
    timeout_winner_uses_points: bool = True,
) -> MatchResult:
    """Play ``matchup`` and return aggregate match-result counts.

    Args:
        record_logs: When ``True``, the returned :class:`MatchResult` carries
            one :class:`GameLog` entry per game.
        timeout_winner_uses_points: When ``True`` (default), capped/unfinished
            games are awarded to the seat with the most prestige (with the
            standard fewer-cards tiebreak), instead of being counted as a tie.
            Splendor 4p often runs long with novice bots, and treating
            timeouts as ties strongly biases ratings toward the lower anchor;
            using a points-based winner gives a better signal of trajectory.
    """
    nP = matchup.num_players
    rotations = _balanced_rotations(nP)
    games_per_perm = matchup.num_games // len(rotations)
    if games_per_perm <= 0:
        raise ValueError(
            f"num_games={matchup.num_games} is too small for "
            f"num_players={nP} (need at least {len(rotations)})"
        )
    total_games = games_per_perm * len(rotations)

    wins_a = 0.0
    wins_b = 0.0
    ties = 0.0
    finished_total = 0
    finished_turns_total = 0.0
    games_played_total = 0
    game_logs: list[GameLog] = []

    for perm_idx, perm in enumerate(rotations):
        seat_to_policy = [0] * BE.MAX_PLAYERS
        for seat in range(nP):
            seat_to_policy[seat] = int(perm[seat])

        engine = BE.BatchedEngine(
            games_per_perm,
            nP,
            device=device,
            seed=matchup.seed + 1000 * perm_idx,
        )
        prev_ended = torch.zeros(
            (games_per_perm,), dtype=torch.bool, device=engine.device
        )
        end_step = torch.full(
            (games_per_perm,),
            fill_value=matchup.max_turns,
            dtype=torch.int32,
            device=engine.device,
        )
        per_game_action_counts: list[list[dict[str, int]]] = (
            [
                [{} for _ in range(BE.MAX_PLAYERS)]
                for _ in range(games_per_perm)
            ]
            if record_logs
            else []
        )
        for turn in range(matchup.max_turns):
            if engine.ended.all():
                break
            cp_pre = engine.current_player.tolist()
            phase_pre = engine.phase.tolist()
            actions = _per_game_actions(
                engine,
                seat_to_policy,
                (policy_a, policy_b),
            )
            if record_logs:
                a_list = actions.tolist()
                ended_list = engine.ended.tolist()
                for b_i in range(games_per_perm):
                    if bool(ended_list[b_i]):
                        continue
                    if int(phase_pre[b_i]) != 0:
                        continue
                    seat = int(cp_pre[b_i])
                    cls = _action_class(int(a_list[b_i]))
                    bucket = per_game_action_counts[b_i][seat]
                    bucket[cls] = bucket.get(cls, 0) + 1
            engine.apply(actions)
            cur_ended = engine.ended
            newly_ended = cur_ended & ~prev_ended
            if newly_ended.any():
                end_step[newly_ended] = turn + 1
            prev_ended = cur_ended

        finished = engine.ended
        pts = engine.points.to(torch.int32)
        bonuses_total = engine.bonuses.sum(dim=-1).to(torch.int32)
        score = pts * 1000 - bonuses_total
        score = torch.where(
            engine.active_mask, score, torch.full_like(score, -(10**9))
        )
        winners = score.argmax(dim=-1)
        winner_role = torch.tensor(
            [seat_to_policy[int(s)] for s in winners.tolist()],
            dtype=torch.int64,
            device=engine.device,
        )
        max_score = score.max(dim=-1, keepdim=True).values
        is_tied = (score == max_score).sum(dim=-1) > 1

        finished_list = finished.tolist()
        winners_list = winners.tolist()
        winner_role_list = winner_role.tolist()
        is_tied_list = is_tied.tolist()
        end_step_list = end_step.tolist()
        pts_full = engine.points.tolist()
        bon_full = engine.bonuses.sum(dim=-1).tolist()
        nobles_full = engine.nobles_claimed.tolist()
        for b_i in range(games_per_perm):
            f = bool(finished_list[b_i])
            tied = bool(is_tied_list[b_i])
            seat = int(winners_list[b_i])
            role = int(winner_role_list[b_i])
            tally_winner = False
            if f and not tied:
                tally_winner = True
            elif (not f) and timeout_winner_uses_points and not tied:
                tally_winner = True
            if tally_winner:
                if role == 0:
                    wins_a += 1.0
                else:
                    wins_b += 1.0
                if f:
                    finished_turns_total += float(end_step_list[b_i])
            else:
                ties += 1.0
            if record_logs:
                game_logs.append(
                    GameLog(
                        matchup=f"{matchup.name_a}-vs-{matchup.name_b}",
                        num_players=nP,
                        seed=matchup.seed + 1000 * perm_idx + b_i,
                        seat_role=[
                            seat_to_policy[s] if s < nP else -1
                            for s in range(BE.MAX_PLAYERS)
                        ],
                        final_points=[int(p) for p in pts_full[b_i]],
                        final_cards=[int(c) for c in bon_full[b_i]],
                        final_nobles=[int(n) for n in nobles_full[b_i]],
                        winner_seat=seat,
                        winner_role=role,
                        finished=f,
                        ended_step=int(end_step_list[b_i]),
                        action_counts_per_seat=per_game_action_counts[b_i],
                    )
                )
        finished_total += int(finished.sum().item())
        games_played_total += games_per_perm

    avg_turns = (
        finished_turns_total / max(1, finished_total) if finished_total > 0 else 0.0
    )
    return MatchResult(
        name_a=matchup.name_a,
        name_b=matchup.name_b,
        games_played=games_played_total,
        games_finished=finished_total,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        avg_finished_turns=avg_turns,
        game_logs=game_logs,
    )


@dataclasses.dataclass
class TournamentReport:
    matches: list[MatchResult]
    ratings: dict[str, float]
    games_per_entity: dict[str, int]
    wall_seconds: float
    summary: str
    ratings_per_pc: dict[int, dict[str, float]] = dataclasses.field(default_factory=dict)
    games_per_entity_per_pc: dict[int, dict[str, int]] = dataclasses.field(
        default_factory=dict
    )


def run_tournament(
    factories: Mapping[str, Callable[[], Any]],
    *,
    num_games: int = 64,
    num_players: int = 2,
    max_turns: int = 200,
    seed: int = 1234,
    anchors: Mapping[str, float] | None = None,
    matchups: Sequence[tuple[str, str]] | None = None,
    device: str = "cpu",
    player_counts: Sequence[int] | None = None,
    record_logs: bool = False,
    timeout_winner_uses_points: bool = True,
    cache_dir: pathlib.Path | str | None = None,
    workers: int = 1,
    participant_player_counts: Mapping[str, Sequence[int]] | None = None,
) -> TournamentReport:
    """Run a round-robin (or explicitly-specified) tournament over policies.

    Args:
        factories: Mapping name -> zero-arg callable returning a fresh
            ``choose`` callable. Names must be unique. The fixed anchor
            entities ``random`` and ``heuristic`` should be present so the
            rating fit has anchored constraints.
        num_games: Per-matchup game count. Distributed evenly across seat
            permutations.
        num_players: 2, 3, or 4.
        max_turns: Cap on phase-transitions per game; capped games count as
            ties.
        seed: Base seed for engines; matchups offset from this for variety.
        anchors: Override the {random, heuristic} anchor ratings if needed.
        matchups: Optional explicit matchup list as ``(name_a, name_b)``
            tuples. If ``None`` we run the full round-robin (every unordered
            pair once).
        device: Torch device.
        participant_player_counts: Optional mapping from entity name to the
            set of player counts in which it may participate. When provided,
            any matchup whose pair includes an entity disallowed at the
            current player count is silently skipped. Useful for restricting
            a trained net to 2-player matchups while still running a full
            multi-pc tournament for the heuristic candidates.

    Returns:
        ``TournamentReport`` with the raw per-matchup counts, the fitted
        ratings, and a brief text summary.

    Note:
        When ``player_counts`` is provided (e.g. ``(2, 3, 4)``), the
        round-robin runs once per player count and all match results feed a
        single anchored rating fit. Pairwise free-for-all 3p / 4p games are
        played as "two-role" matchups (each seat is randomly assigned role
        a or b such that all four seat permutations are covered evenly), so
        the result is comparable across player counts.
    """
    names = list(factories.keys())
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate policy names: {names!r}")
    if matchups is None:
        match_pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
    else:
        match_pairs = list(matchups)

    if player_counts is None:
        player_counts_list = [num_players]
    else:
        player_counts_list = list(player_counts)
    for pc in player_counts_list:
        if pc not in (2, 3, 4):
            raise ValueError(f"player_counts must be in (2,3,4); got {pc}")

    allowed_pcs: dict[str, set[int]] = {}
    if participant_player_counts is not None:
        for ent, pcs in participant_player_counts.items():
            allowed_pcs[ent] = {int(x) for x in pcs}

    def _pair_allowed_at(na: str, nb: str, pc: int) -> bool:
        for ent in (na, nb):
            allow = allowed_pcs.get(ent)
            if allow is not None and pc not in allow:
                return False
        return True

    # Lazy-build policies because we may skip ones whose every matchup is cached.
    policies_cache: dict[str, tuple[PolicyChoose, dict[str, Any]]] = {}

    def _get_policy(name: str) -> tuple[PolicyChoose, dict[str, Any]]:
        if name not in policies_cache:
            built = factories[name]()
            policies_cache[name] = _coerce_policy(built)
        return policies_cache[name]

    def _get_info(name: str) -> dict[str, Any]:
        return _get_policy(name)[1]

    cache_dir_path: pathlib.Path | None
    if cache_dir is None:
        cache_dir_path = None
    else:
        cache_dir_path = pathlib.Path(cache_dir)

    t0 = time.monotonic()
    matches: list[MatchResult] = []
    results_rows: list[dict] = []
    games_per_entity: dict[str, int] = {n: 0 for n in names}
    games_per_entity_per_pc: dict[int, dict[str, int]] = {pc: {} for pc in player_counts_list}
    matches_per_pc: dict[int, list[MatchResult]] = {pc: [] for pc in player_counts_list}
    results_rows_per_pc: dict[int, list[dict]] = {pc: [] for pc in player_counts_list}
    cache_hits = 0
    cache_misses = 0
    # Plan tasks first so cache hits resolve in O(1) without acquiring policy
    # objects, and so the parallel batch is just the matchups that need to run.
    tasks: list[dict[str, Any]] = []
    for pc_idx, pc in enumerate(player_counts_list):
        for k, (na, nb) in enumerate(match_pairs):
            if not _pair_allowed_at(na, nb, pc):
                continue
            matchup_seed = seed + 7919 * k + 4801 * pc_idx
            tasks.append(
                {
                    "pc": pc,
                    "na": na,
                    "nb": nb,
                    "seed": matchup_seed,
                }
            )

    resolved: list[MatchResult | None] = [None] * len(tasks)
    pending_indices: list[int] = []
    cache_keys: list[str | None] = [None] * len(tasks)
    for idx, task in enumerate(tasks):
        m = Matchup(
            name_a=task["na"],
            name_b=task["nb"],
            num_games=num_games,
            num_players=task["pc"],
            max_turns=max_turns,
            seed=task["seed"],
        )
        if cache_dir_path is not None:
            info_a = _get_info(task["na"])
            info_b = _get_info(task["nb"])
            cache_key = MC.make_cache_key(
                name_a=task["na"],
                name_b=task["nb"],
                info_a=info_a,
                info_b=info_b,
                num_players=task["pc"],
                num_games=num_games,
                max_turns=max_turns,
                seed=task["seed"],
                timeout_winner_uses_points=timeout_winner_uses_points,
                extra={"record_logs": bool(record_logs)},
            )
            cache_keys[idx] = cache_key
            hit = MC.load(cache_dir_path, cache_key)
            if hit.hit and hit.payload is not None:
                resolved[idx] = _result_from_payload(
                    hit.payload, task["na"], task["nb"]
                )
                cache_hits += 1
                continue
        pending_indices.append(idx)
        cache_misses += 1

    # Per-thread torch limits so workers don't oversubscribe a shared box.
    # We keep N workers * 1 thread = N total threads.
    torch_threads_global = torch.get_num_threads()
    if workers > 1:
        torch.set_num_threads(max(1, torch_threads_global // workers))

    def _run_task(idx: int) -> MatchResult:
        task = tasks[idx]
        m = Matchup(
            name_a=task["na"],
            name_b=task["nb"],
            num_games=num_games,
            num_players=task["pc"],
            max_turns=max_turns,
            seed=task["seed"],
        )
        # Build fresh per-task policies so threads don't share mutable state.
        choose_a, _ = _coerce_policy(factories[task["na"]]())
        choose_b, _ = _coerce_policy(factories[task["nb"]]())
        result = play_matchup(
            m,
            choose_a,
            choose_b,
            device=device,
            record_logs=record_logs,
            timeout_winner_uses_points=timeout_winner_uses_points,
        )
        if cache_dir_path is not None and cache_keys[idx] is not None:
            MC.save(
                cache_dir_path,
                cache_keys[idx],
                _payload_from_result(result),
            )
        return result

    if pending_indices:
        if workers > 1:
            with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(_run_task, idx): idx for idx in pending_indices
                }
                for fut in _cf.as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    resolved[idx] = fut.result()
        else:
            for idx in pending_indices:
                resolved[idx] = _run_task(idx)

    if workers > 1:
        torch.set_num_threads(torch_threads_global)

    for idx, task in enumerate(tasks):
        result = resolved[idx]
        if result is None:
            raise RuntimeError(f"task {idx} did not resolve")
        pc = task["pc"]
        na = task["na"]
        nb = task["nb"]
        matches.append(result)
        matches_per_pc[pc].append(result)
        R.add_match_result(
            results_rows, na, nb, result.wins_a, result.wins_b, result.ties
        )
        R.add_match_result(
            results_rows_per_pc[pc],
            na,
            nb,
            result.wins_a,
            result.wins_b,
            result.ties,
        )
        games_per_entity[na] = games_per_entity.get(na, 0) + result.games_played
        games_per_entity[nb] = games_per_entity.get(nb, 0) + result.games_played
        games_per_entity_per_pc[pc][na] = (
            games_per_entity_per_pc[pc].get(na, 0) + result.games_played
        )
        games_per_entity_per_pc[pc][nb] = (
            games_per_entity_per_pc[pc].get(nb, 0) + result.games_played
        )

    ratings = R.fit_anchored_ratings(
        results_rows,
        anchors=dict(anchors) if anchors else None,
    )
    ratings_per_pc: dict[int, dict[str, float]] = {}
    for pc in player_counts_list:
        ratings_per_pc[pc] = R.fit_anchored_ratings(
            results_rows_per_pc[pc],
            anchors=dict(anchors) if anchors else None,
        )
    wall = time.monotonic() - t0

    lines: list[str] = []
    pc_label = (
        f"{num_players}-player"
        if len(player_counts_list) == 1
        else "across " + "/".join(str(pc) for pc in player_counts_list) + " players"
    )
    cache_msg = (
        f"  cache: hits={cache_hits} misses={cache_misses}"
        if cache_dir_path is not None
        else "  cache: disabled"
    )
    lines.append(
        f"tournament: {len(tasks)} matchups played "
        f"({len(match_pairs)} pairs x {len(player_counts_list)} player-counts "
        f"= {len(match_pairs) * len(player_counts_list)} configured), "
        f"{num_games} games each ({pc_label}), wall={wall:.1f}s\n{cache_msg}"
    )
    lines.append("ratings (anchored, aggregated):")
    for n, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"  {n:>32s}  {r:8.1f}  games={games_per_entity.get(n, 0)}"
        )
    if len(player_counts_list) > 1:
        lines.append("ratings (anchored, by player count):")
        # Header
        header = "  " + " " * 32 + "  " + "  ".join(
            f"{pc}p={'':>6s}" for pc in player_counts_list
        )
        lines.append(header.rstrip())
        for n, _ in sorted(ratings.items(), key=lambda kv: -kv[1]):
            row_parts = [f"  {n:>32s}"]
            for pc in player_counts_list:
                rpc = ratings_per_pc[pc].get(n, float("nan"))
                row_parts.append(f"  {pc}p={rpc:8.1f}")
            lines.append("".join(row_parts))
    lines.append("matchups (a vs b):")
    for pc in player_counts_list:
        if len(player_counts_list) > 1:
            lines.append(f"  -- {pc}-player --")
        for m in matches_per_pc[pc]:
            wa = m.wins_a / max(1, m.games_played)
            wb = m.wins_b / max(1, m.games_played)
            tie = m.ties / max(1, m.games_played)
            lines.append(
                f"  {m.name_a:>24s} vs {m.name_b:<24s}  "
                f"a={wa:.3f} b={wb:.3f} t={tie:.3f}  "
                f"finished={m.games_finished}/{m.games_played}  "
                f"avg_turns={m.avg_finished_turns:.1f}"
            )
    summary = "\n".join(lines)

    return TournamentReport(
        matches=matches,
        ratings=ratings,
        games_per_entity=games_per_entity,
        wall_seconds=wall,
        summary=summary,
        ratings_per_pc=ratings_per_pc,
        games_per_entity_per_pc=games_per_entity_per_pc,
    )


def analyze_logs_for_candidate(
    matches: Sequence[MatchResult],
    candidate: str,
    *,
    num_players: int | None = None,
) -> str:
    """Aggregate game logs and summarize how often ``candidate`` lost,
    grouped by player count and opponent.

    Reports:
    * Win/tie/loss share
    * Average final points + cards + nobles for candidate vs opponents
    * Average action-class mix for candidate vs opponents
    * Average game length and timeout rate

    Useful for guiding the next candidate iteration: if the candidate is
    spending many actions on ``take3``/``take2`` but ending with low cards
    and points, it is hoarding tokens. If it has many ``reserve_grid`` but
    few ``buy_*`` actions it is over-reserving. The output is a multi-line
    text report intended for direct printing.
    """
    relevant: list[tuple[MatchResult, GameLog, int]] = []
    for m in matches:
        for log in m.game_logs:
            if num_players is not None and log.num_players != num_players:
                continue
            cand_role = -1
            if m.name_a == candidate:
                cand_role = 0
            elif m.name_b == candidate:
                cand_role = 1
            else:
                continue
            relevant.append((m, log, cand_role))
    if not relevant:
        return f"no game logs for candidate={candidate!r}"

    bucket_pc: dict[int, list[tuple[MatchResult, GameLog, int]]] = {}
    for m, log, role in relevant:
        bucket_pc.setdefault(log.num_players, []).append((m, log, role))

    lines: list[str] = []
    lines.append(f"== game-log analysis for {candidate} ==")
    for pc in sorted(bucket_pc):
        rows = bucket_pc[pc]
        wins = sum(1 for _, log, role in rows if log.winner_role == role and log.finished)
        capped = sum(1 for _, log, _ in rows if not log.finished)
        ties = sum(
            1
            for _, log, role in rows
            if log.finished
            and log.winner_role != role
            and not (log.final_points[log.winner_seat] > max(
                log.final_points[s]
                for s in range(log.num_players)
                if log.seat_role[s] == role
            ))
        )
        losses = len(rows) - wins - capped
        avg_cand_pts = 0.0
        avg_opp_pts = 0.0
        avg_cand_cards = 0.0
        avg_opp_cards = 0.0
        avg_cand_nobles = 0.0
        avg_opp_nobles = 0.0
        action_total: dict[str, int] = {}
        action_total_opp: dict[str, int] = {}
        cand_seat_count = 0
        opp_seat_count = 0
        ended_step_total = 0
        for _m, log, role in rows:
            for seat in range(log.num_players):
                if log.seat_role[seat] == role:
                    avg_cand_pts += log.final_points[seat]
                    avg_cand_cards += log.final_cards[seat]
                    avg_cand_nobles += log.final_nobles[seat]
                    cand_seat_count += 1
                    for cls, cnt in log.action_counts_per_seat[seat].items():
                        action_total[cls] = action_total.get(cls, 0) + cnt
                else:
                    avg_opp_pts += log.final_points[seat]
                    avg_opp_cards += log.final_cards[seat]
                    avg_opp_nobles += log.final_nobles[seat]
                    opp_seat_count += 1
                    for cls, cnt in log.action_counts_per_seat[seat].items():
                        action_total_opp[cls] = action_total_opp.get(cls, 0) + cnt
            ended_step_total += log.ended_step

        denom_c = max(1, cand_seat_count)
        denom_o = max(1, opp_seat_count)
        avg_step = ended_step_total / max(1, len(rows))
        cap_rate = capped / max(1, len(rows))
        lines.append(
            f"  {pc}-player: games={len(rows)}  wins={wins}  losses={losses}  "
            f"capped={capped}  avg_step={avg_step:.1f}  cap_rate={cap_rate:.2f}"
        )
        lines.append(
            f"    candidate per-seat: pts={avg_cand_pts/denom_c:.2f}  "
            f"cards={avg_cand_cards/denom_c:.2f}  "
            f"nobles={avg_cand_nobles/denom_c:.2f}"
        )
        lines.append(
            f"    opponent  per-seat: pts={avg_opp_pts/denom_o:.2f}  "
            f"cards={avg_opp_cards/denom_o:.2f}  "
            f"nobles={avg_opp_nobles/denom_o:.2f}"
        )
        # Action-class share normalized.
        def _fmt_actions(act: dict[str, int]) -> str:
            tot = sum(act.values())
            if tot == 0:
                return "no actions recorded"
            keys = sorted(act, key=lambda k: -act[k])
            parts = [
                f"{k}={act[k]}({act[k] / tot:.0%})" for k in keys
            ]
            return " ".join(parts)

        lines.append(f"    candidate actions: {_fmt_actions(action_total)}")
        lines.append(f"    opponent  actions: {_fmt_actions(action_total_opp)}")
    return "\n".join(lines)

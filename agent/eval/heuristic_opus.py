"""Candidate "heuristic-opus" Splendor bots.

Built incrementally, each candidate refining the previous one based on the
strategy report in `experimental/mloeppky/splendor/strategy.md`.

All candidates implement the player-policy contract used by the rest of the
codebase: ``choose(engine: BatchedEngine) -> torch.Tensor`` returning a
``(B,)`` int64 action tensor that is legal for every game in the batch.

Candidate progression (see ``HeuristicOpusV*`` classes):

* V1 -- "Tall, target-driven greedy". Scores affordable cards by points, picks
  a single point-bearing target, and steers token-taking toward that target
  rather than spreading colors uniformly.
* V2 -- V1 plus opponent-denial reserves and noble-aware bonus weighting:
  reserve high-PV cards an opponent could buy next turn, prefer bonuses that
  unlock a noble, and avoid useless reserves.
* V3 -- V2 plus endgame mode: when any player is within 4 points of 15,
  switch to greedy point chasing, smarter discards, and last-round denial
  evaluation against the player who triggered the round.
* V4 -- V3 plus a one-ply lookahead: for the top-K legal actions, simulate
  a single ply on a cloned ``BatchedEngine`` and pick the action whose
  resulting state has the highest static value. Slow, but still much faster
  than MCTS and significantly stronger than V3 in self-play.

The implementation pulls per-game state from the batched engine into Python
lists once per ``choose`` call, runs a per-game scorer, and writes the
resulting actions back as a single ``int64`` tensor. This is plenty fast for
tournaments (the engine itself dominates wall time at B=64-256).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..env import cards as C


_NUM_COLORS: int = C.NUM_COLORS
_GOLD: int = C.GOLD_INDEX
_NUM_TIERS: int = BE.NUM_TIERS
_NUM_GRID_SLOTS: int = BE.NUM_GRID_SLOTS
_MAX_RESERVED: int = BE.MAX_RESERVED
_MAX_NOBLE_SLOTS: int = BE.MAX_NOBLE_SLOTS
_WINNING_POINTS: int = BE.WINNING_POINTS
_TOKEN_LIMIT: int = BE.TOKEN_LIMIT
_TAKE3_COMBOS: tuple[tuple[int, int, int], ...] = A.TAKE3_COMBOS


@dataclasses.dataclass(frozen=True)
class _CardData:
    card_id: int
    level: int
    bonus: int
    points: int
    cost: tuple[int, int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _NobleData:
    noble_id: int
    points: int
    requirement: tuple[int, int, int, int, int]


_CARDS: tuple[_CardData, ...] = tuple(
    _CardData(
        card_id=c.card_id,
        level=c.level,
        bonus=c.bonus,
        points=c.points,
        cost=tuple(c.cost),
    )
    for c in C.CARDS
)
_NOBLES: tuple[_NobleData, ...] = tuple(
    _NobleData(
        noble_id=n.noble_id,
        points=n.points,
        requirement=tuple(n.requirement),
    )
    for n in C.NOBLES
)


@dataclasses.dataclass
class _GameView:
    """Pure-Python snapshot of one game in the batched engine.

    The engine's per-game state is small (5 colors, 3 tiers x 4 slots, etc.),
    so we copy it once per ``choose`` call and let the heuristic operate on
    plain Python primitives. This keeps the scorer easy to reason about and
    sidesteps tensor indexing overhead.
    """

    num_players: int
    cp: int
    phase: int
    points: list[int]
    bonuses: list[list[int]]
    tokens: list[list[int]]
    reserved: list[list[int]]
    grid: list[list[int]]
    nobles: list[int]
    gem_pool: list[int]
    deck_top: list[int]
    last_trigger: int
    turns_since_trigger: int


def _extract_views(engine: BE.BatchedEngine) -> list[_GameView]:
    B = engine.batch_size
    nP = engine.num_players
    points = engine.points.tolist()
    bonuses = engine.bonuses.tolist()
    tokens = engine.tokens.tolist()
    reserved = engine.reserved.tolist()
    grid = engine.grid_card.tolist()
    nobles = engine.noble_ids.tolist()
    gem_pool = engine.gem_pool.tolist()
    deck_top = engine.deck_top.tolist()
    cp = engine.current_player.tolist()
    phase = engine.phase.tolist()
    last_trigger = engine.last_trigger.tolist()
    turns_since_trigger = engine.turns_since_trigger.tolist()
    out: list[_GameView] = []
    for b in range(B):
        out.append(
            _GameView(
                num_players=nP,
                cp=int(cp[b]),
                phase=int(phase[b]),
                points=[int(points[b][p]) for p in range(nP)],
                bonuses=[[int(x) for x in bonuses[b][p]] for p in range(nP)],
                tokens=[[int(x) for x in tokens[b][p]] for p in range(nP)],
                reserved=[[int(x) for x in reserved[b][p]] for p in range(nP)],
                grid=[[int(x) for x in grid[b][t]] for t in range(_NUM_TIERS)],
                nobles=[int(x) for x in nobles[b]],
                gem_pool=[int(x) for x in gem_pool[b]],
                deck_top=[int(x) for x in deck_top[b]],
                last_trigger=int(last_trigger[b]),
                turns_since_trigger=int(turns_since_trigger[b]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Affordability and "distance" helpers operating on a player's pure state.
# ---------------------------------------------------------------------------


def _purchase_cost(
    card: _CardData, bonuses: Sequence[int]
) -> tuple[int, int, int, int, int]:
    """Returns the actual token-equivalent cost after subtracting bonuses."""
    return tuple(max(0, card.cost[c] - bonuses[c]) for c in range(_NUM_COLORS))


def _affordable(
    card: _CardData,
    bonuses: Sequence[int],
    tokens: Sequence[int],
) -> bool:
    """Cost minus bonus minus colored-tokens, with gold covering shortfall."""
    deficit = 0
    for c in range(_NUM_COLORS):
        need = max(0, card.cost[c] - bonuses[c])
        deficit += max(0, need - tokens[c])
    return deficit <= tokens[_GOLD]


def _gold_needed(
    card: _CardData,
    bonuses: Sequence[int],
    tokens: Sequence[int],
) -> int:
    deficit = 0
    for c in range(_NUM_COLORS):
        need = max(0, card.cost[c] - bonuses[c])
        deficit += max(0, need - tokens[c])
    return deficit


def _missing_per_color(
    card: _CardData,
    bonuses: Sequence[int],
    tokens: Sequence[int],
) -> tuple[int, int, int, int, int]:
    """Per-color shortfall ignoring gold (used for take-token planning)."""
    return tuple(
        max(0, max(0, card.cost[c] - bonuses[c]) - tokens[c])
        for c in range(_NUM_COLORS)
    )


def _turns_to_afford(
    card: _CardData,
    bonuses: Sequence[int],
    tokens: Sequence[int],
    pool: Sequence[int],
) -> int:
    """Optimistic lower bound on the number of take-3-token turns to afford ``card``.

    Each turn nets at most 3 tokens from non-empty piles. Gold and bonuses
    already in hand reduce the need 1-for-1. Piles that are empty in the pool
    cannot be drawn from.
    """
    need_total = 0
    for c in range(_NUM_COLORS):
        need = max(0, card.cost[c] - bonuses[c])
        # The player can already cover ``min(need, tokens[c])`` from existing
        # colored tokens; the remainder can be partially covered by gold.
        deficit = max(0, need - tokens[c])
        # If a pile is empty in the pool we still might have the token
        # already; the deficit only counts what we must still acquire.
        need_total += deficit
    # Gold acts as a wildcard.
    need_total = max(0, need_total - tokens[_GOLD])
    if need_total == 0:
        return 0
    return (need_total + 2) // 3


# ---------------------------------------------------------------------------
# Card / noble feature scoring.
# ---------------------------------------------------------------------------


def _noble_color_pressure(view: _GameView) -> tuple[float, ...]:
    """For each color c, how much does our player still need it for *any* noble.

    The score is the sum, over noble tiles still on the board, of the
    remaining requirement in that color (capped at the noble's per-color
    threshold). A higher value means the color is on the critical path to a
    noble, so we should weight it more when picking targets.
    """
    pressure = [0.0] * _NUM_COLORS
    cp_bonuses = view.bonuses[view.cp]
    for nslot in range(_MAX_NOBLE_SLOTS):
        nid = view.nobles[nslot]
        if nid < 0:
            continue
        n = _NOBLES[nid]
        for c in range(_NUM_COLORS):
            need = max(0, n.requirement[c] - cp_bonuses[c])
            pressure[c] += float(need)
    return tuple(pressure)


def _bonus_color_helpfulness(
    color: int,
    view: _GameView,
    noble_pressure: Sequence[float],
) -> float:
    """How useful is gaining one more bonus of ``color`` for our player?

    Combines:
    * Discount on currently visible high-PV cards needing this color.
    * Noble pressure on this color.
    * Diminishing returns: 6th+ bonus of one color is usually wasted.
    """
    cp_bonuses = view.bonuses[view.cp]
    have = cp_bonuses[color]
    # Discount usefulness on visible cards. Tier 2/3 only, since we don't
    # bother chasing tier-1 cards.
    discount = 0.0
    for tier in range(1, _NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            card = _CARDS[cid]
            need = max(0, card.cost[color] - have)
            if need > 0 and card.points > 0:
                discount += float(card.points) * 0.4
    np_score = float(noble_pressure[color]) * 0.3
    diminishing = 0.0 if have <= 4 else -1.0 * (have - 4)
    return discount + np_score + diminishing


def _card_target_score(
    card: _CardData,
    view: _GameView,
    noble_pressure: Sequence[float],
) -> float:
    """Static score of ``card`` as a buying target for our current player.

    Higher is better. This is the central heuristic that drives V1+. We bias
    heavily toward points-per-token ("PPT") and reachability.
    """
    cp_tokens = view.tokens[view.cp]
    cp_bonuses = view.bonuses[view.cp]
    cost_total = sum(card.cost)
    # PPT (Strategy section 3): higher is better. Add a small floor so
    # 0-point cards are not scored as 0/cost = 0.
    ppt = card.points / max(1, cost_total)
    score = card.points * 100.0 + ppt * 50.0
    # Reachability bonus: prefer cards we can almost buy now.
    distance = _turns_to_afford(card, cp_bonuses, cp_tokens, view.gem_pool)
    score -= distance * 12.0
    if _affordable(card, cp_bonuses, cp_tokens):
        score += 25.0
    # Noble synergy: if buying this card pushes us toward an active noble,
    # inflate the score modestly.
    score += _bonus_color_helpfulness(card.bonus, view, noble_pressure) * 1.5
    # Avoid 0-PV tier-1 cards in mid/late game; they're the "wide" trap.
    if card.level == 1 and card.points == 0:
        score -= 25.0
    return score


# ---------------------------------------------------------------------------
# Discard / noble-pick / take helpers shared across all candidates.
# ---------------------------------------------------------------------------


def _choose_discard_action(view: _GameView) -> int:
    """Discard the *least useful* token. Never discard gold while colored
    tokens remain. Among colored tokens, discard whichever color we already
    own the most of as a bonus (least marginal value)."""
    cp_tokens = view.tokens[view.cp]
    cp_bonuses = view.bonuses[view.cp]
    best_kind = -1
    best_score = -1e9
    for kind in range(6):
        if cp_tokens[kind] <= 0:
            continue
        if kind == _GOLD:
            score = -1000.0
        else:
            # Higher score = better discard candidate.
            # Penalize colors we have few of (we want to keep them).
            score = float(cp_bonuses[kind]) - 0.05 * float(cp_tokens[kind])
        if score > best_score:
            best_score = score
            best_kind = kind
    if best_kind < 0:
        # Fallback: any non-zero kind including gold (mask said this is legal).
        for kind in range(6):
            if cp_tokens[kind] > 0:
                return A.DISCARD_BASE + kind
        return A.DISCARD_BASE  # unreachable when phase==1 was set
    return A.DISCARD_BASE + best_kind


def _choose_noble_pick_action(view: _GameView) -> int:
    """All visiting nobles are worth +3 PV. Prefer the noble whose colors are
    rarest in the engine's residual demand so we keep flexibility, but the
    PV outcome is identical regardless of choice."""
    # Pick lowest occupied slot the player qualifies for.
    cp_bonuses = view.bonuses[view.cp]
    for nslot in range(_MAX_NOBLE_SLOTS):
        nid = view.nobles[nslot]
        if nid < 0:
            continue
        req = _NOBLES[nid].requirement
        if all(cp_bonuses[c] >= req[c] for c in range(_NUM_COLORS)):
            return A.PICK_NOBLE_BASE + nslot
    return A.PICK_NOBLE_BASE


def _take3_score_for_target(
    combo: tuple[int, int, int],
    target_missing: Sequence[int],
    view: _GameView,
    noble_pressure: Sequence[float],
) -> float:
    """How much progress does taking ``combo`` make on the target card?

    Each color in the combo that is still missing from the target counts as
    +1 progress. Colors not in the combo but missing get 0. Bonus reward for
    colors that are also under noble pressure to break ties between
    target-redundant combos."""
    score = 0.0
    cp_tokens = view.tokens[view.cp]
    pool = view.gem_pool
    for c in combo:
        if pool[c] <= 0:
            continue  # Pile empty; engine treats this as "skip".
        score += float(min(target_missing[c], 1))  # 0 or 1 of progress
        # Slight bonus for noble pressure to disambiguate.
        score += 0.05 * float(noble_pressure[c])
        # Tiny penalty for already-flush colors (avoid stockpiling >5 of one
        # color if we can avoid it).
        if cp_tokens[c] >= 4:
            score -= 0.4
    # Penalize taking a color we *don't* need at all.
    for c in combo:
        if target_missing[c] == 0 and noble_pressure[c] == 0.0:
            score -= 0.2
    return score


def _take2_score_for_target(
    color: int,
    target_missing: Sequence[int],
    view: _GameView,
    noble_pressure: Sequence[float],
) -> float:
    """How much progress does taking 2 of ``color`` make?"""
    if view.gem_pool[color] < 4:
        return -1.0
    progress = float(min(target_missing[color], 2))
    score = progress * 1.5  # take-2 of a needed color is strong
    score += 0.05 * float(noble_pressure[color])
    cp_tokens = view.tokens[view.cp]
    if cp_tokens[color] >= 4:
        score -= 1.0
    return score


# ---------------------------------------------------------------------------
# Action enumeration / utility helpers.
# ---------------------------------------------------------------------------


def _legal_main_actions(view: _GameView, mask_row: list[bool]) -> list[int]:
    """All phase-0 legal action indices for this game."""
    return [a for a, ok in enumerate(mask_row) if ok and a < A.MAIN_ACTIONS_END]


def _list_affordable_buys(view: _GameView, mask_row: list[bool]) -> list[tuple[int, int]]:
    """Returns (action_index, card_id) pairs for currently-legal buy actions."""
    out: list[tuple[int, int]] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            a = A.BUY_GRID_BASE + tier * _NUM_GRID_SLOTS + s
            if not mask_row[a]:
                continue
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            out.append((a, cid))
    cp_reserved = view.reserved[view.cp]
    for r in range(_MAX_RESERVED):
        a = A.BUY_RESERVED_BASE + r
        if not mask_row[a]:
            continue
        cid = cp_reserved[r]
        if cid < 0:
            continue
        out.append((a, cid))
    return out


def _list_visible_grid_cards(view: _GameView) -> list[tuple[int, int, int, int]]:
    """Returns (tier, slot, action_index_for_buy, card_id) for every visible grid card."""
    out: list[tuple[int, int, int, int]] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            buy_a = A.BUY_GRID_BASE + tier * _NUM_GRID_SLOTS + s
            out.append((tier, s, buy_a, cid))
    return out


def _select_target_card(
    view: _GameView,
    noble_pressure: Sequence[float],
) -> _CardData | None:
    """Pick the best buying target across grid + own reserved cards.

    Returns the underlying ``_CardData`` so the take-token planner can read
    the cost, or ``None`` if no plausible target exists."""
    best_card: _CardData | None = None
    best_score = -1e9
    cp_bonuses = view.bonuses[view.cp]
    cp_tokens = view.tokens[view.cp]
    candidates: list[_CardData] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            candidates.append(_CARDS[cid])
    for r in range(_MAX_RESERVED):
        cid = view.reserved[view.cp][r]
        if cid < 0:
            continue
        candidates.append(_CARDS[cid])
    for card in candidates:
        score = _card_target_score(card, view, noble_pressure)
        # Heavier reachability penalty: if we are 4+ turns away, this is not
        # a near-term target and we should not steer tokens for it.
        distance = _turns_to_afford(card, cp_bonuses, cp_tokens, view.gem_pool)
        score -= max(0, distance - 1) * 6.0
        if score > best_score:
            best_score = score
            best_card = card
    return best_card


# ---------------------------------------------------------------------------
# V1 -- Tall, target-driven greedy.
# ---------------------------------------------------------------------------


class HeuristicOpusV1:
    """Tall, target-driven greedy bot.

    Decision tree (phase 0):
    1. If any *point-bearing* affordable card is visible, buy the best one.
    2. Else if any 0-PV affordable tier-1 card whose bonus completes a noble
       *and* the noble can finish in <=2 more cards, buy it.
    3. Else select a single target card and try to make progress:
        a. ``take 2`` of the bottleneck color if legal.
        b. ``take 3`` different colors with the highest target-progress score.
        c. Reserve the target with gold if reserves < 3 and gold remains.
    4. Discard / noble actions handled by shared helpers.
    """

    name: str = "heuristic_opus_v1"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 1}

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        mask = engine.legal_action_mask().tolist()
        views = _extract_views(engine)
        actions: list[int] = []
        for view, mask_row in zip(views, mask, strict=True):
            actions.append(self._choose_one(view, mask_row))
        return torch.tensor(actions, dtype=torch.int64, device=engine.device)

    def _choose_one(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.phase == 1:
            return _choose_discard_action(view)
        if view.phase == 2:
            return _choose_noble_pick_action(view)
        # phase 0: main action.
        if not any(mask_row[a] for a in range(A.MAIN_ACTIONS_END)):
            return A.PASS_ACTION
        return self._choose_main(view, mask_row)

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        noble_pressure = _noble_color_pressure(view)
        # Step 1: best affordable point-bearing card.
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy

        # Step 2: pick target and steer tokens / reserve toward it.
        target = _select_target_card(view, noble_pressure)
        if target is not None:
            cp_bonuses = view.bonuses[view.cp]
            cp_tokens = view.tokens[view.cp]
            missing = _missing_per_color(target, cp_bonuses, cp_tokens)
            # Try take2.
            best_a, best_score = -1, -1e9
            for color in range(_NUM_COLORS):
                a = A.TAKE2_BASE + color
                if not mask_row[a]:
                    continue
                score = _take2_score_for_target(color, missing, view, noble_pressure)
                if score > best_score:
                    best_score = score
                    best_a = a
            # Try take3.
            for combo_idx, combo in enumerate(_TAKE3_COMBOS):
                a = A.TAKE3_BASE + combo_idx
                if not mask_row[a]:
                    continue
                score = _take3_score_for_target(combo, missing, view, noble_pressure)
                if score > best_score:
                    best_score = score
                    best_a = a
            # Reserve target (for gold + denial bridging) if take options are weak.
            if best_score <= 0.0:
                reserve_a = self._best_reserve(view, mask_row, noble_pressure)
                if reserve_a is not None:
                    return reserve_a
            if best_a >= 0:
                return best_a

        # Step 3: fallback -- reserve a point-bearing card to build, or take any
        # legal token action even if not target-aligned.
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION

    def _best_affordable_buy(
        self,
        view: _GameView,
        mask_row: list[bool],
        noble_pressure: Sequence[float],
    ) -> int | None:
        candidates = _list_affordable_buys(view, mask_row)
        best_a, best_cid, best_score = -1, -1, -1e9
        for a, cid in candidates:
            card = _CARDS[cid]
            score = card.points * 200.0 + _card_target_score(
                card, view, noble_pressure
            )
            if card.points == 0 and card.level == 1:
                if noble_pressure[card.bonus] <= 0:
                    score -= 60.0
            if score > best_score:
                best_score = score
                best_a = a
                best_cid = cid
        if best_a < 0:
            return None
        chosen_card = _CARDS[best_cid]
        if chosen_card.points > 0:
            return best_a
        if noble_pressure[chosen_card.bonus] > 0:
            return best_a
        return None

    def _best_reserve(
        self,
        view: _GameView,
        mask_row: list[bool],
        noble_pressure: Sequence[float],
    ) -> int | None:
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        if view.gem_pool[_GOLD] <= 0:
            # Reserving without gold is rarely worth it for V1; prefer tokens.
            return None
        # Reserve the highest-PV grid card we can't yet afford.
        best_a = -1
        best_score = -1e9
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points <= 0:
                    continue
                if _affordable(card, cp_bonuses, cp_tokens):
                    continue
                score = card.points * 80.0 - 5.0 * _turns_to_afford(
                    card, cp_bonuses, cp_tokens, view.gem_pool
                )
                if score > best_score:
                    best_score = score
                    best_a = a
        if best_a < 0:
            return None
        return best_a


# ---------------------------------------------------------------------------
# V2 -- denial-aware reserves, smarter discards.
# ---------------------------------------------------------------------------


class HeuristicOpusV2(HeuristicOpusV1):
    """V1 plus opponent-denial reserves and noble-aware bonus weighting.

    Behavioural changes vs V1:
    * Before any other reserve, scan opponents in the order they will play and
      reserve any high-PV card that an opponent could buy on their next turn.
    * When picking among non-point-bearing buys, count noble pressure and
      visible-card discount value, not just raw bonus completeness.
    * Discards keep gold last and never discard a token color we still need
      for the chosen target (if any).
    """

    name: str = "heuristic_opus_v2"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 2}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        noble_pressure = _noble_color_pressure(view)
        # Always check for a denial reserve first; if an opponent is about to
        # claim a high-PV card it almost always beats taking tokens.
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial

        return super()._choose_main(view, mask_row)

    def _denial_reserve(
        self, view: _GameView, mask_row: list[bool]
    ) -> int | None:
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        # No-op when the gold pile is empty: reserving without gold rarely
        # helps us, but it still denies the card. Allow it only when threat
        # is clearly large.
        threat_threshold = 4.0  # point-equivalent loss to opponent
        # Predict opponents' affordability of every visible high-PV card.
        threats: list[tuple[float, int]] = []
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points < 3:
                    continue
                # Highest threat is the next opponent to act.
                worst_threat = 0.0
                for offset in range(1, view.num_players):
                    seat = (view.cp + offset) % view.num_players
                    o_bonuses = view.bonuses[seat]
                    o_tokens = view.tokens[seat]
                    if _affordable(card, o_bonuses, o_tokens):
                        # The closer in the rotation the more dangerous.
                        threat_score = float(card.points) * (
                            1.0 - 0.15 * (offset - 1)
                        )
                        # We won't get to act before that opponent (in 2p, the
                        # next opponent is the immediate next seat).
                        if threat_score > worst_threat:
                            worst_threat = threat_score
                if worst_threat <= 0.0:
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                threats.append((worst_threat, a))
        if not threats:
            return None
        threats.sort(reverse=True)
        worst, a = threats[0]
        if worst < threat_threshold:
            return None
        return a


# ---------------------------------------------------------------------------
# V3 -- endgame-aware behaviour.
# ---------------------------------------------------------------------------


class HeuristicOpusV3(HeuristicOpusV2):
    """V2 plus endgame reasoning.

    When any seat has 11+ points, switch to "shortest path to 15":
    * Buying any point-bearing affordable card is preferred even at low PPT.
    * Reserves pivot to either grabbing gold (to enable a winning buy this
      round) or denying the leading player.
    * The take-token planner shifts toward whichever colors close the
      remaining gap on the highest-PV affordable target.
    """

    name: str = "heuristic_opus_v3"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 3}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if not self._is_endgame(view):
            return super()._choose_main(view, mask_row)
        return self._endgame_main(view, mask_row)

    def _is_endgame(self, view: _GameView) -> bool:
        if view.last_trigger >= 0:
            return True
        max_pts = max(view.points)
        return max_pts >= 11

    def _endgame_main(self, view: _GameView, mask_row: list[bool]) -> int:
        noble_pressure = _noble_color_pressure(view)
        # 1. Buy the highest-PV affordable card outright.
        candidates = _list_affordable_buys(view, mask_row)
        if candidates:
            best_a, best_pv, best_tie = -1, -1, -1e9
            for a, cid in candidates:
                card = _CARDS[cid]
                tie = _card_target_score(card, view, noble_pressure)
                if (card.points, tie) > (best_pv, best_tie):
                    best_pv = card.points
                    best_tie = tie
                    best_a = a
            if best_pv > 0:
                return best_a

        # 2. Denial-reserve the leader's likely buys.
        leader = max(range(view.num_players), key=lambda p: view.points[p])
        if leader != view.cp:
            denial = self._endgame_denial(view, mask_row, leader)
            if denial is not None:
                return denial

        # 3. Plan tokens to enable a winning buy.
        target = _select_target_card(view, noble_pressure)
        if target is not None:
            cp_bonuses = view.bonuses[view.cp]
            cp_tokens = view.tokens[view.cp]
            missing = _missing_per_color(target, cp_bonuses, cp_tokens)
            best_a, best_score = -1, -1e9
            for color in range(_NUM_COLORS):
                a = A.TAKE2_BASE + color
                if not mask_row[a]:
                    continue
                score = _take2_score_for_target(
                    color, missing, view, noble_pressure
                )
                if score > best_score:
                    best_score = score
                    best_a = a
            for combo_idx, combo in enumerate(_TAKE3_COMBOS):
                a = A.TAKE3_BASE + combo_idx
                if not mask_row[a]:
                    continue
                score = _take3_score_for_target(
                    combo, missing, view, noble_pressure
                )
                if score > best_score:
                    best_score = score
                    best_a = a
            if best_a >= 0:
                return best_a

        # Fall back on V2 logic if nothing fits.
        return super()._choose_main(view, mask_row)

    def _endgame_denial(
        self,
        view: _GameView,
        mask_row: list[bool],
        leader: int,
    ) -> int | None:
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        leader_bonuses = view.bonuses[leader]
        leader_tokens = view.tokens[leader]
        leader_pts = view.points[leader]
        best_a, best_score = -1, -1e9
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points == 0:
                    continue
                if leader_pts + card.points < _WINNING_POINTS:
                    continue
                if not _affordable(card, leader_bonuses, leader_tokens):
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                # Higher-PV cards are more critical to deny.
                score = float(card.points) * 100.0
                if score > best_score:
                    best_score = score
                    best_a = a
        if best_a < 0:
            return None
        return best_a


# ---------------------------------------------------------------------------
# V4 -- one-ply lookahead on top-K candidate actions.
# ---------------------------------------------------------------------------


def _static_value(view: _GameView) -> float:
    """Static evaluation for our current player's resulting state.

    Aggregates points + cheap proxies for engine strength. Used by the V4
    one-ply lookahead, where higher == better for the acting player."""
    cp = view.cp
    pts = view.points[cp]
    val = float(pts) * 100.0
    bonuses = view.bonuses[cp]
    val += sum(bonuses) * 6.0
    # Diminishing returns on bonuses past 5 of one color.
    for c in range(_NUM_COLORS):
        if bonuses[c] > 5:
            val -= (bonuses[c] - 5) * 4.0
    tokens = view.tokens[cp]
    val += sum(tokens[:_NUM_COLORS]) * 1.5 + tokens[_GOLD] * 4.0
    # Reserved-card potential.
    for r in range(_MAX_RESERVED):
        cid = view.reserved[cp][r]
        if cid >= 0:
            val += _CARDS[cid].points * 8.0
    # Light penalty for opponents being close to victory.
    for p in range(view.num_players):
        if p == cp:
            continue
        if view.points[p] >= 11:
            val -= (view.points[p] - 10) * 30.0
    return val


class HeuristicOpusV4(HeuristicOpusV3):
    """V3 plus a one-ply lookahead.

    For each game we generate the V3 baseline action plus the top-K legal
    alternative actions (by V3 main-action score), simulate one ply on a
    cloned engine, and pick whichever resulting state has the best
    ``_static_value`` for our current player."""

    name: str = "heuristic_opus_v4"

    def __init__(self, top_k: int = 8) -> None:
        self._top_k = max(2, top_k)

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 4, "top_k": self._top_k}

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        baseline = super().choose(engine)
        # For each game collect a candidate-action list (baseline plus several
        # top-scoring V3 alternatives). Then evaluate them all with a single
        # cloned-engine ply and pick the best per game.
        mask = engine.legal_action_mask()
        views = _extract_views(engine)
        baseline_list: list[int] = baseline.tolist()
        # Build per-game candidate lists.
        candidates: list[list[int]] = []
        for b, view in enumerate(views):
            mask_row = mask[b].tolist()
            if view.phase != 0:
                # No lookahead in non-main phases (action space is small and
                # the static heuristic is already optimal).
                candidates.append([baseline_list[b]])
                continue
            cs = self._candidate_actions(view, mask_row, baseline_list[b])
            candidates.append(cs)
        # Run one ply per candidate.
        chosen = self._evaluate_candidates(engine, candidates)
        return torch.tensor(chosen, dtype=torch.int64, device=engine.device)

    def _candidate_actions(
        self,
        view: _GameView,
        mask_row: list[bool],
        baseline: int,
    ) -> list[int]:
        """List of action indices to evaluate for this game.

        Always includes the baseline. Adds: every affordable buy, every
        legal take action that scores positive against the chosen target,
        and any denial-relevant reserve.
        """
        out: set[int] = {baseline}
        out.update(a for a, _cid in _list_affordable_buys(view, mask_row))
        # Take token actions.
        for a in range(A.TAKE3_BASE, A.TAKE3_BASE + A.TAKE3_COUNT):
            if mask_row[a]:
                out.add(a)
        for a in range(A.TAKE2_BASE, A.TAKE2_BASE + A.TAKE2_COUNT):
            if mask_row[a]:
                out.add(a)
        # Reserves only the high-PV ones to keep K small.
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                if _CARDS[cid].points >= 2:
                    out.add(a)
        # Cap by top-K via baseline-score sorting (cheap).
        if len(out) <= self._top_k:
            return sorted(out)
        # Always keep the baseline. Score the rest crudely.
        scored: list[tuple[float, int]] = []
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        np_pressure = _noble_color_pressure(view)
        for a in out:
            if a == baseline:
                continue
            scored.append((self._cheap_score(a, view, cp_bonuses, cp_tokens, np_pressure), a))
        scored.sort(reverse=True)
        keep = [baseline] + [a for _, a in scored[: self._top_k - 1]]
        return sorted(set(keep))

    def _cheap_score(
        self,
        a: int,
        view: _GameView,
        cp_bonuses: Sequence[int],
        cp_tokens: Sequence[int],
        noble_pressure: Sequence[float],
    ) -> float:
        if A.BUY_GRID_BASE <= a < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
            x = a - A.BUY_GRID_BASE
            cid = view.grid[x // _NUM_GRID_SLOTS][x % _NUM_GRID_SLOTS]
            return _CARDS[cid].points * 1000.0
        if A.BUY_RESERVED_BASE <= a < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
            cid = view.reserved[view.cp][a - A.BUY_RESERVED_BASE]
            if cid < 0:
                return -1e9
            return _CARDS[cid].points * 1000.0
        if A.RESERVE_GRID_BASE <= a < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
            x = a - A.RESERVE_GRID_BASE
            cid = view.grid[x // _NUM_GRID_SLOTS][x % _NUM_GRID_SLOTS]
            if cid < 0:
                return -1e9
            return _CARDS[cid].points * 5.0
        if A.TAKE2_BASE <= a < A.TAKE2_BASE + A.TAKE2_COUNT:
            color = a - A.TAKE2_BASE
            return 4.0 + 0.5 * float(noble_pressure[color])
        if A.TAKE3_BASE <= a < A.TAKE3_BASE + A.TAKE3_COUNT:
            combo = _TAKE3_COMBOS[a - A.TAKE3_BASE]
            return 3.0 + 0.5 * sum(noble_pressure[c] for c in combo)
        return 0.0

    def _evaluate_candidates(
        self,
        engine: BE.BatchedEngine,
        candidates: list[list[int]],
    ) -> list[int]:
        """Apply each candidate on a cloned engine and pick the best by static value."""
        # Build a single replicated batch.
        max_k = max(len(cs) for cs in candidates)
        # Pad each candidate list with its first action so all games have
        # the same K. The padded extras evaluate to redundant clones; we just
        # ignore them when picking the best.
        padded: list[list[int]] = []
        for cs in candidates:
            if len(cs) == max_k:
                padded.append(list(cs))
            else:
                padded.append(list(cs) + [cs[0]] * (max_k - len(cs)))
        # Replicate the engine: each game becomes ``max_k`` copies side by side.
        replicated = engine.repeat_interleave(max_k)
        action_tensor = torch.tensor(
            [a for cs in padded for a in cs],
            dtype=torch.int64,
            device=engine.device,
        )
        replicated.apply(action_tensor)
        # Extract resulting views and score from the perspective of the
        # *original* current player.
        rep_views = _extract_views(replicated)
        cp_orig_list = engine.current_player.tolist()
        chosen: list[int] = []
        for i in range(engine.batch_size):
            cp_orig = int(cp_orig_list[i])
            best_score = -1e18
            best_a = candidates[i][0]
            for k in range(len(candidates[i])):
                rv = rep_views[i * max_k + k]
                val = self._post_state_value(rv, cp_orig)
                if val > best_score:
                    best_score = val
                    best_a = candidates[i][k]
            chosen.append(best_a)
        return chosen

    def _post_state_value(self, view: _GameView, our_seat: int) -> float:
        # Replace the view's perspective with our seat for scoring.
        proxy = _GameView(
            num_players=view.num_players,
            cp=our_seat,
            phase=view.phase,
            points=view.points,
            bonuses=view.bonuses,
            tokens=view.tokens,
            reserved=view.reserved,
            grid=view.grid,
            nobles=view.nobles,
            gem_pool=view.gem_pool,
            deck_top=view.deck_top,
            last_trigger=view.last_trigger,
            turns_since_trigger=view.turns_since_trigger,
        )
        return _static_value(proxy)


# ---------------------------------------------------------------------------
# V5/V6 helpers.
# ---------------------------------------------------------------------------


def _noble_progress(view: _GameView, seat: int) -> tuple[int, int]:
    """Return (best_remaining_bonuses, count_nobles_satisfied) for ``seat``.

    ``best_remaining_bonuses`` = the smallest sum-of-missing-color requirements
    across all unclaimed nobles for this seat (lower is better; 0 means a
    noble is already qualified).
    """
    bonuses = view.bonuses[seat]
    best_remaining = 99
    count_satisfied = 0
    for nslot in range(_MAX_NOBLE_SLOTS):
        nid = view.nobles[nslot]
        if nid < 0:
            continue
        n = _NOBLES[nid]
        rem = sum(max(0, n.requirement[c] - bonuses[c]) for c in range(_NUM_COLORS))
        if rem == 0:
            count_satisfied += 1
        if rem < best_remaining:
            best_remaining = rem
    if best_remaining == 99:
        best_remaining = 0
    return best_remaining, count_satisfied


def _best_noble_for_path(view: _GameView, seat: int) -> _NobleData | None:
    """Pick the noble closest to completion for ``seat``, ignoring claimed."""
    bonuses = view.bonuses[seat]
    best: _NobleData | None = None
    best_rem = 99
    for nslot in range(_MAX_NOBLE_SLOTS):
        nid = view.nobles[nslot]
        if nid < 0:
            continue
        n = _NOBLES[nid]
        rem = sum(max(0, n.requirement[c] - bonuses[c]) for c in range(_NUM_COLORS))
        if rem < best_rem:
            best_rem = rem
            best = n
    return best


class HeuristicOpusV5(HeuristicOpusV3):
    """Player-count-aware adaptive racer.

    Key behavioural fixes versus V1-V3 informed by tournament analysis:

    * 3-player and 4-player Splendor reward broader bonus accumulation
      (more nobles, more piles each turn). V5 buys point-zero tier-1 cards
      that progress the *closest* unclaimed noble or that bridge to a
      Tier-3 target, where V1-V3 would have skipped them in favour of
      taking tokens.
    * Denial weight is scaled by ``1 / max(1, num_players - 1)``: in 2p
      denial is great, in 3p+ wasting an action on a deny while two other
      seats race usually loses tempo.
    * Endgame mode only triggers after ``last_trigger >= 0`` *or* a
      player has 13+ points (V3's 11+ trigger was too aggressive at 3p).
    * Take-token planning uses a *two-target* missing-color set rather
      than just one, so token taking is robust when the immediate target
      is reserved by an opponent on the next turn.
    """

    name: str = "heuristic_opus_v5"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 5}

    # -- override endgame trigger --
    def _is_endgame(self, view: _GameView) -> bool:
        if view.last_trigger >= 0:
            return True
        max_pts = max(view.points)
        return max_pts >= 13

    # -- override main-action selection --
    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)

        noble_pressure = _noble_color_pressure(view)
        # 1. Denial reserves only when threat is large *relative to player count*.
        denial = self._denial_reserve_pc(view, mask_row)
        if denial is not None:
            return denial

        # 2. Best buy across visible + reserved (point-bearing or noble-helping).
        buy = self._best_buy_v5(view, mask_row, noble_pressure)
        if buy is not None:
            return buy

        # 3. Plan tokens against the top-2 targets simultaneously.
        target_a = _select_target_card(view, noble_pressure)
        targets: list[_CardData] = []
        if target_a is not None:
            targets.append(target_a)
        # Pick a second target whose color profile differs.
        target_b = self._second_target(view, noble_pressure, target_a)
        if target_b is not None:
            targets.append(target_b)

        if targets:
            cp_bonuses = view.bonuses[view.cp]
            cp_tokens = view.tokens[view.cp]
            # Combined missing vector: max of per-color missing across targets.
            combined_missing = [0] * _NUM_COLORS
            for t in targets:
                miss = _missing_per_color(t, cp_bonuses, cp_tokens)
                for c in range(_NUM_COLORS):
                    combined_missing[c] = max(combined_missing[c], miss[c])
            best_a, best_score = -1, -1e9
            for color in range(_NUM_COLORS):
                a = A.TAKE2_BASE + color
                if not mask_row[a]:
                    continue
                score = _take2_score_for_target(
                    color, combined_missing, view, noble_pressure
                )
                if score > best_score:
                    best_score = score
                    best_a = a
            for combo_idx, combo in enumerate(_TAKE3_COMBOS):
                a = A.TAKE3_BASE + combo_idx
                if not mask_row[a]:
                    continue
                score = _take3_score_for_target(
                    combo, combined_missing, view, noble_pressure
                )
                if score > best_score:
                    best_score = score
                    best_a = a
            if best_a >= 0 and best_score > 0:
                return best_a

        # 4. Reserve a high-PV target while gold is available, else fall back.
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        # 5. Any legal main action.
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION

    def _best_buy_v5(
        self,
        view: _GameView,
        mask_row: list[bool],
        noble_pressure: Sequence[float],
    ) -> int | None:
        """Buy a point-bearing card if any; else buy a 0-PV card whose bonus
        directly progresses the closest noble or unlocks a near-target Tier-3
        purchase. Otherwise returns None.
        """
        candidates = _list_affordable_buys(view, mask_row)
        if not candidates:
            return None
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        # First pass: any point-bearing buy.
        best_pv_a, best_pv_score = -1, -1e9
        for a, cid in candidates:
            card = _CARDS[cid]
            if card.points <= 0:
                continue
            score = card.points * 200.0 + _card_target_score(
                card, view, noble_pressure
            )
            if score > best_pv_score:
                best_pv_score = score
                best_pv_a = a
        if best_pv_a >= 0:
            return best_pv_a

        # Second pass: 0-PV buys that help nobles or upcoming Tier-3 buys.
        # Identify the closest noble for our seat.
        target_noble = _best_noble_for_path(view, view.cp)
        # And the strongest Tier-3 target on the board.
        tier3_target: _CardData | None = None
        tier3_best = -1e9
        for tier in range(2, _NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points < 4:
                    continue
                score = _card_target_score(card, view, noble_pressure)
                if score > tier3_best:
                    tier3_best = score
                    tier3_target = card

        best_zero_a, best_zero_score = -1, -1e9
        for a, cid in candidates:
            card = _CARDS[cid]
            if card.points > 0:
                continue
            score = 0.0
            if target_noble is not None:
                if cp_bonuses[card.bonus] < target_noble.requirement[card.bonus]:
                    score += 80.0
            if tier3_target is not None:
                # Does this bonus reduce the tier-3 cost in a tight spot?
                missing = _missing_per_color(
                    tier3_target, cp_bonuses, cp_tokens
                )
                if missing[card.bonus] > 0:
                    score += 40.0
            # Mild bonus for any card that increases bonus diversity past 1.
            if cp_bonuses[card.bonus] == 0:
                score += 15.0
            if score > best_zero_score:
                best_zero_score = score
                best_zero_a = a
        if best_zero_a < 0:
            return None
        if best_zero_score < 30.0:
            return None
        return best_zero_a

    def _denial_reserve_pc(
        self, view: _GameView, mask_row: list[bool]
    ) -> int | None:
        """Like V2 denial, but the threshold scales with player count."""
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        # 2p threshold 4 (PV gets denied), 3p ~6, 4p ~8.
        pc_factor = max(1, view.num_players - 1)
        threat_threshold = 4.0 * pc_factor
        threats: list[tuple[float, int]] = []
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points < 3:
                    continue
                worst_threat = 0.0
                for offset in range(1, view.num_players):
                    seat = (view.cp + offset) % view.num_players
                    if _affordable(card, view.bonuses[seat], view.tokens[seat]):
                        threat_score = float(card.points) * (
                            1.0 - 0.15 * (offset - 1)
                        )
                        if threat_score > worst_threat:
                            worst_threat = threat_score
                if worst_threat <= 0.0:
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                threats.append((worst_threat, a))
        if not threats:
            return None
        threats.sort(reverse=True)
        worst, a = threats[0]
        if worst < threat_threshold:
            return None
        return a

    def _second_target(
        self,
        view: _GameView,
        noble_pressure: Sequence[float],
        first: _CardData | None,
    ) -> _CardData | None:
        if first is None:
            return None
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        first_missing = _missing_per_color(first, cp_bonuses, cp_tokens)
        best_card: _CardData | None = None
        best_score = -1e9
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.card_id == first.card_id:
                    continue
                score = _card_target_score(card, view, noble_pressure)
                # Bonus for color overlap (we'd progress both targets).
                miss = _missing_per_color(card, cp_bonuses, cp_tokens)
                overlap = sum(1 for c in range(_NUM_COLORS) if miss[c] and first_missing[c])
                score += overlap * 5.0
                if score > best_score:
                    best_score = score
                    best_card = card
        return best_card


# ---------------------------------------------------------------------------
# V6 -- shortest-path planner.
# ---------------------------------------------------------------------------


def _est_turns_to_15(view: _GameView, seat: int) -> float:
    """Cheap optimistic estimate of how many actions the seat needs to reach 15.

    For each visible point-bearing card we estimate ``turns_to_afford`` plus
    1 (the buy itself). We pick the smallest cumulative cost combination
    greedily that sums to >=15 PV. Open to overestimation but useful for
    relative comparisons between seats."""
    bonuses = list(view.bonuses[seat])
    tokens = list(view.tokens[seat])
    pool = view.gem_pool
    pts_left = max(0, _WINNING_POINTS - view.points[seat])
    if pts_left <= 0:
        return 0.0
    candidates: list[_CardData] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            c = _CARDS[cid]
            if c.points > 0:
                candidates.append(c)
    for r in range(_MAX_RESERVED):
        cid = view.reserved[seat][r]
        if cid < 0:
            continue
        c = _CARDS[cid]
        if c.points > 0:
            candidates.append(c)
    if not candidates:
        return 100.0  # hopeless
    plan: list[_CardData] = []
    while pts_left > 0 and candidates:
        # Pick card with smallest distance/PV ratio.
        best, best_score = None, 1e9
        for c in candidates:
            d = _turns_to_afford(c, bonuses, tokens, pool) + 1
            if c.points <= 0:
                continue
            score = d / max(1, c.points)
            if score < best_score:
                best_score = score
                best = c
        if best is None:
            break
        plan.append(best)
        pts_left -= best.points
        candidates.remove(best)
        # Pretend we bought it -- gain bonus + reduce tokens by net cost.
        for col in range(_NUM_COLORS):
            need = max(0, best.cost[col] - bonuses[col])
            spent = min(need, tokens[col])
            tokens[col] -= spent
        bonuses[best.bonus] += 1
    if pts_left > 0:
        return 100.0
    total_actions = 0.0
    bonuses2 = list(view.bonuses[seat])
    tokens2 = list(view.tokens[seat])
    for c in plan:
        total_actions += _turns_to_afford(c, bonuses2, tokens2, pool) + 1
        for col in range(_NUM_COLORS):
            need = max(0, c.cost[col] - bonuses2[col])
            spent = min(need, tokens2[col])
            tokens2[col] -= spent
        bonuses2[c.bonus] += 1
    return total_actions


class HeuristicOpusV7(HeuristicOpusV3):
    """V3 with player-count-aware denial and one extra refinement.

    Tournament data showed V3 was the strongest baseline at 3p/4p but its
    denial threshold (any threat >= 4 PV) was too aggressive at 3p+ where
    spending an action denying one opponent lets the other(s) race ahead.
    V7 scales the denial threshold by ``num_players - 1`` so 2p stays
    aggressive (threshold 4) while 3p uses 8 and 4p uses 12.

    V7 also uses the *bottleneck-color* take2 trick from the V6 attempt
    but ONLY when the chosen target is at least 4 turns away by colored-
    token distance. This keeps mid-game tokens efficient without breaking
    the early-game racer balance V3 already tunes well.
    """

    name: str = "heuristic_opus_v7"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 7}

    def _denial_reserve(
        self, view: _GameView, mask_row: list[bool]
    ) -> int | None:
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        pc_factor = max(1, view.num_players - 1)
        threat_threshold = 4.0 * pc_factor
        threats: list[tuple[float, int]] = []
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points < 3:
                    continue
                worst_threat = 0.0
                for offset in range(1, view.num_players):
                    seat = (view.cp + offset) % view.num_players
                    if _affordable(card, view.bonuses[seat], view.tokens[seat]):
                        threat_score = float(card.points) * (
                            1.0 - 0.15 * (offset - 1)
                        )
                        if threat_score > worst_threat:
                            worst_threat = threat_score
                if worst_threat <= 0.0:
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                threats.append((worst_threat, a))
        if not threats:
            return None
        threats.sort(reverse=True)
        worst, a = threats[0]
        if worst < threat_threshold:
            return None
        return a

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        # Denial uses the new pc-scaled threshold (overrides V2's _denial_reserve
        # only via the V2._choose_main path; we replicate it here so V7 is
        # self-contained for the non-endgame path).
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        # Fall through to V1 logic for buy/target/take/reserve.
        return HeuristicOpusV1._choose_main(self, view, mask_row)


def _take2_post_distance(
    color: int,
    target: _CardData,
    view: _GameView,
) -> tuple[int, int]:
    """Return (post-distance, post-deficit) after applying a take-2 of color.

    Distance is ``_turns_to_afford(target)`` after the take. Deficit is the
    raw token shortfall (without gold) so ties on distance can be broken by
    nominal progress. Returns ``(99, 99)`` if the move is illegal (pile <4).
    """
    if view.gem_pool[color] < 4:
        return (99, 99)
    cp_bonuses = view.bonuses[view.cp]
    cp_tokens = list(view.tokens[view.cp])
    cp_tokens[color] += 2
    deficit = 0
    for c in range(_NUM_COLORS):
        need = max(0, target.cost[c] - cp_bonuses[c])
        deficit += max(0, need - cp_tokens[c])
    deficit_after_gold = max(0, deficit - cp_tokens[_GOLD])
    distance = 0 if deficit_after_gold == 0 else (deficit_after_gold + 2) // 3
    return (distance, deficit)


def _take3_post_distance(
    combo: tuple[int, int, int],
    target: _CardData,
    view: _GameView,
) -> tuple[int, int]:
    """Same as ``_take2_post_distance`` but for a take-3 combo."""
    cp_bonuses = view.bonuses[view.cp]
    cp_tokens = list(view.tokens[view.cp])
    for c in combo:
        if view.gem_pool[c] <= 0:
            continue
        cp_tokens[c] += 1
    deficit = 0
    for c in range(_NUM_COLORS):
        need = max(0, target.cost[c] - cp_bonuses[c])
        deficit += max(0, need - cp_tokens[c])
    deficit_after_gold = max(0, deficit - cp_tokens[_GOLD])
    distance = 0 if deficit_after_gold == 0 else (deficit_after_gold + 2) // 3
    return (distance, deficit)


def _bridge_buy(
    view: _GameView,
    mask_row: list[bool],
    target: _CardData,
) -> int | None:
    """Return a buy-action index for a 0-PV card whose bonus shortens
    the target's affordability distance by >=1, or ``None``.

    A "bridge buy" trades a tempo-neutral action (an affordable 0-PV card)
    for a permanent +1 bonus that strictly reduces the target's
    ``turns_to_afford``. We require strict improvement so the bridge buy
    can't loop or stall.
    """
    cp_bonuses = view.bonuses[view.cp]
    cp_tokens = view.tokens[view.cp]
    base_distance = _turns_to_afford(target, cp_bonuses, cp_tokens, view.gem_pool)
    if base_distance <= 0:
        return None
    candidates = _list_affordable_buys(view, mask_row)
    best_a = -1
    best_save = 0
    for a, cid in candidates:
        card = _CARDS[cid]
        if card.points > 0:
            continue
        if cp_bonuses[card.bonus] >= target.cost[card.bonus]:
            continue
        # Pretend we bought it: bonus +=1, tokens reduced by net cost.
        new_bonuses = list(cp_bonuses)
        new_bonuses[card.bonus] += 1
        new_tokens = list(cp_tokens)
        gold_left = new_tokens[_GOLD]
        for c in range(_NUM_COLORS):
            need = max(0, card.cost[c] - cp_bonuses[c])
            spent = min(need, new_tokens[c])
            new_tokens[c] -= spent
            deficit = need - spent
            if deficit > 0:
                spent_gold = min(deficit, gold_left)
                gold_left -= spent_gold
        new_tokens[_GOLD] = gold_left
        new_distance = _turns_to_afford(target, new_bonuses, new_tokens, view.gem_pool)
        save = base_distance - new_distance
        if save > best_save:
            best_save = save
            best_a = a
    if best_a < 0 or best_save < 1:
        return None
    return best_a


# ---------------------------------------------------------------------------
# V8 -- bridge buys + post-action target distance for take-tokens.
# ---------------------------------------------------------------------------


class HeuristicOpusV8(HeuristicOpusV7):
    """V7 plus two refinements driven by tournament analysis.

    1. **Zero-PV "bridge buys".** Right after the no-point-bearing-affordable-
       buy step, if a 0-PV card on the grid is affordable and its bonus
       strictly shortens our chosen target's ``turns_to_afford`` by >=1, buy
       it. This converts a discounted token-take action into a permanent
       discount that compounds across remaining buys, which in self-play has
       been strictly stronger than spending the same tempo on yet another
       take-3 round.

    2. **Take-token chosen by post-action target distance.** Instead of the
       per-color additive heuristic in ``_take{2,3}_score_for_target``,
       enumerate every legal take-2/take-3 action, compute the resulting
       ``turns_to_afford`` against the chosen target, and pick the action
       that minimizes it. Ties broken by raw deficit, then by noble
       pressure on the picked colors.
    """

    name: str = "heuristic_opus_v8"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 8}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        # Step 1: best affordable point-bearing card.
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        # Step 2: pick a target.
        target = _select_target_card(view, noble_pressure)
        # Step 3 (new): bridge buy if it strictly shortens the target.
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        # Step 4: take tokens by post-action target distance.
        if target is not None:
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        # Step 5: fallback -- reserve a high-PV card or pass.
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION

    def _best_take_for_target(
        self,
        view: _GameView,
        mask_row: list[bool],
        target: _CardData,
        noble_pressure: Sequence[float],
    ) -> int:
        """Pick the take-token action minimizing target distance.

        Tie-break order:
        1. Smallest post-take distance (turns_to_afford).
        2. Smallest post-take raw deficit (so taking 3 distinct colors of
           which only 1 is needed beats taking 0 needed colors).
        3. Largest sum of noble pressure on the colors taken.
        4. Lowest action index (deterministic).
        """
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        base_distance = _turns_to_afford(target, cp_bonuses, cp_tokens, view.gem_pool)
        best_a = -1
        best_key: tuple[int, int, float, int] | None = None
        for color in range(_NUM_COLORS):
            a = A.TAKE2_BASE + color
            if not mask_row[a]:
                continue
            distance, deficit = _take2_post_distance(color, target, view)
            np_bonus = float(noble_pressure[color]) * 2.0
            # Penalize stockpiling: if we already have >=4 of this color, the
            # take is mostly wasted.
            stockpile_pen = 4 if cp_tokens[color] >= 4 else 0
            key = (distance + stockpile_pen, deficit, -np_bonus, a)
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        for combo_idx, combo in enumerate(_TAKE3_COMBOS):
            a = A.TAKE3_BASE + combo_idx
            if not mask_row[a]:
                continue
            distance, deficit = _take3_post_distance(combo, target, view)
            np_bonus = sum(float(noble_pressure[c]) for c in combo)
            stockpile_pen = sum(2 for c in combo if cp_tokens[c] >= 4)
            key = (distance + stockpile_pen, deficit, -np_bonus, a)
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        # If nothing improves over base distance and we already have a high
        # token count, skip take-tokens to avoid burning tempo.
        if best_key is None:
            return -1
        post_distance = best_key[0]
        if post_distance >= base_distance and sum(cp_tokens[:_NUM_COLORS]) >= 8:
            return -1
        return best_a


def _list_point_card_targets(view: _GameView) -> list[_CardData]:
    """Return all visible (grid + own reserved) cards with points >= 1.

    Used by V9's path planner. We exclude 0-PV cards because the planner
    is a race-to-15 search; bridge buys are handled separately by V8's
    ``_bridge_buy``. Capping at points-bearing reduces the candidate set
    typically to 6-10 cards, which keeps the O(N^2) pair search cheap.
    """
    out: list[_CardData] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            card = _CARDS[cid]
            if card.points >= 1:
                out.append(card)
    for r in range(_MAX_RESERVED):
        cid = view.reserved[view.cp][r]
        if cid < 0:
            continue
        card = _CARDS[cid]
        if card.points >= 1:
            out.append(card)
    return out


def _path_target_v9(
    view: _GameView,
    noble_pressure: Sequence[float],
) -> _CardData | None:
    """Pick the immediate buy-target for a shortest-path-to-15 plan.

    Search every 1-card and 2-card combination of visible/reserved
    point-bearing cards. For each combination, pretend we buy them in
    order of cheapest-first, computing cumulative ``turns_to_afford``
    (using running bonuses + tokens) plus 1 action per buy. Pick the
    combination with the smallest total actions that reaches 15 PV. Return
    the *first* card to buy (the one with smaller individual distance).

    If no combination reaches 15 within a sane budget, fall back to the
    single best individual card by ``_card_target_score`` (V1 logic).
    """
    cp = view.cp
    pts_have = view.points[cp]
    pts_needed = max(0, _WINNING_POINTS - pts_have)
    if pts_needed <= 0:
        return None
    cands = _list_point_card_targets(view)
    if not cands:
        return None
    cp_bonuses = list(view.bonuses[cp])
    cp_tokens = list(view.tokens[cp])
    pool = view.gem_pool

    def _path_actions(plan: list[_CardData]) -> tuple[float, _CardData]:
        bonuses = list(cp_bonuses)
        tokens = list(cp_tokens)
        total = 0.0
        first = plan[0]
        for c in plan:
            d = _turns_to_afford(c, bonuses, tokens, pool) + 1
            total += d
            for col in range(_NUM_COLORS):
                need = max(0, c.cost[col] - bonuses[col])
                spent = min(need, tokens[col])
                tokens[col] -= spent
            bonuses[c.bonus] += 1
        return (total, first)

    best_total = 1e9
    best_first: _CardData | None = None
    # Single-card plans (only useful if a single card already worth >= pts_needed).
    for c in cands:
        if c.points < pts_needed:
            continue
        total, _ = _path_actions([c])
        if total < best_total:
            best_total = total
            best_first = c

    # Two-card plans summing to >= pts_needed.
    n = len(cands)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cands[i].points + cands[j].points < pts_needed:
                continue
            # Try both buy orders, taking the cheaper one.
            t_ij, _ = _path_actions([cands[i], cands[j]])
            t_ji, _ = _path_actions([cands[j], cands[i]])
            if t_ij <= t_ji:
                total, first = t_ij, cands[i]
            else:
                total, first = t_ji, cands[j]
            # Tie-break on noble synergy of the first card.
            adjusted = total - 0.05 * float(noble_pressure[first.bonus])
            if adjusted < best_total:
                best_total = adjusted
                best_first = first

    if best_first is None:
        return _select_target_card(view, noble_pressure)
    return best_first


# ---------------------------------------------------------------------------
# V9 -- race-to-15 path planner.
# ---------------------------------------------------------------------------


class HeuristicOpusV9(HeuristicOpusV8):
    """V8 plus a 2-card race-to-15 path planner for target selection.

    V8 picks a single buy-target by static PPT-driven scoring, which is
    myopic when 2 visible cards together reach 15 PV faster than the
    best individual card on its own. V9 enumerates 1-card and 2-card
    plans across visible+reserved point-bearing cards and picks the one
    with the minimum estimated total actions (cumulative
    ``turns_to_afford`` + 1 buy each). The first card of the winning
    plan becomes the immediate target. V8's bridge-buy and take-token
    logic operate against that target unchanged.

    The 2-card search is O(N^2) over visible point-bearing cards (N is
    typically 6-10) and runs once per ``choose`` call, so the overhead
    is negligible against engine.step.
    """

    name: str = "heuristic_opus_v9"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 9}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = _path_target_v9(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


def _opponent_alternative_buys(view: _GameView, exclude_cid: int) -> int:
    """Count visible point-bearing cards (excluding ``exclude_cid``) that
    *some* opponent can afford right now.

    Used by V10 to skip a denial reserve when the threatened opponent has
    multiple alternative buys available -- denying one card just reroutes
    the threat to another, so the reserve is wasted tempo.
    """
    count = 0
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0 or cid == exclude_cid:
                continue
            card = _CARDS[cid]
            if card.points <= 0:
                continue
            for offset in range(1, view.num_players):
                seat = (view.cp + offset) % view.num_players
                if _affordable(card, view.bonuses[seat], view.tokens[seat]):
                    count += 1
                    break
    return count


def _noble_progress_bridge(
    view: _GameView,
    mask_row: list[bool],
) -> int | None:
    """Buy a 0-PV affordable card if its bonus closes the gap on a noble we
    can plausibly claim within 2 more bonus picks.

    Returns the buy-action index or ``None``. Used by V10 as a secondary
    bridge after the strict turn-saving bridge fails to fire. The bound
    "2 more bonus picks" prevents wasting actions on a noble that's still
    far away."""
    cp_bonuses = view.bonuses[view.cp]
    candidates = _list_affordable_buys(view, mask_row)
    if not candidates:
        return None
    best_a = -1
    best_score = -1
    for nslot in range(_MAX_NOBLE_SLOTS):
        nid = view.nobles[nslot]
        if nid < 0:
            continue
        n = _NOBLES[nid]
        rem_total = sum(
            max(0, n.requirement[c] - cp_bonuses[c]) for c in range(_NUM_COLORS)
        )
        if rem_total == 0 or rem_total > 3:
            continue
        for a, cid in candidates:
            card = _CARDS[cid]
            if card.points > 0:
                continue
            need = n.requirement[card.bonus] - cp_bonuses[card.bonus]
            if need <= 0:
                continue
            score = need * 10 + (4 - rem_total)
            if score > best_score:
                best_score = score
                best_a = a
    if best_a < 0:
        return None
    return best_a


# ---------------------------------------------------------------------------
# V10 -- V8 + selective denial + noble-bridge buys.
# ---------------------------------------------------------------------------


class HeuristicOpusV10(HeuristicOpusV8):
    """V8 plus two refinements.

    1. **Selective denial.** V8/V7 reserve any card scoring above the
       pc-scaled threat threshold. V10 also requires that the threatened
       opponent have <= 1 other affordable point-bearing card on the
       board; otherwise reserving is wasted tempo (the opponent just buys
       a different one). At 4p this routinely flips a useless deny into
       progressing our own target.
    2. **Noble-bridge buys.** After the strict turn-saving bridge buy
       fails to fire, look for a 0-PV affordable card whose bonus closes
       our path toward a noble that's <=3 bonus picks away. This widens
       early-game color diversity in 3p+ without the V5 over-correction
       (we still prioritize point-bearing buys and turn-saving bridges
       first).
    """

    name: str = "heuristic_opus_v10"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 10}

    def _denial_reserve(
        self, view: _GameView, mask_row: list[bool]
    ) -> int | None:
        cp_reserved = view.reserved[view.cp]
        if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
            return None
        pc_factor = max(1, view.num_players - 1)
        threat_threshold = 4.0 * pc_factor
        threats: list[tuple[float, int, int]] = []
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points < 3:
                    continue
                worst_threat = 0.0
                for offset in range(1, view.num_players):
                    seat = (view.cp + offset) % view.num_players
                    if _affordable(card, view.bonuses[seat], view.tokens[seat]):
                        threat_score = float(card.points) * (
                            1.0 - 0.15 * (offset - 1)
                        )
                        if threat_score > worst_threat:
                            worst_threat = threat_score
                if worst_threat <= 0.0:
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if not mask_row[a]:
                    continue
                threats.append((worst_threat, a, cid))
        if not threats:
            return None
        threats.sort(reverse=True)
        worst, a, cid = threats[0]
        if worst < threat_threshold:
            return None
        # Selectivity: skip if the opponent has 2+ alternative buys; the deny
        # would just reroute, not block.
        alts = _opponent_alternative_buys(view, exclude_cid=cid)
        if alts >= 2:
            return None
        return a

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = _select_target_card(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        # Secondary bridge: a 0-PV affordable card progressing a near noble.
        nb = _noble_progress_bridge(view, mask_row)
        if nb is not None:
            return nb
        if target is not None:
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


def _self_target_reserve(
    view: _GameView,
    mask_row: list[bool],
    target: _CardData,
) -> int | None:
    """Return a reserve action that locks ``target`` for us, or ``None``.

    Conditions:
    * we're not already at the 3-card reserve cap
    * gold is available (so the reserve gives us +1 gold)
    * the target is 1-2 turns from affordability (otherwise tokens are better)
    * the target is on the grid (we can't reserve a card already in our reserve)
    * an opponent could plausibly afford it within ~3 turns (else it'll just sit
      there and the reserve is wasted tempo)

    Locking a critical target with gold is usually +1 EV when the target is
    contested and within close reach.
    """
    cp = view.cp
    cp_reserved = view.reserved[cp]
    if sum(1 for x in cp_reserved if x >= 0) >= _MAX_RESERVED:
        return None
    if view.gem_pool[_GOLD] <= 0:
        return None
    cp_bonuses = view.bonuses[cp]
    cp_tokens = view.tokens[cp]
    distance = _turns_to_afford(target, cp_bonuses, cp_tokens, view.gem_pool)
    if distance < 1 or distance > 2:
        return None
    # Find the grid position of the target.
    grid_a = -1
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            if cid == target.card_id:
                grid_a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                break
        if grid_a >= 0:
            break
    if grid_a < 0:
        return None
    if not mask_row[grid_a]:
        return None
    # Check opponent contention.
    contested = False
    for offset in range(1, view.num_players):
        seat = (cp + offset) % view.num_players
        # An opponent is plausibly racing for this card if they're within a
        # few colored-token gap of affordability OR they've reserved a similar
        # card themselves.
        o_bonuses = view.bonuses[seat]
        o_tokens = view.tokens[seat]
        o_distance = _turns_to_afford(target, o_bonuses, o_tokens, view.gem_pool)
        if o_distance <= 3:
            contested = True
            break
    if not contested:
        return None
    return grid_a


# ---------------------------------------------------------------------------
# V11 -- V8 + "self-target reserve" for gold + lock-in when contested.
# ---------------------------------------------------------------------------


class HeuristicOpusV11(HeuristicOpusV8):
    """V8 plus a self-target reserve action for gold + lock-in.

    When our chosen target is 1-2 turns from affordability and opponents
    are also racing for it, reserving the target ourselves picks up a
    gold token *and* removes it from the grid so no opponent can buy it.
    The +1 gold doubles as a wildcard, often shaving an extra turn off
    the affording timeline. V8 misses this because its reserve fallback
    only fires for unaffordable point-bearing cards as a last resort.

    The check is conservative -- distance must be exactly 1 or 2 turns
    away, gold must be in the pool, and at least one opponent must be
    within 3 turns of affording the same card. Otherwise V8 logic
    (take tokens against the target) is strictly better.
    """

    name: str = "heuristic_opus_v11"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 11}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = _select_target_card(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
            self_reserve = _self_target_reserve(view, mask_row, target)
            if self_reserve is not None:
                return self_reserve
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


def _opponent_min_distance(view: _GameView, card: _CardData) -> int:
    """Smallest ``turns_to_afford`` for ``card`` across all opponents.

    Used by V14's contention-aware target picker. A small value means the
    card is highly contested -- racing for it risks the opponent buying
    it first. Cards with no opponent in range get distance ``99``.
    """
    best = 99
    for offset in range(1, view.num_players):
        seat = (view.cp + offset) % view.num_players
        d = _turns_to_afford(card, view.bonuses[seat], view.tokens[seat], view.gem_pool)
        if d < best:
            best = d
    return best


def _select_target_card_v14(
    view: _GameView,
    noble_pressure: Sequence[float],
) -> _CardData | None:
    """Like ``_select_target_card`` but penalizes targets contested by
    opponents who could afford them sooner than us.

    The contention penalty is a function of ``opponent_distance - our_distance``:
    if an opponent can afford the card before we can, the card is heavily
    penalized; if we're equal-distance, mildly penalized; if we're closer,
    no penalty.
    """
    best_card: _CardData | None = None
    best_score = -1e9
    cp_bonuses = view.bonuses[view.cp]
    cp_tokens = view.tokens[view.cp]
    candidates: list[_CardData] = []
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            candidates.append(_CARDS[cid])
    for r in range(_MAX_RESERVED):
        cid = view.reserved[view.cp][r]
        if cid < 0:
            continue
        candidates.append(_CARDS[cid])
    for card in candidates:
        score = _card_target_score(card, view, noble_pressure)
        our_distance = _turns_to_afford(card, cp_bonuses, cp_tokens, view.gem_pool)
        score -= max(0, our_distance - 1) * 6.0
        # Contention penalty: only apply for cards on the grid (reserved cards
        # cannot be taken by opponents).
        on_grid = any(
            view.grid[tier][s] == card.card_id
            for tier in range(_NUM_TIERS)
            for s in range(_NUM_GRID_SLOTS)
        )
        if on_grid:
            opp_distance = _opponent_min_distance(view, card)
            gap = our_distance - opp_distance
            if gap > 0:
                # Opponent reaches it sooner; large penalty.
                score -= 25.0 * gap
            elif gap == 0 and our_distance >= 1:
                # Tie: mild penalty (we might lose tempo to denial reserves).
                score -= 8.0
        if score > best_score:
            best_score = score
            best_card = card
    return best_card


# ---------------------------------------------------------------------------
# V14 -- V10 + opponent-contention-aware target selection.
# ---------------------------------------------------------------------------


class HeuristicOpusV14(HeuristicOpusV10):
    """V10 plus a contention-aware target picker.

    V10 picks the highest-PPT reachable card on the board, regardless of
    whether an opponent could buy it sooner. V14 adds a penalty when an
    opponent's ``turns_to_afford`` is <= ours: chasing a contested card
    we'll likely lose burns tokens we can't recover. The penalty is
    proportional to how many turns the opponent leads us by.

    All other V10 behavior (selective denial, noble-bridge buys, V8's
    bridge buy + take-for-target) is unchanged.
    """

    name: str = "heuristic_opus_v14"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 14}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = _select_target_card_v14(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        nb = _noble_progress_bridge(view, mask_row)
        if nb is not None:
            return nb
        if target is not None:
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


# ---------------------------------------------------------------------------
# V13 -- V10's denial/bridge logic + V9's path planner.
# ---------------------------------------------------------------------------


class HeuristicOpusV13(HeuristicOpusV10):
    """V10 (selective denial + noble-bridge) plus V9's race-to-15 target.

    V10 wins aggregate ratings driven by a 4p edge from selective denials
    and noble-bridge 0-PV buys. V9 wins 3p ratings via the 2-card race-to-
    15 target planner. V13 stacks both: V10's pre-target structure (denial
    skip on substitutability + noble bridge after the strict bridge) is
    preserved, but the immediate target is picked by ``_path_target_v9``
    instead of V8's single-card score. Single-card plans are also covered
    by the path planner so 2p doesn't lose tactical sharpness.
    """

    name: str = "heuristic_opus_v13"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 13}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = _path_target_v9(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        nb = _noble_progress_bridge(view, mask_row)
        if nb is not None:
            return nb
        if target is not None:
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


# ---------------------------------------------------------------------------
# V15 -- adaptive: V13 (path planner) at 2p/3p, V10 (selective denial) at 4p.
# ---------------------------------------------------------------------------


class HeuristicOpusV15(HeuristicOpusV13):
    """Adaptive opus combining V13's path planner and V10's stability.

    Tournament data with 48-game samples consistently showed:
    * V13's race-to-15 path planner wins 2p (and is competitive 3p)
      because tighter games reward multi-card lookahead.
    * V13's path planner regresses at 4p where chaotic multi-opponent
      play invalidates planned paths quickly; V10's single-card target
      with selective denial is steadier.

    V15 routes 2p and 3p through V13's planner (its strongest regime)
    and 4p through V10's single-card flow. Both share the V8 base
    (bridge buys + take-for-target) and V10's denial/noble-bridge
    structure, so the dispatch is purely on target selection.
    """

    name: str = "heuristic_opus_v15"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 15}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players >= 4:
            return HeuristicOpusV10._choose_main(self, view, mask_row)
        return HeuristicOpusV13._choose_main(self, view, mask_row)


class HeuristicOpusV16(HeuristicOpusV13):
    """V15 minus path planning at 3p.

    Deep tournament data showed V8 beats V15 head-to-head at 3p
    (V15 = 0.448 win rate with ~6% ties), suggesting V13's path planner
    hurts in 3p chaos similar to 4p. V16 tests: keep V13's planner only
    at 2p (where it wins clearly), use V8 at 3p, V10 at 4p.
    """

    name: str = "heuristic_opus_v16"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 16}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players == 2:
            return HeuristicOpusV13._choose_main(self, view, mask_row)
        if view.num_players == 3:
            return HeuristicOpusV8._choose_main(self, view, mask_row)
        return HeuristicOpusV10._choose_main(self, view, mask_row)


# ---------------------------------------------------------------------------
# V17 -- V15 + 1-ply lookahead for buy / take / reserve.
# ---------------------------------------------------------------------------


class HeuristicOpusV17(HeuristicOpusV15):
    """V15 plus 1-ply opponent-aware lookahead.

    Three stacked refinements over V15. Each one is gated to a specific
    decision step (buy / take / reserve) so that V15's adaptive 2/3/4-player
    dispatch and existing structure are unchanged when the new logic is
    not load-bearing.

    1. **Opponent-aware buy.** ``_best_affordable_buy`` is overridden:
       for each affordable buy candidate, we simulate the buy on a cloned
       view (clearing the bought slot, conservatively skipping the random
       deck refill) and compute the maximum PV any opponent could buy from
       the resulting state. The buy's score is penalized by this post-buy
       threat. Net effect: at equal PV/PPT, prefer the buy that absorbs
       the biggest exposed card.

    2. **Smart self-reserve.** When V15 falls all the way through to
       ``_best_reserve``, V17 first checks whether the path planner's
       chosen target is high-PV and far away (turns_to_afford >= 4),
       gold is available in the pool, and a reserve slot is open. If so,
       reserve that target for the gold + lock-in. V11 tried a broader
       version of this and regressed; V17's gating keeps the cost low.

    3. **Take-3 opponent-drain bonus.** ``_best_take_for_target`` is
       extended to break ties among take combos by preferring those that
       drain colors opponents need most for their own best targets. Pure
       tie-break (does not change the primary "post-take distance"
       ordering); cheap to compute and keeps tempo intact.
    """

    name: str = "heuristic_opus_v17"
    _LOOKAHEAD_THREAT_WEIGHT: float = 35.0
    _SELF_RESERVE_MIN_PV: int = 3
    _SELF_RESERVE_MIN_DIST: int = 4
    _TAKE_DRAIN_TIE_WEIGHT: float = 0.05

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 17}

    def _best_affordable_buy(
        self,
        view: _GameView,
        mask_row: list[bool],
        noble_pressure: Sequence[float],
    ) -> int | None:
        candidates = _list_affordable_buys(view, mask_row)
        if not candidates:
            return None
        best_a, best_cid, best_score = -1, -1, -1e9
        for a, cid in candidates:
            card = _CARDS[cid]
            score = card.points * 200.0 + _card_target_score(
                card, view, noble_pressure
            )
            if card.points == 0 and card.level == 1:
                if noble_pressure[card.bonus] <= 0:
                    score -= 60.0
            future = _shallow_copy_view(view)
            _apply_buy_clear_slot(future, view.cp, a, card)
            post_threat = _opponent_max_pv_threat(future, view.cp)
            score -= self._LOOKAHEAD_THREAT_WEIGHT * post_threat
            if score > best_score:
                best_score = score
                best_a = a
                best_cid = cid
        if best_a < 0:
            return None
        chosen_card = _CARDS[best_cid]
        if chosen_card.points > 0:
            return best_a
        if noble_pressure[chosen_card.bonus] > 0:
            return best_a
        return None

    def _best_take_for_target(
        self,
        view: _GameView,
        mask_row: list[bool],
        target: _CardData,
        noble_pressure: Sequence[float],
    ) -> int:
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        base_distance = _turns_to_afford(target, cp_bonuses, cp_tokens, view.gem_pool)
        opp_demand = _opponent_color_demand(view, view.cp)
        best_a = -1
        best_key: tuple[float, int, float, int] | None = None
        for color in range(_NUM_COLORS):
            a = A.TAKE2_BASE + color
            if not mask_row[a]:
                continue
            distance, deficit = _take2_post_distance(color, target, view)
            np_bonus = float(noble_pressure[color]) * 2.0
            stockpile_pen = 4 if cp_tokens[color] >= 4 else 0
            drain = float(opp_demand[color]) * 2.0
            primary = distance + stockpile_pen
            tie_secondary = -np_bonus - self._TAKE_DRAIN_TIE_WEIGHT * drain
            key = (float(primary), deficit, tie_secondary, a)
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        for combo_idx, combo in enumerate(_TAKE3_COMBOS):
            a = A.TAKE3_BASE + combo_idx
            if not mask_row[a]:
                continue
            distance, deficit = _take3_post_distance(combo, target, view)
            np_bonus = sum(float(noble_pressure[c]) for c in combo)
            stockpile_pen = sum(2 for c in combo if cp_tokens[c] >= 4)
            drain = sum(float(opp_demand[c]) for c in combo)
            primary = distance + stockpile_pen
            tie_secondary = -np_bonus - self._TAKE_DRAIN_TIE_WEIGHT * drain
            key = (float(primary), deficit, tie_secondary, a)
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        if best_key is None:
            return -1
        post_distance = best_key[0]
        if post_distance >= base_distance and sum(cp_tokens[:_NUM_COLORS]) >= 8:
            return -1
        return best_a

    def _smart_self_reserve(
        self,
        view: _GameView,
        mask_row: list[bool],
        target: _CardData | None,
    ) -> int | None:
        """Reserve our own far high-PV target if conditions match.

        Gating (avoiding V11's regression):
          * target points >= 3 (don't waste reserves on cheap cards)
          * turns_to_afford >= 4 (close targets reach by token-take faster)
          * gold available in the pool (the value of reserving is the gold)
          * at least one reserved slot free
          * we currently have < 2 reserved cards (don't clog reserves)
          * not in endgame (last_trigger or max_pts >= 11)
        """
        if target is None or target.points < self._SELF_RESERVE_MIN_PV:
            return None
        if view.gem_pool[_GOLD] <= 0:
            return None
        cp_reserved = view.reserved[view.cp]
        used = sum(1 for x in cp_reserved if x >= 0)
        if used >= _MAX_RESERVED:
            return None
        if used >= 2:
            return None
        if view.last_trigger >= 0:
            return None
        if any(p >= 11 for p in view.points):
            return None
        cp_bonuses = view.bonuses[view.cp]
        cp_tokens = view.tokens[view.cp]
        dist = _turns_to_afford(target, cp_bonuses, cp_tokens, view.gem_pool)
        if dist < self._SELF_RESERVE_MIN_DIST:
            return None
        # Find the action ID for reserving this card.
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                if view.grid[tier][s] != target.card_id:
                    continue
                a = A.RESERVE_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                if mask_row[a]:
                    return a
        return None

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        # Endgame / denial / best-affordable-buy paths live in V15 / V13 /
        # V10 / V8 and now use this class's overridden _best_affordable_buy
        # (via virtual dispatch). We extend with smart-self-reserve at the
        # very end of the take-target step.
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        if view.num_players >= 4:
            target = _select_target_card(view, noble_pressure)
        else:
            target = _path_target_v9(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        nb = _noble_progress_bridge(view, mask_row)
        if nb is not None:
            return nb
        if target is not None:
            best_a = self._best_take_for_target(view, mask_row, target, noble_pressure)
            if best_a >= 0:
                return best_a
        # New: smart self-reserve before falling back to V8's _best_reserve.
        sr = self._smart_self_reserve(view, mask_row, target)
        if sr is not None:
            return sr
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


def _opponent_best_buy_action(
    view: _GameView, opp_seat: int
) -> tuple[int, _CardData] | None:
    """Single-ply: opponent's best affordable buy by V8-style score.

    Score = ``points * 200`` only (we omit V13's PPT term to keep the
    inner-loop cost low; the 2-ply lookahead is dominated by the PV
    component for opponent picks anyway). Ties broken by lower action
    index (deterministic).
    """
    bonuses = view.bonuses[opp_seat]
    tokens = view.tokens[opp_seat]
    best_a, best_card, best_score = -1, None, -1e9
    for tier in range(_NUM_TIERS):
        for s in range(_NUM_GRID_SLOTS):
            cid = view.grid[tier][s]
            if cid < 0:
                continue
            card = _CARDS[cid]
            if not _affordable(card, bonuses, tokens):
                continue
            score = card.points * 200.0
            if score > best_score:
                best_score = score
                best_a = A.BUY_GRID_BASE + tier * _NUM_GRID_SLOTS + s
                best_card = card
    for r in range(_MAX_RESERVED):
        cid = view.reserved[opp_seat][r]
        if cid < 0:
            continue
        card = _CARDS[cid]
        if not _affordable(card, bonuses, tokens):
            continue
        score = card.points * 200.0
        if score > best_score:
            best_score = score
            best_a = A.BUY_RESERVED_BASE + r
            best_card = card
    if best_a < 0 or best_card is None:
        return None
    return (best_a, best_card)


# ---------------------------------------------------------------------------
# V18 -- V15 + V17's lookahead at 3p only.
# ---------------------------------------------------------------------------


class HeuristicOpusV18(HeuristicOpusV15):
    """V15 with V17's 1-ply lookahead applied only at 3 players.

    V17 tournament (96 games / matchup, 2/3/4p) showed:
    * 2p: lookahead hurt (V15=2715 -> V17=2693, -22 ELO)
    * 3p: lookahead helped (V15=2633 -> V17=2652, +19 ELO)
    * 4p: lookahead hurt (V15=2604 -> V17=2579, -25 ELO)

    Per-pc fits suggest that at 2p the threat-aware buy is too conservative
    against a single very PV-greedy opponent (penalizing the buys our path
    plan wants), and at 4p the max-over-3-opponents threat is inflated
    because not all opponents threaten in parallel. At 3p the lookahead is
    a clean win. V18 routes 3p through V17 and keeps V15 elsewhere.
    """

    name: str = "heuristic_opus_v18"

    def __init__(self) -> None:
        self._v17 = HeuristicOpusV17()

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 18}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players == 3:
            return self._v17._choose_main(view, mask_row)
        return HeuristicOpusV15._choose_main(self, view, mask_row)


# ---------------------------------------------------------------------------
# V19 -- V18 with 2-ply minimax at 3p.
# ---------------------------------------------------------------------------


class HeuristicOpusV19(HeuristicOpusV18):
    """V18 plus a 2-ply minimax buy lookahead at 3 players.

    V17/V18's 1-ply lookahead penalizes buys by ``max-PV opponent could
    afford post-buy``. That's a coarse threat estimate -- it doesn't
    distinguish "opponent could afford a 4-PV card but they have nothing
    better than that" vs "opponent has a clear best card". V19 simulates
    each subsequent opponent's actual best buy response (highest-PV
    affordable, V8-style scoring) and scores the position by
    ``our_PV_gained - max_opp_PV_gained``. This catches situations where
    our buy A leaves card B exposed AND opponent can actually grab it
    (where the 1-ply threat scan would've correctly flagged B but with
    the same penalty as a benign exposure that no opponent could reach).

    Restricted to 3p, mirroring V18's gating where lookahead has
    consistently helped.
    """

    name: str = "heuristic_opus_v19"
    _OUR_GAIN_WEIGHT: float = 80.0
    _OPP_GAIN_WEIGHT: float = 60.0

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 19}

    def _best_affordable_buy_2ply(
        self,
        view: _GameView,
        mask_row: list[bool],
        noble_pressure: Sequence[float],
    ) -> int | None:
        candidates = _list_affordable_buys(view, mask_row)
        if not candidates:
            return None
        initial_pts = list(view.points)
        best_a, best_cid, best_score = -1, -1, -1e9
        for a, cid in candidates:
            card = _CARDS[cid]
            base = card.points * 200.0 + _card_target_score(
                card, view, noble_pressure
            )
            if card.points == 0 and card.level == 1:
                if noble_pressure[card.bonus] <= 0:
                    base -= 60.0
            future = _shallow_copy_view(view)
            _apply_buy_clear_slot(future, view.cp, a, card)
            for offset in range(1, view.num_players):
                opp_seat = (view.cp + offset) % view.num_players
                opp_best = _opponent_best_buy_action(future, opp_seat)
                if opp_best is None:
                    continue
                opp_a, opp_card = opp_best
                _apply_buy_clear_slot(future, opp_seat, opp_a, opp_card)
            our_gain = future.points[view.cp] - initial_pts[view.cp]
            opp_gain = max(
                future.points[s] - initial_pts[s]
                for s in range(view.num_players)
                if s != view.cp
            )
            score = base + self._OUR_GAIN_WEIGHT * our_gain - self._OPP_GAIN_WEIGHT * opp_gain
            if score > best_score:
                best_score = score
                best_a = a
                best_cid = cid
        if best_a < 0:
            return None
        chosen_card = _CARDS[best_cid]
        if chosen_card.points > 0:
            return best_a
        if noble_pressure[chosen_card.bonus] > 0:
            return best_a
        return None

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players != 3:
            return HeuristicOpusV15._choose_main(self, view, mask_row)
        return self._choose_main_2ply(view, mask_row, _path_target_v9)

    def _choose_main_2ply(
        self,
        view: _GameView,
        mask_row: list[bool],
        target_picker: Callable[[_GameView, Sequence[float]], _CardData | None],
    ) -> int:
        """V13-style dispatch with V19's 2-ply buy lookahead substituted in.

        Factored out so that V20 (and any future variant) can reuse the
        same control flow with a different target_picker (e.g. V13's
        path planner at 2p/3p, V10's single-card scorer at 4p).
        """
        if self._is_endgame(view):
            return self._endgame_main(view, mask_row)
        noble_pressure = _noble_color_pressure(view)
        denial = self._denial_reserve(view, mask_row)
        if denial is not None:
            return denial
        best_buy = self._best_affordable_buy_2ply(view, mask_row, noble_pressure)
        if best_buy is not None:
            return best_buy
        target = target_picker(view, noble_pressure)
        if target is not None:
            bridge = _bridge_buy(view, mask_row, target)
            if bridge is not None:
                return bridge
        nb = _noble_progress_bridge(view, mask_row)
        if nb is not None:
            return nb
        if target is not None:
            best_a = self._best_take_for_target(
                view, mask_row, target, noble_pressure
            )
            if best_a >= 0:
                return best_a
        reserve_a = self._best_reserve(view, mask_row, noble_pressure)
        if reserve_a is not None:
            return reserve_a
        for a in range(A.MAIN_ACTIONS_END):
            if mask_row[a]:
                return a
        return A.PASS_ACTION


# ---------------------------------------------------------------------------
# V20 -- V19's 2-ply minimax extended to 2p (and 3p).
# ---------------------------------------------------------------------------


class HeuristicOpusV20(HeuristicOpusV19):
    """V19 with 2-ply minimax buy lookahead extended to 2 players too.

    V17's 1-ply threat scan regressed at 2p (-22 ELO) because the
    "max-PV opponent could afford" estimate was too pessimistic when
    a single opponent could only realistically grab one specific card
    (the planner penalized unrelated buys for the same threat). V19's
    true 2-ply minimax actually applies the opponent's best response
    instead of taking the max -- so a buy that "exposes" a card the
    opponent will indeed grab gets penalized; one that exposes a card
    they wouldn't pick (because they have a better buy) doesn't.

    V20 hypothesis: with the more accurate 2-ply model, the regression
    at 2p disappears and the lookahead becomes a clean win across the
    two player counts where it sees real benefit (2p and 3p). V10
    single-card logic remains at 4p (max-over-3-opponents inflation
    isn't fixed by 2-ply alone).
    """

    name: str = "heuristic_opus_v20"

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 20}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players >= 4:
            return HeuristicOpusV10._choose_main(self, view, mask_row)
        return self._choose_main_2ply(view, mask_row, _path_target_v9)


# ---------------------------------------------------------------------------
# V12 -- player-count adaptive: V8 at 2p, V9 at 3p, V10 at 4p.
# ---------------------------------------------------------------------------


class HeuristicOpusV12(HeuristicOpusV8):
    """Adaptive opus: dispatches by player count to the strongest sub-bot.

    Tournament data with 48 games per matchup showed:
    * V8 wins 2-player (single-target PPT + bridge buys + take-for-target).
    * V9 wins 3-player (its 2-card race-to-15 path planner sees mid-game
      paths that single-card scoring misses).
    * V10 wins 4-player (selective denial + noble-bridge buys handle the
      multi-opponent chaos better than V8's static logic).

    V12 routes to whichever sub-strategy is empirically strongest at the
    current ``num_players``. The endgame check, denial check, and
    affordable-buy step are all shared via ``HeuristicOpusV8``; only the
    target-selection / bridge / take logic varies.
    """

    name: str = "heuristic_opus_v12"

    def __init__(self) -> None:
        self._v9 = HeuristicOpusV9()
        self._v10 = HeuristicOpusV10()

    def info(self) -> dict[str, Any]:
        return {"kind": "heuristic_opus", "version": 12}

    def _choose_main(self, view: _GameView, mask_row: list[bool]) -> int:
        if view.num_players == 2:
            return HeuristicOpusV8._choose_main(self, view, mask_row)
        if view.num_players == 3:
            return self._v9._choose_main(view, mask_row)
        return self._v10._choose_main(view, mask_row)


def _shallow_copy_view(view: _GameView) -> _GameView:
    return _GameView(
        num_players=view.num_players,
        cp=view.cp,
        phase=view.phase,
        points=list(view.points),
        bonuses=[list(b) for b in view.bonuses],
        tokens=[list(t) for t in view.tokens],
        reserved=[list(r) for r in view.reserved],
        grid=[list(g) for g in view.grid],
        nobles=list(view.nobles),
        gem_pool=list(view.gem_pool),
        deck_top=list(view.deck_top),
        last_trigger=view.last_trigger,
        turns_since_trigger=view.turns_since_trigger,
    )


def _apply_buy(view: _GameView, seat: int, card: _CardData) -> None:
    bonuses = view.bonuses[seat]
    tokens = view.tokens[seat]
    for c in range(_NUM_COLORS):
        need = max(0, card.cost[c] - bonuses[c])
        spent = min(need, tokens[c])
        tokens[c] -= spent
        view.gem_pool[c] += spent
        deficit = need - spent
        if deficit > 0:
            spent_gold = min(deficit, tokens[_GOLD])
            tokens[_GOLD] -= spent_gold
            view.gem_pool[_GOLD] += spent_gold
    bonuses[card.bonus] += 1
    view.points[seat] += card.points


def _grid_loc_from_buy_action(action_id: int) -> tuple[int, int] | None:
    if A.BUY_GRID_BASE <= action_id < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
        idx = action_id - A.BUY_GRID_BASE
        return idx // _NUM_GRID_SLOTS, idx % _NUM_GRID_SLOTS
    return None


def _reserved_slot_from_buy_action(action_id: int) -> int | None:
    if A.BUY_RESERVED_BASE <= action_id < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
        return action_id - A.BUY_RESERVED_BASE
    return None


def _apply_buy_clear_slot(
    view: _GameView, seat: int, action_id: int, card: _CardData
) -> None:
    """Forward-simulate a buy: pay tokens, add bonus/points, clear the slot.

    Conservatively skips deck refill (deck-top is hidden), so the grid slot
    is treated as empty post-buy. This is appropriate for one-ply opponent
    threat analysis: the opponent cannot buy a card we just removed.
    """
    _apply_buy(view, seat, card)
    grid_loc = _grid_loc_from_buy_action(action_id)
    if grid_loc is not None:
        tier, slot = grid_loc
        view.grid[tier][slot] = -1
        return
    res_slot = _reserved_slot_from_buy_action(action_id)
    if res_slot is not None:
        view.reserved[seat][res_slot] = -1


def _opponent_max_pv_threat(view: _GameView, our_seat: int) -> float:
    """Max PV any non-cp opponent could buy from this state in one ply.

    Scans each opponent's affordable point-bearing buys (grid + their own
    reserved). Returns the largest single-card PV gain available. Used by
    V17's lookahead buy scoring to prefer buys that absorb high exposures.
    """
    threat = 0.0
    for seat in range(view.num_players):
        if seat == our_seat:
            continue
        opp_bonuses = view.bonuses[seat]
        opp_tokens = view.tokens[seat]
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points <= 0:
                    continue
                if not _affordable(card, opp_bonuses, opp_tokens):
                    continue
                pv = float(card.points)
                if pv > threat:
                    threat = pv
        for r in range(_MAX_RESERVED):
            cid = view.reserved[seat][r]
            if cid < 0:
                continue
            card = _CARDS[cid]
            if card.points <= 0:
                continue
            if not _affordable(card, opp_bonuses, opp_tokens):
                continue
            pv = float(card.points)
            if pv > threat:
                threat = pv
    return threat


def _opponent_color_demand(view: _GameView, our_seat: int) -> tuple[int, ...]:
    """Per-color count of "tokens opponents still need" toward their best target.

    Approximation: for each opponent, find their highest PPT-equivalent
    affordable-or-near point-bearing card (cheapest weighted option), and
    sum the per-color shortfall of that card across all opponents. Used by
    V17 to bias take-3 combos toward draining the colors opponents need.
    """
    demand = [0] * _NUM_COLORS
    for seat in range(view.num_players):
        if seat == our_seat:
            continue
        opp_bonuses = view.bonuses[seat]
        opp_tokens = view.tokens[seat]
        # Pick the opponent's best-looking visible point-bearing card by
        # smallest turns_to_afford with tie on highest PV.
        best: tuple[float, _CardData] | None = None
        for tier in range(_NUM_TIERS):
            for s in range(_NUM_GRID_SLOTS):
                cid = view.grid[tier][s]
                if cid < 0:
                    continue
                card = _CARDS[cid]
                if card.points <= 0:
                    continue
                d = _turns_to_afford(card, opp_bonuses, opp_tokens, view.gem_pool)
                key = (d, -card.points)
                if best is None or key < (best[0], -best[1].points):
                    best = (d, card)
        for r in range(_MAX_RESERVED):
            cid = view.reserved[seat][r]
            if cid < 0:
                continue
            card = _CARDS[cid]
            if card.points <= 0:
                continue
            d = _turns_to_afford(card, opp_bonuses, opp_tokens, view.gem_pool)
            key = (d, -card.points)
            if best is None or key < (best[0], -best[1].points):
                best = (d, card)
        if best is None:
            continue
        target = best[1]
        miss = _missing_per_color(target, opp_bonuses, opp_tokens)
        for c in range(_NUM_COLORS):
            demand[c] += int(miss[c])
    return tuple(demand)


def _apply_reserve(view: _GameView, seat: int, card: _CardData) -> None:
    res = view.reserved[seat]
    for r in range(_MAX_RESERVED):
        if res[r] < 0:
            res[r] = card.card_id
            break
    if view.gem_pool[_GOLD] > 0:
        view.gem_pool[_GOLD] -= 1
        view.tokens[seat][_GOLD] += 1


# ---------------------------------------------------------------------------
# Public registry: name -> factory.
# ---------------------------------------------------------------------------

_CANDIDATE_FACTORIES: dict[str, Callable[[], Any]] = {
    HeuristicOpusV1.name: HeuristicOpusV1,
    HeuristicOpusV2.name: HeuristicOpusV2,
    HeuristicOpusV3.name: HeuristicOpusV3,
    HeuristicOpusV4.name: HeuristicOpusV4,
    HeuristicOpusV5.name: HeuristicOpusV5,
    HeuristicOpusV7.name: HeuristicOpusV7,
    HeuristicOpusV8.name: HeuristicOpusV8,
    HeuristicOpusV9.name: HeuristicOpusV9,
    HeuristicOpusV10.name: HeuristicOpusV10,
    HeuristicOpusV11.name: HeuristicOpusV11,
    HeuristicOpusV12.name: HeuristicOpusV12,
    HeuristicOpusV13.name: HeuristicOpusV13,
    HeuristicOpusV14.name: HeuristicOpusV14,
    HeuristicOpusV15.name: HeuristicOpusV15,
    HeuristicOpusV16.name: HeuristicOpusV16,
    HeuristicOpusV17.name: HeuristicOpusV17,
    HeuristicOpusV18.name: HeuristicOpusV18,
    HeuristicOpusV19.name: HeuristicOpusV19,
    HeuristicOpusV20.name: HeuristicOpusV20,
}


def list_candidate_names() -> list[str]:
    return list(_CANDIDATE_FACTORIES)


def make_candidate(name: str) -> Any:
    if name not in _CANDIDATE_FACTORIES:
        raise ValueError(
            f"unknown heuristic-opus candidate {name!r}; "
            f"valid: {sorted(_CANDIDATE_FACTORIES)}"
        )
    return _CANDIDATE_FACTORIES[name]()

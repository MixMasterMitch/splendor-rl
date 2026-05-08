"""Reference single-game Splendor engine in pure Python.

The source of truth for rule interpretation. Used to validate the batched
engine via parity tests. Not performance critical.

Phases:
  0: main action (take/reserve/buy/pass)
  1: discard down to 10 tokens (one token per action)
  2: pick one of several qualifying nobles (rare)

After a phase-0 action the engine internally transitions to phase 1 if the
acting player has more than 10 tokens, then transitions to phase 2 if the
acting player qualifies for 2+ nobles simultaneously. Zero or one qualifying
noble is auto-claimed with no agent decision.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import List, Optional, Tuple

from . import actions as A
from . import cards as C

WINNING_POINTS = 15
MAX_TOKENS_PER_PLAYER = 10


@dataclasses.dataclass
class PlayerState:
    tokens: List[int]  # length 6: [W,B,G,R,K,Gold]
    bonuses: List[int]  # length 5: [W,B,G,R,K]
    reserved: List[int]  # card_ids; -1 = empty slot. length MAX_RESERVED
    reserved_hidden: List[bool]  # True if reserved via blind
    points: int
    nobles: int  # count claimed

    @staticmethod
    def empty() -> "PlayerState":
        return PlayerState(
            tokens=[0, 0, 0, 0, 0, 0],
            bonuses=[0, 0, 0, 0, 0],
            reserved=[-1, -1, -1],
            reserved_hidden=[False, False, False],
            points=0,
            nobles=0,
        )

    def total_tokens(self) -> int:
        return sum(self.tokens)

    def reserved_count(self) -> int:
        return sum(1 for x in self.reserved if x >= 0)


@dataclasses.dataclass
class GameState:
    num_players: int
    gem_pool: List[int]  # length 6
    grid: List[List[int]]  # grid[tier][slot] = card_id or -1
    decks: List[List[int]]  # decks[tier] is a list of card_ids remaining; top is last
    nobles: List[int]  # noble_ids currently on table (not yet claimed); -1 = claimed slot
    players: List[PlayerState]
    current_player: int
    first_player: int  # who went first this game (round boundary)
    phase: int
    turn_count: int
    last_round_trigger_player: int  # -1 if not triggered; otherwise the player who first crossed 15
    round_actions_since_trigger: int  # turns elapsed since trigger (for NN feature / diagnostics)
    ended: bool

    def copy(self) -> "GameState":
        return copy.deepcopy(self)


def create_game(num_players: int, rng: Optional[random.Random] = None) -> GameState:
    """Deterministic setup driven by rng (random.Random)."""
    if rng is None:
        rng = random.Random(0)
    assert num_players in (2, 3, 4)

    supply = list(C.token_supply_for_players(num_players))
    num_nobles = C.num_nobles_for_players(num_players)

    # Build decks sorted by tier; shuffle each tier
    decks: List[List[int]] = [[], [], []]
    for card in C.CARDS:
        decks[card.level - 1].append(card.card_id)
    for d in decks:
        rng.shuffle(d)

    grid: List[List[int]] = [[-1, -1, -1, -1] for _ in range(3)]
    for t in range(3):
        for s in range(A.NUM_GRID_SLOTS):
            if decks[t]:
                grid[t][s] = decks[t].pop()

    noble_pool = [n.noble_id for n in C.NOBLES]
    rng.shuffle(noble_pool)
    nobles = noble_pool[:num_nobles]
    # Pad to MAX_NOBLE_SLOTS with -1 for fixed-shape state
    while len(nobles) < A.MAX_NOBLE_SLOTS:
        nobles.append(-1)

    players = [PlayerState.empty() for _ in range(num_players)]

    first_player = rng.randrange(num_players)
    return GameState(
        num_players=num_players,
        gem_pool=supply,
        grid=grid,
        decks=decks,
        nobles=nobles,
        players=players,
        current_player=first_player,
        first_player=first_player,
        phase=0,
        turn_count=0,
        last_round_trigger_player=-1,
        round_actions_since_trigger=0,
        ended=False,
    )


def grid_card_id(state: GameState, tier: int, slot: int) -> int:
    return state.grid[tier][slot]


def _card(card_id: int) -> C.Card:
    return C.CARDS[card_id]


def _noble(nid: int) -> C.Noble:
    return C.NOBLES[nid]


def _payment_for_card(player: PlayerState, card: C.Card) -> Optional[Tuple[List[int], int]]:
    """Given a card, compute the minimal token/gold payment.

    Returns (spent_colored[5], spent_gold) or None if not affordable.
    spent_colored[i] = min(cost_i - bonus_i, tokens_i) clamped to >= 0.
    """
    spent = [0] * 5
    gold_needed = 0
    for i in range(5):
        required = card.cost[i] - player.bonuses[i]
        if required <= 0:
            continue
        pay_color = min(required, player.tokens[i])
        spent[i] = pay_color
        deficit = required - pay_color
        if deficit > 0:
            gold_needed += deficit
    if gold_needed > player.tokens[C.GOLD_INDEX]:
        return None
    return spent, gold_needed


def legal_action_mask(state: GameState) -> List[bool]:
    mask = [False] * A.NUM_ACTIONS
    if state.ended:
        return mask

    p = state.players[state.current_player]
    if state.phase == 0:
        # Take 3 different
        non_empty = [i for i in range(5) if state.gem_pool[i] > 0]
        for i, combo in enumerate(A.TAKE3_COMBOS):
            if all(state.gem_pool[c] > 0 for c in combo):
                mask[A.take3_index(i)] = True
            elif len(non_empty) < 3 and set(combo).issubset(set(range(5))):
                # Degenerate case: fewer than 3 piles; take from as many as possible.
                # Represent by allowing any combo whose available subset matches the maximal
                # available subset. We simplify by enabling every combo that includes the
                # maximal available set; engine applies only available pickups.
                pass
        # When fewer than 3 piles are non-empty, allow any TAKE3 combo whose selected
        # colors are a superset of the available colors (so the combo is effectively
        # "take from all available piles"). This keeps the action space simple while
        # covering the edge case.
        if len(non_empty) < 3:
            avail = set(non_empty)
            for i, combo in enumerate(A.TAKE3_COMBOS):
                if avail.issubset(set(combo)):
                    mask[A.take3_index(i)] = True

        # Take 2 same
        for c in range(5):
            if state.gem_pool[c] >= 4:
                mask[A.take2_index(c)] = True

        # Reserve grid
        can_reserve = p.reserved_count() < A.MAX_RESERVED
        if can_reserve:
            for t in range(3):
                for s in range(A.NUM_GRID_SLOTS):
                    if state.grid[t][s] >= 0:
                        mask[A.reserve_grid_index(t, s)] = True
            for t in range(3):
                if state.decks[t]:
                    mask[A.reserve_blind_index(t)] = True

        # Buy grid
        for t in range(3):
            for s in range(A.NUM_GRID_SLOTS):
                cid = state.grid[t][s]
                if cid >= 0 and _payment_for_card(p, _card(cid)) is not None:
                    mask[A.buy_grid_index(t, s)] = True

        # Buy reserved
        for r in range(A.MAX_RESERVED):
            cid = p.reserved[r]
            if cid >= 0 and _payment_for_card(p, _card(cid)) is not None:
                mask[A.buy_reserved_index(r)] = True

        # Token-limit constraint: if taking tokens would exceed 10 and we have
        # no way to discard later, we still allow; discard phase handles it.
        # No further modifications.

        # Pass only if nothing else is legal (should be effectively never).
        if not any(mask):
            mask[A.PASS_ACTION] = True

    elif state.phase == 1:
        # Discard any held token kind with count > 0
        for k in range(6):
            if p.tokens[k] > 0:
                mask[A.discard_index(k)] = True

    elif state.phase == 2:
        # Pick one of the qualifying nobles
        for ns in range(A.MAX_NOBLE_SLOTS):
            nid = state.nobles[ns]
            if nid < 0:
                continue
            req = _noble(nid).requirement
            if all(p.bonuses[c] >= req[c] for c in range(5)):
                mask[A.pick_noble_index(ns)] = True

    return mask


def _refill_grid_slot(state: GameState, tier: int, slot: int) -> None:
    if state.decks[tier]:
        state.grid[tier][slot] = state.decks[tier].pop()
    else:
        state.grid[tier][slot] = -1


def _qualifying_nobles(state: GameState, player_idx: int) -> List[int]:
    p = state.players[player_idx]
    result = []
    for ns in range(A.MAX_NOBLE_SLOTS):
        nid = state.nobles[ns]
        if nid < 0:
            continue
        req = _noble(nid).requirement
        if all(p.bonuses[c] >= req[c] for c in range(5)):
            result.append(ns)
    return result


def _claim_noble(state: GameState, player_idx: int, nslot: int) -> None:
    nid = state.nobles[nslot]
    assert nid >= 0
    n = _noble(nid)
    p = state.players[player_idx]
    p.points += n.points
    p.nobles += 1
    state.nobles[nslot] = -1


def _advance_after_action(state: GameState) -> None:
    """After a main-phase action, handle discard, nobles, and turn rotation."""
    p = state.players[state.current_player]

    # Phase 1: discard if over token limit
    if p.total_tokens() > MAX_TOKENS_PER_PLAYER:
        state.phase = 1
        return

    # Phase 2: nobles
    qual = _qualifying_nobles(state, state.current_player)
    if len(qual) >= 2:
        state.phase = 2
        return
    if len(qual) == 1:
        _claim_noble(state, state.current_player, qual[0])

    _end_turn(state)


def _end_turn(state: GameState) -> None:
    """Advance to next player and check end-of-game."""
    # Check if this player has crossed 15 and no trigger yet
    cur = state.current_player
    p = state.players[cur]
    if state.last_round_trigger_player < 0 and p.points >= WINNING_POINTS:
        state.last_round_trigger_player = cur

    state.turn_count += 1
    state.phase = 0

    nxt = (cur + 1) % state.num_players
    state.current_player = nxt

    # Game ends when current_player wraps back to first_player, meaning the
    # round is complete and everyone has had equal turns. This correctly handles
    # the case where the last player in a round triggers — the game ends
    # immediately since the round is already complete.
    if state.last_round_trigger_player >= 0:
        state.round_actions_since_trigger += 1
        if state.current_player == state.first_player:
            state.ended = True


def apply_action(state: GameState, action: int, rng: Optional[random.Random] = None) -> None:
    """Mutates state by applying the given (legal) action."""
    if state.ended:
        raise RuntimeError("game has ended")
    mask = legal_action_mask(state)
    if not mask[action]:
        raise ValueError(f"illegal action {action} ({A.action_name(action)})")
    if rng is None:
        rng = random.Random()

    p = state.players[state.current_player]

    if state.phase == 0:
        if A.TAKE3_BASE <= action < A.TAKE3_BASE + A.TAKE3_COUNT:
            combo = A.TAKE3_COMBOS[action - A.TAKE3_BASE]
            for c in combo:
                if state.gem_pool[c] > 0:
                    state.gem_pool[c] -= 1
                    p.tokens[c] += 1
            _advance_after_action(state)
            return

        if A.TAKE2_BASE <= action < A.TAKE2_BASE + A.TAKE2_COUNT:
            c = action - A.TAKE2_BASE
            assert state.gem_pool[c] >= 4
            state.gem_pool[c] -= 2
            p.tokens[c] += 2
            _advance_after_action(state)
            return

        if A.RESERVE_GRID_BASE <= action < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
            x = action - A.RESERVE_GRID_BASE
            tier, slot = x // A.NUM_GRID_SLOTS, x % A.NUM_GRID_SLOTS
            cid = state.grid[tier][slot]
            assert cid >= 0
            empty_rs = [r for r in range(A.MAX_RESERVED) if p.reserved[r] < 0]
            assert empty_rs
            rs = empty_rs[0]
            p.reserved[rs] = cid
            p.reserved_hidden[rs] = False
            if state.gem_pool[C.GOLD_INDEX] > 0:
                state.gem_pool[C.GOLD_INDEX] -= 1
                p.tokens[C.GOLD_INDEX] += 1
            _refill_grid_slot(state, tier, slot)
            _advance_after_action(state)
            return

        if A.RESERVE_BLIND_BASE <= action < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT:
            tier = action - A.RESERVE_BLIND_BASE
            assert state.decks[tier]
            cid = state.decks[tier].pop()
            empty_rs = [r for r in range(A.MAX_RESERVED) if p.reserved[r] < 0]
            rs = empty_rs[0]
            p.reserved[rs] = cid
            p.reserved_hidden[rs] = True
            if state.gem_pool[C.GOLD_INDEX] > 0:
                state.gem_pool[C.GOLD_INDEX] -= 1
                p.tokens[C.GOLD_INDEX] += 1
            _advance_after_action(state)
            return

        if A.BUY_GRID_BASE <= action < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
            x = action - A.BUY_GRID_BASE
            tier, slot = x // A.NUM_GRID_SLOTS, x % A.NUM_GRID_SLOTS
            cid = state.grid[tier][slot]
            assert cid >= 0
            card = _card(cid)
            pay = _payment_for_card(p, card)
            assert pay is not None
            spent_colored, spent_gold = pay
            for c in range(5):
                p.tokens[c] -= spent_colored[c]
                state.gem_pool[c] += spent_colored[c]
            p.tokens[C.GOLD_INDEX] -= spent_gold
            state.gem_pool[C.GOLD_INDEX] += spent_gold
            p.bonuses[card.bonus] += 1
            p.points += card.points
            _refill_grid_slot(state, tier, slot)
            _advance_after_action(state)
            return

        if A.BUY_RESERVED_BASE <= action < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
            rs = action - A.BUY_RESERVED_BASE
            cid = p.reserved[rs]
            assert cid >= 0
            card = _card(cid)
            pay = _payment_for_card(p, card)
            assert pay is not None
            spent_colored, spent_gold = pay
            for c in range(5):
                p.tokens[c] -= spent_colored[c]
                state.gem_pool[c] += spent_colored[c]
            p.tokens[C.GOLD_INDEX] -= spent_gold
            state.gem_pool[C.GOLD_INDEX] += spent_gold
            p.bonuses[card.bonus] += 1
            p.points += card.points
            p.reserved[rs] = -1
            p.reserved_hidden[rs] = False
            _advance_after_action(state)
            return

        if action == A.PASS_ACTION:
            _advance_after_action(state)
            return

        raise ValueError(f"bad main action {action}")

    if state.phase == 1:
        assert A.DISCARD_BASE <= action < A.DISCARD_BASE + A.DISCARD_COUNT
        k = action - A.DISCARD_BASE
        assert p.tokens[k] > 0
        p.tokens[k] -= 1
        state.gem_pool[k] += 1
        if p.total_tokens() <= MAX_TOKENS_PER_PLAYER:
            # Done discarding; proceed to noble check
            qual = _qualifying_nobles(state, state.current_player)
            if len(qual) >= 2:
                state.phase = 2
                return
            if len(qual) == 1:
                _claim_noble(state, state.current_player, qual[0])
            _end_turn(state)
        return

    if state.phase == 2:
        assert A.PICK_NOBLE_BASE <= action < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT
        ns = action - A.PICK_NOBLE_BASE
        _claim_noble(state, state.current_player, ns)
        _end_turn(state)
        return


def final_scores(state: GameState) -> List[Tuple[int, int]]:
    """Returns list of (points, -cards_bought) per player for ranking. Highest is best."""
    result = []
    for p in state.players:
        cards_bought = sum(p.bonuses)
        result.append((p.points, -cards_bought))
    return result


def winners(state: GameState) -> List[int]:
    """Returns list of player indices that share the top rank (ties possible)."""
    assert state.ended
    scores = final_scores(state)
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best]

"""Renders BatchedEngine state as natural-language prompt for LLM consumption."""

from __future__ import annotations

from typing import Any, List

from agent.env import actions as A
from agent.env import cards as C
from agent.env import batched_engine as BE

# Short color labels for compact rendering
_COLOR_SHORT = ("W", "B", "G", "R", "K")
_COLOR_FULL = ("White", "Blue", "Green", "Red", "Black")
_TOKEN_LABELS = ("W", "B", "G", "R", "K", "Gold")

# Phase names
_PHASE_NAMES = {0: "Main", 1: "Discard (must return tokens to 10)", 2: "Noble Pick"}


def _format_cost(cost: tuple[int, ...]) -> str:
    """Format a card cost as compact string like '2W 1B 3R'."""
    parts = []
    for i, c in enumerate(cost):
        if c > 0:
            parts.append(f"{c}{_COLOR_SHORT[i]}")
    return " ".join(parts) if parts else "Free"


def _net_cost(card: C.Card, bonuses: list[int], tokens: list[int]) -> str:
    """Compute the net out-of-pocket token cost after applying bonuses.

    Returns a compact string like '3B 1R' or 'Free'.
    """
    parts = []
    for i, c in enumerate(card.cost):
        remaining = max(0, c - bonuses[i])
        if remaining > 0:
            parts.append(f"{remaining}{_COLOR_SHORT[i]}")
    return " ".join(parts) if parts else "Free"


def _is_buyable(card: C.Card, bonuses: list[int], tokens: list[int]) -> bool:
    """Check if a player can afford a card given bonuses, tokens, and gold."""
    gold = tokens[5] if len(tokens) > 5 else 0
    total_deficit = 0
    for i, c in enumerate(card.cost):
        remaining = max(0, c - bonuses[i])
        shortfall = max(0, remaining - tokens[i])
        total_deficit += shortfall
    return total_deficit <= gold


def _card_detail(card: C.Card) -> str:
    """Format a single card's details (without net cost)."""
    cost_str = _format_cost(card.cost)
    return f"[Card {card.card_id}] Cost: {cost_str} | Points: {card.points} | Bonus: {_COLOR_FULL[card.bonus]}"


def _card_detail_with_buyability(card: C.Card, bonuses: list[int], tokens: list[int]) -> str:
    """Format a card's details including net cost and buyability tag."""
    base_str = _format_cost(card.cost)
    net_str = _net_cost(card, bonuses, tokens)
    buyable = _is_buyable(card, bonuses, tokens)
    tag = "[BUYABLE]" if buyable else "[CANNOT AFFORD]"
    return (
        f"[Card {card.card_id}] Base: {base_str} | Net Cost: {net_str} "
        f"| Points: {card.points} | Bonus: {_COLOR_FULL[card.bonus]} {tag}"
    )


def _card_detail_with_net(card: C.Card, bonuses: list[int]) -> str:
    """Format a card's details including the net cost after bonuses."""
    base_str = _format_cost(card.cost)
    net_str = _net_cost(card, bonuses, [])
    return (
        f"[Card {card.card_id}] Base: {base_str} | Net Cost: {net_str} "
        f"| Points: {card.points} | Bonus: {_COLOR_FULL[card.bonus]}"
    )


class GameStateRenderer:
    """Renders engine state as natural-language prompt for LLM consumption."""

    def __init__(self) -> None:
        # Pre-index cards and nobles for fast lookup
        self._cards = {c.card_id: c for c in C.CARDS}
        self._nobles = {n.noble_id: n for n in C.NOBLES}

    def render(
        self,
        engine: BE.BatchedEngine,
        seat: int,
        batch_idx: int = 0,
    ) -> str:
        """Produce the full user prompt for the given player's perspective."""
        b = batch_idx
        current_player = int(engine.current_player[b])
        phase = int(engine.phase[b])
        num_players = engine.num_players

        # Cache current player's bonuses and tokens for net cost / buyability calculations
        self._current_bonuses = engine.bonuses[b, seat].tolist()
        self._current_tokens = engine.tokens[b, seat].tolist()

        lines: list[str] = []
        lines.append(f"=== GAME STATE (Your turn - Player {seat}) ===")
        lines.append(f"Phase: {_PHASE_NAMES.get(phase, 'Unknown')}")

        # --- End-Game Alert ---
        max_opponent_points = 0
        max_opponent_id = -1
        for p in range(num_players):
            if p == seat:
                continue
            pts = int(engine.points[b, p])
            if pts > max_opponent_points:
                max_opponent_points = pts
                max_opponent_id = p
        my_points = int(engine.points[b, seat])
        highest_points = max(my_points, max_opponent_points)
        if highest_points >= 13:
            if max_opponent_points >= my_points and max_opponent_id >= 0:
                lines.append(
                    f"[🚨 ALERT: FINAL ROUND IMMINENT - Player {max_opponent_id} "
                    f"has {max_opponent_points} points! BUY THE HIGHEST-POINT CARD YOU CAN!]"
                )
            else:
                lines.append(
                    f"[🚨 ALERT: FINAL ROUND IMMINENT - You have {my_points} points! "
                    f"Maximize your score NOW!]"
                )
        lines.append("")

        # --- Gem Pool ---
        lines.append("--- Gem Pool ---")
        pool = engine.gem_pool[b].tolist()
        pool_parts = [f"{_COLOR_FULL[i]}: {pool[i]}" for i in range(5)]
        pool_parts.append(f"Gold: {pool[5]}")
        lines.append(", ".join(pool_parts))
        lines.append("")

        # --- Grid (Available Cards) ---
        lines.append("--- Grid (Available Cards) ---")
        for tier in range(3):
            lines.append(f"Tier {tier + 1}:")
            for slot in range(4):
                card_id = int(engine.grid_card[b, tier, slot])
                if card_id >= 0 and card_id in self._cards:
                    card = self._cards[card_id]
                    lines.append(
                        f"  Slot {slot}: {_card_detail_with_buyability(card, self._current_bonuses, self._current_tokens)}"
                    )
                else:
                    lines.append(f"  Slot {slot}: [Empty]")

        # Deck remaining counts
        deck_counts = []
        for tier in range(3):
            deck_counts.append(f"Tier{tier + 1}={int(engine.deck_top[b, tier])}")
        lines.append(f"Deck remaining: {', '.join(deck_counts)}")
        lines.append("")

        # --- Nobles ---
        lines.append("--- Nobles ---")
        for slot in range(5):
            noble_id = int(engine.noble_ids[b, slot])
            if noble_id >= 0 and noble_id in self._nobles:
                noble = self._nobles[noble_id]
                req_parts = []
                for i, r in enumerate(noble.requirement):
                    if r > 0:
                        req_parts.append(f"{r}{_COLOR_SHORT[i]}")
                lines.append(f"  Noble {slot}: Requires {' '.join(req_parts)} | 3 points")
        lines.append("")

        # --- Your State ---
        lines.append(f"--- Your State (Player {seat}) ---")
        self._render_player_full(lines, engine, b, seat)
        lines.append("")

        # --- Opponent(s) ---
        for p in range(num_players):
            if p == seat:
                continue
            lines.append(f"--- Opponent (Player {p}) ---")
            self._render_opponent(lines, engine, b, p)
            self._render_opponent_threats(lines, engine, b, p)
            lines.append("")

        # --- Legal Actions ---
        lines.append(self.render_legal_actions(engine, batch_idx=b))

        return "\n".join(lines)

    def _render_player_full(
        self, lines: list[str], engine: BE.BatchedEngine, b: int, seat: int
    ) -> None:
        """Render full details for the current player (tokens, bonuses, points, reserved)."""
        tokens = engine.tokens[b, seat].tolist()
        bonuses = engine.bonuses[b, seat].tolist()
        points = int(engine.points[b, seat])

        total_tokens = sum(tokens)
        tok_str = " ".join(f"{_TOKEN_LABELS[i]}:{tokens[i]}" for i in range(6))
        lines.append(f"Tokens ({total_tokens}/10 Max): {tok_str}")
        if total_tokens >= 8:
            lines.append(
                f"  ⚠️ You have {total_tokens} tokens. Taking gems will likely force a discard!"
            )

        bon_str = " ".join(f"{_COLOR_SHORT[i]}:{bonuses[i]}" for i in range(5))
        lines.append(f"Bonuses: {bon_str}")

        lines.append(f"Points: {points}")

        # Reserved cards with full details including buyability
        reserved_cards = []
        for r in range(3):
            card_id = int(engine.reserved[b, seat, r])
            if card_id >= 0 and card_id in self._cards:
                card = self._cards[card_id]
                reserved_cards.append(
                    _card_detail_with_buyability(card, bonuses, tokens)
                )
        if reserved_cards:
            lines.append(f"Reserved: {'; '.join(reserved_cards)}")
        else:
            lines.append("Reserved: None")

    def _render_opponent(
        self, lines: list[str], engine: BE.BatchedEngine, b: int, player: int
    ) -> None:
        """Render opponent state (tokens, bonuses, points, reserved count only)."""
        tokens = engine.tokens[b, player].tolist()
        bonuses = engine.bonuses[b, player].tolist()
        points = int(engine.points[b, player])

        tok_str = " ".join(f"{_TOKEN_LABELS[i]}:{tokens[i]}" for i in range(6))
        lines.append(f"Tokens: {tok_str}")

        bon_str = " ".join(f"{_COLOR_SHORT[i]}:{bonuses[i]}" for i in range(5))
        lines.append(f"Bonuses: {bon_str}")

        lines.append(f"Points: {points}")

        # Only show count of reserved cards (face-down)
        reserved_count = sum(
            1 for r in range(3) if int(engine.reserved[b, player, r]) >= 0
        )
        if reserved_count > 0:
            cards_word = "card" if reserved_count == 1 else "cards"
            lines.append(f"Reserved: {reserved_count} face-down {cards_word}")
        else:
            lines.append("Reserved: None")

    def _render_opponent_threats(
        self, lines: list[str], engine: BE.BatchedEngine, b: int, player: int
    ) -> None:
        """Render synthesized threat intelligence for an opponent.

        Includes:
        - Alert when opponent is 1 card away from a noble
        - List of 1+ point cards on the grid the opponent can afford next turn
        """
        bonuses = engine.bonuses[b, player].tolist()
        tokens = engine.tokens[b, player].tolist()
        alerts: list[str] = []

        # Check nobles: is this opponent 1 bonus away from any noble?
        for slot in range(5):
            noble_id = int(engine.noble_ids[b, slot])
            if noble_id < 0 or noble_id not in self._nobles:
                continue
            noble = self._nobles[noble_id]
            deficit = 0
            for i, r in enumerate(noble.requirement):
                shortfall = r - bonuses[i]
                if shortfall > 0:
                    deficit += shortfall
            if deficit == 1:
                alerts.append(f"[ALERT: 1 card away from Noble {slot}]")

        # Find grid cards worth 1+ points that this opponent can afford
        buyable: list[str] = []
        gold = tokens[5]
        for tier in range(3):
            for slot_idx in range(4):
                card_id = int(engine.grid_card[b, tier, slot_idx])
                if card_id < 0 or card_id not in self._cards:
                    continue
                card = self._cards[card_id]
                if card.points < 1:
                    continue
                # Check affordability: net cost after bonuses, covered by tokens + gold
                total_deficit = 0
                for i, c in enumerate(card.cost):
                    remaining = max(0, c - bonuses[i])
                    shortfall = max(0, remaining - tokens[i])
                    total_deficit += shortfall
                if total_deficit <= gold:
                    net_str = _net_cost(card, bonuses, tokens)
                    buyable.append(
                        f"t{tier+1}s{slot_idx} [Card {card.card_id}] "
                        f"{card.points}pt {_COLOR_FULL[card.bonus]} (net {net_str})"
                    )

        if alerts or buyable:
            for a in alerts:
                lines.append(a)
            if buyable:
                lines.append(f"Can buy next turn (1+ pts): {'; '.join(buyable)}")

    def render_legal_actions(
        self,
        engine: BE.BatchedEngine,
        batch_idx: int = 0,
    ) -> str:
        """Render the legal actions as a numbered list with descriptions.

        Applies smart filtering:
        - take3 actions that yield 0 gems are hidden
        - take3 actions are deduplicated: if multiple combos yield the same
          actual gem set, only the one with the most gems is shown; dominated
          actions (subset of another's yield) are dropped.
        """
        b = batch_idx
        mask = engine.legal_action_mask()
        legal = mask[b].tolist()
        pool = engine.gem_pool[b].tolist()

        lines: list[str] = []
        lines.append("=== LEGAL ACTIONS ===")

        # --- Collect take3 actions and deduplicate by actual yield ---
        take3_candidates: list[tuple[int, frozenset[int]]] = []  # (action_idx, actual_colors)
        for action_idx in range(A.TAKE3_BASE, A.TAKE3_BASE + A.TAKE3_COUNT):
            if not legal[action_idx]:
                continue
            combo = A.TAKE3_COMBOS[action_idx - A.TAKE3_BASE]
            actual = frozenset(c for c in combo if pool[c] > 0)
            if not actual:
                continue  # yields 0 gems — hide entirely
            take3_candidates.append((action_idx, actual))

        # Remove dominated actions: if action A yields {G} and action B yields {G, R},
        # drop A because B is strictly better.
        take3_shown: list[int] = []
        # Sort by yield size descending so we check larger sets first
        take3_candidates.sort(key=lambda x: -len(x[1]))
        kept_yields: list[frozenset[int]] = []
        for action_idx, actual in take3_candidates:
            # Check if this yield is a subset of any already-kept yield
            dominated = any(actual <= kept for kept in kept_yields)
            if not dominated:
                take3_shown.append(action_idx)
                kept_yields.append(actual)

        # Render take3 actions
        for action_idx in take3_shown:
            combo = A.TAKE3_COMBOS[action_idx - A.TAKE3_BASE]
            actual = [c for c in combo if pool[c] > 0]
            short = [_COLOR_SHORT[c] for c in actual]
            colors = [_COLOR_FULL[c] for c in actual]
            if len(actual) == 3:
                lines.append(
                    f"{action_idx}: take3({','.join(short)}) - "
                    f"Take 1 {' + 1 '.join(colors)} token"
                )
            elif len(actual) == 2:
                lines.append(
                    f"{action_idx}: take_gems({','.join(short)}) - "
                    f"Take 1 {' + 1 '.join(colors)} token (only {len(actual)} available)"
                )
            else:
                lines.append(
                    f"{action_idx}: take_gems({short[0]}) - "
                    f"Take 1 {colors[0]} token (only color available)"
                )

        # --- Render all other legal actions normally ---
        for action_idx in range(A.NUM_ACTIONS):
            if not legal[action_idx]:
                continue
            # Skip take3 — already handled above
            if A.TAKE3_BASE <= action_idx < A.TAKE3_BASE + A.TAKE3_COUNT:
                continue
            desc = self._describe_action(action_idx, engine, b)
            if desc:
                lines.append(f"{action_idx}: {desc}")

        lines.append("")
        lines.append("Choose one action by responding with just the action number.")
        return "\n".join(lines)

    def _describe_action(self, action_idx: int, engine: BE.BatchedEngine, b: int) -> str:
        """Produce a human-readable description for a single action."""
        pool = engine.gem_pool[b].tolist()

        if A.TAKE3_BASE <= action_idx < A.TAKE3_BASE + A.TAKE3_COUNT:
            # Handled by render_legal_actions; shouldn't reach here normally
            combo = A.TAKE3_COMBOS[action_idx - A.TAKE3_BASE]
            short = [_COLOR_SHORT[c] for c in combo]
            colors = [_COLOR_FULL[c] for c in combo]
            return f"take3({','.join(short)}) - Take 1 {' + 1 '.join(colors)} token"

        if A.TAKE2_BASE <= action_idx < A.TAKE2_BASE + A.TAKE2_COUNT:
            c = action_idx - A.TAKE2_BASE
            return f"take2({_COLOR_SHORT[c]}) - Take 2 {_COLOR_FULL[c]} tokens"

        if A.RESERVE_GRID_BASE <= action_idx < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
            x = action_idx - A.RESERVE_GRID_BASE
            tier, slot = x // 4, x % 4
            card_id = int(engine.grid_card[b, tier, slot])
            card_info = ""
            if card_id >= 0 and card_id in self._cards:
                card = self._cards[card_id]
                card_info = f" {_card_detail(card)}"
            gold_note = "" if pool[5] > 0 else " (no gold available)"
            return f"reserve_grid(t{tier+1},s{slot}) - Reserve{card_info}{gold_note}"

        if A.RESERVE_BLIND_BASE <= action_idx < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT:
            tier = action_idx - A.RESERVE_BLIND_BASE
            gold_note = "" if pool[5] > 0 else " (no gold available)"
            return f"reserve_blind(t{tier+1}) - Reserve top card from Tier {tier+1} deck (face-down){gold_note}"

        if A.BUY_GRID_BASE <= action_idx < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
            x = action_idx - A.BUY_GRID_BASE
            tier, slot = x // 4, x % 4
            card_id = int(engine.grid_card[b, tier, slot])
            card_info = ""
            if card_id >= 0 and card_id in self._cards:
                card = self._cards[card_id]
                card_info = f" {_card_detail_with_net(card, self._current_bonuses)}"
            return f"buy_grid(t{tier+1},s{slot}) - Buy{card_info}"

        if A.BUY_RESERVED_BASE <= action_idx < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
            rslot = action_idx - A.BUY_RESERVED_BASE
            seat = int(engine.current_player[b])
            card_id = int(engine.reserved[b, seat, rslot])
            card_info = ""
            if card_id >= 0 and card_id in self._cards:
                card = self._cards[card_id]
                card_info = f" {_card_detail_with_net(card, self._current_bonuses)}"
            return f"buy_reserved(r{rslot}) - Buy reserved{card_info}"

        if action_idx == A.PASS_ACTION:
            return "pass - Pass (no legal actions available)"

        if A.DISCARD_BASE <= action_idx < A.DISCARD_BASE + A.DISCARD_COUNT:
            token_kind = action_idx - A.DISCARD_BASE
            label = _TOKEN_LABELS[token_kind]
            full = _COLOR_FULL[token_kind] if token_kind < 5 else "Gold"
            return f"discard({label}) - Return 1 {full} token"

        if A.PICK_NOBLE_BASE <= action_idx < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT:
            nslot = action_idx - A.PICK_NOBLE_BASE
            noble_id = int(engine.noble_ids[b, nslot])
            noble_info = ""
            if noble_id >= 0 and noble_id in self._nobles:
                noble = self._nobles[noble_id]
                req_parts = [f"{r}{_COLOR_SHORT[i]}" for i, r in enumerate(noble.requirement) if r > 0]
                noble_info = f" (Requires {' '.join(req_parts)})"
            return f"pick_noble(n{nslot}) - Claim noble{noble_info} for 3 points"

        return f"action({action_idx})"

    def render_history_summary(
        self,
        steps: List[dict[str, Any]],
        last_n: int = 5,
    ) -> str:
        """Render a summary of the last N moves."""
        if not steps:
            return ""

        recent = steps[-last_n:]
        lines: list[str] = []
        lines.append("--- Recent Moves ---")
        for step in recent:
            player = step.get("player", "?")
            action_name = step.get("action_name", A.action_name(step.get("action", 0)))
            detail = step.get("action_detail", {})
            kind = detail.get("kind", "")
            lines.append(f"  Player {player}: {action_name}")

        return "\n".join(lines)

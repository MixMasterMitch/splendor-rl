"""Validates RULES.md section 11 checks for card and noble data."""

from __future__ import annotations

from agent.env import cards as C


def test_cards_count_and_distribution():
    cards = C.load_default_cards()
    C.validate_cards(cards)


def test_nobles_count_and_distribution():
    nobles = C.load_default_nobles()
    C.validate_nobles(nobles)


def test_token_supply():
    C.validate_token_supply()
    assert C.token_supply_for_players(2) == (4, 4, 4, 4, 4, 5)
    assert C.token_supply_for_players(3) == (5, 5, 5, 5, 5, 5)
    assert C.token_supply_for_players(4) == (7, 7, 7, 7, 7, 5)


def test_num_nobles_for_players():
    assert C.num_nobles_for_players(2) == 3
    assert C.num_nobles_for_players(3) == 4
    assert C.num_nobles_for_players(4) == 5


def test_card_costs_nonnegative_and_bounded():
    for c in C.load_default_cards():
        assert all(0 <= v for v in c.cost)
        assert sum(c.cost) >= 3, f"card {c.card_id} total cost < 3: {c.cost}"


def test_noble_requirements_sum_to_8_or_9():
    for n in C.load_default_nobles():
        assert sum(n.requirement) in (8, 9), f"noble {n.name} bad req {n.requirement}"

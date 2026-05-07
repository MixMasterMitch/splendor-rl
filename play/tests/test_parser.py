"""Tests for play.llm.parser.ActionParser."""

from __future__ import annotations

import pytest

from play.llm.parser import ActionParser


@pytest.fixture()
def parser() -> ActionParser:
    return ActionParser()


@pytest.fixture()
def legal_actions() -> list[int]:
    return [0, 5, 12, 30, 42]


class TestStrategy1_StandaloneNumber:
    """Strategy 1: line matching just a number."""

    def test_single_number_on_line(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("5", legal_actions) == 5

    def test_number_with_whitespace(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("  30  ", legal_actions) == 30

    def test_number_among_other_lines(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("Some reasoning\n42\nDone", legal_actions) == 42

    def test_number_not_in_legal_actions(self, parser: ActionParser, legal_actions: list[int]):
        # 99 is not legal, so strategy 1 fails and falls through
        assert parser.parse("99", legal_actions) is None


class TestStrategy2_Patterns:
    """Strategy 2: patterns like 'Action N', 'action: N', 'I choose N'."""

    def test_action_n(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("I would pick Action 30 because it's strong", legal_actions) == 30

    def test_action_colon_n(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("My choice is action: 12", legal_actions) == 12

    def test_i_choose_n(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("I choose 42 as my move", legal_actions) == 42

    def test_case_insensitive(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("ACTION 5", legal_actions) == 5

    def test_pattern_not_in_legal(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("Action 99", legal_actions) is None


class TestStrategy3_FirstLegalInteger:
    """Strategy 3 was removed — bare integers in prose are no longer matched.

    The parser now only matches standalone numbers on their own line (strategy 1)
    or explicit patterns like 'Action N' (strategy 2). This prevents false positives
    when the LLM mentions numbers in its reasoning text.
    """

    def test_bare_integer_in_prose_not_matched(self, parser: ActionParser, legal_actions: list[int]):
        # Bare integers in prose should NOT be matched (could be reasoning text)
        assert parser.parse("Looking at the board, 30 seems optimal", legal_actions) is None

    def test_bare_integers_in_prose_not_matched(self, parser: ActionParser, legal_actions: list[int]):
        # Neither 99 nor 42 should match when embedded in prose
        assert parser.parse("Option 99 is bad, but 42 works", legal_actions) is None


class TestStrategy4_NoValidAction:
    """Strategy 4: return None if no valid action found."""

    def test_no_numbers(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("I am not sure what to do", legal_actions) is None

    def test_empty_response(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("", legal_actions) is None

    def test_only_non_legal_numbers(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("Maybe 99 or 100", legal_actions) is None


class TestEdgeCases:
    """Edge cases and priority ordering."""

    def test_empty_legal_actions(self, parser: ActionParser):
        assert parser.parse("5", []) is None

    def test_strategy1_priority_over_strategy3(self, parser: ActionParser, legal_actions: list[int]):
        # Strategy 1 (standalone line) should take priority
        assert parser.parse("I like 42 but\n5\nis better", legal_actions) == 5

    def test_strategy2_priority_over_strategy3(self, parser: ActionParser, legal_actions: list[int]):
        # "Action 30" matches strategy 2 before strategy 3 finds "5"
        assert parser.parse("I think Action 30 is best, also 5 is ok", legal_actions) == 30

    def test_rejects_action_not_in_legal_set(self, parser: ActionParser):
        assert parser.parse("Action 99", [0, 5, 12]) is None

    def test_multiline_with_reasoning(self, parser: ActionParser, legal_actions: list[int]):
        response = """Let me analyze the board state.
The gem pool has plenty of white and green tokens.
I think buying the card at tier 1 slot 0 is optimal.

Action 30"""
        assert parser.parse(response, legal_actions) == 30

    def test_zero_is_valid_action(self, parser: ActionParser, legal_actions: list[int]):
        assert parser.parse("0", legal_actions) == 0

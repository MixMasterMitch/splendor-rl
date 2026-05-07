"""Tests for play.llm.rate_limiter."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from play.llm.rate_limiter import LLMRateLimiter, RateLimitExceeded


@pytest.fixture()
def limiter() -> LLMRateLimiter:
    return LLMRateLimiter(max_games_per_hour=10)


class TestCheckAndRecord:
    """Tests for check_and_record method."""

    def test_allows_first_game(self, limiter: LLMRateLimiter):
        # Should not raise
        limiter.check_and_record("alice")

    def test_allows_up_to_max_games(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

    def test_raises_on_exceeding_limit(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_and_record("alice")

        assert exc_info.value.username == "alice"
        assert exc_info.value.retry_after_seconds > 0

    def test_different_users_independent(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

        # Bob should still be able to play
        limiter.check_and_record("bob")

    def test_expired_entries_pruned(self, limiter: LLMRateLimiter):
        # Simulate entries from over an hour ago
        past = time.time() - 3700  # 3700 seconds ago (> 1 hour)
        limiter._timestamps["alice"] = [past + i for i in range(10)]

        # Should succeed because all entries are expired
        limiter.check_and_record("alice")

    def test_mixed_expired_and_valid(self, limiter: LLMRateLimiter):
        now = time.time()
        # 5 expired entries + 5 valid entries = only 5 count
        expired = [now - 4000 + i for i in range(5)]
        valid = [now - 100 + i for i in range(5)]
        limiter._timestamps["alice"] = expired + valid

        # Should allow 5 more games (only 5 valid entries)
        for _ in range(5):
            limiter.check_and_record("alice")

        # 11th should fail
        with pytest.raises(RateLimitExceeded):
            limiter.check_and_record("alice")

    def test_error_message_includes_username(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

        with pytest.raises(RateLimitExceeded, match="alice"):
            limiter.check_and_record("alice")

    def test_error_message_includes_retry_info(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

        with pytest.raises(RateLimitExceeded, match="Try again in"):
            limiter.check_and_record("alice")


class TestRemaining:
    """Tests for remaining method."""

    def test_no_entries_returns_max(self, limiter: LLMRateLimiter):
        remaining, seconds = limiter.remaining("alice")
        assert remaining == 10
        assert seconds == 0.0

    def test_after_one_game(self, limiter: LLMRateLimiter):
        limiter.check_and_record("alice")
        remaining, seconds = limiter.remaining("alice")
        assert remaining == 9
        assert seconds > 0.0
        assert seconds <= 3600.0

    def test_at_limit(self, limiter: LLMRateLimiter):
        for _ in range(10):
            limiter.check_and_record("alice")

        remaining, seconds = limiter.remaining("alice")
        assert remaining == 0
        assert seconds > 0.0

    def test_expired_entries_not_counted(self, limiter: LLMRateLimiter):
        now = time.time()
        limiter._timestamps["alice"] = [now - 4000]  # expired

        remaining, seconds = limiter.remaining("alice")
        assert remaining == 10
        assert seconds == 0.0


class TestCustomLimit:
    """Tests with custom max_games_per_hour."""

    def test_custom_limit_of_3(self):
        limiter = LLMRateLimiter(max_games_per_hour=3)
        for _ in range(3):
            limiter.check_and_record("alice")

        with pytest.raises(RateLimitExceeded):
            limiter.check_and_record("alice")

    def test_remaining_with_custom_limit(self):
        limiter = LLMRateLimiter(max_games_per_hour=5)
        limiter.check_and_record("alice")
        remaining, _ = limiter.remaining("alice")
        assert remaining == 4

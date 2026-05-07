"""Per-user rate limiting for LLM game creation."""

from __future__ import annotations

import time


class RateLimitExceeded(Exception):
    """Raised when a user has exceeded the LLM game rate limit."""

    def __init__(self, username: str, retry_after_seconds: float) -> None:
        self.username = username
        self.retry_after_seconds = retry_after_seconds
        minutes = retry_after_seconds / 60
        super().__init__(
            f"Rate limit exceeded for user '{username}'. "
            f"You have used all 10 LLM games in the last hour. "
            f"Try again in {minutes:.1f} minutes."
        )


class LLMRateLimiter:
    """Per-user rate limiting for LLM game creation.

    Tracks timestamps of game creation per user in an in-memory dict.
    Enforces a maximum number of LLM games per rolling hour window.
    """

    WINDOW_SECONDS: float = 3600.0  # 1 hour

    def __init__(self, max_games_per_hour: int = 10) -> None:
        self._max_games = max_games_per_hour
        self._timestamps: dict[str, list[float]] = {}

    def _prune(self, username: str) -> list[float]:
        """Remove expired entries (older than 1 hour) and return remaining."""
        now = time.time()
        cutoff = now - self.WINDOW_SECONDS
        entries = self._timestamps.get(username, [])
        valid = [ts for ts in entries if ts > cutoff]
        self._timestamps[username] = valid
        return valid

    def check_and_record(self, username: str) -> None:
        """Record a new LLM game for the user.

        Raises RateLimitExceeded if the user has >= max_games_per_hour
        LLM games in the last hour.
        """
        entries = self._prune(username)

        if len(entries) >= self._max_games:
            # Oldest entry determines when a slot frees up
            oldest = min(entries)
            retry_after = oldest + self.WINDOW_SECONDS - time.time()
            raise RateLimitExceeded(username, max(0.0, retry_after))

        self._timestamps[username].append(time.time())

    def remaining(self, username: str) -> tuple[int, float]:
        """Return (remaining_games, seconds_until_oldest_entry_expires).

        If the user has no entries, returns (max_games, 0.0).
        If the user has entries but hasn't hit the limit, returns
        (remaining_slots, seconds_until_oldest_expires).
        """
        entries = self._prune(username)

        remaining_games = self._max_games - len(entries)

        if not entries:
            return (remaining_games, 0.0)

        oldest = min(entries)
        seconds_until_reset = oldest + self.WINDOW_SECONDS - time.time()
        return (remaining_games, max(0.0, seconds_until_reset))

"""Derive ``UserIdentity`` from HTTP headers (local dev proxy or any reverse proxy)."""

from __future__ import annotations

import dataclasses
import re
from typing import Mapping

_USERNAME_MAX_LEN = 32
_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


@dataclasses.dataclass(frozen=True)
class UserIdentity:
    """Screen name chosen by the player; carries no cryptographic trust."""

    username: str


def _norm_header(headers: Mapping[str, str], key: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == key.lower():
            return v if v else None
    return None


def normalize_username(raw: str) -> str:
    """Normalize and validate a username from user input."""
    s = raw.strip()
    if not s:
        raise ValueError("username is empty")
    if len(s) > _USERNAME_MAX_LEN:
        raise ValueError(f"username must be at most {_USERNAME_MAX_LEN} characters")
    if _USERNAME_PATTERN.fullmatch(s) is None:
        raise ValueError(
            "username may only contain ASCII letters, digits, underscore (_), hyphen (-)",
        )
    return s


def identity_from_headers(headers: Mapping[str, str]) -> UserIdentity:
    """Build identity from request headers.

    Local dev:

        X-Splendor-Username: required; non-empty after trim; see ``normalize_username``.

    A gateway can inject the same header on behalf of a signed-in identity later.
    """
    raw = _norm_header(headers, "X-Splendor-Username")
    if raw is None:
        raise ValueError("missing X-Splendor-Username")
    username = normalize_username(raw)
    return UserIdentity(username=username)


def human_entity_id(identity: UserIdentity) -> str:
    """Stable rating entity id for this user."""
    return f"human:{identity.username}"

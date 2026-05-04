"""Append-only journal for human-readable decisions during training.

The implementing agent writes a short entry at every decision point describing
observation, action, and rationale so a fresh session can orient itself via
`scripts/status.py`.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from typing import Optional


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def append_entry(journal_path: pathlib.Path, title: str, body: str = "") -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(journal_path, "a") as f:
        f.write(f"\n## {_now()} - {title}\n\n")
        if body:
            f.write(body.rstrip() + "\n")


def read_last_entry(journal_path: pathlib.Path) -> Optional[str]:
    if not journal_path.exists():
        return None
    with open(journal_path) as f:
        content = f.read()
    blocks = content.split("\n## ")
    if not blocks:
        return None
    last = blocks[-1]
    if not last.strip():
        return None
    return "## " + last if not last.startswith("## ") else last

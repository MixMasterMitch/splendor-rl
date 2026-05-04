"""Durable JSON storage for interactive play (local dev; DynamoDB-shaped API)."""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from typing import Any, Iterable, Literal


GameStatus = Literal["in_flight", "completed", "aborted", "human_turn", "ai_thinking"]


class JsonPlayStore:
    """File-backed store under ``root/games/*.json`` and ``root/users/*.json``."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = pathlib.Path(root)
        self._games_dir = self._root / "games"
        self._users_dir = self._root / "users"
        self._games_dir.mkdir(parents=True, exist_ok=True)
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def _atomic_write_json(self, path: pathlib.Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def game_path(self, game_id: str) -> pathlib.Path:
        return self._games_dir / f"{game_id}.json"

    def user_rating_path(self, username: str) -> pathlib.Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in username)
        return self._users_dir / f"{safe}.json"

    def load_game(self, game_id: str) -> dict[str, Any] | None:
        p = self.game_path(game_id)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def save_game(self, record: dict[str, Any]) -> None:
        game_id = str(record["game_id"])
        record = {**record, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self._lock:
            self._atomic_write_json(self.game_path(game_id), record)

    def list_games_for_user(
        self,
        username: str,
        status: Iterable[GameStatus] | None = None,
    ) -> list[dict[str, Any]]:
        want = set(status) if status is not None else None
        out: list[dict[str, Any]] = []
        if not self._games_dir.exists():
            return out
        for p in sorted(self._games_dir.glob("*.json")):
            try:
                with open(p) as f:
                    row = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("user_sub") != username:
                continue
            st = row.get("status")
            if want is not None and st not in want:
                continue
            out.append(row)
        out.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
        return out

    def load_user_rating_blob(self, username: str) -> dict[str, Any] | None:
        p = self.user_rating_path(username)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def save_user_rating_blob(self, username: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write_json(self.user_rating_path(username), data)

    def list_all_user_rating_blobs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self._users_dir.exists():
            return out
        for p in sorted(self._users_dir.glob("*.json")):
            try:
                with open(p) as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        return out

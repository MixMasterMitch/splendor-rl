"""Persistent cache for tournament match results.

A matchup is fully deterministic given:
* both policies' identities (name + ``info()`` dict),
* the engine seed,
* the matchup configuration (player count, game count, max-turn cap), and
* the source code of every Python module that influences play (the
  candidate definitions and the reference bots).

This module hashes those inputs into a stable cache key and stores the
resulting :class:`tournament.MatchResult` JSON on disk. A subsequent run
that touches the same matchup -- e.g. iterating on V7 while leaving V1
unchanged -- only re-plays the matchups whose code or parameters
actually changed; the rest are reloaded from cache in milliseconds.

The cache key intentionally includes a SHA-256 of the source text of
``heuristic_opus.py`` and ``bots.py``. If you edit either file, all
cached results referencing the affected policies are invalidated
automatically. This is a correctness guarantee: stale results never
sneak into a new tournament.

Cache layout (under ``cache_dir``)::

    <cache_dir>/v1/<sha-prefix>/<full-key>.json

The split keeps directory entries shallow on filesystems with large
``readdir`` cost, and the ``v1`` namespace lets future schema changes
land alongside legacy files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Mapping

_SCHEMA_VERSION: str = "v1"


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_source_files() -> list[pathlib.Path]:
    """All Python files whose contents affect candidate play.

    Currently: ``agent/eval/heuristic_opus.py`` and ``agent/eval/bots.py``.
    The function uses ``__file__`` of this module to anchor the paths,
    which works whether the code runs from a Bazel runfiles tree or a
    raw checkout.
    """
    here = pathlib.Path(__file__).resolve().parent
    out: list[pathlib.Path] = []
    for name in ("heuristic_opus.py", "bots.py"):
        p = here / name
        if p.exists():
            out.append(p)
    return out


def code_fingerprint() -> str:
    """SHA-256 over the candidate source files, plus a schema tag."""
    h = hashlib.sha256()
    h.update(_SCHEMA_VERSION.encode("ascii"))
    for path in _candidate_source_files():
        h.update(b"\n")
        h.update(str(path.name).encode("utf-8"))
        h.update(b":")
        h.update(_file_sha256(path).encode("ascii"))
    return h.hexdigest()


def _stable_json(obj: Any) -> str:
    """JSON dump with sorted keys for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def make_cache_key(
    *,
    name_a: str,
    name_b: str,
    info_a: Mapping[str, Any],
    info_b: Mapping[str, Any],
    num_players: int,
    num_games: int,
    max_turns: int,
    seed: int,
    timeout_winner_uses_points: bool,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable hex key for a fully-specified matchup."""
    payload: dict[str, Any] = {
        "schema": _SCHEMA_VERSION,
        "code": code_fingerprint(),
        "name_a": name_a,
        "name_b": name_b,
        "info_a": dict(info_a),
        "info_b": dict(info_b),
        "num_players": num_players,
        "num_games": num_games,
        "max_turns": max_turns,
        "seed": seed,
        "twup": bool(timeout_winner_uses_points),
    }
    if extra:
        payload["extra"] = dict(extra)
    blob = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _key_path(cache_dir: pathlib.Path, key: str) -> pathlib.Path:
    return cache_dir / _SCHEMA_VERSION / key[:2] / f"{key}.json"


@dataclasses.dataclass
class CacheLookupResult:
    hit: bool
    payload: dict[str, Any] | None


def load(cache_dir: pathlib.Path, key: str) -> CacheLookupResult:
    path = _key_path(cache_dir, key)
    if not path.exists():
        return CacheLookupResult(hit=False, payload=None)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return CacheLookupResult(hit=False, payload=None)
    if not isinstance(data, dict):
        return CacheLookupResult(hit=False, payload=None)
    if data.get("schema") != _SCHEMA_VERSION:
        return CacheLookupResult(hit=False, payload=None)
    return CacheLookupResult(hit=True, payload=data)


def save(cache_dir: pathlib.Path, key: str, payload: dict[str, Any]) -> None:
    path = _key_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_with_meta = dict(payload)
    payload_with_meta.setdefault("schema", _SCHEMA_VERSION)
    payload_with_meta.setdefault("saved_at", time.time())
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload_with_meta, f, indent=2)
    os.replace(tmp, path)

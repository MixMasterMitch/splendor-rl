"""Simple league of past checkpoints plus aggregate match results.

A "league" is a directory with numbered checkpoint files and a manifest JSON
describing checkpoint metadata plus an aggregated head-to-head result table.
Checkpoint ratings are fit in batch from that result table, anchored so
`random=1000` and `heuristic=2500`. This makes the resulting leaderboard
agnostic to the order in which the matches were played.

The exploiter is a separate network trained to specifically target the latest
main agent (its reward signal is the agent's loss, not general self-play).
For V1 we expose the hook to alternate training between "main" and
"exploiter" slots; the exploiter plays only against the main agent.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
from typing import List, Optional

import torch

from ..net import model as M
from . import checkpointing as CK
from . import ranking as R


class League:
    def __init__(
        self,
        root: pathlib.Path,
        max_entries: int | None = None,
        keep_recent: int = 8,
        anchors: Optional[dict[str, float]] = None,
    ):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "league.json"
        self._net_cache: dict[tuple[str, str], M.SplendorNet] = {}
        self.max_entries = max_entries
        self.keep_recent = keep_recent
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
        self.manifest.setdefault("entries", [])
        self.manifest.setdefault("results", [])
        self.manifest.setdefault("anchors", dict(anchors or R.DEFAULT_ANCHORS))
        self._migrate_absolute_paths()
        self._migrate_legacy_manifest_if_needed()

    def _resolve_path(self, stored: str) -> pathlib.Path:
        """Resolve a stored path (relative to league root) to an absolute path."""
        p = pathlib.Path(stored)
        if p.is_absolute():
            return p
        return self.root / p

    def _to_relative_path(self, p: pathlib.Path | str) -> str:
        """Convert an absolute path to a relative path from the league root.

        If the path lives under ``self.root``, store it as a relative path so
        the workspace is portable across machines.  Otherwise fall back to the
        absolute string (shouldn't happen in practice).
        """
        try:
            return str(pathlib.Path(p).relative_to(self.root))
        except ValueError:
            return str(p)

    def _migrate_absolute_paths(self) -> None:
        """One-time migration: convert any legacy absolute paths to relative."""
        changed = False
        for entry in self.manifest.get("entries", []):
            raw = entry.get("path", "")
            p = pathlib.Path(raw)
            if p.is_absolute():
                # Try to find the file under self.root by its filename.
                candidate = self.root / p.name
                if candidate.exists():
                    entry["path"] = p.name
                    changed = True
                else:
                    # File doesn't exist locally either way; store relative
                    # form so future loads give a clear error.
                    try:
                        entry["path"] = str(p.relative_to(self.root))
                    except ValueError:
                        entry["path"] = p.name
                    changed = True
        if changed:
            self._save_manifest()

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def _entry_strength_key(self, entry: dict) -> tuple[float, float, float, float, float]:
        return (
            self._entry_rating(entry),
            float(entry.get("score_hint", 0.0)),
            float(entry.get("winrate_vs_heuristic", 0.0)) + 0.5 * float(
                entry.get("ties_vs_heuristic", 0.0)
            ),
            float(entry.get("finished_vs_heuristic", 0.0)),
            -float(
                entry.get(
                    "avg_finished_step_vs_heuristic",
                    entry.get("avg_turns_vs_heuristic", float("inf")),
                )
            ),
        )

    def _entry_entity_id(self, idx: int) -> str:
        return f"ckpt:{idx}"

    def _entry_rating(self, entry: dict) -> float:
        return float(
            entry.get(
                "rating",
                self.manifest["anchors"].get("heuristic", R.HEURISTIC_ANCHOR_RATING),
            )
        )

    def _active_entity_ids(self) -> set[str]:
        out = set(self.manifest["anchors"])
        for entry in self.manifest["entries"]:
            out.add(self._entry_entity_id(int(entry["idx"])))
        return out

    def _record_entry_baselines_from_metadata(self, entry: dict) -> bool:
        entity = self._entry_entity_id(int(entry["idx"]))
        wrote_any = False
        use_rank = "rank_winrate_vs_random" in entry or "rank_winrate_vs_heuristic" in entry
        prefix = "rank_" if use_rank else ""
        total_games = 512 if use_rank else 256
        for opponent in ("random", "heuristic"):
            winrate_key = f"{prefix}winrate_vs_{opponent}"
            if winrate_key not in entry:
                continue
            tie_key = f"{prefix}ties_vs_{opponent}"
            wins = int(round(total_games * float(entry.get(winrate_key, 0.0))))
            ties = int(round(total_games * float(entry.get(tie_key, 0.0))))
            losses = max(int(total_games) - wins - ties, 0)
            self.record_result(entity, opponent, float(wins), float(losses), float(ties))
            wrote_any = True
        return wrote_any

    def _migrate_legacy_manifest_if_needed(self) -> None:
        if self.manifest.get("results"):
            return
        wrote_any = False
        for entry in self.manifest["entries"]:
            wrote_any = self._record_entry_baselines_from_metadata(entry) or wrote_any
        if wrote_any:
            self.recompute_ratings()

    def _drop_entry(self, entry: dict) -> None:
        path = self._resolve_path(entry["path"])
        if path.exists():
            path.unlink()
        for key in list(self._net_cache):
            if key[0] == str(path):
                del self._net_cache[key]

    def _prune_results(self) -> None:
        keep = self._active_entity_ids()
        self.manifest["results"] = [
            row
            for row in self.manifest["results"]
            if row["a"] in keep and row["b"] in keep
        ]

    def _prune_entries(self) -> None:
        if self.max_entries is None:
            return
        entries = list(self.manifest["entries"])
        if len(entries) <= self.max_entries:
            return
        keep_recent = min(max(self.keep_recent, 0), self.max_entries)
        recent = entries[-keep_recent:] if keep_recent > 0 else []
        older = entries[:-keep_recent] if keep_recent > 0 else entries
        keep_best = max(self.max_entries - len(recent), 0)
        best = sorted(older, key=self._entry_strength_key, reverse=True)[:keep_best]
        keep_paths = {entry["path"] for entry in recent}
        keep_paths.update(entry["path"] for entry in best)
        new_entries: list[dict] = []
        for entry in entries:
            if entry["path"] in keep_paths:
                new_entries.append(entry)
            else:
                self._drop_entry(entry)
        self.manifest["entries"] = new_entries
        self._prune_results()

    def add_checkpoint(
        self,
        net: M.SplendorNet,
        tag: str,
        metadata: Optional[dict] = None,
    ) -> pathlib.Path:
        next_idx = (
            max((int(entry["idx"]) for entry in self.manifest["entries"]), default=-1) + 1
        )
        idx = next_idx
        path = self.root / f"ckpt_{idx:05d}_{tag}.pt"
        torch.save(
            {
                "net": net.state_dict(),
                "config": CK.net_config_dict(net),
            },
            path,
        )
        self.manifest["entries"].append(
            {
                "idx": idx,
                "tag": tag,
                "path": self._to_relative_path(path),
                "rating": float(self.manifest["anchors"]["heuristic"]),
                "games": 0,
                "hidden": int(net.hidden),
                "arch": str(net.arch),
            }
        )
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    self.manifest["entries"][-1][key] = value
        self._prune_entries()
        self._save_manifest()
        return path

    def list_entries(self) -> List[dict]:
        return list(self.manifest["entries"])

    def latest_entry(self) -> Optional[dict]:
        if not self.manifest["entries"]:
            return None
        return self.manifest["entries"][-1]

    def entry_by_idx(self, idx: int) -> Optional[dict]:
        for entry in self.manifest["entries"]:
            if int(entry["idx"]) == idx:
                return entry
        return None

    def load_cached_net(
        self,
        path: str | pathlib.Path,
        device: torch.device | str,
    ) -> M.SplendorNet:
        resolved = self._resolve_path(str(path))
        path_str = str(resolved)
        device_t = torch.device(device)
        key = (path_str, str(device_t))
        cached = self._net_cache.get(key)
        if cached is not None:
            return cached
        net, _ = CK.load_net_from_checkpoint(pathlib.Path(path_str), map_location=device_t)
        net = net.to(device_t)
        net.eval()
        self._net_cache[key] = net
        return net

    def sample_opponent(self, rng: Optional[random.Random] = None) -> Optional[dict]:
        if not self.manifest["entries"]:
            return None
        if rng is None:
            rng = random.Random()
        # Weight by recency (newer = heavier) times a softened rating factor.
        # Ratings are anchored at random=1000, heuristic=2500, so we center the
        # sampling weights around the heuristic anchor rather than absolute 0.
        entries = self.manifest["entries"]
        weights = []
        for i, e in enumerate(entries):
            recency = 1.0 + i
            rel = (self._entry_rating(e) - R.HEURISTIC_ANCHOR_RATING) / 800.0
            rel = max(min(rel, 4.0), -4.0)
            weights.append(recency * (10 ** rel))
        tot = sum(weights)
        r = rng.random() * tot
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                return entries[i]
        return entries[-1]

    def rating_candidates(self, exclude_idx: int | None = None, limit: int = 4) -> List[dict]:
        if limit <= 0:
            return []
        entries = [
            entry
            for entry in self.manifest["entries"]
            if exclude_idx is None or int(entry["idx"]) != exclude_idx
        ]
        if not entries:
            return []
        recent_n = max(1, limit // 2)
        best_n = max(limit - recent_n, 0)
        recent = list(reversed(entries[-recent_n:]))
        best = sorted(entries, key=self._entry_strength_key, reverse=True)[:best_n]
        out: list[dict] = []
        seen: set[int] = set()
        for group in (recent, best):
            for entry in group:
                idx = int(entry["idx"])
                if idx in seen:
                    continue
                seen.add(idx)
                out.append(entry)
                if len(out) >= limit:
                    return out
        return out

    def record_result(
        self,
        entity_a: str,
        entity_b: str,
        wins_a: float,
        wins_b: float,
        ties: float,
    ) -> None:
        R.add_match_result(
            self.manifest["results"],
            entity_a,
            entity_b,
            wins_a,
            wins_b,
            ties,
        )

    def record_checkpoint_baselines(
        self,
        idx: int,
        row: dict,
        rank_games: int,
        eval_games: int,
    ) -> None:
        entity = self._entry_entity_id(idx)
        use_rank = "rank_winrate_vs_random" in row or "rank_winrate_vs_heuristic" in row
        prefix = "rank_" if use_rank else ""
        total_games = rank_games if use_rank else eval_games
        for opponent in ("random", "heuristic", "heuristic_opus"):
            winrate_key = f"{prefix}winrate_vs_{opponent}"
            if winrate_key not in row:
                continue
            tie_key = f"{prefix}ties_vs_{opponent}"
            wins = int(round(total_games * float(row.get(winrate_key, 0.0))))
            ties = int(round(total_games * float(row.get(tie_key, 0.0))))
            losses = max(total_games - wins - ties, 0)
            self.record_result(entity, opponent, float(wins), float(losses), float(ties))

    def recompute_ratings(self, extra_anchors: dict[str, float] | None = None) -> dict[str, float]:
        initial = {
            self._entry_entity_id(int(entry["idx"])): self._entry_rating(entry)
            for entry in self.manifest["entries"]
        }
        anchors = dict(self.manifest["anchors"])
        if extra_anchors:
            anchors.update(extra_anchors)
        ratings = R.fit_anchored_ratings(
            self.manifest["results"],
            anchors=anchors,
            initial=initial,
        )
        games_by_entity: dict[str, int] = {key: 0 for key in ratings}
        for row in self.manifest["results"]:
            total = int(round(float(row.get("games", 0.0))))
            games_by_entity[row["a"]] = games_by_entity.get(row["a"], 0) + total
            games_by_entity[row["b"]] = games_by_entity.get(row["b"], 0) + total
        for entry in self.manifest["entries"]:
            entity = self._entry_entity_id(int(entry["idx"]))
            if entity in ratings:
                entry["rating"] = float(ratings[entity])
                entry.pop("elo", None)
                entry["games"] = int(games_by_entity.get(entity, 0))
        # Persist fitted ratings for floating entities (non-anchor, non-checkpoint
        # participants like heuristic_opus that appear in results).
        entry_entities = {self._entry_entity_id(int(e["idx"])) for e in self.manifest["entries"]}
        anchor_entities = set(self.manifest["anchors"])
        floating: dict[str, dict] = {}
        for entity, rating in ratings.items():
            if entity not in entry_entities and entity not in anchor_entities:
                floating[entity] = {
                    "rating": float(rating),
                    "games": int(games_by_entity.get(entity, 0)),
                }
        if floating:
            self.manifest["floating_entities"] = floating
        self.manifest["rating_system"] = "anchored_bt"
        self._save_manifest()
        return ratings

    def update_rating(self, idx: int, new_rating: float, games_played: int) -> None:
        e = self.entry_by_idx(idx)
        if e is None:
            return
        e["rating"] = new_rating
        e.pop("elo", None)
        e["games"] = games_played
        self._save_manifest()

    def update_entry_fields(self, idx: int, fields: dict) -> None:
        entry = self.entry_by_idx(idx)
        if entry is None:
            return
        for key, value in fields.items():
            entry[key] = value
        self._save_manifest()


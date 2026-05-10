"""Simple league of past checkpoints plus aggregate match results.

A "league" is a directory with numbered checkpoint files and a manifest JSON
describing checkpoint metadata plus an aggregated head-to-head result table.
Checkpoint ratings are fit per player count from that result table, anchored so
`random=1000`. Per-PC ratings are calibrated by (n-1) and averaged to produce
a combined rating. This makes the resulting leaderboard agnostic to the order
in which the matches were played.

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

    def _entry_available(self, entry: dict) -> bool:
        """Return True if the entry's checkpoint file exists on disk."""
        path_str = entry.get("path", "")
        if not path_str:
            return False
        return self._resolve_path(path_str).exists()

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
                R.DEFAULT_INITIAL_RATING,
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
        entry["active"] = False
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
        """Prune checkpoint *files* from disk to save space, but keep all
        entries and results in the manifest for comprehensive rating history.

        Only the most recent ``keep_recent`` plus the best-rated older entries
        (up to ``max_entries`` total) retain their files on disk. Pruned entries
        remain in the JSON so their pairwise results continue to inform ratings.

        Preservation criterion: an entry is kept if it is in the top-K by
        *any* per-PC rating (rating_2p, rating_3p, rating_4p) OR by combined
        rating. This preserves diversity in the self-play opponent pool.
        """
        if self.max_entries is None:
            return
        entries = list(self.manifest["entries"])
        if len(entries) <= self.max_entries:
            return
        keep_recent = min(max(self.keep_recent, 0), self.max_entries)
        recent = entries[-keep_recent:] if keep_recent > 0 else []
        older = entries[:-keep_recent] if keep_recent > 0 else entries
        keep_best = max(self.max_entries - len(recent), 0)

        # Build per-PC top-K sets: keep entries that are best at *any* PC.
        # Allocate slots across combined + per-PC ratings.
        per_pc_slots = max(keep_best // 4, 1)  # slots per dimension
        combined_slots = keep_best  # combined also gets full budget (overlap is fine)

        keep_set: set[str] = set()

        # Top-K by combined rating
        by_combined = sorted(older, key=self._entry_strength_key, reverse=True)
        for entry in by_combined[:combined_slots]:
            keep_set.add(entry["path"])

        # Top-K by each per-PC rating
        for pc_key in ("rating_2p", "rating_3p", "rating_4p"):
            by_pc = sorted(
                older,
                key=lambda e, k=pc_key: float(e.get(k, 0.0)),
                reverse=True,
            )
            for entry in by_pc[:per_pc_slots]:
                keep_set.add(entry["path"])

        keep_paths = {entry["path"] for entry in recent}
        keep_paths.update(keep_set)

        # Delete checkpoint files for entries we no longer need on disk,
        # but keep the entries themselves in the manifest for rating history.
        for entry in entries:
            if entry["path"] not in keep_paths:
                self._drop_entry(entry)

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
                "rating": float(R.DEFAULT_INITIAL_RATING),
                "games": 0,
                "hidden": int(net.hidden),
                "arch": str(net.arch),
                "active": True,
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
        # Only consider entries whose checkpoint files still exist on disk.
        entries = [e for e in self.manifest["entries"] if self._entry_available(e)]
        if not entries:
            return None
        # Weight by recency (newer = heavier) times a softened rating factor.
        # Ratings are centered around DEFAULT_INITIAL_RATING so we use that
        # as the reference point for sampling weights.
        weights = []
        for i, e in enumerate(entries):
            recency = 1.0 + i
            rel = (self._entry_rating(e) - R.DEFAULT_INITIAL_RATING) / 800.0
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
        # Only consider entries whose checkpoint files still exist on disk.
        entries = [
            entry
            for entry in self.manifest["entries"]
            if (exclude_idx is None or int(entry["idx"]) != exclude_idx)
            and self._entry_available(entry)
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
        ties: float = 0.0,
        num_players: int = 2,
    ) -> None:
        R.add_match_result(
            self.manifest["results"],
            entity_a,
            entity_b,
            wins_a,
            wins_b,
            ties,
            num_players=num_players,
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
        ratings_data = R.compute_ratings(
            self.manifest["results"],
            anchors=anchors,
            initial=initial,
        )
        # Extract combined ratings
        ratings: dict[str, float] = {}
        for entity, data in ratings_data.items():
            if data["rating"] is not None:
                ratings[entity] = round(data["rating"])

        # Count actual games per entity.
        # In a K-player game, one physical game produces (K-1) pairwise
        # result entries per participant.  Each result row for player count
        # `pc` contributes (wins_a + wins_b + ties) pairwise entries, but
        # those represent the same physical games — so divide by (pc - 1)
        # to avoid overcounting.
        games_by_entity: dict[str, float] = {key: 0.0 for key in ratings}
        for row in self.manifest["results"]:
            row_total = 0.0
            for pc in (2, 3, 4):
                wa = float(row.get(f"wins_a_{pc}p", 0))
                wb = float(row.get(f"wins_b_{pc}p", 0))
                ties = float(row.get(f"ties_{pc}p", 0))
                pairwise_count = wa + wb + ties
                if pairwise_count > 0:
                    row_total += pairwise_count / (pc - 1)
            # Legacy format fallback
            if row_total == 0:
                row_total = float(row.get("games", 0))
            games_by_entity[row["a"]] = games_by_entity.get(row["a"], 0.0) + row_total
            games_by_entity[row["b"]] = games_by_entity.get(row["b"], 0.0) + row_total

        for entry in self.manifest["entries"]:
            entity = self._entry_entity_id(int(entry["idx"]))
            if entity in ratings:
                entry["rating"] = round(float(ratings[entity]))
                entry.pop("elo", None)
                entry["games"] = int(games_by_entity.get(entity, 0))
                # Store per-PC ratings on the entry for visibility
                data = ratings_data.get(entity, {})
                for pc in (2, 3, 4):
                    cal_key = f"calibrated_{pc}p"
                    if cal_key in data:
                        entry[f"rating_{pc}p"] = round(data[cal_key])

        # Persist fitted ratings for floating entities (non-anchor, non-checkpoint
        # participants like heuristic_opus that appear in results).
        entry_entities = {self._entry_entity_id(int(e["idx"])) for e in self.manifest["entries"]}
        anchor_entities = set(self.manifest["anchors"])
        floating: dict[str, dict] = {}
        for entity, data in ratings_data.items():
            if entity not in entry_entities and entity not in anchor_entities:
                floating[entity] = {
                    "rating": round(data["rating"]),
                    "games": int(games_by_entity.get(entity, 0)),
                }
                for pc in (2, 3, 4):
                    cal_key = f"calibrated_{pc}p"
                    if cal_key in data:
                        floating[entity][f"rating_{pc}p"] = round(data[cal_key])
        if floating:
            self.manifest["floating_entities"] = floating
        self.manifest["rating_system"] = "anchored_bt_per_pc"
        self._save_manifest()
        return ratings

    def update_rating(self, idx: int, new_rating: float, games_played: int) -> None:
        e = self.entry_by_idx(idx)
        if e is None:
            return
        e["rating"] = round(new_rating)
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


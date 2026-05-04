"""Durable run directory management.

Every training or eval invocation creates or attaches to a `Run` which owns:
- `events.log` - append-only, human-tailable, structured log.
- `metrics.jsonl` - one JSON per eval point with all scalar metrics.
- `state.json` - resumable run state (iter, wall_s, phase, last ckpt path).
- `heartbeat.json` - updated periodically to let watchers detect hangs.
- `config.yaml` - frozen hyperparameters and code git SHA (written once).
- `checkpoints/` and `replay/` - binary artifacts.
- `journal.md` - human-readable narrative of decisions.
- `commands/cmd-<n>.log` - per-invocation numbered log.

The Run class is process-safe for a single writer. Multiple readers are fine.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def _default_runs_root() -> str:
    """Resolve the ``runs/`` directory.

    If ``SPLENDOR_AGENT_RUNS_ROOT`` is set, it wins; otherwise fall back to
    ``agent/runs/`` relative to this file's package location.
    """
    env_root = os.environ.get("SPLENDOR_AGENT_RUNS_ROOT")
    if env_root:
        return env_root
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
    )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


class Run:
    def __init__(self, run_id: str, runs_root: Optional[str] = None, create_ok: bool = True):
        self.run_id = run_id
        self.root = pathlib.Path(runs_root or _default_runs_root()) / run_id
        self.events_path = self.root / "events.log"
        self.metrics_path = self.root / "metrics.jsonl"
        self.state_path = self.root / "state.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.config_path = self.root / "config.yaml"
        self.journal_path = self.root / "journal.md"
        self.ckpt_dir = self.root / "checkpoints"
        self.replay_dir = self.root / "replay"
        self.samples_dir = self.root / "samples"
        self.commands_dir = self.root / "commands"
        self._start_wall = time.monotonic()
        if not self.root.exists():
            if not create_ok:
                raise FileNotFoundError(f"run {run_id} does not exist at {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
            for d in (self.ckpt_dir, self.replay_dir, self.samples_dir, self.commands_dir):
                d.mkdir(exist_ok=True)
        else:
            for d in (self.ckpt_dir, self.replay_dir, self.samples_dir, self.commands_dir):
                d.mkdir(exist_ok=True)

        self._cmd_log_fp = self._open_command_log()
        self.event("run_attached", {"run_id": run_id, "root": str(self.root)})

    def _open_command_log(self):
        existing = sorted(self.commands_dir.glob("cmd-*.log"))
        next_n = len(existing)
        path = self.commands_dir / f"cmd-{next_n:04d}.log"
        return open(path, "a", buffering=1)

    def close(self):
        try:
            self._cmd_log_fp.close()
        except Exception:
            pass

    def event(self, name: str, fields: Optional[Dict[str, Any]] = None, level: str = "INFO"):
        record = {
            "t": _now_iso(),
            "lvl": level,
            "event": name,
            "fields": fields or {},
        }
        line = json.dumps(record, default=str) + "\n"
        with open(self.events_path, "a") as f:
            f.write(line)
        self._cmd_log_fp.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def metric(self, row: Dict[str, Any]) -> None:
        if "t" not in row:
            row = {"t": _now_iso(), **row}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, default=str, indent=2)
        os.replace(tmp, self.state_path)

    def read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        with open(self.state_path) as f:
            return json.load(f)

    def write_heartbeat(self, fields: Dict[str, Any]) -> None:
        data = {"t": _now_iso(), **fields}
        tmp = self.heartbeat_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, default=str)
        os.replace(tmp, self.heartbeat_path)

    def read_heartbeat(self) -> Dict[str, Any]:
        if not self.heartbeat_path.exists():
            return {}
        with open(self.heartbeat_path) as f:
            return json.load(f)

    def write_config_if_missing(self, config: Dict[str, Any]) -> None:
        if self.config_path.exists():
            return
        data = {"git_sha": _git_sha(), "config": config}
        with open(self.config_path, "w") as f:
            for k, v in data.items():
                f.write(f"{k}: {json.dumps(v)}\n")

    def read_recent_events(self, n: int = 20) -> list:
        if not self.events_path.exists():
            return []
        with open(self.events_path) as f:
            lines = f.readlines()
        return lines[-n:]

    def read_recent_metrics(self, n: int = 5) -> list:
        if not self.metrics_path.exists():
            return []
        with open(self.metrics_path) as f:
            lines = f.readlines()
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def wall_s(self) -> float:
        return time.monotonic() - self._start_wall

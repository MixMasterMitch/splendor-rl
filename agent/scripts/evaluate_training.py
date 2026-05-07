"""Evaluate recent training progress.

Produces a concise analysis of training runs: rating trends, winrates vs
reference bots, wall-time efficiency, and league standings. Designed to be
invoked by a human or an agent skill for a quick health check.

Usage:
    python -m agent.scripts.evaluate_training [--run-id RUN_ID] [--last N]

Without --run-id, scans all runs under agent/runs/ and reports on the most
recently active one. --last controls how many metric rows to analyze (default: 20).

Output is plain text suitable for terminal or agent consumption.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any


def _default_runs_root() -> pathlib.Path:
    env_root = os.environ.get("SPLENDOR_AGENT_RUNS_ROOT")
    if env_root:
        return pathlib.Path(env_root)
    return pathlib.Path(__file__).resolve().parent.parent / "runs"


def _find_most_recent_run(runs_root: pathlib.Path) -> str | None:
    """Find the run with the most recent heartbeat or state file."""
    best_run = None
    best_mtime = 0.0
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        # Skip tune trial dirs and league
        if entry.name.startswith("tune_trial") or entry.name == "league":
            continue
        for marker in ("heartbeat.json", "state.json", "metrics.jsonl"):
            p = entry / marker
            if p.exists():
                mt = p.stat().st_mtime
                if mt > best_mtime:
                    best_mtime = mt
                    best_run = entry.name
    return best_run


def _load_metrics(metrics_path: pathlib.Path, last_n: int) -> list[dict]:
    """Load the last N metric rows from a JSONL file."""
    if not metrics_path.exists():
        return []
    with open(metrics_path) as f:
        lines = f.readlines()
    rows = []
    for line in lines[-last_n:]:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def _load_config(config_path: pathlib.Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        for line in f:
            if line.startswith("config:"):
                raw = line[len("config:"):].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {}
    return {}


def _load_league(runs_root: pathlib.Path) -> dict:
    league_path = runs_root / "league" / "league.json"
    if not league_path.exists():
        return {}
    with open(league_path) as f:
        return json.load(f)


def _load_heartbeat(hb_path: pathlib.Path) -> dict:
    if not hb_path.exists():
        return {}
    with open(hb_path) as f:
        return json.load(f)


def _format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _trend_arrow(values: list[float]) -> str:
    """Return a simple trend indicator based on first vs last third."""
    if len(values) < 3:
        return "—"
    third = max(1, len(values) // 3)
    early = sum(values[:third]) / third
    late = sum(values[-third:]) / third
    diff = late - early
    if abs(diff) < 0.01:
        return "→"
    return "↑" if diff > 0 else "↓"


def _analyze_metrics(metrics: list[dict]) -> dict[str, Any]:
    """Extract key statistics from metric rows."""
    analysis: dict[str, Any] = {}

    if not metrics:
        return analysis

    # Iteration range
    iters = [m.get("iter", 0) for m in metrics]
    analysis["iter_range"] = (min(iters), max(iters))
    analysis["num_evals"] = len(metrics)

    # Time span
    timestamps = []
    for m in metrics:
        t = m.get("t")
        if t:
            try:
                timestamps.append(datetime.fromisoformat(t))
            except (ValueError, TypeError):
                pass
    if len(timestamps) >= 2:
        span = (timestamps[-1] - timestamps[0]).total_seconds() / 60
        analysis["time_span_min"] = span

    # Elapsed from first metric
    elapsed_vals = [m.get("elapsed_min", 0) for m in metrics if "elapsed_min" in m]
    if elapsed_vals:
        analysis["total_elapsed_min"] = max(elapsed_vals)

    # Winrates
    for opponent in ("random", "heuristic", "heuristic_opus"):
        key = f"winrate_vs_{opponent}"
        vals = [m[key] for m in metrics if key in m]
        if vals:
            analysis[f"{opponent}_winrate_latest"] = vals[-1]
            analysis[f"{opponent}_winrate_mean"] = sum(vals) / len(vals)
            analysis[f"{opponent}_winrate_trend"] = _trend_arrow(vals)

    # Game length (avg turns vs heuristic_opus as proxy for efficiency)
    turn_key = "avg_turns_vs_heuristic_opus"
    turn_vals = [m[turn_key] for m in metrics if turn_key in m]
    if turn_vals:
        analysis["avg_game_length_latest"] = turn_vals[-1]
        analysis["avg_game_length_trend"] = _trend_arrow(
            [-v for v in turn_vals]  # invert so shorter = up
        )

    # Eval throughput
    gps_vals = [m.get("eval_games_per_s", 0) for m in metrics if "eval_games_per_s" in m]
    if gps_vals:
        analysis["eval_games_per_s"] = sum(gps_vals) / len(gps_vals)

    return analysis


def _format_league_table(league: dict, top_n: int = 10) -> str:
    """Format the top league entries as a compact table."""
    entries = league.get("entries", [])
    if not entries:
        return "  (no league entries)"

    # Sort by rating descending
    ranked = sorted(entries, key=lambda e: e.get("rating", 0), reverse=True)[:top_n]

    lines = []
    lines.append(f"  {'#':<3} {'Idx':<6} {'Tag':<10} {'Rating':<8} {'Games':<6}")
    lines.append(f"  {'—'*3} {'—'*6} {'—'*10} {'—'*8} {'—'*6}")
    for i, e in enumerate(ranked, 1):
        lines.append(
            f"  {i:<3} {e.get('idx', '?'):<6} {e.get('tag', '?'):<10} "
            f"{e.get('rating', 0):<8.1f} {e.get('games', 0):<6}"
        )
    return "\n".join(lines)


def evaluate_training(
    run_id: str | None = None,
    last_n: int = 20,
    runs_root: pathlib.Path | None = None,
) -> str:
    """Main evaluation logic. Returns a formatted report string."""
    root = runs_root or _default_runs_root()

    if run_id is None:
        run_id = _find_most_recent_run(root)
        if run_id is None:
            return "No training runs found."

    run_dir = root / run_id
    if not run_dir.exists():
        return f"Run '{run_id}' not found at {run_dir}"

    # Load data
    state = _load_state(run_dir / "state.json")
    config = _load_config(run_dir / "config.yaml")
    metrics = _load_metrics(run_dir / "metrics.jsonl", last_n)
    heartbeat = _load_heartbeat(run_dir / "heartbeat.json")
    league = _load_league(root)
    analysis = _analyze_metrics(metrics)

    # Build report
    sections: list[str] = []

    # Header
    sections.append(f"Training Evaluation: {run_id}")
    sections.append("=" * (len(sections[0])))

    # Run status
    current_iter = state.get("iter", "?")
    decision = state.get("decision", "?")
    max_iters = config.get("max_iters", "?")
    sections.append(f"\nStatus: iter {current_iter}/{max_iters} | decision: {decision}")

    # Config summary
    if config:
        arch = config.get("arch", "?")
        hidden = config.get("hidden", "?")
        sp_games = config.get("selfplay_games", "?")
        sp_sims = config.get("selfplay_sims", "?")
        lr = config.get("lr", "?")
        device = config.get("device", "?")
        sections.append(
            f"Config: {arch}/{hidden} | selfplay {sp_games}g×{sp_sims}sims | "
            f"lr={lr} | device={device}"
        )

    # Elapsed time
    if "total_elapsed_min" in analysis:
        sections.append(f"Wall time: {_format_duration(analysis['total_elapsed_min'])}")

    # Heartbeat freshness
    if heartbeat:
        hb_t = heartbeat.get("t")
        if hb_t:
            try:
                hb_dt = datetime.fromisoformat(hb_t)
                age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                if age < 120:
                    sections.append(f"Heartbeat: active ({age:.0f}s ago)")
                elif age < 3600:
                    sections.append(f"Heartbeat: stale ({age/60:.0f}min ago)")
                else:
                    sections.append(f"Heartbeat: cold ({age/3600:.1f}h ago)")
            except (ValueError, TypeError):
                pass

    # Performance vs reference bots
    if analysis:
        sections.append("\nPerformance (last {} evals, iters {}-{}):".format(
            analysis.get("num_evals", "?"),
            analysis.get("iter_range", ("?", "?"))[0],
            analysis.get("iter_range", ("?", "?"))[1],
        ))

        for opponent, label in [
            ("random", "Random"),
            ("heuristic", "Heuristic"),
            ("heuristic_opus", "Opus"),
        ]:
            wr = analysis.get(f"{opponent}_winrate_latest")
            mean = analysis.get(f"{opponent}_winrate_mean")
            trend = analysis.get(f"{opponent}_winrate_trend", "—")
            if wr is not None:
                sections.append(
                    f"  vs {label:<12} {wr*100:5.1f}% (avg {mean*100:.1f}%) {trend}"
                )

        gl = analysis.get("avg_game_length_latest")
        gl_trend = analysis.get("avg_game_length_trend", "—")
        if gl is not None:
            sections.append(f"  Avg game length: {gl:.1f} turns {gl_trend}")

        gps = analysis.get("eval_games_per_s")
        if gps:
            sections.append(f"  Eval throughput: {gps:.1f} games/s")

    # League standings
    if league.get("entries"):
        sections.append(f"\nLeague Top-10 (anchors: random=1000, heuristic=2500):")
        sections.append(_format_league_table(league, top_n=10))

        # Best rating
        entries = league["entries"]
        best = max(entries, key=lambda e: e.get("rating", 0))
        sections.append(
            f"\n  Peak: ckpt {best.get('idx')} (tag {best.get('tag')}) "
            f"rating {best.get('rating', 0):.1f}"
        )

    # Quick assessment
    sections.append("\nAssessment:")
    if analysis:
        opus_wr = analysis.get("heuristic_opus_winrate_latest", 0)
        opus_trend = analysis.get("heuristic_opus_winrate_trend", "→")
        if opus_wr >= 0.75:
            sections.append("  Strong play vs Opus (≥75% winrate).")
        elif opus_wr >= 0.60:
            sections.append("  Solid play vs Opus (60-75% winrate).")
        else:
            sections.append("  Struggling vs Opus (<60% winrate).")

        if opus_trend == "↑":
            sections.append("  Trend: improving.")
        elif opus_trend == "↓":
            sections.append("  Trend: declining — may need hyperparameter adjustment.")
        else:
            sections.append("  Trend: stable/flat.")

        if league.get("entries"):
            best_rating = max(e.get("rating", 0) for e in league["entries"])
            if best_rating > 2700:
                sections.append(f"  League peak {best_rating:.0f} — strong overall.")
            elif best_rating > 2600:
                sections.append(f"  League peak {best_rating:.0f} — good, room to grow.")
            else:
                sections.append(f"  League peak {best_rating:.0f} — early stage.")
    else:
        sections.append("  No metrics available for assessment.")

    return "\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate recent training progress and produce a summary report."
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run ID to evaluate. If omitted, uses the most recently active run.",
    )
    p.add_argument(
        "--last",
        type=int,
        default=20,
        help="Number of recent metric rows to analyze (default: 20).",
    )
    p.add_argument(
        "--runs-root",
        default=None,
        help="Override the runs directory (default: agent/runs/).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = pathlib.Path(args.runs_root) if args.runs_root else None
    report = evaluate_training(
        run_id=args.run_id,
        last_n=args.last,
        runs_root=runs_root,
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

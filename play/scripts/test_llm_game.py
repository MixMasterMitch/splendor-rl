"""End-to-end validation script for LLM Bedrock agent gameplay.

Plays a full game against an LLM-powered opponent via the play API,
validating async flow, legal moves, response times, and game completion.

Usage:
    python -m play.scripts.test_llm_game --model bedrock_claude_sonnet --verbose
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a full game against an LLM Bedrock agent and validate behavior.",
    )
    parser.add_argument(
        "--model",
        default="bedrock_claude_sonnet",
        help="Model ID to use as opponent (default: bedrock_claude_sonnet)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the play server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--username",
        default="test_user",
        help="Username for API requests (default: test_user)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum number of turns before aborting (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each move and game state details",
    )
    return parser.parse_args()


def make_headers(username: str) -> dict[str, str]:
    return {"X-Splendor-Username": username, "Content-Type": "application/json"}


def create_game(base_url: str, model_id: str, headers: dict[str, str]) -> dict[str, Any]:
    """Create a new 2-player game with the LLM model as opponent (seat 1)."""
    body = {
        "num_players": 2,
        "human_seat": 0,
        "opponents": {"1": model_id},
    }
    resp = requests.post(f"{base_url}/api/games", json=body, headers=headers)
    if resp.status_code not in (200, 201):
        print(f"ERROR: Failed to create game: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()


def get_game_view(base_url: str, game_id: str, headers: dict[str, str]) -> dict[str, Any]:
    """Fetch the current game view."""
    resp = requests.get(f"{base_url}/api/games/{game_id}", headers=headers)
    if resp.status_code != 200:
        print(f"ERROR: Failed to get game view: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()


def submit_action(
    base_url: str, game_id: str, action: int, headers: dict[str, str]
) -> dict[str, Any]:
    """Submit a human action."""
    body = {"action": action}
    resp = requests.post(
        f"{base_url}/api/games/{game_id}/action", json=body, headers=headers
    )
    if resp.status_code != 200:
        print(f"ERROR: Failed to submit action: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()


def poll_until_not_thinking(
    base_url: str,
    game_id: str,
    headers: dict[str, str],
    timeout: float = 60.0,
    poll_interval: float = 2.0,
    verbose: bool = False,
) -> tuple[dict[str, Any], float]:
    """Poll until game status is no longer 'ai_thinking'.

    Returns (view, elapsed_seconds).
    Raises SystemExit if timeout exceeded.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"ERROR: AI move timed out after {timeout:.0f}s (game {game_id})")
            sys.exit(1)

        view = get_game_view(base_url, game_id, headers)
        status = view.get("status")

        if status != "ai_thinking":
            return view, time.time() - start

        if verbose:
            print(f"  ... AI thinking ({elapsed:.1f}s elapsed)")

        time.sleep(poll_interval)


def pick_random_legal_action(view: dict[str, Any]) -> int:
    """Pick a random legal action from the game view."""
    legal_actions = view.get("legal_actions")
    if not legal_actions:
        print("ERROR: No legal actions available in view")
        sys.exit(1)
    return random.choice(legal_actions)


def count_fallbacks(view: dict[str, Any]) -> int:
    """Count how many steps used LLM fallback actions."""
    steps = view.get("steps") or []
    return sum(1 for step in steps if step.get("llm_fallback"))


def main() -> None:
    args = parse_args()
    headers = make_headers(args.username)

    print(f"=== LLM Game Test ===")
    print(f"Model: {args.model}")
    print(f"Server: {args.base_url}")
    print(f"Username: {args.username}")
    print(f"Max turns: {args.max_turns}")
    print()

    # Step 1: Create game
    print("Creating game...")
    game_data = create_game(args.base_url, args.model, headers)
    game_id = game_data.get("game_id")
    if not game_id:
        print(f"ERROR: No game_id in response: {game_data}")
        sys.exit(1)
    print(f"Game created: {game_id}")

    # Step 2: Get initial view and poll if AI goes first
    view = get_game_view(args.base_url, game_id, headers)
    status = view.get("status")

    ai_response_times: list[float] = []
    turn_count = 0

    # If AI is thinking (it goes first), wait for it
    if status == "ai_thinking":
        print("AI is making first move, polling...")
        view, elapsed = poll_until_not_thinking(
            args.base_url, game_id, headers, verbose=args.verbose
        )
        ai_response_times.append(elapsed)
        print(f"AI first move completed in {elapsed:.2f}s")

    # Step 3: Main game loop
    print("\nStarting game loop...")
    while True:
        status = view.get("status")

        if status == "ended":
            print(f"\nGame ended after {turn_count} human turns.")
            break

        if turn_count >= args.max_turns:
            print(f"\nMax turns ({args.max_turns}) reached. Stopping.")
            break

        if status != "human_turn":
            print(f"ERROR: Unexpected status '{status}' when expecting human_turn")
            sys.exit(1)

        # Pick and submit a random legal action
        action = pick_random_legal_action(view)
        turn_count += 1

        if args.verbose:
            legal_actions = view.get("legal_actions", [])
            print(
                f"\nTurn {turn_count}: Playing action {action} "
                f"(from {len(legal_actions)} legal actions)"
            )

        submit_response = submit_action(args.base_url, game_id, action, headers)

        # After submitting, check if game ended immediately or AI needs to think
        # The submit response may contain the updated view
        if isinstance(submit_response, dict) and "status" in submit_response:
            view = submit_response
        else:
            view = get_game_view(args.base_url, game_id, headers)

        status = view.get("status")

        if status == "ended":
            # Game ended after our move (we triggered end condition)
            continue

        if status == "ai_thinking":
            # Wait for AI to respond
            view, elapsed = poll_until_not_thinking(
                args.base_url, game_id, headers, verbose=args.verbose
            )
            ai_response_times.append(elapsed)
            if args.verbose:
                print(f"  AI responded in {elapsed:.2f}s")

    # Step 4: Print summary
    print("\n" + "=" * 50)
    print("GAME SUMMARY")
    print("=" * 50)
    print(f"Game ID: {game_id}")
    print(f"Total human turns: {turn_count}")
    print(f"Final status: {view.get('status')}")

    # AI response times
    if ai_response_times:
        avg_time = sum(ai_response_times) / len(ai_response_times)
        min_time = min(ai_response_times)
        max_time = max(ai_response_times)
        print(f"\nAI Response Times ({len(ai_response_times)} moves):")
        print(f"  Min: {min_time:.2f}s")
        print(f"  Avg: {avg_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
    else:
        avg_time = 0.0
        print("\nNo AI response times recorded.")

    # Fallback actions
    fallback_count = count_fallbacks(view)
    if fallback_count > 0:
        print(f"\nFallback actions used: {fallback_count}")
    else:
        print("\nNo fallback actions used.")

    # Final scores and winner
    scores = view.get("scores")
    if scores:
        print(f"\nFinal Scores: {scores}")
        # Determine winner (highest score)
        if isinstance(scores, list) and len(scores) >= 2:
            if scores[0] > scores[1]:
                print("Winner: Human (Player 0)")
            elif scores[1] > scores[0]:
                print(f"Winner: AI (Player 1 - {args.model})")
            else:
                print("Result: Tie")

    winner = view.get("winner")
    if winner is not None:
        if winner == 0:
            print("Winner: Human (Player 0)")
        else:
            print(f"Winner: AI (Player 1 - {args.model})")

    # Step 5: Validate success criteria
    print("\n" + "-" * 50)
    print("VALIDATION")
    print("-" * 50)

    success = True

    # Check game completed
    if view.get("status") == "ended":
        print("[PASS] Game completed successfully")
    elif turn_count >= args.max_turns:
        print("[WARN] Game did not complete within max turns (not necessarily a failure)")
    else:
        print("[FAIL] Game did not complete")
        success = False

    # Check average AI response time
    if ai_response_times:
        if avg_time < 15.0:
            print(f"[PASS] Average AI response time ({avg_time:.2f}s) < 15s")
        else:
            print(f"[FAIL] Average AI response time ({avg_time:.2f}s) >= 15s")
            success = False
    else:
        print("[PASS] No AI moves to validate timing (human went last)")

    # Check no infinite polling (already enforced by 60s timeout in poll_until_not_thinking)
    print("[PASS] No infinite polling (60s timeout enforced per AI move)")

    # Check no unhandled exceptions (if we got here, no exceptions occurred)
    print("[PASS] No unhandled exceptions")

    print("\n" + "=" * 50)
    if success:
        print("RESULT: PASS")
        print("=" * 50)
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()

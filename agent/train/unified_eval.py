"""Unified evaluation: one big eval every N iterations.

Runs 2048 games mixing 2p/3p/4p with random seat assignment. The eval agent
plays alongside opponents drawn from: random, heuristic, heuristic_opus, and
4 random league ML checkpoints. All pairwise results (including non-eval
agents vs each other) feed into the rating system.

Designed to run async on CPU via a subprocess so GPU selfplay continues.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import random as stdlib_random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import model as M
from ..search import gumbel_mcts as G
from . import checkpointing as CK
from . import ranking as R

logger = logging.getLogger(__name__)


def _infer_arch(state_dict: dict[str, torch.Tensor]) -> tuple[int, str]:
    """Infer ``(hidden, arch)`` from a SplendorNet state_dict.

    The ``flat`` architecture has a ``flat_trunk.0.weight`` key while the
    ``attn`` architecture has ``g_trunk.0.weight``.  Hidden size is derived
    from the policy head's input dimension (``policy_head.0.weight`` has
    shape ``[hidden, hidden*2]``).
    """
    if "flat_trunk.0.weight" in state_dict:
        arch = "flat"
    elif "g_trunk.0.weight" in state_dict:
        arch = "attn"
    else:
        raise ValueError("Cannot infer arch from state_dict keys")

    # policy_head.0.weight has shape (hidden, hidden*2)
    hidden = int(state_dict["policy_head.0.weight"].shape[0])
    return hidden, arch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UnifiedEvalConfig:
    """Configuration for the unified eval."""

    total_games: int = 512
    num_sims: int = 64
    max_turns: int = 200
    # Player count weights: 2p=1.0, 3p=0.5, 4p=1.0
    weight_2p: float = 1.0
    weight_3p: float = 0.5
    weight_4p: float = 1.0
    # Number of league opponents to sample
    league_opponents: int = 4
    timeout_s: float = 900.0  # 15 minutes for the bigger eval


# ---------------------------------------------------------------------------
# Game distribution
# ---------------------------------------------------------------------------


def _distribute_games(total: int, w2: float, w3: float, w4: float) -> Tuple[int, int, int]:
    """Distribute total games across 2p/3p/4p by weight."""
    total_weight = w2 + w3 + w4
    n2 = int(round(total * w2 / total_weight))
    n3 = int(round(total * w3 / total_weight))
    n4 = total - n2 - n3
    return n2, n3, n4


# ---------------------------------------------------------------------------
# Pairwise result extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PairwiseResult:
    """A single pairwise result from a multiplayer game."""

    winner: str
    loser: str
    weight: float  # 1/(num_players - 1) normalization
    num_players: int = 2


def extract_pairwise_results(
    seat_names: List[List[str]],
    winner_seats: torch.Tensor,
    finished: torch.Tensor,
    num_players_per_game: int,
) -> List[PairwiseResult]:
    """Extract pairwise results from finished games.

    For a game with N players where player A wins:
      A beats B with weight 1.0
      A beats C with weight 1.0
      etc.

    This produces (N-1) pairwise records per game, each with weight 1.0.
    The game-count logic in recompute_ratings divides by (N-1) to recover
    the actual number of physical games, so we must NOT normalize here.

    Only finished games contribute results.
    """
    results: List[PairwiseResult] = []
    B = winner_seats.shape[0]

    for b in range(B):
        if not finished[b]:
            continue
        winner_idx = int(winner_seats[b].item())
        winner_name = seat_names[b][winner_idx]
        for p in range(num_players_per_game):
            if p == winner_idx:
                continue
            loser_name = seat_names[b][p]
            # Skip self-play: in multiplayer games the same entity can
            # occupy multiple seats; recording X-beats-X is meaningless.
            if loser_name == winner_name:
                continue
            results.append(PairwiseResult(
                winner=winner_name,
                loser=loser_name,
                weight=1.0,
                num_players=num_players_per_game,
            ))
    return results


# ---------------------------------------------------------------------------
# Multiplayer game runner
# ---------------------------------------------------------------------------


def _run_multiplayer_games(
    num_players: int,
    num_games: int,
    policies: Dict[str, Callable[[BE.BatchedEngine], torch.Tensor]],
    eval_agent_name: str,
    seed: int,
    max_turns: int,
) -> Tuple[List[PairwiseResult], Dict[str, float]]:
    """Run a batch of multiplayer games with mixed policies.

    Each game randomly assigns seats from the available policies.
    The eval agent is guaranteed one seat per game; remaining seats
    are filled by random sampling from other policies.

    Returns (pairwise_results, summary_metrics).
    """
    if num_games == 0:
        return [], {}

    device = "cpu"
    rng = stdlib_random.Random(seed)
    other_names = [n for n in policies if n != eval_agent_name]

    # Assign seats: eval agent gets a random seat, others fill remaining
    seat_names: List[List[str]] = []
    for _ in range(num_games):
        seats = [""] * num_players
        eval_seat = rng.randint(0, num_players - 1)
        seats[eval_seat] = eval_agent_name
        for p in range(num_players):
            if p == eval_seat:
                continue
            seats[p] = rng.choice(other_names)
        seat_names.append(seats)

    # Build seat->policy mapping tensor for efficient dispatch
    # Group games by seat assignment pattern for batched execution
    engine = BE.BatchedEngine(num_games, num_players, device=device, seed=seed)

    # Track game end
    game_end_step = torch.full(
        (num_games,), fill_value=max_turns, dtype=torch.int32, device=device
    )
    prev_ended = torch.zeros(num_games, dtype=torch.bool, device=device)

    # Play games turn by turn
    turn = 0
    while turn < max_turns and (~engine.ended).any():
        alive = ~engine.ended
        if not alive.any():
            break

        cp = engine.current_player.to(torch.long)

        # Group games by which policy is active
        actions = torch.full(
            (num_games,), A.PASS_ACTION, dtype=torch.int64, device=device
        )

        # Determine which policy each game needs
        policy_for_game: Dict[str, List[int]] = {}
        for b in range(num_games):
            if not alive[b]:
                continue
            player_idx = int(cp[b].item())
            name = seat_names[b][player_idx]
            if name not in policy_for_game:
                policy_for_game[name] = []
            policy_for_game[name].append(b)

        # Execute each policy on its subset
        for name, game_indices in policy_for_game.items():
            idx = torch.tensor(game_indices, dtype=torch.long, device=device)
            sub_engine = engine.index_select(idx)
            sub_actions = policies[name](sub_engine)
            actions.index_copy_(0, idx, sub_actions)

        engine.apply(actions)

        cur_ended = engine.ended
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

    # Determine winners
    pts = engine.points.to(torch.int32)
    bonuses_total = engine.bonuses.sum(dim=-1).to(torch.int32)
    score = pts * 1000 - bonuses_total
    score = torch.where(engine.active_mask, score, torch.full_like(score, -(10**9)))
    winner_seats = score.argmax(dim=-1)  # (B,)
    finished = engine.ended

    # Extract pairwise results
    pairwise = extract_pairwise_results(
        seat_names, winner_seats, finished, num_players
    )

    # Summary metrics
    finished_count = int(finished.sum().item())
    eval_wins = 0
    eval_games_finished = 0
    for b in range(num_games):
        if not finished[b]:
            continue
        eval_games_finished += 1
        winner_idx = int(winner_seats[b].item())
        if seat_names[b][winner_idx] == eval_agent_name:
            eval_wins += 1

    avg_turns = float(game_end_step.to(torch.float32).sum().item()) / max(num_games, 1)

    metrics = {
        f"games_{num_players}p": num_games,
        f"finished_{num_players}p": finished_count,
        f"eval_winrate_{num_players}p": eval_wins / max(eval_games_finished, 1),
        f"avg_turns_{num_players}p": avg_turns,
    }

    return pairwise, metrics


# ---------------------------------------------------------------------------
# Unified eval orchestrator
# ---------------------------------------------------------------------------


def run_unified_eval(
    state_dict: Dict[str, torch.Tensor],
    league_checkpoint_paths: List[str],
    config: UnifiedEvalConfig,
    seed: int,
) -> Dict:
    """Run the full unified eval. Designed to run in a subprocess on CPU.

    Args:
        state_dict: The eval agent's network weights.
        league_checkpoint_paths: Paths to league checkpoint files to use as opponents.
        config: Eval configuration.
        seed: Random seed.

    Returns:
        Dict with keys:
            - "pairwise": list of {winner, loser, weight} dicts
            - "metrics": summary metrics dict
            - "eval_wall_s": wall time
    """
    from ..eval import bots as B
    from ..eval import heuristic_opus as HO

    t_start = time.monotonic()

    # Load eval agent
    hidden, arch = _infer_arch(state_dict)
    eval_net = M.SplendorNet(hidden=hidden, arch=arch)
    eval_net.load_state_dict(state_dict)
    eval_net.eval()

    eval_agent_name = "eval_agent"

    # Build policy callables
    def _make_net_policy(net: M.SplendorNet, num_sims: int):
        """Create a policy callable from a net."""
        def choose(engine: BE.BatchedEngine) -> torch.Tensor:
            with torch.no_grad():
                act, _ = G.gumbel_root_act(engine, net, num_sims=num_sims)
            return act
        return choose

    def _make_bot_policy(bot):
        return bot.choose

    policies: Dict[str, Callable[[BE.BatchedEngine], torch.Tensor]] = {
        eval_agent_name: _make_net_policy(eval_net, config.num_sims),
        "random": _make_bot_policy(B.RandomBot(seed=seed)),
        "heuristic": _make_bot_policy(B.HeuristicBot()),
        "heuristic_opus": _make_bot_policy(HO.HeuristicOpusV15()),
    }

    # Load league opponents
    league_names: List[str] = []
    for i, path in enumerate(league_checkpoint_paths):
        try:
            net, _ = CK.load_net_from_checkpoint(
                __import__("pathlib").Path(path), map_location="cpu"
            )
            net.eval()
            name = f"league_{i}"
            policies[name] = _make_net_policy(net, config.num_sims)
            league_names.append(name)
        except Exception as e:
            logger.warning("Failed to load league checkpoint %s: %s", path, e)

    # Distribute games
    n2, n3, n4 = _distribute_games(
        config.total_games, config.weight_2p, config.weight_3p, config.weight_4p
    )

    all_pairwise: List[PairwiseResult] = []
    all_metrics: Dict[str, float] = {}

    # Run games for each player count
    for num_players, num_games in [(2, n2), (3, n3), (4, n4)]:
        if num_games == 0:
            continue
        pairwise, metrics = _run_multiplayer_games(
            num_players=num_players,
            num_games=num_games,
            policies=policies,
            eval_agent_name=eval_agent_name,
            seed=seed + num_players * 1000,
            max_turns=config.max_turns,
        )
        all_pairwise.extend(pairwise)
        all_metrics.update(metrics)

    wall_s = time.monotonic() - t_start
    all_metrics["eval_wall_s"] = round(wall_s, 3)
    all_metrics["eval_games_total"] = float(n2 + n3 + n4)

    # Aggregate pairwise into {(winner, loser, num_players): weight} for the rating system
    pairwise_agg: Dict[Tuple[str, str, int], float] = {}
    for pr in all_pairwise:
        key = (pr.winner, pr.loser, pr.num_players)
        pairwise_agg[key] = pairwise_agg.get(key, 0.0) + pr.weight

    # Convert to serializable list
    pairwise_list = [
        {"winner": k[0], "loser": k[1], "num_players": k[2], "weight": v}
        for k, v in pairwise_agg.items()
    ]

    return {
        "pairwise": pairwise_list,
        "metrics": all_metrics,
        "eval_wall_s": wall_s,
        "league_opponents": league_names,
    }


# ---------------------------------------------------------------------------
# Async subprocess wrapper
# ---------------------------------------------------------------------------


def _unified_eval_worker(
    queue: multiprocessing.Queue,
    state_dict: Dict[str, torch.Tensor],
    league_checkpoint_paths: List[str],
    config: UnifiedEvalConfig,
    iteration: int,
    seed: int,
) -> None:
    """Target function for the unified eval subprocess. Runs on CPU."""
    try:
        results = run_unified_eval(state_dict, league_checkpoint_paths, config, seed)
        queue.put((iteration, results))
    except Exception as exc:
        import traceback
        queue.put((iteration, {"error": str(exc), "traceback": traceback.format_exc()}))


class UnifiedEvalHandle:
    """Manages a single unified eval subprocess."""

    def __init__(self, config: UnifiedEvalConfig) -> None:
        self._config = config
        self._process: Optional[multiprocessing.Process] = None
        self._mp_ctx = multiprocessing.get_context("forkserver")
        self._queue: multiprocessing.Queue = self._mp_ctx.Queue()
        self._iteration_tag: Optional[int] = None

    def is_active(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def launch(
        self,
        net_state_dict: Dict[str, torch.Tensor],
        league_checkpoint_paths: List[str],
        iteration: int,
        seed: int,
    ) -> bool:
        """Spawn eval subprocess. Returns True on success, False if already active."""
        if self.is_active():
            return False
        self._reap()

        self._process = self._mp_ctx.Process(
            target=_unified_eval_worker,
            args=(
                self._queue,
                net_state_dict,
                league_checkpoint_paths,
                self._config,
                iteration,
                seed,
            ),
            daemon=True,
        )
        self._iteration_tag = iteration
        self._process.start()
        return True

    def try_collect(self) -> Optional[Tuple[int, Dict]]:
        """Non-blocking check for completed eval results."""
        if self._process is None:
            return None
        result = self._drain_one()
        if result is not None:
            self._reap()
            return result
        if not self._process.is_alive():
            exitcode = self._process.exitcode
            logger.warning(
                "Unified eval subprocess exited with code %s (iteration %s)",
                exitcode, self._iteration_tag,
            )
            self._reap()
            return None
        return None

    def wait_and_collect(self, timeout: float | None = None) -> Optional[Tuple[int, Dict]]:
        """Blocking wait for the active subprocess."""
        if self._process is None:
            return None
        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        self._process.join(timeout=effective_timeout)
        if self._process.is_alive():
            logger.warning(
                "Unified eval subprocess did not finish within %.1fs — terminating",
                effective_timeout,
            )
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
        result = self._drain_one()
        self._reap()
        return result

    def cleanup(self) -> None:
        """Kill any active subprocess and drain the queue."""
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
            self._reap()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def _drain_one(self) -> Optional[Tuple[int, Dict]]:
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    def _reap(self) -> None:
        if self._process is not None:
            if not self._process.is_alive():
                self._process.join(timeout=1)
            self._process = None
            self._iteration_tag = None

"""Async CPU evaluation subprocess.

Manages a forked subprocess that runs the eval ladder on CPU while the GPU
continues selfplay and learning.  At most one eval subprocess is active at
any time.  Results are communicated back via a ``multiprocessing.Queue``.

Lifecycle (from the training loop's perspective):
    1. ``handle.launch(snapshot, iteration, seed)`` — spawns the worker.
    2. ``handle.try_collect()`` — non-blocking poll at each iteration boundary.
    3. ``handle.wait_and_collect(timeout)`` — blocking drain at loop exit.
    4. ``handle.cleanup()`` — kill stragglers and drain the queue.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import multiprocessing.queues
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AsyncEvalConfig:
    """Configuration for the async eval subprocess."""

    eval_games: int = 256
    eval_sims: int = 4
    eval_max_turns: int = 200
    num_players: int = 2
    timeout_s: float = 300.0  # 5-minute default


# ---------------------------------------------------------------------------
# Worker (runs in the forked child process)
# ---------------------------------------------------------------------------

def _eval_worker(
    queue: multiprocessing.Queue,
    state_dict: dict[str, torch.Tensor],
    hidden: int,
    arch: str,
    config: AsyncEvalConfig,
    iteration: int,
    seed: int,
) -> None:
    """Target function for the eval subprocess.  Runs entirely on CPU."""
    try:
        from ..net.model import SplendorNet
        from ..eval.ladder import evaluate

        net = SplendorNet(hidden=hidden, arch=arch)
        net.load_state_dict(state_dict)
        net.eval()

        results = evaluate(
            net,
            num_players=config.num_players,
            num_games=config.eval_games,
            device="cpu",
            num_sims=config.eval_sims,
            max_turns=config.eval_max_turns,
            seed=seed,
        )
        queue.put((iteration, results))
    except Exception as exc:
        # Error sentinel so the parent can distinguish success from failure.
        queue.put((iteration, {"error": str(exc)}))


# ---------------------------------------------------------------------------
# Handle (used by the training loop in the parent process)
# ---------------------------------------------------------------------------

class AsyncEvalHandle:
    """Manages a single eval subprocess lifecycle."""

    def __init__(self, config: AsyncEvalConfig) -> None:
        self._config = config
        self._process: Optional[multiprocessing.Process] = None
        # Use forkserver context: fork deadlocks when CUDA has been
        # initialised in the parent, and spawn cannot pickle fork-context
        # semaphores.  forkserver is safe with CUDA and still fast.
        self._mp_ctx = multiprocessing.get_context("forkserver")
        self._queue: multiprocessing.Queue = self._mp_ctx.Queue()
        self._iteration_tag: Optional[int] = None

    # -- status -------------------------------------------------------------

    def is_active(self) -> bool:
        """True if an eval subprocess is currently running."""
        return self._process is not None and self._process.is_alive()

    # -- launch -------------------------------------------------------------

    def launch(
        self,
        net_state_dict: dict[str, torch.Tensor],
        iteration: int,
        seed: int,
    ) -> bool:
        """Spawn an eval subprocess.

        Returns ``True`` on success, ``False`` if a subprocess is already
        active (single-active constraint).
        """
        if self.is_active():
            return False

        # Clean up any zombie from a previous run.
        self._reap()

        # Infer architecture from the state_dict keys.
        hidden, arch = _infer_arch(net_state_dict)

        ctx = self._mp_ctx
        self._process = ctx.Process(
            target=_eval_worker,
            args=(
                self._queue,
                net_state_dict,
                hidden,
                arch,
                self._config,
                iteration,
                seed,
            ),
            daemon=True,
        )
        self._iteration_tag = iteration
        self._process.start()
        return True

    # -- collection ---------------------------------------------------------

    def try_collect(self) -> Optional[tuple[int, dict]]:
        """Non-blocking check for completed eval results.

        Returns ``(iteration_tag, results_dict)`` or ``None``.
        """
        if self._process is None:
            return None

        # Check the queue first (result may already be there even if the
        # process hasn't been joined yet).
        result = self._drain_one()
        if result is not None:
            self._reap()
            return result

        # If the process has exited but the queue is empty, something went
        # wrong (e.g. the child was killed by the OS).
        if not self._process.is_alive():
            exitcode = self._process.exitcode
            logger.warning(
                "Eval subprocess exited with code %s but produced no result "
                "(iteration %s)",
                exitcode,
                self._iteration_tag,
            )
            self._reap()
            return None

        return None

    def wait_and_collect(self, timeout: float | None = None) -> Optional[tuple[int, dict]]:
        """Blocking wait for the active subprocess.

        Used at loop termination.  Joins with *timeout*, then terminates /
        kills if the child is still alive.
        """
        if self._process is None:
            return None

        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        self._process.join(timeout=effective_timeout)

        if self._process.is_alive():
            logger.warning(
                "Eval subprocess did not finish within %.1fs — terminating "
                "(iteration %s)",
                effective_timeout,
                self._iteration_tag,
            )
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                logger.warning("Eval subprocess still alive after terminate — killing")
                self._process.kill()
                self._process.join(timeout=5)

        result = self._drain_one()
        self._reap()
        return result

    # -- cleanup ------------------------------------------------------------

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

        # Drain any leftover items so the queue's background thread can exit.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    # -- internals ----------------------------------------------------------

    def _drain_one(self) -> Optional[tuple[int, dict]]:
        """Try to get exactly one result from the queue (non-blocking)."""
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    def _reap(self) -> None:
        """Join a finished process and reset internal state."""
        if self._process is not None:
            if not self._process.is_alive():
                self._process.join(timeout=1)
            self._process = None
            self._iteration_tag = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

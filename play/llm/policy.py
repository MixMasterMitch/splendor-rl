"""LLMBedrockPolicy: PlayerPolicy implementation backed by AWS Bedrock Claude."""

from __future__ import annotations

import logging
import random
from typing import Any

import torch

from agent.env import batched_engine as BE
from play.llm.bedrock_client import BedrockClient, BedrockUnavailableError
from play.llm.parser import ActionParser
from play.llm.prompts import (
    CLARIFYING_REPROMPT,
    CLARIFYING_REPROMPT_DEBUG,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DEBUG,
)
from play.llm.renderer import GameStateRenderer

logger = logging.getLogger(__name__)


def _random_legal_action(engine: BE.BatchedEngine, batch_idx: int = 0) -> int:
    """Select a uniform random legal action from the engine's legal action mask."""
    mask = engine.legal_action_mask()  # (B, NA)
    legal_indices = mask[batch_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(legal_indices, int):
        legal_indices = [legal_indices]
    return random.choice(legal_indices)


class LLMBedrockPolicy:
    """PlayerPolicy implementation backed by AWS Bedrock Claude.

    Orchestrates the render → invoke → parse pipeline for each turn.
    On parse failure: retries once with a clarifying re-prompt.
    On second failure or API error: falls back to a uniform random legal action.

    When debug=True, the LLM is asked to provide a one-sentence justification
    alongside its action choice. The last justification is available via
    the `last_reasoning` property.
    """

    def __init__(
        self,
        model_id: str,
        bedrock_model_id: str,
        region: str = "us-west-2",
        include_history: bool = False,
        timeout: float = 30.0,
        debug: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._bedrock_model_id = bedrock_model_id
        self._region = region
        self._include_history = include_history
        self._timeout = timeout
        self._debug = debug
        self._max_tokens = max_tokens
        self._last_reasoning: str | None = None
        self._last_raw_response: str | None = None
        self._last_user_prompt: str | None = None

        self._renderer = GameStateRenderer()
        self._client = BedrockClient(region=region)
        self._parser = ActionParser()

    @property
    def last_reasoning(self) -> str | None:
        """The justification from the most recent LLM move (debug mode only)."""
        return self._last_reasoning

    @property
    def last_raw_response(self) -> str | None:
        """The full raw text response from the most recent LLM call."""
        return self._last_raw_response

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        """Returns (1,) int64 action tensor. Batch size must be 1 for LLM policy."""
        action = self._choose_action(engine, game_history=None)
        return torch.tensor([action], dtype=torch.int64)

    def choose_async(
        self,
        engine: BE.BatchedEngine,
        game_history: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Async-friendly version called from background thread.

        Accepts optional game_history for the include_history flag.
        """
        action = self._choose_action(engine, game_history=game_history)
        return torch.tensor([action], dtype=torch.int64)

    def info(self) -> dict[str, Any]:
        """Returns policy metadata for logging."""
        return {
            "kind": "llm_bedrock",
            "model_id": self._model_id,
            "bedrock_model_id": self._bedrock_model_id,
            "region": self._region,
            "include_history": self._include_history,
            "timeout": self._timeout,
            "debug": self._debug,
            "max_tokens": self._max_tokens,
        }

    def _choose_action(
        self,
        engine: BE.BatchedEngine,
        game_history: list[dict[str, Any]] | None = None,
    ) -> int:
        """Core logic: render → invoke → parse with retry and fallback."""
        self._last_reasoning = None
        self._last_raw_response = None
        self._last_user_prompt = None
        batch_idx = 0
        seat = int(engine.current_player[batch_idx])

        # Get legal actions list
        mask = engine.legal_action_mask()  # (B, NA)
        legal_actions = mask[batch_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(legal_actions, int):
            legal_actions = [legal_actions]

        # Select prompts based on debug mode
        system_prompt = SYSTEM_PROMPT_DEBUG if self._debug else SYSTEM_PROMPT
        clarifying = CLARIFYING_REPROMPT_DEBUG if self._debug else CLARIFYING_REPROMPT

        # Render the game state prompt
        user_prompt = self._renderer.render(engine, seat, batch_idx=batch_idx)

        # Optionally append history summary
        if self._include_history and game_history:
            history_text = self._renderer.render_history_summary(game_history, last_n=5)
            if history_text:
                user_prompt = user_prompt + "\n\n" + history_text

        # Attempt 1: invoke and parse
        self._last_user_prompt = user_prompt
        try:
            response = self._client.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=self._bedrock_model_id,
                max_tokens=self._max_tokens,
            )
            self._last_raw_response = response.text

            if self._debug:
                result = self._parser.parse_with_reasoning(response.text, legal_actions)
                if result.action is not None:
                    self._last_reasoning = result.reasoning
                    logger.warning(
                        "[LLM] action=%d latency=%.0fms thinking=%r",
                        result.action,
                        response.latency_ms,
                        result.reasoning,
                    )
                    return result.action
                # If truncated, don't trust fallback number extraction from reasoning
                if response.stop_reason == "max_tokens":
                    logger.warning(
                        "[LLM] response truncated (max_tokens hit), retrying with clarifying prompt"
                    )
            else:
                action = self._parser.parse(response.text, legal_actions)
                if action is not None:
                    logger.warning(
                        "[LLM] action=%d latency=%.0fms",
                        action,
                        response.latency_ms,
                    )
                    return action

            # Attempt 2: retry with clarifying re-prompt
            logger.warning(
                "[LLM] parse failure on first attempt, response=%r — retrying",
                response.text,
            )
            retry_prompt = user_prompt + "\n\n" + clarifying
            response2 = self._client.invoke(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                model_id=self._bedrock_model_id,
                max_tokens=self._max_tokens,
            )

            if self._debug:
                result2 = self._parser.parse_with_reasoning(response2.text, legal_actions)
                if result2.action is not None:
                    self._last_reasoning = result2.reasoning
                    logger.warning(
                        "[LLM] action=%d (retry) latency=%.0fms thinking=%r",
                        result2.action,
                        response2.latency_ms,
                        result2.reasoning,
                    )
                    return result2.action
            else:
                action2 = self._parser.parse(response2.text, legal_actions)
                if action2 is not None:
                    logger.warning(
                        "[LLM] action=%d (retry) latency=%.0fms",
                        action2,
                        response2.latency_ms,
                    )
                    return action2

            # Both attempts failed — fall back to random
            logger.warning(
                "[LLM] parse failure on BOTH attempts, falling back to random. "
                "retry_response=%r",
                response2.text,
            )
            self._last_reasoning = "[fallback: parse failure]"
            return _random_legal_action(engine, batch_idx)

        except BedrockUnavailableError as e:
            logger.warning(
                "Bedrock unavailable (model=%s): %s — falling back to random",
                self._bedrock_model_id,
                e,
            )
            self._last_reasoning = f"[fallback: {e}]"
            return _random_legal_action(engine, batch_idx)
        except Exception as e:
            logger.error(
                "Unexpected error in LLM policy (model=%s): %s — falling back to random",
                self._bedrock_model_id,
                e,
                exc_info=True,
            )
            self._last_reasoning = f"[fallback: {e}]"
            return _random_legal_action(engine, batch_idx)

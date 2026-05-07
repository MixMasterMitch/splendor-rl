"""LLM Bedrock Agent package for Splendor AI opponents."""

from play.llm.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DEBUG,
    SYSTEM_PROMPT_NO_COT,
    CLARIFYING_REPROMPT,
    CLARIFYING_REPROMPT_DEBUG,
    CLARIFYING_REPROMPT_NO_COT,
)

__all__ = [
    "LLMBedrockPolicy",
    "GameStateRenderer",
    "BedrockClient",
    "ActionParser",
    "ParseResult",
    "LLMRateLimiter",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_DEBUG",
    "SYSTEM_PROMPT_NO_COT",
    "CLARIFYING_REPROMPT",
    "CLARIFYING_REPROMPT_DEBUG",
    "CLARIFYING_REPROMPT_NO_COT",
]


def __getattr__(name: str):
    """Lazy imports for classes implemented in submodules."""
    if name == "LLMBedrockPolicy":
        from play.llm.policy import LLMBedrockPolicy
        return LLMBedrockPolicy
    if name == "GameStateRenderer":
        from play.llm.renderer import GameStateRenderer
        return GameStateRenderer
    if name == "BedrockClient":
        from play.llm.bedrock_client import BedrockClient
        return BedrockClient
    if name == "ActionParser":
        from play.llm.parser import ActionParser
        return ActionParser
    if name == "ParseResult":
        from play.llm.parser import ParseResult
        return ParseResult
    if name == "LLMRateLimiter":
        from play.llm.rate_limiter import LLMRateLimiter
        return LLMRateLimiter
    raise AttributeError(f"module 'play.llm' has no attribute {name!r}")

"""Action parser for extracting action indices from LLM response text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of parsing an LLM response, optionally including reasoning."""

    action: int | None
    reasoning: str | None = None


class ActionParser:
    """Extracts action index from LLM response text."""

    # Patterns for strategy 2: common phrasing the LLM might use
    _PATTERNS = [
        re.compile(r"\baction\s+(\d+)\b", re.IGNORECASE),
        re.compile(r"\baction:\s*(\d+)\b", re.IGNORECASE),
        re.compile(r"\bI choose\s+(\d+)\b", re.IGNORECASE),
    ]

    def parse(
        self,
        response_text: str,
        legal_actions: list[int],
    ) -> int | None:
        """Extract action index from response. Returns None on failure.

        Parsing strategy (ordered):
        1. Look for a line matching just a number (e.g., "5" on its own line)
        2. Look for patterns like "Action N", "action: N", "I choose N"
        3. Look for the first integer in the response that is in legal_actions
        4. Return None if no valid action found
        """
        if not legal_actions:
            return None

        legal_set = set(legal_actions)

        # Strategy 1: line matching just a number
        for line in response_text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"\d+", stripped):
                value = int(stripped)
                if value in legal_set:
                    return value

        # Strategy 2: known patterns
        for pattern in self._PATTERNS:
            match = pattern.search(response_text)
            if match:
                value = int(match.group(1))
                if value in legal_set:
                    return value

        # Strategy 3: no valid action found
        return None

    def parse_with_reasoning(
        self,
        response_text: str,
        legal_actions: list[int],
    ) -> ParseResult:
        """Extract action index and reasoning from response.

        Expects format:
            THINKING: <reasoning>
            ACTION: <number>

        Falls back to standard parse if the format doesn't match.
        """
        if not legal_actions:
            return ParseResult(action=None)

        legal_set = set(legal_actions)

        # Try to parse the structured THINKING:/ACTION: format
        thinking: str | None = None
        action_val: int | None = None

        # Collect thinking text (may span multiple lines until ACTION: is found)
        thinking_lines: list[str] = []
        in_thinking = False

        for line in response_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                num_str = stripped[len("ACTION:"):].strip()
                m = re.match(r"(\d+)", num_str)
                if m:
                    val = int(m.group(1))
                    if val in legal_set:
                        action_val = val
                in_thinking = False
            elif stripped.upper().startswith("THINKING:"):
                thinking_lines = [stripped[len("THINKING:"):].strip()]
                in_thinking = True
            elif in_thinking:
                thinking_lines.append(stripped)

        if thinking_lines:
            thinking = " ".join(thinking_lines).strip()

        if action_val is not None:
            return ParseResult(action=action_val, reasoning=thinking)

        # Fall back: try first line as number, second as reasoning
        lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
        if lines and re.fullmatch(r"\d+", lines[0]):
            value = int(lines[0])
            if value in legal_set:
                reasoning = " ".join(lines[1:]).strip() if len(lines) > 1 else None
                return ParseResult(action=value, reasoning=reasoning)

        # Final fallback: standard parse (no reasoning extracted)
        action = self.parse(response_text, legal_actions)
        reasoning = None
        if action is not None and lines:
            non_numeric = [l for l in lines if not re.fullmatch(r"\d+", l)]
            if non_numeric:
                r = non_numeric[0]
                # Strip THINKING: prefix if present
                if r.upper().startswith("THINKING:"):
                    r = r[len("THINKING:"):].strip()
                reasoning = r
        return ParseResult(action=action, reasoning=reasoning)

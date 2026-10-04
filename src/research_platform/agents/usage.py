"""What one agent call cost: tokens as the provider reported them, and an estimate in money.

Section 13 asks for "tokens and estimated cost per job". Tokens are read from what the
model provider returned with each response, never estimated. Cost is an estimate by
construction: tokens multiplied by the configured price of the configured model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentCallStats:
    """Accumulates across the attempts one agent call needed to produce its contract."""

    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def observe(self, usage: Mapping[str, Any] | None) -> None:
        """Count one attempt and whatever token usage the provider reported for it.

        A provider that reports nothing contributes no tokens rather than a guess, and a
        malformed figure is ignored rather than allowed to fail the agent call it was
        only describing.
        """
        self.attempts += 1
        if not usage:
            return
        self.prompt_tokens += _count(usage.get("prompt_tokens"))
        self.completion_tokens += _count(usage.get("completion_tokens"))

    @property
    def corrections(self) -> int:
        """How many attempts were corrections of an earlier, rejected response."""
        return max(0, self.attempts - 1)


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


@dataclass(frozen=True)
class TokenPricing:
    """The price of the configured model, in USD per million tokens."""

    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_million_usd < 0 or self.output_per_million_usd < 0:
            raise ValueError("a token price cannot be negative")

    def cost_of(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_million_usd
            + completion_tokens * self.output_per_million_usd
        ) / 1_000_000

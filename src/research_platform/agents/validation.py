"""Parse an agent response into its contract, with bounded correction.

Section 12 requires that an invalid agent output rejects the state transition and
requests a bounded schema correction. Bounded is the important word: the agent is told
precisely what failed and given a fixed number of attempts, after which the transition
fails rather than the platform retrying indefinitely against a model that cannot produce
the shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

DEFAULT_MAX_ATTEMPTS = 3


class SchemaCorrectionExhausted(RuntimeError):
    """Raised when an agent could not produce its contract within the attempt budget."""

    def __init__(self, contract: str, attempts: int, failures: list[str]) -> None:
        super().__init__(
            f"{contract} was not produced in {attempts} attempts; last failure: {failures[-1]}"
        )
        self.contract = contract
        self.attempts = attempts
        self.failures = failures


def describe_validation_failure(error: ValidationError) -> str:
    """Describe what to fix in terms an agent can act on.

    Only the field path and the reason are reported. The rejected value is deliberately
    omitted, because echoing it back invites the agent to reproduce content that may have
    come from an untrusted source.
    """
    problems = [
        f"{'.'.join(str(part) for part in problem['loc']) or '(root)'}: {problem['msg']}"
        for problem in error.errors()
    ]
    return "; ".join(problems)


def parse_agent_output[Contract: BaseModel](contract: type[Contract], raw: str) -> Contract:
    """Parse one agent response, raising ValidationError when it does not conform."""
    text = raw.strip()
    if not text:
        raise ValueError("the agent returned an empty response")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"the agent response is not valid JSON: {error.msg}") from error
    return contract.model_validate(document)


@dataclass
class BoundedSchemaCorrection:
    """Ask an agent for its contract, correcting at most a fixed number of times."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    failures: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("at least one attempt must be allowed")

    def resolve[Contract: BaseModel](
        self,
        contract: type[Contract],
        produce: Callable[[str | None], str],
    ) -> Contract:
        """Call ``produce`` until it returns the contract, or the budget is spent.

        ``produce`` receives ``None`` on the first attempt and the previous failure
        description afterwards, so a correction round names exactly what to fix.
        """
        self.failures.clear()
        correction: str | None = None
        for _ in range(self.max_attempts):
            raw = produce(correction)
            try:
                return parse_agent_output(contract, raw)
            except ValidationError as error:
                correction = describe_validation_failure(error)
            except ValueError as error:
                correction = str(error)
            self.failures.append(correction)
        raise SchemaCorrectionExhausted(contract.__name__, self.max_attempts, self.failures)

    @property
    def attempts_used(self) -> int:
        return len(self.failures)

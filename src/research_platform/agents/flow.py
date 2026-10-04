"""One agent step, as a CrewAI Flow: request, judge, and correct a bounded number of times.

Section 13 places a "CrewAI Flow stage" between a Temporal activity and the agent task it
runs. This is that stage. A Temporal activity starts one flow per agent step; the flow
asks the agent for its contract, judges the response, and routes a rejected one back for
correction until it is accepted or the attempt budget is spent (section 12).

Running inside a Flow is not only for the trace. A CrewAI agent asked to ``kickoff`` from
code that is already inside an event loop - which is where every Temporal activity runs -
hands back an unawaited coroutine instead of an answer. A Flow runs its steps where the
agent can answer directly, so the same agent call works from an activity, a test and a
script alike.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.lite_agent_output import LiteAgentOutput
from pydantic import BaseModel, Field, ValidationError

from research_platform.agents.usage import AgentCallStats
from research_platform.agents.validation import (
    SchemaCorrectionExhausted,
    describe_validation_failure,
    parse_agent_output,
)
from research_platform.observability.tracing import get_tracer

ACCEPTED = "accepted"
REJECTED = "rejected"
EXHAUSTED = "exhausted"


class ContractState(BaseModel):
    """Where one agent step has got to."""

    attempts: int = 0
    raw: str = ""
    failures: list[str] = Field(default_factory=list)


class AgentContractFlow(Flow[ContractState]):
    """Ask one agent for one contract, correcting an invalid response a bounded number of times.

    ``verify`` rejects a well-formed response whose references are not real, under the
    same attempt budget as a response that fails its schema: both are an invalid agent
    output. Only the field path and the reason of a failure are sent back to the agent,
    never the rejected value, which may have come from an untrusted source.
    """

    def __init__(
        self,
        *,
        agent: Any,
        role: str,
        contract: type[BaseModel],
        instructions: str,
        max_attempts: int,
        verify: Callable[[Any], object] | None = None,
        stats: AgentCallStats | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("at least one attempt must be allowed")
        # CrewAI's own console panels and hosted tracing are turned off: this platform
        # reports through OpenTelemetry, and an agent step must not print or phone home.
        super().__init__(suppress_flow_events=True, tracing=False)
        self._agent = agent
        self._role = role
        self._contract = contract
        self._instructions = instructions
        self._max_attempts = max_attempts
        self._verify = verify
        self._stats = stats
        self._accepted: BaseModel | None = None

    @property
    def accepted(self) -> BaseModel:
        """The contract the agent produced, or the reason it never did."""
        if self._accepted is None:
            raise SchemaCorrectionExhausted(
                self._contract.__name__, self.state.attempts, list(self.state.failures)
            )
        return self._accepted

    def _ask(self, stage: str, message: str) -> None:
        self.state.attempts += 1
        with get_tracer().start_as_current_span("crewai.flow.stage") as span:
            span.set_attribute("flow.stage", stage)
            span.set_attribute("agent_role", self._role)
            span.set_attribute("flow.attempt", self.state.attempts)
            output = self._agent.kickoff(message)
            if not isinstance(output, LiteAgentOutput):
                raise TypeError("agent.kickoff did not return an agent output")
            if self._stats is not None:
                self._stats.observe(output.usage_metrics)
            self.state.raw = output.raw

    @start()
    def request(self) -> None:
        self._ask("request", self._instructions)

    @router(or_(request, "correct"))
    def judge(self) -> str:
        with get_tracer().start_as_current_span("crewai.flow.stage") as span:
            span.set_attribute("flow.stage", "judge")
            span.set_attribute("agent_role", self._role)
            try:
                parsed = parse_agent_output(self._contract, self.state.raw)
                if self._verify is not None:
                    self._verify(parsed)
            except ValidationError as error:
                failure = describe_validation_failure(error)
            except ValueError as error:
                failure = str(error)
            else:
                self._accepted = parsed
                span.set_attribute("flow.outcome", ACCEPTED)
                return ACCEPTED
            self.state.failures.append(failure)
            outcome = EXHAUSTED if self.state.attempts >= self._max_attempts else REJECTED
            span.set_attribute("flow.outcome", outcome)
            return outcome

    @listen(REJECTED)
    def correct(self) -> None:
        self._ask(
            "correct",
            f"{self._instructions}\n\n"
            "Your previous response was rejected: "
            f"{self.state.failures[-1]}\n"
            "Respond again with the corrected JSON only, and nothing else.",
        )

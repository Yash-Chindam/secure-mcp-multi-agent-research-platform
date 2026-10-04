import json
from unittest.mock import patch

import pytest
from crewai.lite_agent_output import LiteAgentOutput
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from research_platform.agents.contracts import ResearchPlan
from research_platform.agents.crew import (
    AGENT_SPECS,
    build_agent,
    describe_contract,
    request_agent_output,
)
from research_platform.agents.provenance import EvidenceClaims
from research_platform.agents.validation import SchemaCorrectionExhausted
from research_platform.domain.tasks import AgentRole

VALID_PLAN = json.dumps(
    {
        "tasks": [
            {
                "objective": "Collect the vendor pricing page",
                "assigned_agent": AgentRole.RESEARCHER.value,
                "evidence_requirements": ["a dated pricing page"],
            }
        ],
        "rationale": "Pricing must be sourced before it can be compared.",
    }
)

PLAN_WITH_A_CYCLE = json.dumps(
    {
        "tasks": [
            {
                "objective": "First",
                "assigned_agent": AgentRole.RESEARCHER.value,
                "evidence_requirements": ["a source"],
                "depends_on": [1],
            },
            {
                "objective": "Second",
                "assigned_agent": AgentRole.ANALYST.value,
                "evidence_requirements": ["a source"],
                "depends_on": [0],
            },
        ],
        "rationale": "Circular.",
    }
)


class FakeAgent:
    """Stands in for a crewai.Agent, returning scripted kickoff responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.messages: list[str] = []

    def kickoff(self, message: str) -> LiteAgentOutput:
        self.messages.append(message)
        return LiteAgentOutput(raw=next(self._responses), agent_role="researcher")


def test_every_role_has_a_spec_naming_its_contract() -> None:
    assert set(AGENT_SPECS) == set(AgentRole)
    assert AGENT_SPECS[AgentRole.RESEARCHER].contract is EvidenceClaims
    assert AGENT_SPECS[AgentRole.PLANNER].contract is ResearchPlan


def test_build_agent_uses_the_role_spec() -> None:
    agent = build_agent(AgentRole.PLANNER, llm="gpt-4o-mini")

    assert agent.role == AGENT_SPECS[AgentRole.PLANNER].role
    assert agent.goal == AGENT_SPECS[AgentRole.PLANNER].goal
    assert agent.tools == []


def test_a_conforming_first_response_needs_no_correction() -> None:
    agent = FakeAgent([VALID_PLAN])

    plan = request_agent_output(agent, AgentRole.PLANNER, instructions="Plan the research.")  # type: ignore[arg-type]

    assert isinstance(plan, ResearchPlan)
    assert len(agent.messages) == 1


def test_an_invalid_response_is_corrected_with_the_failure_named() -> None:
    agent = FakeAgent([PLAN_WITH_A_CYCLE, VALID_PLAN])

    plan = request_agent_output(agent, AgentRole.PLANNER, instructions="Plan the research.")  # type: ignore[arg-type]

    assert isinstance(plan, ResearchPlan)
    assert len(agent.messages) == 2
    assert "dependency cycle" in agent.messages[1]


def test_correction_is_bounded_and_raises_when_exhausted() -> None:
    agent = FakeAgent([PLAN_WITH_A_CYCLE, PLAN_WITH_A_CYCLE, PLAN_WITH_A_CYCLE])

    with pytest.raises(SchemaCorrectionExhausted) as error:
        request_agent_output(
            agent,
            AgentRole.PLANNER,
            instructions="Plan the research.",
            max_attempts=3,  # type: ignore[arg-type]
        )

    assert error.value.contract == "ResearchPlan"
    assert len(agent.messages) == 3


def test_an_agent_that_returns_something_other_than_an_answer_is_an_error() -> None:
    async def _coro() -> LiteAgentOutput:
        return LiteAgentOutput(raw=VALID_PLAN, agent_role="researcher")

    pending = _coro()

    class AsyncAgent:
        def kickoff(self, message: str) -> object:
            return pending

    try:
        with pytest.raises(TypeError, match="did not return an agent output"):
            request_agent_output(AsyncAgent(), AgentRole.PLANNER, instructions="Plan the research.")  # type: ignore[arg-type]
    finally:
        pending.close()


def test_each_attempt_is_a_traced_flow_stage() -> None:
    """Section 13: a CrewAI Flow stage sits between the activity and the agent task."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent = FakeAgent(["not json", VALID_PLAN])

    with patch("research_platform.agents.flow.get_tracer", lambda: provider.get_tracer("test")):
        request_agent_output(agent, AgentRole.PLANNER, instructions="Plan the research.")

    stages = [
        (span.attributes["flow.stage"], span.attributes.get("flow.outcome"))
        for span in exporter.get_finished_spans()
        if span.name == "crewai.flow.stage"
    ]
    assert stages == [
        ("request", None),
        ("judge", "rejected"),
        ("correct", None),
        ("judge", "accepted"),
    ]
    assert all(span.attributes["agent_role"] == "planner" for span in exporter.get_finished_spans())


def test_a_flow_must_allow_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="at least one attempt"):
        request_agent_output(
            FakeAgent([VALID_PLAN]), AgentRole.PLANNER, instructions="Plan.", max_attempts=0
        )


@pytest.mark.parametrize("role", list(AgentRole))
def test_every_agent_is_shown_the_fields_of_the_contract_it_is_held_to(role: AgentRole) -> None:
    """A model told only the contract's name cannot guess what to return."""
    contract = AGENT_SPECS[role].contract

    described = describe_contract(contract)

    assert contract.__name__ in described
    for field in contract.model_fields:
        assert f'"{field}"' in described


def test_the_contract_is_part_of_what_the_agent_is_asked() -> None:
    agent = FakeAgent([VALID_PLAN])

    request_agent_output(agent, AgentRole.PLANNER, instructions="Plan the research.")  # type: ignore[arg-type]

    asked = agent.messages[0]
    assert asked.startswith("Plan the research.")
    assert '"evidence_requirements"' in asked
    assert '"assigned_agent"' in asked

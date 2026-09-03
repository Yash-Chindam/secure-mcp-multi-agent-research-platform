import json

import pytest
from crewai.lite_agent_output import LiteAgentOutput

from research_platform.agents.contracts import EvidenceSubmission, ResearchPlan
from research_platform.agents.crew import AGENT_SPECS, build_agent, request_agent_output
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
    assert AGENT_SPECS[AgentRole.RESEARCHER].contract is EvidenceSubmission
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
    assert agent.messages == ["Plan the research."]


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


def test_request_agent_output_raises_when_kickoff_returns_a_coroutine() -> None:
    async def _coro() -> LiteAgentOutput:
        return LiteAgentOutput(raw=VALID_PLAN, agent_role="researcher")

    pending = _coro()

    class AsyncAgent:
        def kickoff(self, message: str) -> object:
            return pending

    try:
        with pytest.raises(TypeError, match="CrewAI Flow"):
            request_agent_output(AsyncAgent(), AgentRole.PLANNER, instructions="Plan the research.")  # type: ignore[arg-type]
    finally:
        pending.close()

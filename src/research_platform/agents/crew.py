"""The five CrewAI specialists from section 7, each bound to its output contract.

Sequencing which role runs when - the planner before the researchers, the critic after
the analyst, a durable pause for reviewer approval - belongs to the workflow that drives
a job, not to this module. What lives here is narrower: for one role and one piece of
work, ask the agent for its contract and correct it a bounded number of times before
giving up (section 12). Each such step runs as a CrewAI Flow
(``research_platform.agents.flow``), which is the "CrewAI Flow stage" section 13 places
under a Temporal activity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from crewai import Agent
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    ResearchPlan,
    ResearchReport,
)
from research_platform.agents.flow import AgentContractFlow
from research_platform.agents.provenance import EvidenceClaims
from research_platform.agents.usage import AgentCallStats
from research_platform.agents.validation import DEFAULT_MAX_ATTEMPTS
from research_platform.domain.tasks import AgentRole


class KickoffAgent(Protocol):
    """The one thing ``request_agent_output`` needs from an agent.

    A real ``crewai.Agent`` satisfies this structurally, and so does a fake built for a
    test - neither needs a live LLM to exercise the bounded-correction loop around it.
    """

    def kickoff(self, message: str) -> LiteAgentOutput | object: ...


@dataclass(frozen=True)
class AgentSpec:
    """The role, goal, backstory and output contract section 7 assigns to a specialist."""

    role: str
    goal: str
    backstory: str
    contract: type[BaseModel]


AGENT_SPECS: dict[AgentRole, AgentSpec] = {
    AgentRole.PLANNER: AgentSpec(
        role="Research planner",
        goal=(
            "Decompose the research assignment into ordered tasks with explicit evidence "
            "requirements, assigning each to the specialist who should carry it out."
        ),
        backstory=(
            "You read a research request and the metadata of the tools available to the "
            "crew, and you break the work into tasks the researcher, analyst, critic and "
            "reporter can execute. You never call a tool yourself."
        ),
        contract=ResearchPlan,
    ),
    AgentRole.RESEARCHER: AgentSpec(
        role="Researcher",
        goal="Collect dated, source-attributed evidence for one task using only approved tools.",
        backstory=(
            "You gather evidence through the tools you were given, never from your own "
            "recollection, and you record what you could not find rather than guessing at it. "
            "Every excerpt you submit is quoted exactly from a tool result and names that "
            "result's tool_invocation_id, because the platform checks each one."
        ),
        contract=EvidenceClaims,
    ),
    AgentRole.ANALYST: AgentSpec(
        role="Analyst",
        goal="Compare the collected evidence and perform any calculations the findings depend on.",
        backstory=(
            "You turn evidence into claims. Every claim you propose names the evidence and "
            "calculations it rests on, so a critic can check it without redoing your work."
        ),
        contract=AnalysisResult,
    ),
    AgentRole.CRITIC: AgentSpec(
        role="Critic",
        goal=(
            "Judge each proposed claim against the evidence and name what the research "
            "failed to cover."
        ),
        backstory=(
            "You are the last check before a person sees this research. A claim you cannot "
            "verify against evidence is not supported, and a contradiction you find must name "
            "the evidence it conflicts with."
        ),
        contract=CriticReview,
    ),
    AgentRole.REPORTER: AgentSpec(
        role="Reporter",
        goal=(
            "Write the final report from claims the critic supported, citing evidence for "
            "every factual sentence."
        ),
        backstory=(
            "You write for a reader who will follow every citation back to its source, so you "
            "never state a fact you cannot cite and you say plainly when the report is partial."
        ),
        contract=ResearchReport,
    ),
}


def build_agent(
    role: AgentRole,
    *,
    llm: str | BaseLLM,
    tools: list[BaseTool] | None = None,
) -> Agent:
    """Build the CrewAI agent for one role, with only the tools its role may use.

    The capability gateway enforces tool access regardless of what an agent is handed
    here (section 11); restricting the list keeps the agent's own tool selection honest
    about what it is allowed to attempt, and keeps the planner - a metadata-only role -
    from being offered a tool it could never be authorized to run.
    """
    spec = AGENT_SPECS[role]
    return Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        llm=llm,
        tools=tools or [],
        allow_delegation=False,
    )


def _contract_flow(
    agent: KickoffAgent,
    role: AgentRole,
    *,
    instructions: str,
    max_attempts: int,
    verify: Callable[[Any], object] | None,
    stats: AgentCallStats | None,
) -> AgentContractFlow:
    return AgentContractFlow(
        agent=agent,
        role=role.value,
        contract=AGENT_SPECS[role].contract,
        instructions=instructions,
        max_attempts=max_attempts,
        verify=verify,
        stats=stats,
    )


def request_agent_output(
    agent: KickoffAgent,
    role: AgentRole,
    *,
    instructions: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    verify: Callable[[Any], object] | None = None,
    stats: AgentCallStats | None = None,
) -> BaseModel:
    """Ask an agent for its contract, correcting an invalid response a bounded number of times.

    The step runs as a CrewAI Flow (``research_platform.agents.flow``). ``verify``
    rejects a well-formed response whose references are not real (see
    ``research_platform.agents.checks``), under the same attempt budget. ``stats`` is
    filled in with every attempt made and the tokens each one used, whether or not the
    call succeeds in the end.

    Raises ``SchemaCorrectionExhausted`` when the agent cannot produce a conforming
    response within the attempt budget, so the caller can reject the state transition
    rather than advance the workflow on an invalid output (section 12).
    """
    flow = _contract_flow(
        agent,
        role,
        instructions=instructions,
        max_attempts=max_attempts,
        verify=verify,
        stats=stats,
    )
    flow.kickoff()
    return flow.accepted


async def request_agent_output_async(
    agent: KickoffAgent,
    role: AgentRole,
    *,
    instructions: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    verify: Callable[[Any], object] | None = None,
    stats: AgentCallStats | None = None,
) -> BaseModel:
    """``request_agent_output`` for a caller already inside an event loop, such as an activity."""
    flow = _contract_flow(
        agent,
        role,
        instructions=instructions,
        max_attempts=max_attempts,
        verify=verify,
        stats=stats,
    )
    await flow.kickoff_async()
    return flow.accepted

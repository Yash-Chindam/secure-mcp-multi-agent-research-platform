"""The five CrewAI specialists from section 7, each bound to its output contract.

Sequencing which role runs when - the planner before the researchers, the critic after
the analyst, a durable pause for reviewer approval - belongs to the workflow that drives
a job, not to this module. What lives here is narrower: for one role and one piece of
work, ask the agent for its contract and correct it a bounded number of times before
giving up (section 12), using ``Agent.kickoff`` so a role can be resolved on its own
without assembling a CrewAI ``Task``/``Crew`` around it.
"""

from __future__ import annotations

from dataclasses import dataclass

from crewai import Agent
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    ResearchPlan,
    ResearchReport,
)
from research_platform.agents.validation import DEFAULT_MAX_ATTEMPTS, BoundedSchemaCorrection
from research_platform.domain.tasks import AgentRole


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
            "recollection, and you record what you could not find rather than guessing at it."
        ),
        contract=EvidenceSubmission,
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


def request_agent_output(
    agent: Agent,
    role: AgentRole,
    *,
    instructions: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> BaseModel:
    """Ask an agent for its contract, correcting an invalid response a bounded number of times.

    Raises ``SchemaCorrectionExhausted`` when the agent cannot produce a conforming
    response within the attempt budget, so the caller can reject the state transition
    rather than advance the workflow on an invalid output (section 12).
    """
    spec = AGENT_SPECS[role]
    correction = BoundedSchemaCorrection(max_attempts=max_attempts)

    def produce(previous_failure: str | None) -> str:
        if previous_failure is None:
            message = instructions
        else:
            message = (
                f"{instructions}\n\n"
                "Your previous response did not conform to the required schema: "
                f"{previous_failure}\n"
                "Respond again with the corrected JSON only, and nothing else."
            )
        output = agent.kickoff(message)
        if not isinstance(output, LiteAgentOutput):
            raise TypeError(
                "agent.kickoff returned a coroutine; call it from inside a CrewAI Flow instead"
            )
        return output.raw

    return correction.resolve(spec.contract, produce)

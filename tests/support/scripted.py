"""A scripted crew that does what an honest one does, with no model behind it.

The platform no longer takes an agent's word for its evidence: a researcher has to call a
tool and quote what came back, and every later agent has to cite identifiers that exist.
So a test crew cannot just return canned JSON with invented identifiers - it has to go
through the same motions. These agents do: the researcher really calls its fetch tool and
quotes the result, and the analyst and reporter read the evidence identifiers out of the
prompt they are given, exactly where a model would find them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from crewai.lite_agent_output import LiteAgentOutput
from crewai.tools import BaseTool

from research_platform.domain.tasks import AgentRole
from research_platform.mcp.gateway import ExecutionRequest

SOURCE_URL = "https://vendor.test/pricing"
SOURCE_TEXT = "Vendor pricing is 20 USD per seat."
CLAIM = "The vendor charges 20 USD per seat."

INVOCATION_ID = re.compile(r"\[tool_invocation_id: ([0-9a-f-]{36})\]")
EVIDENCE_ID = re.compile(r"\[([0-9a-f]{8}-[0-9a-f-]{27})\]")

PLAN = json.dumps(
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


class StubExecutor:
    """Stands in for the MCP servers: every call returns the vendor's pricing sentence."""

    def execute(self, request: ExecutionRequest) -> str:
        return SOURCE_TEXT


class ScriptedAgent:
    """Returns whatever ``respond`` computes for the message it was given."""

    def __init__(self, respond: Callable[[str], str]) -> None:
        self._respond = respond
        self.tools: list[BaseTool] = []
        self.messages: list[str] = []

    def kickoff(self, message: str) -> LiteAgentOutput:
        self.messages.append(message)
        return LiteAgentOutput(raw=self._respond(message), agent_role="agent")


class ScriptedResearcher(ScriptedAgent):
    """Fetches the pricing page with its tool, then quotes what the tool returned."""

    def __init__(
        self, tools: list[BaseTool], *, excerpt: str = SOURCE_TEXT, url: str = SOURCE_URL
    ) -> None:
        super().__init__(self._research)
        self.tools = tools
        self._excerpt = excerpt
        self._url = url

    def _research(self, _message: str) -> str:
        fetch = next(tool for tool in self.tools if tool.name == "web_research_fetch")
        returned = str(fetch.run(url=self._url))
        match = INVOCATION_ID.search(returned)
        assert match is not None, f"the tool result named no invocation: {returned!r}"
        return json.dumps(
            {"claims": [{"excerpt": self._excerpt, "tool_invocation_id": match.group(1)}]}
        )


def first_evidence_id(message: str) -> str:
    match = EVIDENCE_ID.search(message)
    assert match is not None, "the prompt listed no evidence identifier"
    return match.group(1)


def analysis_citing(evidence_id: str) -> str:
    return json.dumps(
        {
            "findings": [
                {"claim": CLAIM, "supporting_evidence_ids": [evidence_id], "confidence": 0.9}
            ]
        }
    )


def review(*, requires_reviewer: bool = False) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"claim": CLAIM, "verdict": "supported", "reasoning": "Matches the source."}
            ],
            "coverage_gaps": ["enterprise pricing was not sourced"] if requires_reviewer else [],
        }
    )


def report_citing(evidence_id: str) -> str:
    return json.dumps(
        {
            "title": "Vendor pricing",
            "sections": [{"heading": "Pricing", "body": f"{CLAIM[:-1]} [{evidence_id}]."}],
        }
    )


def honest_crew(
    *, requires_reviewer: bool = False
) -> Callable[[AgentRole, list[BaseTool]], ScriptedAgent]:
    """An agent factory whose five roles produce a verifiable, fully cited job."""

    def build_agent(role: AgentRole, tools: list[BaseTool]) -> ScriptedAgent:
        if role is AgentRole.PLANNER:
            return ScriptedAgent(lambda _message: PLAN)
        if role is AgentRole.RESEARCHER:
            return ScriptedResearcher(tools)
        if role is AgentRole.ANALYST:
            return ScriptedAgent(lambda message: analysis_citing(first_evidence_id(message)))
        if role is AgentRole.CRITIC:
            return ScriptedAgent(lambda _message: review(requires_reviewer=requires_reviewer))
        return ScriptedAgent(lambda message: report_citing(first_evidence_id(message)))

    return build_agent

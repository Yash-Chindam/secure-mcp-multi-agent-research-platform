"""A canned model behind real CrewAI agents.

``support.scripted`` replaces the whole agent. This replaces only the model: the agents
are the real ``crewai.Agent`` objects ``build_agent`` creates, with their real prompts,
their real tool-calling loop and the real ``CapabilityTool`` objects, and this stands
where the model provider would. It reads the conversation the way a model does and
answers in the format CrewAI asks for, so everything between the platform and the model
runs for real - which a scripted agent skips entirely.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM

INVOCATION_ID = re.compile(r"\[tool_invocation_id: ([0-9a-f-]{36})\]")
EVIDENCE_ID = re.compile(r"\[([0-9a-f]{8}-[0-9a-f-]{27})\]")

SOURCE_URL = "https://vendor.test/pricing"
SOURCE_TEXT = "Vendor pricing is 20 USD per seat."
CLAIM = "The vendor charges 20 USD per seat."

PLAN = {
    "tasks": [
        {
            "objective": "Collect the vendor pricing page",
            "assigned_agent": "researcher",
            "evidence_requirements": ["a dated pricing page"],
        }
    ],
    "rationale": "Pricing must be sourced before it can be compared.",
}


def final(answer: dict[str, Any]) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {json.dumps(answer)}"


class CannedModel(BaseLLM):
    """Plays the model for all five roles, telling them apart by their system prompt."""

    def __init__(self, *, first_plan_is_malformed: bool = False) -> None:
        super().__init__(model="canned")
        self.calls: list[str] = []
        self._first_plan_is_malformed = first_plan_is_malformed

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 32_000

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        conversation = "\n".join(str(message.get("content", "")) for message in messages)
        role = self._role_of(str(messages[0].get("content", "")))
        self.calls.append(role)
        return getattr(self, f"_{role}")(conversation)

    async def acall(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        return self.call(messages, *args, **kwargs)

    @staticmethod
    def _role_of(system_prompt: str) -> str:
        for title, role in (
            ("Research planner", "planner"),
            ("Researcher", "researcher"),
            ("Analyst", "analyst"),
            ("Critic", "critic"),
            ("Reporter", "reporter"),
        ):
            if f"You are {title}" in system_prompt:
                return role
        raise AssertionError(f"unrecognised agent prompt: {system_prompt[:80]!r}")

    def _planner(self, conversation: str) -> str:
        if self._first_plan_is_malformed and "previous response was rejected" not in conversation:
            return final({"tasks": [], "rationale": "nothing to do"})
        return final(PLAN)

    def _researcher(self, conversation: str) -> str:
        called = INVOCATION_ID.search(conversation)
        if called is None:
            return (
                "Thought: I need to read the pricing page.\n"
                "Action: web_research_fetch\n"
                f'Action Input: {{"url": "{SOURCE_URL}"}}'
            )
        return final({"claims": [{"excerpt": SOURCE_TEXT, "tool_invocation_id": called.group(1)}]})

    @staticmethod
    def _evidence_id(conversation: str) -> str:
        found = EVIDENCE_ID.search(conversation)
        assert found is not None, "the prompt listed no evidence identifier"
        return found.group(1)

    def _analyst(self, conversation: str) -> str:
        return final(
            {
                "findings": [
                    {
                        "claim": CLAIM,
                        "supporting_evidence_ids": [self._evidence_id(conversation)],
                        "confidence": 0.9,
                    }
                ]
            }
        )

    def _critic(self, _conversation: str) -> str:
        return final(
            {
                "verdicts": [
                    {"claim": CLAIM, "verdict": "supported", "reasoning": "Matches the source."}
                ]
            }
        )

    def _reporter(self, conversation: str) -> str:
        body = f"{CLAIM[:-1]} [{self._evidence_id(conversation)}]."
        return final(
            {"title": "Vendor pricing", "sections": [{"heading": "Pricing", "body": body}]}
        )

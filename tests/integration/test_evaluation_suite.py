"""The section 14 evaluation suite, run over the real pipeline with a scripted crew.

A scripted crew says nothing about how good a model is. What these runs do check is that
the suite measures what it claims to, and that the properties the platform itself
guarantees - citations resolve, partial results are labelled, another tenant gets
nothing - hold whatever the agents do.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from crewai.tools import BaseTool
from support.scripted import (
    CLAIM,
    ScriptedAgent,
    ScriptedResearcher,
    first_evidence_id,
    honest_crew,
)

from research_platform.domain.models import JobStatus, ResearchBudget
from research_platform.domain.tasks import AgentRole
from research_platform.evaluation.__main__ import main, render
from research_platform.evaluation.scenarios import SCENARIOS
from research_platform.evaluation.scoring import task_completion_rate, tool_call_validity_rate
from research_platform.evaluation.suite import (
    CORPUS_DOMAIN,
    Scenario,
    probe_cross_tenant_access,
    run_scenario,
    run_suite,
)
from research_platform.mcp.catalogue import DEFAULT_CAPABILITIES
from research_platform.mcp.servers.backends import SourceDocument

pytestmark = pytest.mark.integration

URL = f"https://{CORPUS_DOMAIN}/pricing"
TEXT = "Vendor pricing is 20 USD per seat."

PRICING = Scenario(
    name="pricing",
    question="What does the vendor charge per seat?",
    documents=(SourceDocument(url=URL, title="Pricing", text=TEXT),),
    expected_tools=frozenset({"web-research.fetch"}),
)


def crew(**replaced: Any) -> Any:
    """An honest crew reading the evaluation corpus, with any role swapped out."""

    def build_agent(role: AgentRole, tools: list[BaseTool]) -> Any:
        if role.value in replaced:
            return replaced[role.value](tools)
        if role is AgentRole.RESEARCHER:
            return ScriptedResearcher(tools, excerpt=TEXT, url=URL)
        return honest_crew()(role, tools)

    return build_agent


@pytest.mark.asyncio
async def test_an_honest_run_completes_and_scores_full_marks() -> None:
    result = await run_scenario(PRICING, crew())

    assert result.status is JobStatus.COMPLETED
    assert result.completed and result.published
    assert result.tools_used == ["web-research.fetch"]
    assert result.tool_selection_accuracy == 1.0
    assert result.citation_correctness == 1.0
    assert result.claim_support_rate == 1.0
    assert result.research_coverage == 1.0
    assert result.contradiction_recall is None
    assert (result.tool_calls, result.invalid_tool_calls) == (1, 0)
    assert result.partial_is_labelled


@pytest.mark.asyncio
async def test_publications_own_re_read_is_not_counted_as_a_tool_the_agents_chose() -> None:
    result = await run_scenario(PRICING, crew())

    assert result.tool_calls == 1


@pytest.mark.asyncio
async def test_a_tool_the_scenario_did_not_expect_lowers_tool_selection() -> None:
    expecting_search = Scenario(
        name="pricing",
        question=PRICING.question,
        documents=PRICING.documents,
        expected_tools=frozenset({"web-research.search", "web-research.fetch"}),
    )

    result = await run_scenario(expecting_search, crew())

    assert result.tool_selection_accuracy == 0.5


@pytest.mark.asyncio
async def test_a_run_that_finds_nothing_is_partial_labelled_and_scores_no_coverage() -> None:
    def fabricating(tools: list[BaseTool]) -> ScriptedResearcher:
        return ScriptedResearcher(tools, excerpt="Vendor pricing is 5 USD per seat.", url=URL)

    result = await run_scenario(PRICING, crew(researcher=fabricating))

    assert result.status is JobStatus.PARTIAL
    assert result.status_detail == "no evidence could be collected"
    assert not result.completed and not result.published
    assert result.research_coverage == 0.0
    assert result.citation_correctness is None
    assert result.partial_is_labelled


@pytest.mark.asyncio
async def test_a_contradiction_the_critic_flags_is_scored_as_recalled() -> None:
    def flagging_critic(_tools: list[BaseTool]) -> ScriptedAgent:
        return ScriptedAgent(
            lambda message: json.dumps(
                {
                    "verdicts": [
                        {
                            "claim": CLAIM,
                            "verdict": "contradicted",
                            "reasoning": "Another source states a different price.",
                            "conflicting_evidence_ids": [first_evidence_id(message)],
                        }
                    ]
                }
            )
        )

    labelled = Scenario(
        name="conflict",
        question=PRICING.question,
        documents=PRICING.documents,
        expected_tools=PRICING.expected_tools,
        known_contradictions=("20 usd per seat", "a fact nobody flagged"),
    )

    result = await run_scenario(labelled, crew(critic=flagging_critic))

    assert result.contradiction_recall == 0.5
    assert result.claim_support_rate == 0.0


@pytest.mark.asyncio
async def test_a_re_read_the_budget_refuses_does_not_stop_the_job_completing() -> None:
    """The one tool call allowed goes to the researcher; the job still publishes."""
    tight = Scenario(
        name="tight",
        question=PRICING.question,
        documents=PRICING.documents,
        expected_tools=PRICING.expected_tools,
        budget=ResearchBudget(max_tool_calls=1),
    )

    result = await run_scenario(tight, crew())

    assert result.status is JobStatus.COMPLETED
    assert result.published
    assert result.tool_calls == 1


@pytest.mark.asyncio
async def test_the_suite_aggregates_scenarios_and_checks_the_design_targets() -> None:
    def fabricating(tools: list[BaseTool]) -> ScriptedResearcher:
        return ScriptedResearcher(tools, excerpt="Not on the page.", url=URL)

    def sometimes_honest(role: AgentRole, tools: list[BaseTool]) -> Any:
        return crew()(role, tools)

    honest = await run_suite([PRICING, PRICING], sometimes_honest)
    mixed_results = [
        await run_scenario(PRICING, crew()),
        await run_scenario(PRICING, crew(researcher=fabricating)),
    ]

    assert honest.task_completion_rate == 1.0
    assert honest.citation_correctness == 1.0
    assert honest.tool_call_validity_rate == 1.0
    assert honest.cross_tenant_successes == 0
    assert honest.mean_cost_usd_per_completed_report == 0.0
    assert honest.mean_active_seconds_per_completed_report is not None
    assert honest.meets_targets
    assert set(honest.targets) == {
        "at least 95% schema-valid tool calls",
        "every published factual claim linked to evidence",
        "zero successful cross-tenant access attempts",
        "partial output clearly identified",
    }
    assert [result.completed for result in mixed_results] == [True, False]


@pytest.mark.asyncio
async def test_a_suite_needs_at_least_one_scenario() -> None:
    with pytest.raises(ValueError, match="at least one scenario"):
        await run_suite([], crew())


def test_every_way_another_tenant_tries_to_reach_a_job_is_blocked() -> None:
    probes = probe_cross_tenant_access()

    assert len(probes) >= 18
    assert [probe.attempt for probe in probes if not probe.blocked] == []
    assert len({probe.attempt for probe in probes}) == len(probes)


def test_the_built_in_scenarios_are_answerable_by_the_tools_they_expect() -> None:
    known = {f"{capability.server}.{capability.name}" for capability in DEFAULT_CAPABILITIES}

    assert len({scenario.name for scenario in SCENARIOS}) == len(SCENARIOS) >= 4
    for scenario in SCENARIOS:
        assert scenario.expected_tools <= known
        assert scenario.documents
        assert all(
            document.url.startswith(f"https://{CORPUS_DOMAIN}/") for document in scenario.documents
        )
    assert any(scenario.known_contradictions for scenario in SCENARIOS)


def test_the_command_prints_the_scores_and_passes_when_targets_are_met(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([], agents=crew(), scenarios=[PRICING])

    printed = capsys.readouterr().out
    assert code == 0
    assert "pricing: completed" in printed
    assert "task completion rate:          100%" in printed
    assert "contradiction recall:          n/a" in printed
    assert "[met] zero successful cross-tenant access attempts" in printed
    assert "MISSED" not in printed


def test_the_command_can_print_the_whole_report_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--json"], agents=crew(), scenarios=[PRICING])

    report = json.loads(capsys.readouterr().out)
    assert report["scenarios"][0]["name"] == "pricing"
    assert report["cross_tenant_successes"] == 0


@pytest.mark.asyncio
async def test_a_missed_target_is_shown_as_missed_and_fails_the_command() -> None:
    report = await run_suite([PRICING], crew())
    missed = report.model_copy(
        update={"targets": {**report.targets, "at least 95% schema-valid tool calls": False}}
    )

    assert not missed.meets_targets
    assert "[MISSED] at least 95% schema-valid tool calls" in render(missed)


def test_completion_and_tool_call_validity_are_plain_fractions() -> None:
    assert task_completion_rate(completed=3, total=4) == 0.75
    assert tool_call_validity_rate(calls=20, invalid=1) == 0.95
    assert tool_call_validity_rate(calls=0, invalid=0) == 1.0
    with pytest.raises(ValueError, match="at least one job"):
        task_completion_rate(completed=0, total=0)
    with pytest.raises(ValueError, match="between zero and the total"):
        task_completion_rate(completed=5, total=4)
    with pytest.raises(ValueError, match="between zero and the number of calls"):
        tool_call_validity_rate(calls=1, invalid=2)

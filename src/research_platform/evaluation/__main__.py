"""Run the section 14 evaluation suite: ``python -m research_platform.evaluation``.

Uses the model in ``RESEARCH_AGENT_LLM`` and the built-in scenarios. Prints the scores
and whether each design target was met, and exits non-zero when one was not, so the
suite can gate a release.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from research_platform.agents.crew import build_agent
from research_platform.agents.usage import TokenPricing
from research_platform.evaluation.scenarios import SCENARIOS
from research_platform.evaluation.suite import AgentFactory, EvaluationReport, Scenario, run_suite
from research_platform.settings import Settings, load_settings


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render(report: EvaluationReport) -> str:
    """The report as plain text: one line per scenario, then the scores and targets."""
    lines = ["Scenarios"]
    for result in report.scenarios:
        detail = f" - {result.status_detail}" if result.status_detail else ""
        lines.append(f"  {result.name}: {result.status.value}{detail}")
    cost = report.mean_cost_usd_per_completed_report
    seconds = report.mean_active_seconds_per_completed_report
    lines += [
        "",
        "Scores",
        f"  task completion rate:          {_rate(report.task_completion_rate)}",
        f"  tool selection accuracy:       {_rate(report.tool_selection_accuracy)}",
        f"  schema-valid tool calls:       {_rate(report.tool_call_validity_rate)}",
        f"  citation correctness:          {_rate(report.citation_correctness)}",
        f"  claim support rate:            {_rate(report.claim_support_rate)}",
        f"  research coverage:             {_rate(report.research_coverage)}",
        f"  contradiction recall:          {_rate(report.contradiction_recall)}",
        f"  cross-tenant attempts blocked: "
        f"{len(report.cross_tenant_probes) - report.cross_tenant_successes}"
        f"/{len(report.cross_tenant_probes)}",
        f"  cost per completed report:     {'n/a' if cost is None else f'{cost:.4f} USD'}",
        f"  time per completed report:     {'n/a' if seconds is None else f'{seconds:.1f} s'}",
        "",
        "Design targets",
    ]
    lines += [
        f"  [{'met' if met else 'MISSED'}] {target}" for target, met in report.targets.items()
    ]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    agents: AgentFactory | None = None,
    scenarios: Sequence[Scenario] = SCENARIOS,
    settings: Settings | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research_platform.evaluation",
        description="Score the research pipeline against the section 14 evaluation criteria.",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    arguments = parser.parse_args(argv)

    resolved = settings or load_settings()
    factory = agents or (lambda role, tools: build_agent(role, llm=resolved.agent_llm, tools=tools))
    pricing = TokenPricing(
        input_per_million_usd=resolved.llm_input_cost_per_million_usd,
        output_per_million_usd=resolved.llm_output_cost_per_million_usd,
    )
    report = asyncio.run(run_suite(scenarios, factory, pricing=pricing))
    print(report.model_dump_json(indent=2) if arguments.json else render(report))
    return 0 if report.meets_targets else 1


if __name__ == "__main__":
    sys.exit(main())

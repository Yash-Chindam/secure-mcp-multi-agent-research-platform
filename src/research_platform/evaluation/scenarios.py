"""The labelled scenarios the evaluation suite runs by default.

Each one is a question, a small fixed corpus served by the web research server, and the
ground truth a run is scored against. The corpus is fixed so two runs of the same model
see the same sources, and so the suite needs no network beyond the model itself.
"""

from __future__ import annotations

from research_platform.evaluation.suite import CORPUS_DOMAIN, Scenario
from research_platform.mcp.servers.backends import SourceDocument

SEARCH_AND_FETCH = frozenset({"web-research.search", "web-research.fetch"})


def _page(path: str, title: str, text: str) -> SourceDocument:
    return SourceDocument(url=f"https://{CORPUS_DOMAIN}/{path}", title=title, text=text)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="single-source-fact",
        question="What does the Northwind team plan cost per seat per month?",
        documents=(
            _page(
                "northwind/pricing",
                "Northwind pricing",
                "Northwind pricing, effective 1 March 2026. The team plan costs 20 USD per "
                "seat per month, billed annually. The enterprise plan is priced on request.",
            ),
        ),
        expected_tools=SEARCH_AND_FETCH,
    ),
    Scenario(
        name="two-source-comparison",
        question=("Which is cheaper per seat per month on its team plan: Northwind or Contoso?"),
        documents=(
            _page(
                "northwind/pricing",
                "Northwind pricing",
                "Northwind pricing, effective 1 March 2026. The team plan costs 20 USD per "
                "seat per month.",
            ),
            _page(
                "contoso/pricing",
                "Contoso pricing",
                "Contoso pricing, effective 15 January 2026. The team plan costs 16 USD per "
                "seat per month.",
            ),
        ),
        expected_tools=SEARCH_AND_FETCH,
    ),
    Scenario(
        name="conflicting-sources",
        question="How many seats does the Northwind team plan include at minimum?",
        documents=(
            _page(
                "northwind/pricing",
                "Northwind pricing",
                "Northwind pricing. The team plan has a minimum of 5 seats.",
            ),
            _page(
                "northwind/faq",
                "Northwind FAQ",
                "Northwind frequently asked questions. The team plan has a minimum of 10 seats.",
            ),
        ),
        expected_tools=SEARCH_AND_FETCH,
        known_contradictions=("minimum",),
    ),
    Scenario(
        name="unanswerable-from-corpus",
        question="What uptime does the Northwind enterprise plan guarantee in its SLA?",
        documents=(
            _page(
                "northwind/pricing",
                "Northwind pricing",
                "Northwind pricing. The team plan costs 20 USD per seat per month. The "
                "enterprise plan is priced on request.",
            ),
        ),
        expected_tools=SEARCH_AND_FETCH,
    ),
)

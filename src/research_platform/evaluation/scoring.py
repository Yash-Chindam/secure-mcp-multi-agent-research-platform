"""Section 14 evaluation metrics, computed from data the platform already produces.

Section 14 lists what the platform should be measured against without prescribing how;
this module answers "how" for the metrics that have an unambiguous definition in terms
of the domain models the pipeline already produces - a ``ResearchReport``, a
``CriticReview``, the evidence requirements a plan asked for.

``research_platform.evaluation.suite`` runs the pipeline over labelled scenarios and
feeds these functions what each job recorded. Cross-tenant access prevention is scored
there too, by probing a real job from a second tenant. Recovery after a controlled
failure is a property of a *running* Temporal workflow, not of a job's finished output,
and is proven by ``tests/integration/test_workflow_recovery.py``.

``contradiction_recall`` and ``tool_selection_accuracy`` need a labelled scenario's
ground truth as an input - a live job has no ground truth to compare itself against, so
these are for a scripted evaluation run over known scenarios, not for scoring production
traffic.
"""

from __future__ import annotations

from uuid import UUID

from research_platform.agents.contracts import CriticReview, ResearchReport, unsupported_citations
from research_platform.domain.models import CriticVerdict


def citation_correctness(
    report: ResearchReport, *, available_evidence_ids: frozenset[UUID]
) -> float:
    """The fraction of a report's citations that reference evidence that actually exists.

    A model can produce a well-formed identifier for evidence that was never collected
    (``contracts.unsupported_citations`` already checks for exactly this); this turns
    that check into a section 14 score, 1.0 when every citation is genuine and there is
    nothing to report a report cites nothing at all.
    """
    cited = report.cited_evidence_ids
    if not cited:
        return 1.0
    missing = unsupported_citations(report, available_evidence_ids=available_evidence_ids)
    return 1 - (len(missing) / len(cited))


def claim_support_rate(review: CriticReview) -> float:
    """The fraction of claims the critic reviewed that it found supported by evidence.

    ``CriticReview.verdicts`` requires at least one entry, so there is always a
    denominator here - a critic review with nothing to say about isn't a valid one.
    """
    supported = sum(1 for verdict in review.verdicts if verdict.verdict is CriticVerdict.SUPPORTED)
    return supported / len(review.verdicts)


def research_coverage(*, requested: int, unmet: int) -> float:
    """The fraction of evidence requirements a research pass actually satisfied."""
    if requested <= 0:
        raise ValueError("research_coverage needs at least one requested evidence requirement")
    if unmet > requested:
        raise ValueError("unmet requirements cannot exceed the number requested")
    return (requested - unmet) / requested


def contradiction_recall(review: CriticReview, *, known_contradictions: frozenset[str]) -> float:
    """How many of a labelled scenario's known contradictions the critic actually caught.

    ``known_contradictions`` is the set of claims a scenario's ground truth says the
    critic should flag as ``CONTRADICTED``.
    """
    if not known_contradictions:
        return 1.0
    flagged = {
        verdict.claim
        for verdict in review.verdicts
        if verdict.verdict is CriticVerdict.CONTRADICTED
    }
    caught = known_contradictions & flagged
    return len(caught) / len(known_contradictions)


def tool_selection_accuracy(*, used: frozenset[str], expected: frozenset[str]) -> float:
    """How closely the capabilities actually called match a labelled scenario's expectation.

    Scored as the Jaccard index between what was called and what the scenario says should
    have been called: 1.0 when the sets match exactly, 0.0 when they share nothing.
    """
    if not used and not expected:
        return 1.0
    return len(used & expected) / len(used | expected)


def schema_validity_rate(*, attempts: int, exhausted: int) -> float:
    """The fraction of agent responses that conformed to their contract.

    Section 14's design target is at least 95% here. ``attempts`` counts every
    ``request_agent_output`` call across a batch; ``exhausted`` counts the ones that
    raised ``SchemaCorrectionExhausted``. Bounded schema correction already absorbs an
    occasional malformed response before it counts against this, so this is a coarser
    signal than any single call succeeding or failing.
    """
    if attempts <= 0:
        raise ValueError("schema_validity_rate needs at least one attempt")
    if exhausted > attempts:
        raise ValueError("exhausted attempts cannot exceed total attempts")
    return (attempts - exhausted) / attempts


def task_completion_rate(*, completed: int, total: int) -> float:
    """The fraction of jobs that reached ``completed`` rather than partial or failed."""
    if total <= 0:
        raise ValueError("task_completion_rate needs at least one job")
    if not 0 <= completed <= total:
        raise ValueError("completed jobs must be between zero and the total")
    return completed / total


def tool_call_validity_rate(*, calls: int, invalid: int) -> float:
    """The fraction of tool calls whose arguments the tool's schema accepted.

    Section 14's design target is at least 95%. A run that made no tool calls has made
    no invalid ones, so it scores 1.0 rather than being undefined.
    """
    if calls < 0 or not 0 <= invalid <= max(calls, 0):
        raise ValueError("invalid calls must be between zero and the number of calls")
    if calls == 0:
        return 1.0
    return (calls - invalid) / calls

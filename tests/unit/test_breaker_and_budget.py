from datetime import timedelta
from uuid import uuid4

import pytest

from research_platform.domain.models import ResearchBudget
from research_platform.mcp.breaker import (
    BudgetExhausted,
    BudgetLedger,
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
    MutableClock,
)

COOLDOWN = timedelta(seconds=30)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def breaker(clock: MutableClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, cooldown=COOLDOWN, clock=clock)


def test_a_new_circuit_is_closed(breaker: CircuitBreaker) -> None:
    assert breaker.state_of("web-research") is CircuitState.CLOSED
    breaker.ensure_closed("web-research")


def test_the_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CircuitBreaker(failure_threshold=0)


def test_failures_below_the_threshold_keep_the_circuit_closed(breaker: CircuitBreaker) -> None:
    breaker.record_failure("web-research")
    breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.CLOSED


def test_reaching_the_threshold_opens_the_circuit(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.OPEN
    with pytest.raises(CircuitOpen) as error:
        breaker.ensure_closed("web-research")
    assert error.value.server == "web-research"


def test_a_success_resets_the_failure_count(breaker: CircuitBreaker) -> None:
    breaker.record_failure("web-research")
    breaker.record_failure("web-research")
    breaker.record_success("web-research")
    breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.CLOSED


def test_an_open_circuit_only_rests_one_server(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        breaker.record_failure("web-research")

    breaker.ensure_closed("github")


def test_the_cooldown_admits_a_single_probe(breaker: CircuitBreaker, clock: MutableClock) -> None:
    for _ in range(3):
        breaker.record_failure("web-research")
    clock.advance(COOLDOWN)

    assert breaker.state_of("web-research") is CircuitState.HALF_OPEN
    breaker.ensure_closed("web-research")


def test_a_failed_probe_reopens_the_circuit_immediately(
    breaker: CircuitBreaker, clock: MutableClock
) -> None:
    for _ in range(3):
        breaker.record_failure("web-research")
    clock.advance(COOLDOWN)
    breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.OPEN


def test_a_successful_probe_closes_the_circuit(
    breaker: CircuitBreaker, clock: MutableClock
) -> None:
    for _ in range(3):
        breaker.record_failure("web-research")
    clock.advance(COOLDOWN)
    breaker.record_success("web-research")

    assert breaker.state_of("web-research") is CircuitState.CLOSED


def test_the_ledger_starts_empty() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()

    assert ledger.usage(job_id).tool_calls == 0
    assert ledger.remaining_calls(job_id, ResearchBudget(max_tool_calls=5)) == 5


def test_reserving_a_call_consumes_the_allowance() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=2)

    assert ledger.reserve_call(job_id, budget) == 1
    assert ledger.reserve_call(job_id, budget) == 2
    assert ledger.remaining_calls(job_id, budget) == 0


def test_an_exhausted_budget_refuses_new_work() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=1)
    ledger.reserve_call(job_id, budget)

    with pytest.raises(BudgetExhausted) as error:
        ledger.reserve_call(job_id, budget)

    assert error.value.limit == 1


def test_budgets_are_counted_per_job() -> None:
    ledger = BudgetLedger()
    budget = ResearchBudget(max_tool_calls=1)
    ledger.reserve_call(uuid4(), budget)

    assert ledger.reserve_call(uuid4(), budget) == 1


def test_usage_is_returned_as_a_copy() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=5)
    ledger.reserve_call(job_id, budget)

    snapshot = ledger.usage(job_id)
    snapshot.tool_calls = 99

    assert ledger.usage(job_id).tool_calls == 1


def test_cost_accumulates_and_rejects_negative_amounts() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()

    ledger.record_cost(job_id, 1.5)
    ledger.record_cost(job_id, 0.5)

    assert ledger.usage(job_id).cost_usd == 2.0
    with pytest.raises(ValueError, match="cannot be negative"):
        ledger.record_cost(job_id, -1)

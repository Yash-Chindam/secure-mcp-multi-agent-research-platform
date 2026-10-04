"""One contract for budgets and rate limits, in process memory and in a real Redis.

Section 6 gives Redis "limits and selected coordination state". What matters about the
shared implementation is exactly what a single process cannot show: two workers spending
against one total, and neither being able to take the last unit twice.

The Redis parameter is skipped unless ``RESEARCH_TEST_REDIS_URL`` is set; CI provides one.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
import redis
from fastmcp import Client
from fastmcp.exceptions import ToolError

from research_platform.composition import (
    build_budget_ledger,
    build_circuit_breaker,
    describe_limits,
)
from research_platform.domain.invocations import ErrorClass
from research_platform.domain.models import JobUsage, ResearchBudget
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal
from research_platform.mcp.breaker import (
    Breaker,
    BudgetExhausted,
    BudgetLedger,
    Budgets,
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
    LimitStoreUnavailable,
    MutableClock,
)
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityDenied, CapabilityGateway, ExecutionRequest
from research_platform.mcp.servers.backends import SourceDocument, StaticWebBackend
from research_platform.mcp.servers.configured import configure_servers
from research_platform.mcp.servers.web_boundary import (
    DomainPolicy,
    RateLimiter,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)
from research_platform.mcp.servers.web_research import (
    WebResearchService,
    build_web_research_server,
)
from research_platform.persistence.redis_state import (
    RedisBudgetLedger,
    RedisCircuitBreaker,
    RedisRateLimiter,
    connect,
)
from research_platform.settings import Settings

pytestmark = pytest.mark.integration

REDIS_URL = os.getenv("RESEARCH_TEST_REDIS_URL")
SKIP_REASON = "set RESEARCH_TEST_REDIS_URL to run the shared limit tests"
requires_redis = pytest.mark.skipif(not REDIS_URL, reason=SKIP_REASON)

UNREACHABLE = "redis://127.0.0.1:1/0"


def client() -> redis.Redis:
    assert REDIS_URL is not None
    return connect(REDIS_URL)


@pytest.fixture(params=["in-memory", "redis"])
def new_ledger(request: pytest.FixtureRequest) -> Callable[[], Budgets]:
    """Builds ledgers that share one store, the way two workers would."""
    if request.param == "in-memory":
        shared = BudgetLedger()
        return lambda: shared
    if not REDIS_URL:
        pytest.skip(SKIP_REASON)
    prefix = f"test-{uuid4().hex}"
    return lambda: RedisBudgetLedger(client(), prefix=prefix)


@pytest.fixture
def ledger(new_ledger: Callable[[], Budgets]) -> Budgets:
    return new_ledger()


# -- budgets ------------------------------------------------------------------------------------


def test_a_job_nothing_was_spent_on_has_spent_nothing(ledger: Budgets) -> None:
    assert ledger.usage(uuid4()) == JobUsage()


def test_tool_calls_are_claimed_one_at_a_time_up_to_the_budget(ledger: Budgets) -> None:
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=2)

    assert ledger.reserve_call(job_id, budget) == 1
    assert ledger.remaining_calls(job_id, budget) == 1
    assert ledger.reserve_call(job_id, budget) == 2

    with pytest.raises(BudgetExhausted, match="reached its 2 tool-call budget"):
        ledger.reserve_call(job_id, budget)
    assert ledger.usage(job_id).tool_calls == 2
    assert ledger.remaining_calls(job_id, budget) == 0


def test_agent_usage_accumulates_and_returns_the_new_total(ledger: Budgets) -> None:
    job_id = uuid4()

    ledger.record_agent_call(job_id, prompt_tokens=100, completion_tokens=20, cost_usd=0.25)
    total = ledger.record_agent_call(
        job_id, prompt_tokens=50, completion_tokens=5, active_seconds=2.5, corrections=1
    )

    expected = JobUsage(
        agent_calls=2,
        schema_corrections=1,
        prompt_tokens=150,
        completion_tokens=25,
        cost_usd=0.25,
        active_seconds=2.5,
    )
    assert total == expected
    assert ledger.usage(job_id) == expected


def test_tool_calls_and_agent_usage_are_counted_on_the_same_job(ledger: Budgets) -> None:
    job_id = uuid4()

    ledger.reserve_call(job_id, ResearchBudget())
    ledger.record_agent_call(job_id, prompt_tokens=10)

    assert ledger.usage(job_id) == JobUsage(tool_calls=1, agent_calls=1, prompt_tokens=10)


@pytest.mark.parametrize(
    ("spent", "kind"),
    [
        ({"prompt_tokens": 600, "completion_tokens": 400}, "token"),
        ({"cost_usd": 2.0}, "USD cost"),
        ({"active_seconds": 60.0}, "second runtime"),
    ],
)
def test_a_spent_budget_refuses_new_agent_work(
    ledger: Budgets, spent: dict[str, Any], kind: str
) -> None:
    budget = ResearchBudget(max_tokens=1_000, max_cost_usd=2.0, max_runtime_seconds=60)
    job_id = uuid4()
    ledger.ensure_within(job_id, budget)

    ledger.record_agent_call(job_id, **spent)

    with pytest.raises(BudgetExhausted) as refused:
        ledger.ensure_within(job_id, budget)
    assert refused.value.kind == kind


def test_negative_usage_is_refused(ledger: Budgets) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ledger.record_agent_call(uuid4(), completion_tokens=-1)


def test_one_job_never_spends_from_another(ledger: Budgets) -> None:
    budget = ResearchBudget(max_tool_calls=1)
    first, second = uuid4(), uuid4()
    ledger.reserve_call(first, budget)

    assert ledger.reserve_call(second, budget) == 1


def test_two_workers_spend_against_one_total(new_ledger: Callable[[], Budgets]) -> None:
    """The point of sharing: a second worker does not hand the job a second allowance."""
    one_worker, another_worker = new_ledger(), new_ledger()
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=3, max_tokens=1_000)

    one_worker.reserve_call(job_id, budget)
    another_worker.reserve_call(job_id, budget)
    one_worker.reserve_call(job_id, budget)
    one_worker.record_agent_call(job_id, prompt_tokens=1_000)

    with pytest.raises(BudgetExhausted, match="tool-call"):
        another_worker.reserve_call(job_id, budget)
    with pytest.raises(BudgetExhausted, match="token"):
        another_worker.ensure_within(job_id, budget)
    assert another_worker.usage(job_id).tool_calls == 3


def test_concurrent_claims_never_exceed_the_budget(new_ledger: Callable[[], Budgets]) -> None:
    """Forty callers race for ten calls; exactly ten win, whichever store counts them."""
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=10)

    def claim(_: int) -> bool:
        try:
            new_ledger().reserve_call(job_id, budget)
        except BudgetExhausted:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        won = list(pool.map(claim, range(40)))

    assert sum(won) == 10
    assert new_ledger().usage(job_id).tool_calls == 10


# -- rate limits ----------------------------------------------------------------------------------


@pytest.fixture(params=["in-memory", "redis"])
def new_limiter(request: pytest.FixtureRequest) -> Callable[..., RateLimiter]:
    """Builds limiters over one shared window, the way two server replicas would."""
    clock = MutableClock()
    if request.param == "in-memory":
        shared: dict[int, SlidingWindowRateLimiter] = {}

        def in_memory(limit: int = 2) -> RateLimiter:
            return shared.setdefault(
                limit,
                SlidingWindowRateLimiter(limit=limit, window=timedelta(seconds=60), clock=clock),
            )

        in_memory.clock = clock  # type: ignore[attr-defined]
        return in_memory
    if not REDIS_URL:
        pytest.skip(SKIP_REASON)
    prefix = f"test-{uuid4().hex}"

    def shared_through_redis(limit: int = 2) -> RateLimiter:
        return RedisRateLimiter(
            client(), limit=limit, window=timedelta(seconds=60), clock=clock, prefix=prefix
        )

    shared_through_redis.clock = clock  # type: ignore[attr-defined]
    return shared_through_redis


def test_requests_are_allowed_up_to_the_limit_and_then_refused(
    new_limiter: Callable[..., RateLimiter],
) -> None:
    limiter = new_limiter()

    assert limiter.remaining("acme|vendor.test") == 2
    assert limiter.acquire("acme|vendor.test") == 1
    assert limiter.acquire("acme|vendor.test") == 0

    with pytest.raises(RateLimitExceeded, match="exceeded 2 requests per 60s"):
        limiter.acquire("acme|vendor.test")
    assert limiter.remaining("acme|vendor.test") == 0


def test_a_slot_frees_once_its_request_leaves_the_window(
    new_limiter: Callable[..., RateLimiter],
) -> None:
    limiter = new_limiter()
    limiter.acquire("acme|vendor.test")
    new_limiter.clock.advance(timedelta(seconds=30))  # type: ignore[attr-defined]
    limiter.acquire("acme|vendor.test")

    new_limiter.clock.advance(timedelta(seconds=31))  # type: ignore[attr-defined]

    assert limiter.remaining("acme|vendor.test") == 1
    assert limiter.acquire("acme|vendor.test") == 0


def test_one_tenant_cannot_use_up_anothers_allowance(
    new_limiter: Callable[..., RateLimiter],
) -> None:
    limiter = new_limiter()
    limiter.acquire("tenant-a|vendor.test")
    limiter.acquire("tenant-a|vendor.test")

    assert limiter.acquire("tenant-b|vendor.test") == 1


def test_two_replicas_share_one_window(new_limiter: Callable[..., RateLimiter]) -> None:
    one_replica, another_replica = new_limiter(), new_limiter()
    one_replica.acquire("acme|vendor.test")
    another_replica.acquire("acme|vendor.test")

    with pytest.raises(RateLimitExceeded):
        one_replica.acquire("acme|vendor.test")


def test_concurrent_requests_never_exceed_the_limit(
    new_limiter: Callable[..., RateLimiter],
) -> None:
    def request(_: int) -> bool:
        try:
            new_limiter(5).acquire("acme|vendor.test")
        except RateLimitExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(pool.map(request, range(30)))

    assert sum(allowed) == 5


# -- circuit breaking -----------------------------------------------------------------------------

COOLDOWN = timedelta(seconds=30)


@pytest.fixture(params=["in-memory", "redis"])
def new_breaker(request: pytest.FixtureRequest) -> Callable[[], Breaker]:
    """Builds breakers over one shared state, the way two workers would."""
    clock = MutableClock()
    if request.param == "in-memory":
        shared = CircuitBreaker(failure_threshold=3, cooldown=COOLDOWN, clock=clock)

        def in_memory() -> Breaker:
            return shared

        in_memory.clock = clock  # type: ignore[attr-defined]
        return in_memory
    if not REDIS_URL:
        pytest.skip(SKIP_REASON)
    prefix = f"test-{uuid4().hex}"

    def shared_through_redis() -> Breaker:
        return RedisCircuitBreaker(
            client(), failure_threshold=3, cooldown=COOLDOWN, clock=clock, prefix=prefix
        )

    shared_through_redis.clock = clock  # type: ignore[attr-defined]
    return shared_through_redis


def test_a_server_nothing_is_known_about_is_closed(new_breaker: Callable[[], Breaker]) -> None:
    breaker = new_breaker()

    assert breaker.state_of("web-research") is CircuitState.CLOSED
    breaker.ensure_closed("web-research")
    assert breaker.states() == {}


def test_failures_below_the_threshold_leave_the_circuit_closed(
    new_breaker: Callable[[], Breaker],
) -> None:
    breaker = new_breaker()
    breaker.record_failure("web-research")
    breaker.record_failure("web-research")

    breaker.ensure_closed("web-research")
    assert breaker.states() == {"web-research": CircuitState.CLOSED}


def test_reaching_the_threshold_opens_the_circuit_until_the_cooldown_ends(
    new_breaker: Callable[[], Breaker],
) -> None:
    breaker = new_breaker()
    for _ in range(3):
        breaker.record_failure("web-research")

    with pytest.raises(CircuitOpen) as rested:
        breaker.ensure_closed("web-research")
    assert rested.value.until == new_breaker.clock() + COOLDOWN  # type: ignore[attr-defined]

    new_breaker.clock.advance(COOLDOWN)  # type: ignore[attr-defined]

    assert breaker.state_of("web-research") is CircuitState.HALF_OPEN
    breaker.ensure_closed("web-research")


def test_a_success_closes_the_circuit_and_forgets_earlier_failures(
    new_breaker: Callable[[], Breaker],
) -> None:
    breaker = new_breaker()
    breaker.record_failure("web-research")
    breaker.record_failure("web-research")
    breaker.record_success("web-research")

    breaker.record_failure("web-research")
    breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.CLOSED


def test_a_failed_probe_reopens_the_circuit_at_once(new_breaker: Callable[[], Breaker]) -> None:
    breaker = new_breaker()
    for _ in range(3):
        breaker.record_failure("web-research")
    new_breaker.clock.advance(COOLDOWN)  # type: ignore[attr-defined]
    assert breaker.state_of("web-research") is CircuitState.HALF_OPEN

    breaker.record_failure("web-research")

    assert breaker.state_of("web-research") is CircuitState.OPEN


def test_a_successful_probe_closes_the_circuit(new_breaker: Callable[[], Breaker]) -> None:
    breaker = new_breaker()
    for _ in range(3):
        breaker.record_failure("web-research")
    new_breaker.clock.advance(COOLDOWN)  # type: ignore[attr-defined]

    breaker.record_success("web-research")

    assert breaker.states() == {"web-research": CircuitState.CLOSED}


def test_one_failing_server_does_not_rest_another(new_breaker: Callable[[], Breaker]) -> None:
    breaker = new_breaker()
    for _ in range(3):
        breaker.record_failure("github")
    breaker.record_success("web-research")

    breaker.ensure_closed("web-research")
    assert breaker.states() == {"github": CircuitState.OPEN, "web-research": CircuitState.CLOSED}


def test_a_server_one_worker_saw_fail_is_rested_by_every_worker(
    new_breaker: Callable[[], Breaker],
) -> None:
    """Three failures in total open the circuit, whichever workers saw them."""
    one_worker, another_worker = new_breaker(), new_breaker()
    one_worker.record_failure("web-research")
    another_worker.record_failure("web-research")
    one_worker.record_failure("web-research")

    with pytest.raises(CircuitOpen):
        another_worker.ensure_closed("web-research")


# -- Redis specifically ---------------------------------------------------------------------------


@requires_redis
def test_a_limiter_with_no_clock_of_its_own_uses_redis_time() -> None:
    limiter = RedisRateLimiter(
        client(), limit=1, window=timedelta(seconds=60), prefix=f"test-{uuid4().hex}"
    )

    assert limiter.acquire("acme|vendor.test") == 0
    with pytest.raises(RateLimitExceeded):
        limiter.acquire("acme|vendor.test")


def test_a_limit_must_allow_at_least_one_request() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        RedisRateLimiter(redis.Redis.from_url(UNREACHABLE), limit=0)


@requires_redis
def test_a_jobs_usage_expires_long_after_it_was_last_spent_against() -> None:
    prefix = f"test-{uuid4().hex}"
    job_id = uuid4()
    RedisBudgetLedger(client(), prefix=prefix).reserve_call(job_id, ResearchBudget())

    ttl = client().ttl(f"{prefix}:budget:{job_id}")

    assert 29 * 86_400 < int(ttl) <= 30 * 86_400  # type: ignore[arg-type]


def test_a_redis_that_cannot_be_reached_is_a_startup_error() -> None:
    with pytest.raises(LimitStoreUnavailable, match="could not be reached"):
        connect(UNREACHABLE)


def unreachable_ledger() -> RedisBudgetLedger:
    return RedisBudgetLedger(redis.Redis.from_url(UNREACHABLE, decode_responses=True))


def test_a_ledger_that_cannot_reach_its_store_refuses_rather_than_guessing() -> None:
    ledger = unreachable_ledger()
    job_id = uuid4()

    with pytest.raises(LimitStoreUnavailable):
        ledger.reserve_call(job_id, ResearchBudget())
    with pytest.raises(LimitStoreUnavailable):
        ledger.ensure_within(job_id, ResearchBudget())
    with pytest.raises(LimitStoreUnavailable):
        ledger.record_agent_call(job_id, prompt_tokens=1)


def test_a_limiter_that_cannot_reach_its_store_refuses_rather_than_allowing() -> None:
    limiter = RedisRateLimiter(redis.Redis.from_url(UNREACHABLE, decode_responses=True), limit=5)

    with pytest.raises(LimitStoreUnavailable):
        limiter.acquire("acme|vendor.test")
    with pytest.raises(LimitStoreUnavailable):
        limiter.remaining("acme|vendor.test")


def test_a_tool_call_that_cannot_be_counted_is_denied_before_it_is_made() -> None:
    """Fail closed at the gateway: no budget accounting, no call."""

    class NeverCalled:
        def execute(self, _request: ExecutionRequest) -> str:
            raise AssertionError("an unmetered call reached the MCP server")

    recorded = []
    gateway = CapabilityGateway(
        registry=default_registry(),
        executor=NeverCalled(),
        budgets=unreachable_ledger(),
        audit=recorded.append,
    )

    with pytest.raises(CapabilityDenied, match="shared limit store is unavailable"):
        gateway.invoke(
            principal=Principal(tenant_id="acme", subject_id="job:1").for_agent(
                AgentRole.RESEARCHER
            ),
            job_id=uuid4(),
            task_id=uuid4(),
            server="web-research",
            capability_name="fetch",
            arguments={"url": "https://vendor.test/pricing"},
            budget=ResearchBudget(),
        )

    assert recorded[0].error_class is ErrorClass.UPSTREAM_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("fetch", {"tenant_id": "acme", "url": "https://vendor.test/pricing"}),
        ("search", {"tenant_id": "acme", "query": "vendor pricing"}),
    ],
)
async def test_the_web_server_refuses_a_request_it_cannot_rate_limit(
    tool: str, arguments: dict[str, str]
) -> None:
    page = SourceDocument(url="https://vendor.test/pricing", title="Pricing", text="20 USD")
    server = build_web_research_server(
        WebResearchService(
            backend=StaticWebBackend(documents={page.url: page}),
            policy=DomainPolicy(domains=frozenset({"vendor.test"})),
            limiter=RedisRateLimiter(
                redis.Redis.from_url(UNREACHABLE, decode_responses=True), limit=5
            ),
        )
    )

    async with Client(server) as mcp:
        with pytest.raises(ToolError, match="rate limit unavailable"):
            await mcp.call_tool(tool, arguments)


def unreachable_breaker() -> RedisCircuitBreaker:
    return RedisCircuitBreaker(redis.Redis.from_url(UNREACHABLE, decode_responses=True))


def test_a_breaker_that_cannot_read_its_state_does_not_let_the_call_through() -> None:
    with pytest.raises(LimitStoreUnavailable):
        unreachable_breaker().ensure_closed("web-research")


def test_a_breaker_that_cannot_record_an_outcome_does_not_fail_the_call_it_followed() -> None:
    breaker = unreachable_breaker()

    breaker.record_success("web-research")
    breaker.record_failure("web-research")

    assert breaker.states() == {}


def test_a_breaker_needs_a_positive_threshold() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RedisCircuitBreaker(redis.Redis.from_url(UNREACHABLE), failure_threshold=0)


def test_a_call_is_denied_when_the_shared_breaker_cannot_be_read() -> None:
    class NeverCalled:
        def execute(self, _request: ExecutionRequest) -> str:
            raise AssertionError("the call went ahead without knowing the circuit state")

    gateway = CapabilityGateway(
        registry=default_registry(), executor=NeverCalled(), breaker=unreachable_breaker()
    )

    with pytest.raises(CapabilityDenied, match="shared limit store is unavailable"):
        gateway.invoke(
            principal=Principal(tenant_id="acme", subject_id="job:1").for_agent(
                AgentRole.RESEARCHER
            ),
            job_id=uuid4(),
            task_id=uuid4(),
            server="web-research",
            capability_name="fetch",
            arguments={"url": "https://vendor.test/pricing"},
            budget=ResearchBudget(),
        )


@requires_redis
def test_a_deployment_with_redis_shares_its_circuit_state() -> None:
    assert isinstance(build_circuit_breaker(Settings(redis_url=REDIS_URL)), RedisCircuitBreaker)
    assert isinstance(build_circuit_breaker(Settings(redis_url=None)), CircuitBreaker)


# -- wiring -------------------------------------------------------------------------------------


def test_a_deployment_with_no_redis_counts_limits_per_process() -> None:
    settings = Settings(redis_url=None)

    assert isinstance(build_budget_ledger(settings), BudgetLedger)
    assert "per-process" in describe_limits(settings)


@requires_redis
def test_a_deployment_with_redis_shares_its_budgets_and_its_web_rate_limit() -> None:
    settings = Settings(
        redis_url=REDIS_URL, web_allowed_domains="vendor.test", web_requests_per_minute=7
    )

    ledger = build_budget_ledger(settings)
    servers = configure_servers(settings)

    assert isinstance(ledger, RedisBudgetLedger)
    assert "shared through Redis" in describe_limits(settings)
    assert "web-research" in servers.targets

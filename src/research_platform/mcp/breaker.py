"""Per-server circuit breaking and per-job budget accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import UUID

from research_platform.domain.models import JobUsage, ResearchBudget, utc_now

Clock = Callable[[], datetime]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(RuntimeError):
    def __init__(self, server: str, until: datetime) -> None:
        super().__init__(f"circuit for {server} is open until {until.isoformat()}")
        self.server = server
        self.until = until


class BudgetExhausted(RuntimeError):
    """A job has spent one of its four budgets, so no new work is started for it."""

    def __init__(self, job_id: UUID, limit: float, kind: str = "tool-call") -> None:
        super().__init__(f"job {job_id} reached its {limit:g} {kind} budget")
        self.job_id = job_id
        self.limit = limit
        self.kind = kind


@dataclass
class _Circuit:
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at: datetime | None = None


class CircuitBreaker:
    """Stop calling an MCP server that is repeatedly failing, and probe before resuming."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(seconds=30),
        clock: Clock = utc_now,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}
        self._lock = RLock()

    def state_of(self, server: str) -> CircuitState:
        with self._lock:
            return self._refresh(server).state

    def states(self) -> dict[str, CircuitState]:
        """The current state of every server this breaker has seen a call for."""
        with self._lock:
            return {server: self._refresh(server).state for server in list(self._circuits)}

    def ensure_closed(self, server: str) -> None:
        """Raise when the server is being rested, without consuming an upstream call."""
        with self._lock:
            circuit = self._refresh(server)
            if circuit.state is CircuitState.OPEN:
                assert circuit.opened_at is not None
                raise CircuitOpen(server, circuit.opened_at + self._cooldown)

    def record_success(self, server: str) -> None:
        with self._lock:
            self._circuits[server] = _Circuit()

    def record_failure(self, server: str) -> None:
        with self._lock:
            circuit = self._refresh(server)
            if circuit.state is CircuitState.HALF_OPEN:
                self._circuits[server] = _Circuit(
                    failures=self._failure_threshold,
                    state=CircuitState.OPEN,
                    opened_at=self._clock(),
                )
                return
            circuit.failures += 1
            if circuit.failures >= self._failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = self._clock()
            self._circuits[server] = circuit

    def _refresh(self, server: str) -> _Circuit:
        circuit = self._circuits.setdefault(server, _Circuit())
        if circuit.state is CircuitState.OPEN:
            assert circuit.opened_at is not None
            if self._clock() - circuit.opened_at >= self._cooldown:
                circuit.state = CircuitState.HALF_OPEN
        return circuit


class LimitStoreUnavailable(RuntimeError):
    """The store that counts budgets or request rates could not be reached.

    Raised instead of guessing: work that cannot be accounted for is refused.
    """


class Budgets(Protocol):
    """What a budget ledger must do, wherever it keeps its counts."""

    def usage(self, job_id: UUID) -> JobUsage: ...

    def remaining_calls(self, job_id: UUID, budget: ResearchBudget) -> int: ...

    def reserve_call(self, job_id: UUID, budget: ResearchBudget) -> int: ...

    def ensure_within(self, job_id: UUID, budget: ResearchBudget) -> None: ...

    def record_agent_call(
        self,
        job_id: UUID,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        active_seconds: float = 0.0,
        corrections: int = 0,
    ) -> JobUsage: ...


class BudgetLedger:
    """What each job has spent, so an exhausted budget stops new work (section 12).

    Tool calls are claimed before a call is made, so that budget is never exceeded.
    Tokens, cost and working time are only known once an agent call has finished, so
    they are checked before the next one starts: a job may overshoot by the call that
    crossed the line, and is then refused any further agent work.

    This one counts in process memory, which is right for a single worker and for tests.
    ``research_platform.persistence.redis_state.RedisBudgetLedger`` keeps the same
    counts in Redis so every worker spends against one total.
    """

    def __init__(self) -> None:
        self._usage: dict[UUID, JobUsage] = {}
        self._lock = RLock()

    def usage(self, job_id: UUID) -> JobUsage:
        with self._lock:
            return self._usage.get(job_id, JobUsage()).model_copy()

    def remaining_calls(self, job_id: UUID, budget: ResearchBudget) -> int:
        return max(0, budget.max_tool_calls - self.usage(job_id).tool_calls)

    def reserve_call(self, job_id: UUID, budget: ResearchBudget) -> int:
        """Claim one tool call, refusing once the job has spent its allowance."""
        with self._lock:
            recorded = self._usage.setdefault(job_id, JobUsage())
            if recorded.tool_calls >= budget.max_tool_calls:
                raise BudgetExhausted(job_id, budget.max_tool_calls)
            recorded.tool_calls += 1
            return recorded.tool_calls

    def ensure_within(self, job_id: UUID, budget: ResearchBudget) -> None:
        """Refuse new agent work for a job that has spent its tokens, money or time."""
        spent = self.usage(job_id)
        if spent.total_tokens >= budget.max_tokens:
            raise BudgetExhausted(job_id, budget.max_tokens, "token")
        if spent.cost_usd >= budget.max_cost_usd:
            raise BudgetExhausted(job_id, budget.max_cost_usd, "USD cost")
        if spent.active_seconds >= budget.max_runtime_seconds:
            raise BudgetExhausted(job_id, budget.max_runtime_seconds, "second runtime")

    def record_agent_call(
        self,
        job_id: UUID,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        active_seconds: float = 0.0,
        corrections: int = 0,
    ) -> JobUsage:
        """Add what one finished agent call spent, and return the job's new total."""
        if min(prompt_tokens, completion_tokens, cost_usd, active_seconds, corrections) < 0:
            raise ValueError("recorded usage cannot be negative")
        with self._lock:
            recorded = self._usage.setdefault(job_id, JobUsage())
            recorded.agent_calls += 1
            recorded.schema_corrections += corrections
            recorded.prompt_tokens += prompt_tokens
            recorded.completion_tokens += completion_tokens
            recorded.cost_usd += cost_usd
            recorded.active_seconds += active_seconds
            return recorded.model_copy()

    def record_cost(self, job_id: UUID, cost_usd: float) -> None:
        if cost_usd < 0:
            raise ValueError("recorded cost cannot be negative")
        with self._lock:
            self._usage.setdefault(job_id, JobUsage()).cost_usd += cost_usd


@dataclass
class MutableClock:
    """A clock tests can advance without sleeping."""

    now: datetime = field(default_factory=utc_now)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta

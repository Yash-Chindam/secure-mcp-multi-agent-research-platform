"""Per-server circuit breaking and per-job budget accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from uuid import UUID

from research_platform.domain.models import ResearchBudget, utc_now

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
    def __init__(self, job_id: UUID, limit: int) -> None:
        super().__init__(f"job {job_id} reached its {limit} tool-call budget")
        self.job_id = job_id
        self.limit = limit


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


@dataclass
class BudgetUsage:
    tool_calls: int = 0
    cost_usd: float = 0.0


class BudgetLedger:
    """Count tool calls per job so an exhausted budget stops new work (section 12)."""

    def __init__(self) -> None:
        self._usage: dict[UUID, BudgetUsage] = {}
        self._lock = RLock()

    def usage(self, job_id: UUID) -> BudgetUsage:
        with self._lock:
            recorded = self._usage.get(job_id, BudgetUsage())
        return BudgetUsage(tool_calls=recorded.tool_calls, cost_usd=recorded.cost_usd)

    def remaining_calls(self, job_id: UUID, budget: ResearchBudget) -> int:
        return max(0, budget.max_tool_calls - self.usage(job_id).tool_calls)

    def reserve_call(self, job_id: UUID, budget: ResearchBudget) -> int:
        """Claim one tool call, refusing once the job has spent its allowance."""
        with self._lock:
            recorded = self._usage.setdefault(job_id, BudgetUsage())
            if recorded.tool_calls >= budget.max_tool_calls:
                raise BudgetExhausted(job_id, budget.max_tool_calls)
            recorded.tool_calls += 1
            return recorded.tool_calls

    def record_cost(self, job_id: UUID, cost_usd: float) -> None:
        if cost_usd < 0:
            raise ValueError("recorded cost cannot be negative")
        with self._lock:
            self._usage.setdefault(job_id, BudgetUsage()).cost_usd += cost_usd


@dataclass
class MutableClock:
    """A clock tests can advance without sleeping."""

    now: datetime = field(default_factory=utc_now)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta

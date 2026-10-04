"""Limits every worker shares: per-job budgets and per-tenant request rates, in Redis.

Section 6 gives Redis "locks, limits and selected coordination state". A budget counted
inside one worker process is not a budget once a second worker exists: each would allow
the job its full allowance. The same is true of a request rate. Both live here, with
each check-and-claim done in one Lua script so two workers can never both take the last
unit.

Both fail closed. If Redis cannot be reached, ``LimitStoreUnavailable`` is raised and the
caller refuses the work it could not account for, rather than doing it unmetered.

The circuit breaker is here as well: a server one worker has seen fail three times is
rested by every worker, instead of each one spending its own three failures to find out.
Its bookkeeping is the one thing that does not fail closed - see ``RedisCircuitBreaker``.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import redis

from research_platform.domain.models import JobUsage, ResearchBudget
from research_platform.mcp.breaker import (
    BudgetExhausted,
    CircuitOpen,
    CircuitState,
    LimitStoreUnavailable,
)
from research_platform.mcp.servers.web_boundary import RateLimitExceeded

Clock = Callable[[], datetime]

logger = logging.getLogger(__name__)

USAGE_TTL = timedelta(days=30)
"""How long a job's usage is kept after it was last added to.

Long enough to outlast a job parked for a reviewer; refreshed on every write, so a job
still being worked on never loses its count.
"""

_RESERVE_CALL = """
local used = tonumber(redis.call('HGET', KEYS[1], 'tool_calls') or '0')
if used >= tonumber(ARGV[1]) then
    return -1
end
local claimed = redis.call('HINCRBY', KEYS[1], 'tool_calls', 1)
redis.call('EXPIRE', KEYS[1], ARGV[2])
return claimed
"""

_RECORD_AGENT_CALL = """
redis.call('HINCRBY', KEYS[1], 'agent_calls', 1)
redis.call('HINCRBY', KEYS[1], 'schema_corrections', ARGV[1])
redis.call('HINCRBY', KEYS[1], 'prompt_tokens', ARGV[2])
redis.call('HINCRBY', KEYS[1], 'completion_tokens', ARGV[3])
redis.call('HINCRBYFLOAT', KEYS[1], 'cost_usd', ARGV[4])
redis.call('HINCRBYFLOAT', KEYS[1], 'active_seconds', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[6])
return redis.call('HGETALL', KEYS[1])
"""

_ACQUIRE = """
local now = tonumber(ARGV[1])
if now == nil then
    local time = redis.call('TIME')
    now = tonumber(time[1]) * 1000000 + tonumber(time[2])
end
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local used = redis.call('ZCARD', KEYS[1])
if used >= tonumber(ARGV[3]) then
    return -1
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], math.ceil(window / 1000))
return tonumber(ARGV[3]) - used - 1
"""

_REMAINING = """
local now = tonumber(ARGV[1])
if now == nil then
    local time = redis.call('TIME')
    now = tonumber(time[1]) * 1000000 + tonumber(time[2])
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[2]))
return redis.call('ZCARD', KEYS[1])
"""


def connect(url: str) -> redis.Redis:
    """A client for ``url``, verified reachable so a bad address fails at startup."""
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
    try:
        client.ping()
    except redis.RedisError as error:
        raise LimitStoreUnavailable(f"Redis could not be reached: {error}") from error
    return client


def _unavailable(error: redis.RedisError) -> LimitStoreUnavailable:
    return LimitStoreUnavailable(f"the shared limit store is unavailable: {error}")


def _usage_from(fields: dict[str, str]) -> JobUsage:
    return JobUsage(
        tool_calls=int(fields.get("tool_calls", 0)),
        agent_calls=int(fields.get("agent_calls", 0)),
        schema_corrections=int(fields.get("schema_corrections", 0)),
        prompt_tokens=int(fields.get("prompt_tokens", 0)),
        completion_tokens=int(fields.get("completion_tokens", 0)),
        cost_usd=float(fields.get("cost_usd", 0.0)),
        active_seconds=float(fields.get("active_seconds", 0.0)),
    )


class RedisBudgetLedger:
    """``BudgetLedger`` whose counts are shared by every worker."""

    def __init__(self, client: redis.Redis, *, prefix: str = "research") -> None:
        self._client = client
        self._prefix = prefix
        self._reserve = client.register_script(_RESERVE_CALL)
        self._record = client.register_script(_RECORD_AGENT_CALL)

    def _key(self, job_id: UUID) -> str:
        return f"{self._prefix}:budget:{job_id}"

    def usage(self, job_id: UUID) -> JobUsage:
        try:
            fields: Any = self._client.hgetall(self._key(job_id))
        except redis.RedisError as error:
            raise _unavailable(error) from error
        return _usage_from(fields)

    def remaining_calls(self, job_id: UUID, budget: ResearchBudget) -> int:
        return max(0, budget.max_tool_calls - self.usage(job_id).tool_calls)

    def reserve_call(self, job_id: UUID, budget: ResearchBudget) -> int:
        """Claim one tool call. The check and the claim are one atomic step."""
        try:
            claimed = int(
                self._reserve(
                    keys=[self._key(job_id)],
                    args=[budget.max_tool_calls, int(USAGE_TTL.total_seconds())],
                )
            )
        except redis.RedisError as error:
            raise _unavailable(error) from error
        if claimed < 0:
            raise BudgetExhausted(job_id, budget.max_tool_calls)
        return claimed

    def ensure_within(self, job_id: UUID, budget: ResearchBudget) -> None:
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
        if min(prompt_tokens, completion_tokens, cost_usd, active_seconds, corrections) < 0:
            raise ValueError("recorded usage cannot be negative")
        try:
            flat: Any = self._record(
                keys=[self._key(job_id)],
                args=[
                    corrections,
                    prompt_tokens,
                    completion_tokens,
                    repr(float(cost_usd)),
                    repr(float(active_seconds)),
                    int(USAGE_TTL.total_seconds()),
                ],
            )
        except redis.RedisError as error:
            raise _unavailable(error) from error
        return _usage_from(dict(zip(flat[::2], flat[1::2], strict=True)))


class RedisRateLimiter:
    """``SlidingWindowRateLimiter`` whose window is shared by every server replica.

    Time is Redis's own unless a clock is supplied, so replicas whose clocks disagree
    still agree on what falls inside the window.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        limit: int,
        window: timedelta = timedelta(seconds=60),
        clock: Clock | None = None,
        prefix: str = "research",
    ) -> None:
        if limit < 1:
            raise ValueError("a rate limit must allow at least one request")
        self.limit = limit
        self.window = window
        self._client = client
        self._clock = clock
        self._prefix = prefix
        self._acquire = client.register_script(_ACQUIRE)
        self._remaining = client.register_script(_REMAINING)

    def _key(self, key: str) -> str:
        return f"{self._prefix}:rate:{key}"

    def _now(self) -> str:
        """The caller's time in microseconds, or empty to have Redis supply its own."""
        return "" if self._clock is None else str(int(self._clock().timestamp() * 1_000_000))

    def _window(self) -> int:
        return int(self.window.total_seconds() * 1_000_000)

    def remaining(self, key: str) -> int:
        try:
            used = int(self._remaining(keys=[self._key(key)], args=[self._now(), self._window()]))
        except redis.RedisError as error:
            raise _unavailable(error) from error
        return max(0, self.limit - used)

    def acquire(self, key: str) -> int:
        """Claim one request slot, refusing once the shared window is full."""
        try:
            left = int(
                self._acquire(
                    keys=[self._key(key)],
                    args=[self._now(), self._window(), self.limit, secrets.token_hex(8)],
                )
            )
        except redis.RedisError as error:
            raise _unavailable(error) from error
        if left < 0:
            raise RateLimitExceeded(key, self.limit, self.window)
        return left


_NOW = """
local now = tonumber(ARGV[1])
if now == nil then
    local time = redis.call('TIME')
    now = tonumber(time[1]) * 1000000 + tonumber(time[2])
end
"""

_REFRESH = (
    _NOW
    + """
local state = redis.call('HGET', KEYS[1], 'state') or 'closed'
local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
if state == 'open' and now - opened >= tonumber(ARGV[2]) then
    state = 'half_open'
    redis.call('HSET', KEYS[1], 'state', state)
end
"""
)

_CIRCUIT_STATE = _REFRESH + "return {state, string.format('%.0f', opened)}"

_CIRCUIT_FAILURE = (
    _REFRESH
    + """
local threshold = tonumber(ARGV[3])
if state == 'half_open' then
    redis.call(
        'HSET', KEYS[1], 'state', 'open', 'failures', threshold,
        'opened_at', string.format('%.0f', now)
    )
else
    local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
    if failures >= threshold then
        redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', string.format('%.0f', now))
    end
end
redis.call('SADD', KEYS[2], ARGV[4])
redis.call('EXPIRE', KEYS[1], 86400)
return redis.call('HGET', KEYS[1], 'state') or 'closed'
"""
)


class RedisCircuitBreaker:
    """``CircuitBreaker`` whose state is shared by every worker.

    Reading the state fails closed: a worker that cannot learn whether a server is being
    rested does not call it. Recording an outcome does not. By then the call has already
    happened, and losing its result because the bookkeeping failed would be worse than a
    missed count, so a failed write is logged and the outcome of the call stands.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(seconds=30),
        clock: Clock | None = None,
        prefix: str = "research",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._client = client
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock
        self._prefix = prefix
        self._state = client.register_script(_CIRCUIT_STATE)
        self._failure = client.register_script(_CIRCUIT_FAILURE)

    def _key(self, server: str) -> str:
        return f"{self._prefix}:circuit:{server}"

    @property
    def _servers(self) -> str:
        return f"{self._prefix}:circuits"

    def _now(self) -> str:
        return "" if self._clock is None else str(int(self._clock().timestamp() * 1_000_000))

    def _cooldown_us(self) -> int:
        return int(self._cooldown.total_seconds() * 1_000_000)

    def _read(self, server: str) -> tuple[CircuitState, int]:
        try:
            state, opened = self._state(
                keys=[self._key(server)], args=[self._now(), self._cooldown_us()]
            )
        except redis.RedisError as error:
            raise _unavailable(error) from error
        return CircuitState(state), int(float(opened))

    def state_of(self, server: str) -> CircuitState:
        return self._read(server)[0]

    def states(self) -> dict[str, CircuitState]:
        """Every server a call has been recorded for. Empty if the store cannot be read."""
        try:
            servers: Any = self._client.smembers(self._servers)
            return {server: self.state_of(server) for server in sorted(servers)}
        except (redis.RedisError, LimitStoreUnavailable):
            return {}

    def ensure_closed(self, server: str) -> None:
        state, opened = self._read(server)
        if state is CircuitState.OPEN:
            until = datetime.fromtimestamp(opened / 1_000_000, tz=UTC) + self._cooldown
            raise CircuitOpen(server, until)

    def record_success(self, server: str) -> None:
        try:
            pipeline = self._client.pipeline()
            pipeline.delete(self._key(server))
            pipeline.sadd(self._servers, server)
            pipeline.execute()
        except redis.RedisError:
            logger.warning("could not record a successful call to %s", server, exc_info=True)

    def record_failure(self, server: str) -> None:
        try:
            self._failure(
                keys=[self._key(server), self._servers],
                args=[self._now(), self._cooldown_us(), self._threshold, server],
            )
        except redis.RedisError:
            logger.warning("could not record a failed call to %s", server, exc_info=True)

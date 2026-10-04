"""The section 13 metrics, as one small typed wrapper around an OpenTelemetry meter.

Section 13 names what to track without naming instruments, so this module is the one
place that decides the instrument for each: a counter for something that only ever goes
up (calls, denials, injection flags, unsupported citations), a histogram for something
whose distribution matters (latency, approval wait time), an up-down counter for
something that also goes back down (active jobs). Call sites read as what happened, not
as the OpenTelemetry API. Circuit state is an observable gauge read from the breaker
itself when metrics are collected, so it cannot drift from the state calls are actually
being refused under.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from research_platform.mcp.breaker import CircuitState
from research_platform.settings import Settings

SERVICE = "secure-mcp-research-platform"


@dataclass(frozen=True)
class MetricsSetup:
    """What metrics were configured with, for a startup log line."""

    provider: MeterProvider
    exports: bool

    @property
    def description(self) -> str:
        if self.exports:
            return "metrics exported over OTLP"
        return "metrics recorded but not exported (no OTEL exporter endpoint configured)"


def configure_metrics(
    settings: Settings, *, extra_reader: MetricReader | None = None
) -> MetricsSetup:
    """Build the process-wide meter provider and register it as the global default.

    ``extra_reader`` lets a test attach an in-memory reader without also standing up a
    real OTLP collector; production code never passes it.
    """
    readers: list[MetricReader] = []
    exports = bool(settings.otel_exporter_otlp_endpoint)
    if exports:
        exporter = OTLPMetricExporter(endpoint=settings.otlp_signal_endpoint("metrics"))
        readers.append(PeriodicExportingMetricReader(exporter))
    if extra_reader is not None:
        readers.append(extra_reader)
    provider = MeterProvider(
        resource=Resource.create({SERVICE_NAME: SERVICE}), metric_readers=readers
    )
    otel_metrics.set_meter_provider(provider)
    return MetricsSetup(provider=provider, exports=exports)


class PlatformMetrics:
    """The section 13 instruments, built once from a meter.

    Safe to construct before ``configure_metrics`` runs: OpenTelemetry's default meter
    provider hands back instruments that record without exporting, so a component built
    before startup configuration completes never has to guard against a missing meter.
    """

    def __init__(self, meter: otel_metrics.Meter | None = None) -> None:
        meter = meter or otel_metrics.get_meter(SERVICE)

        self.active_jobs = meter.create_up_down_counter(
            "research.jobs.active",
            unit="{job}",
            description="Jobs currently in progress.",
        )
        self.job_queue_age = meter.create_histogram(
            "research.jobs.queue_age",
            unit="s",
            description="Time between a job's creation and its plan starting.",
        )
        self.task_completions = meter.create_counter(
            "research.tasks.completed",
            unit="{task}",
            description="Research tasks that finished, by outcome.",
        )
        self.approval_wait_time = meter.create_histogram(
            "research.review.wait_time",
            unit="s",
            description="How long a job waited for a reviewer decision.",
        )
        self.unsupported_citations = meter.create_counter(
            "research.report.unsupported_citations",
            unit="{citation}",
            description="Citations a report made to evidence that was never recorded.",
        )
        self.source_checks = meter.create_counter(
            "research.report.source_checks",
            unit="{source}",
            description="Cited sources re-read at publication, by whether they had drifted.",
        )
        self.findings = meter.create_counter(
            "research.findings",
            unit="{finding}",
            description="Claims a job settled on, by critic verdict (claim support).",
        )
        self.llm_tokens = meter.create_counter(
            "research.llm.tokens",
            unit="{token}",
            description="Tokens the model provider reported, by agent role and direction.",
        )
        self.llm_cost = meter.create_counter(
            "research.llm.cost",
            unit="USD",
            description="Estimated model cost, from token counts and the configured price.",
        )
        self.agent_corrections = meter.create_counter(
            "research.agent.schema_corrections",
            unit="{attempt}",
            description="Agent responses rejected and sent back for correction.",
        )
        self.activity_retries = meter.create_counter(
            "research.activity.retries",
            unit="{attempt}",
            description="Activity attempts after the first, by activity.",
        )
        self.job_tokens = meter.create_histogram(
            "research.job.tokens",
            unit="{token}",
            description="Tokens one job used by the time it ended.",
        )
        self.job_cost = meter.create_histogram(
            "research.job.cost",
            unit="USD",
            description="Estimated cost of one job by the time it ended.",
        )
        self.job_active_time = meter.create_histogram(
            "research.job.active_time",
            unit="s",
            description="Time agents spent working on one job, excluding reviewer waits.",
        )
        self._meter = meter
        self.mcp_calls = meter.create_counter(
            "mcp.invocation.count",
            unit="{call}",
            description="MCP capability calls, by outcome and error class.",
        )
        self.mcp_call_duration = meter.create_histogram(
            "mcp.invocation.duration",
            unit="ms",
            description="MCP capability call latency.",
        )
        self.permission_denials = meter.create_counter(
            "mcp.invocation.denied",
            unit="{call}",
            description="Calls refused by policy, approval or budget before execution.",
        )
        self.injection_flags = meter.create_counter(
            "mcp.invocation.injection_flags",
            unit="{flag}",
            description="Injection patterns detected in a tool result.",
        )

    def observe_circuits(self, states: Callable[[], dict[str, CircuitState]]) -> None:
        """Report each MCP server's circuit state: 0 closed, 1 half-open, 2 open."""
        levels = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}

        def read(_options: CallbackOptions) -> Iterable[Observation]:
            return [
                Observation(levels[state], {"mcp.server": server})
                for server, state in states().items()
            ]

        self._meter.create_observable_gauge(
            "mcp.circuit.state",
            callbacks=[read],
            unit="{state}",
            description="Circuit state per MCP server: 0 closed, 1 half-open, 2 open.",
        )


_default_metrics: PlatformMetrics | None = None


def get_metrics() -> PlatformMetrics:
    """The process-wide metrics instance, built once against the current meter provider."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = PlatformMetrics()
    return _default_metrics

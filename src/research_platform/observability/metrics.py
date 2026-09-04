"""The section 13 metrics, as one small typed wrapper around an OpenTelemetry meter.

Section 13 names what to track without naming instruments, so this module is the one
place that decides the instrument for each: a counter for something that only ever goes
up (calls, denials, injection flags, unsupported citations), a histogram for something
whose distribution matters (latency, approval wait time), an up-down counter for
something that also goes back down (active jobs). Call sites read as what happened, not
as the OpenTelemetry API. Circuit state is not a separate instrument here: every MCP call
already carries an ``error_class`` attribute, and a circuit held open shows up as a run of
``mcp.invocation.count`` with ``error_class=upstream_unavailable`` - one dimension on the
counter every call already increments, rather than a second instrument to keep in sync
with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

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
        exporter = OTLPMetricExporter(endpoint=settings.otel_exporter_otlp_endpoint)
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


_default_metrics: PlatformMetrics | None = None


def get_metrics() -> PlatformMetrics:
    """The process-wide metrics instance, built once against the current meter provider."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = PlatformMetrics()
    return _default_metrics

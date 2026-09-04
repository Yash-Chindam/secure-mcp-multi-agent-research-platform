"""Distributed tracing across the section 13 hierarchy.

    Research job
      -> Temporal workflow/activity
        -> agent task
          -> MCP invocation
            -> downstream HTTP/database action

Every span in that hierarchy opens under one process-wide tracer, so a single trace
threads through a job's plan, its parallel research tasks, and every capability call
each one makes. An unconfigured OTLP endpoint is stated explicitly rather than silently
defaulted (the same rule ``settings.py`` already applies to authorization): spans are
still created either way, since nothing that opens one needs to know whether exporting
is on, but they are only ever sent somewhere when an endpoint is configured.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Tracer

from research_platform.settings import Settings

SERVICE = "secure-mcp-research-platform"


@dataclass(frozen=True)
class TracingSetup:
    """What tracing was configured with, for a startup log line."""

    provider: TracerProvider
    exports: bool

    @property
    def description(self) -> str:
        if self.exports:
            return "spans exported over OTLP"
        return "spans created but not exported (no OTEL exporter endpoint configured)"


def configure_tracing(
    settings: Settings, *, extra_exporter: SpanExporter | None = None
) -> TracingSetup:
    """Build the process-wide tracer provider and register it as the global default.

    ``extra_exporter`` lets a test attach an in-memory exporter without also standing up
    a real OTLP collector; production code never passes it.
    """
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE}))
    exports = bool(settings.otel_exporter_otlp_endpoint)
    if exports:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    if extra_exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(extra_exporter))
    trace.set_tracer_provider(provider)
    return TracingSetup(provider=provider, exports=exports)


def get_tracer() -> Tracer:
    """The tracer every span in the section 13 hierarchy is opened under.

    Safe to call before ``configure_tracing`` runs: OpenTelemetry's default provider
    creates spans that are simply never exported, so a component built before startup
    configuration completes never has to guard against a missing tracer.
    """
    return trace.get_tracer(SERVICE)

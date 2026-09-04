from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from research_platform.observability.metrics import configure_metrics, get_metrics
from research_platform.observability.tracing import configure_tracing, get_tracer
from research_platform.settings import Settings


def test_tracing_without_an_endpoint_is_stated_explicitly() -> None:
    setup = configure_tracing(Settings())
    try:
        assert setup.exports is False
        assert "not exported" in setup.description
    finally:
        setup.provider.shutdown()


def test_tracing_with_an_endpoint_exports() -> None:
    setup = configure_tracing(Settings(otel_exporter_otlp_endpoint="http://collector.test:4318"))
    try:
        assert setup.exports is True
        assert setup.description == "spans exported over OTLP"
    finally:
        setup.provider.shutdown()


def test_metrics_without_an_endpoint_is_stated_explicitly() -> None:
    setup = configure_metrics(Settings())
    try:
        assert setup.exports is False
        assert "not exported" in setup.description
    finally:
        setup.provider.shutdown()


def test_metrics_with_an_endpoint_exports() -> None:
    setup = configure_metrics(Settings(otel_exporter_otlp_endpoint="http://collector.test:4318"))
    try:
        assert setup.exports is True
        assert setup.description == "metrics exported over OTLP"
    finally:
        setup.provider.shutdown()


def test_an_extra_exporter_receives_spans_alongside_configuration() -> None:
    exporter = InMemorySpanExporter()
    setup = configure_tracing(Settings(), extra_exporter=exporter)
    try:
        setup.provider.get_tracer("test").start_span("probe").end()
        setup.provider.force_flush()
        assert [span.name for span in exporter.get_finished_spans()] == ["probe"]
    finally:
        setup.provider.shutdown()


def test_an_extra_reader_can_be_attached_alongside_configuration() -> None:
    reader = InMemoryMetricReader()
    setup = configure_metrics(Settings(), extra_reader=reader)
    try:
        setup.provider.get_meter("test").create_counter("probe").add(1)
        assert reader.get_metrics_data() is not None
    finally:
        setup.provider.shutdown()


def test_get_tracer_works_before_any_configuration() -> None:
    assert get_tracer() is not None


def test_get_metrics_returns_the_same_process_wide_instance() -> None:
    assert get_metrics() is get_metrics()

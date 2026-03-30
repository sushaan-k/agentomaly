"""Tests for the OpenTelemetry collector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from spectra.instrumentation.otel import OTelCollector
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    LLMCall,
    Severity,
    ToolCall,
)


class _InMemoryExporter(SpanExporter):
    """Simple in-memory span exporter for testing."""

    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _make_provider_and_exporter():
    """Create an OTel TracerProvider with an in-memory exporter."""
    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestOTelCollector:
    def test_init_with_custom_provider(self) -> None:
        provider, _ = _make_provider_and_exporter()
        collector = OTelCollector(service_name="test-spectra", tracer_provider=provider)
        assert collector.service_name == "test-spectra"

    def test_init_default_provider(self) -> None:
        collector = OTelCollector(service_name="spectra-default")
        assert collector.service_name == "spectra-default"

    def test_export_trace_success(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        started = datetime.now(UTC)
        trace = AgentTrace(
            agent_type="test-agent",
            task_id="task-1",
            user_id="user-1",
            session_id="sess-1",
            started_at=started,
            ended_at=started + timedelta(minutes=1),
            tool_calls=[
                ToolCall(tool_name="search_kb", duration_ms=50.0),
                ToolCall(tool_name="respond", duration_ms=30.0),
            ],
            llm_calls=[
                LLMCall(model="gpt-4", total_tokens=1000, duration_ms=200.0),
            ],
            success=True,
        )

        collector.export_trace(trace)
        spans = exporter.spans

        # Should have parent span + 2 tool spans + 1 llm span = 4
        assert len(spans) == 4

        parent_spans = [s for s in spans if s.name.startswith("agent.")]
        assert len(parent_spans) == 1

        tool_spans = [s for s in spans if s.name.startswith("tool.")]
        assert len(tool_spans) == 2

        llm_spans = [s for s in spans if s.name.startswith("llm.")]
        assert len(llm_spans) == 1

    def test_export_trace_failed_sets_error_status(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        trace = AgentTrace(
            agent_type="test-agent",
            success=False,
        )

        collector.export_trace(trace)
        spans = exporter.spans
        assert len(spans) == 1

        from opentelemetry.trace import StatusCode

        assert spans[0].status.status_code == StatusCode.ERROR

    def test_export_trace_no_tool_or_llm_calls(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        trace = AgentTrace(agent_type="empty-agent", success=True)
        collector.export_trace(trace)

        spans = exporter.spans
        assert len(spans) == 1
        assert spans[0].name == "agent.empty-agent"

    def test_export_anomaly_low_severity(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        event = AnomalyEvent(
            trace_id="trace-123",
            agent_type="test-agent",
            detector_type=DetectorType.TOOL_USAGE,
            severity=Severity.LOW,
            title="Minor anomaly",
            description="Something slightly odd",
            score=0.3,
        )

        collector.export_anomaly(event)
        spans = exporter.spans
        assert len(spans) == 1
        assert spans[0].name == "anomaly.tool_usage"

        from opentelemetry.trace import StatusCode

        assert spans[0].status.status_code != StatusCode.ERROR

    def test_export_anomaly_critical_severity(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        event = AnomalyEvent(
            trace_id="trace-456",
            agent_type="test-agent",
            detector_type=DetectorType.INJECTION,
            severity=Severity.CRITICAL,
            title="Injection detected",
            description="Behavioral shift after content ingestion",
            score=0.95,
        )

        collector.export_anomaly(event)
        spans = exporter.spans
        assert len(spans) == 1

        from opentelemetry.trace import StatusCode

        assert spans[0].status.status_code == StatusCode.ERROR

    def test_export_anomaly_high_severity(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        event = AnomalyEvent(
            trace_id="trace-789",
            agent_type="test-agent",
            detector_type=DetectorType.VOLUME,
            severity=Severity.HIGH,
            title="Volume spike",
            description="Too many calls",
            score=0.8,
        )

        collector.export_anomaly(event)
        spans = exporter.spans
        assert len(spans) == 1

        from opentelemetry.trace import StatusCode

        assert spans[0].status.status_code == StatusCode.ERROR

    def test_export_anomaly_medium_severity(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        event = AnomalyEvent(
            trace_id="trace-med",
            agent_type="test-agent",
            detector_type=DetectorType.CONTENT,
            severity=Severity.MEDIUM,
            title="Content anomaly",
            description="Unusual output",
            score=0.5,
        )

        collector.export_anomaly(event)
        spans = exporter.spans
        assert len(spans) == 1

        from opentelemetry.trace import StatusCode

        assert spans[0].status.status_code != StatusCode.ERROR

    def test_trace_attributes(self) -> None:
        started = datetime.now(UTC)
        trace = AgentTrace(
            agent_type="test-agent",
            trace_id="abc123",
            task_id="task-42",
            user_id="user-99",
            session_id="sess-7",
            started_at=started,
            ended_at=started + timedelta(seconds=5),
            tool_calls=[ToolCall(tool_name="search_kb")],
            llm_calls=[LLMCall(total_tokens=500)],
            success=True,
        )
        attrs = OTelCollector._trace_attributes(trace)
        assert attrs["spectra.trace.id"] == "abc123"
        assert attrs["spectra.trace.agent_type"] == "test-agent"
        assert attrs["spectra.trace.task_id"] == "task-42"
        assert attrs["spectra.trace.user_id"] == "user-99"
        assert attrs["spectra.trace.session_id"] == "sess-7"
        assert attrs["spectra.trace.success"] is True
        assert attrs["spectra.trace.tool_call_count"] == 1
        assert attrs["spectra.trace.llm_call_count"] == 1
        assert attrs["spectra.trace.total_tokens"] == 500

    def test_export_trace_with_unknown_model(self) -> None:
        provider, exporter = _make_provider_and_exporter()
        collector = OTelCollector(tracer_provider=provider)

        trace = AgentTrace(
            agent_type="test-agent",
            llm_calls=[LLMCall(model="", total_tokens=100)],
            success=True,
        )
        collector.export_trace(trace)
        spans = exporter.spans
        llm_spans = [s for s in spans if s.name.startswith("llm.")]
        assert len(llm_spans) == 1
        assert llm_spans[0].name == "llm.unknown"

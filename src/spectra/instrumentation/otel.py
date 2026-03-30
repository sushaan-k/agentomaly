"""OpenTelemetry integration for spectra.

Provides an OTelCollector that exports spectra traces and anomaly events
as OpenTelemetry spans, making them visible in any OTel-compatible
observability backend (Jaeger, Grafana Tempo, Datadog, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from spectra.models import AgentTrace, AnomalyEvent

logger = logging.getLogger(__name__)


class OTelCollector:
    """Exports spectra data as OpenTelemetry spans.

    Creates spans for agent traces and anomaly events, attaching
    relevant attributes for searchability in observability backends.

    Args:
        service_name: The service name for the OTel tracer.
        tracer_provider: Optional custom TracerProvider. If not provided,
            the global provider is used.
    """

    def __init__(
        self,
        service_name: str = "spectra",
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        self.service_name = service_name
        if tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(service_name)
        else:
            self._tracer = otel_trace.get_tracer(service_name)

    def export_trace(self, agent_trace: AgentTrace) -> None:
        """Export an agent trace as an OpenTelemetry span hierarchy.

        Creates a parent span for the trace with child spans for each
        tool call and LLM call.

        Args:
            agent_trace: The agent execution trace to export.
        """
        with self._tracer.start_as_current_span(
            name=f"agent.{agent_trace.agent_type}",
            attributes=self._trace_attributes(agent_trace),
        ) as span:
            for tc in agent_trace.tool_calls:
                with self._tracer.start_as_current_span(
                    name=f"tool.{tc.tool_name}",
                    attributes={
                        "spectra.tool.name": tc.tool_name,
                        "spectra.tool.success": tc.success,
                        "spectra.tool.duration_ms": tc.duration_ms,
                    },
                ):
                    pass

            for llm in agent_trace.llm_calls:
                with self._tracer.start_as_current_span(
                    name=f"llm.{llm.model or 'unknown'}",
                    attributes={
                        "spectra.llm.model": llm.model,
                        "spectra.llm.total_tokens": llm.total_tokens,
                        "spectra.llm.duration_ms": llm.duration_ms,
                    },
                ):
                    pass

            if not agent_trace.success:
                span.set_status(StatusCode.ERROR, "Agent task failed")

    def export_anomaly(self, event: AnomalyEvent) -> None:
        """Export an anomaly event as an OpenTelemetry span.

        Args:
            event: The anomaly event to export.
        """
        with self._tracer.start_as_current_span(
            name=f"anomaly.{event.detector_type.value}",
            attributes={
                "spectra.anomaly.event_id": event.event_id,
                "spectra.anomaly.trace_id": event.trace_id,
                "spectra.anomaly.agent_type": event.agent_type,
                "spectra.anomaly.detector": event.detector_type.value,
                "spectra.anomaly.severity": event.severity.value,
                "spectra.anomaly.title": event.title,
                "spectra.anomaly.score": event.score,
            },
        ) as span:
            if event.severity.value in ("HIGH", "CRITICAL"):
                span.set_status(StatusCode.ERROR, event.title)

    @staticmethod
    def _trace_attributes(agent_trace: AgentTrace) -> dict[str, Any]:
        """Build OTel span attributes from an agent trace.

        Args:
            agent_trace: The trace to extract attributes from.

        Returns:
            Dictionary of span attributes.
        """
        return {
            "spectra.trace.id": agent_trace.trace_id,
            "spectra.trace.agent_type": agent_trace.agent_type,
            "spectra.trace.task_id": agent_trace.task_id,
            "spectra.trace.user_id": agent_trace.user_id,
            "spectra.trace.session_id": agent_trace.session_id,
            "spectra.trace.success": agent_trace.success,
            "spectra.trace.tool_call_count": len(agent_trace.tool_calls),
            "spectra.trace.llm_call_count": len(agent_trace.llm_calls),
            "spectra.trace.total_tokens": agent_trace.total_tokens,
            "spectra.trace.duration_ms": agent_trace.duration_ms,
        }

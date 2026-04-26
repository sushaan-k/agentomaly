"""Tests for graceful handling of traces with empty tool call lists."""

from __future__ import annotations

from spectra.detectors.content_anomaly import ContentAnomalyDetector
from spectra.detectors.injection import InjectionDetector
from spectra.detectors.sequence_anomaly import SequenceAnomalyDetector
from spectra.detectors.tool_anomaly import ToolAnomalyDetector
from spectra.detectors.volume_anomaly import VolumeAnomalyDetector
from spectra.models import AgentTrace, Sensitivity
from spectra.profiler.profile import BehavioralProfile


def _empty_trace() -> AgentTrace:
    """Create a trace with zero tool calls and zero LLM calls."""
    return AgentTrace(
        agent_type="test-agent",
        tool_calls=[],
        llm_calls=[],
        output="A normal response.",
    )


class TestEmptyToolCalls:
    """Ensure every detector handles traces with no tool calls without error."""

    def test_tool_anomaly_empty_tools(self, trained_profile: BehavioralProfile) -> None:
        detector = ToolAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        events = detector.analyze(_empty_trace(), trained_profile)
        assert events == []

    def test_sequence_anomaly_empty_tools(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = SequenceAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        events = detector.analyze(_empty_trace(), trained_profile)
        # With no actions, no sequence anomalies should be reported
        assert isinstance(events, list)

    def test_volume_anomaly_empty_tools(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        events = detector.analyze(_empty_trace(), trained_profile)
        # Should not raise division-by-zero errors
        assert isinstance(events, list)

    def test_content_anomaly_empty_tools(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = ContentAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        events = detector.analyze(_empty_trace(), trained_profile)
        assert isinstance(events, list)

    def test_injection_empty_tools(self, trained_profile: BehavioralProfile) -> None:
        detector = InjectionDetector(sensitivity=Sensitivity.MEDIUM)
        events = detector.analyze(_empty_trace(), trained_profile)
        assert events == []

    def test_volume_no_false_positive_when_profile_also_empty(self) -> None:
        """When the profile has 0 mean/std for tool calls and the trace
        also has 0 tool calls, no anomaly should be reported for that
        metric.
        """
        profile = BehavioralProfile(agent_type="zero-tools")
        # volume_stats defaults are all 0.0
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.PARANOID)
        events = detector.analyze(_empty_trace(), profile)
        tool_events = [e for e in events if "Tool call count" in e.title]
        assert tool_events == []

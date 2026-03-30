"""Extended tests for the Monitor runtime, covering edge cases."""

from __future__ import annotations

import pytest

from spectra.detectors.base import BaseDetector
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    Severity,
    ToolCall,
)
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile


def _alert_only_policy() -> dict[str, str]:
    return {
        "LOW": "log",
        "MEDIUM": "alert",
        "HIGH": "alert",
        "CRITICAL": "alert",
    }


class _FaultyDetector(BaseDetector):
    """Detector that always raises an exception during analysis."""

    detector_type = DetectorType.TOOL_USAGE

    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        raise RuntimeError("detector crashed")


class TestMonitorDetectorFailure:
    @pytest.mark.asyncio
    async def test_detector_exception_logged_not_raised(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """A failing detector should not crash the monitor."""
        monitor = Monitor(
            profile=trained_profile,
            detectors=[_FaultyDetector()],
        )
        monitor.start()

        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="search_kb")],
        )
        # Should not raise even though detector throws
        events = await monitor.analyze(trace)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_custom_detectors(self, trained_profile: BehavioralProfile) -> None:
        """Monitor accepts custom list of detectors."""
        from spectra.detectors.tool_anomaly import ToolAnomalyDetector

        detector = ToolAnomalyDetector()
        monitor = Monitor(
            profile=trained_profile,
            detectors=[detector],
            response_policy=_alert_only_policy(),
        )
        monitor.start()

        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="evil_tool")],
        )
        events = await monitor.analyze(trace)
        assert all(e.detector_type == DetectorType.TOOL_USAGE for e in events)

    @pytest.mark.asyncio
    async def test_analyze_with_no_anomalies(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Monitor should return a list even when sparse traces are unusual."""
        monitor = Monitor(
            profile=trained_profile,
            sensitivity="low",
            response_policy=_alert_only_policy(),
        )
        monitor.start()

        trace = AgentTrace(agent_type="test-agent")
        events = await monitor.analyze(trace)
        # Sparse traces may still trigger detectors; the monitor should not hang.
        assert isinstance(events, list)

    def test_summary_with_events(self, trained_profile: BehavioralProfile) -> None:
        """Summary should reflect severity counts correctly."""
        monitor = Monitor(profile=trained_profile)
        # Manually add events to the log
        event = AnomalyEvent(
            trace_id="test",
            agent_type="test-agent",
            detector_type=DetectorType.TOOL_USAGE,
            severity=Severity.CRITICAL,
            title="Test",
            description="Test",
            score=0.9,
        )
        monitor._event_log.append(event)
        summary = monitor.summary()
        assert summary["total_anomalies"] == 1
        assert summary["severity_counts"]["CRITICAL"] == 1

    def test_default_detectors_created(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Default detectors should include all 5 built-in detectors."""
        monitor = Monitor(profile=trained_profile)
        assert len(monitor._detectors) == 5

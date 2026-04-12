"""Tests for anomaly severity trend tracking."""

from __future__ import annotations

import pytest

from spectra.models import AgentTrace, AnomalyEvent, DetectorType, Severity
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile
from spectra.trend import Trend, TrendTracker


def _make_event(severity: Severity) -> AnomalyEvent:
    """Create a minimal anomaly event with the given severity."""
    return AnomalyEvent(
        trace_id="t1",
        agent_type="test-agent",
        detector_type=DetectorType.TOOL_USAGE,
        severity=severity,
        title="test",
        description="test event",
        score=0.5,
    )


class TestTrendTracker:
    def test_insufficient_data(self) -> None:
        tracker = TrendTracker(window_size=10)
        assert tracker.get_trend() == Trend.INSUFFICIENT_DATA

    def test_insufficient_data_with_few_events(self) -> None:
        tracker = TrendTracker(window_size=10)
        for _ in range(3):
            tracker.record(_make_event(Severity.LOW))
        assert tracker.get_trend() == Trend.INSUFFICIENT_DATA

    def test_stable_trend(self) -> None:
        tracker = TrendTracker(window_size=10)
        for _ in range(10):
            tracker.record(_make_event(Severity.MEDIUM))
        assert tracker.get_trend() == Trend.STABLE

    def test_escalating_trend(self) -> None:
        tracker = TrendTracker(window_size=10)
        for _ in range(5):
            tracker.record(_make_event(Severity.LOW))
        for _ in range(5):
            tracker.record(_make_event(Severity.CRITICAL))
        assert tracker.get_trend() == Trend.ESCALATING

    def test_de_escalating_trend(self) -> None:
        tracker = TrendTracker(window_size=10)
        for _ in range(5):
            tracker.record(_make_event(Severity.CRITICAL))
        for _ in range(5):
            tracker.record(_make_event(Severity.LOW))
        assert tracker.get_trend() == Trend.DE_ESCALATING

    def test_record_many(self) -> None:
        tracker = TrendTracker(window_size=10)
        events = [_make_event(Severity.MEDIUM) for _ in range(6)]
        tracker.record_many(events)
        assert tracker.get_trend() == Trend.STABLE

    def test_current_mean_severity_empty(self) -> None:
        tracker = TrendTracker(window_size=10)
        assert tracker.current_mean_severity() is None

    def test_current_mean_severity(self) -> None:
        tracker = TrendTracker(window_size=10)
        tracker.record(_make_event(Severity.LOW))
        tracker.record(_make_event(Severity.CRITICAL))
        # LOW=1.0, CRITICAL=4.0 => mean=2.5
        assert tracker.current_mean_severity() == 2.5

    def test_clear(self) -> None:
        tracker = TrendTracker(window_size=10)
        tracker.record(_make_event(Severity.HIGH))
        tracker.clear()
        assert tracker.current_mean_severity() is None

    def test_snapshot(self) -> None:
        tracker = TrendTracker(window_size=10)
        for _ in range(6):
            tracker.record(_make_event(Severity.MEDIUM))
        snap = tracker.snapshot()
        assert snap["trend"] == "stable"
        assert snap["event_count"] == 6
        assert snap["window_size"] == 10
        assert snap["mean_severity"] == 2.0

    def test_window_size_validation(self) -> None:
        with pytest.raises(ValueError, match="window_size must be at least 4"):
            TrendTracker(window_size=2)

    def test_rolling_window_evicts_old_events(self) -> None:
        tracker = TrendTracker(window_size=6)
        # Fill with LOW events then overflow with CRITICAL
        for _ in range(6):
            tracker.record(_make_event(Severity.LOW))
        for _ in range(6):
            tracker.record(_make_event(Severity.CRITICAL))
        # Window should only contain CRITICAL (all LOW evicted)
        assert tracker.get_trend() == Trend.STABLE
        assert tracker.current_mean_severity() == 4.0


class TestMonitorTrend:
    def test_get_trend_initial(self, trained_profile: BehavioralProfile) -> None:
        monitor = Monitor(profile=trained_profile)
        assert monitor.get_trend() == Trend.INSUFFICIENT_DATA

    @pytest.mark.asyncio
    async def test_get_trend_after_anomalies(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "LOW": "log",
                "MEDIUM": "log",
                "HIGH": "log",
                "CRITICAL": "log",
            },
        )
        monitor.start()
        # Analyze the same anomalous trace multiple times to build up events
        for _ in range(3):
            await monitor.analyze(anomalous_trace)
        # Should have enough data points for a non-insufficient result
        trend = monitor.get_trend()
        assert trend in (Trend.ESCALATING, Trend.STABLE, Trend.DE_ESCALATING)

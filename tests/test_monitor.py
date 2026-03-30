"""Tests for the main Monitor runtime."""

from __future__ import annotations

import pytest

from spectra.exceptions import MonitorNotRunningError
from spectra.models import AgentTrace, Sensitivity, Severity, ToolCall
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile


def _alert_only_policy() -> dict[str, str]:
    return {
        "LOW": "log",
        "MEDIUM": "alert",
        "HIGH": "alert",
        "CRITICAL": "alert",
    }


class TestMonitor:
    def test_start_stop(self, trained_profile: BehavioralProfile) -> None:
        monitor = Monitor(profile=trained_profile)
        assert not monitor.is_running

        monitor.start()
        assert monitor.is_running

        monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_analyze_requires_start(
        self, trained_profile: BehavioralProfile
    ) -> None:
        monitor = Monitor(profile=trained_profile)
        trace = AgentTrace(agent_type="test-agent")
        with pytest.raises(MonitorNotRunningError):
            await monitor.analyze(trace)

    @pytest.mark.asyncio
    async def test_analyze_normal_trace(
        self,
        trained_profile: BehavioralProfile,
        normal_trace: AgentTrace,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            sensitivity="low",
        )
        monitor.start()
        events = await monitor.analyze(normal_trace)
        critical = [e for e in events if e.severity == Severity.CRITICAL]
        assert len(critical) == 0

    @pytest.mark.asyncio
    async def test_analyze_anomalous_trace(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            sensitivity="medium",
            response_policy=_alert_only_policy(),
        )
        monitor.start()
        events = await monitor.analyze(anomalous_trace)
        assert len(events) > 0
        assert any(e.severity == Severity.CRITICAL for e in events)

    @pytest.mark.asyncio
    async def test_event_log(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            response_policy=_alert_only_policy(),
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)
        assert len(monitor.event_log) > 0

    @pytest.mark.asyncio
    async def test_clear_event_log(
        self,
        trained_profile: BehavioralProfile,
        anomalous_trace: AgentTrace,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            response_policy=_alert_only_policy(),
        )
        monitor.start()
        await monitor.analyze(anomalous_trace)
        monitor.clear_event_log()
        assert len(monitor.event_log) == 0

    def test_summary(self, trained_profile: BehavioralProfile) -> None:
        monitor = Monitor(
            profile=trained_profile,
            sensitivity="high",
        )
        summary = monitor.summary()
        assert summary["running"] is False
        assert summary["agent_type"] == "test-agent"
        assert summary["sensitivity"] == "high"
        assert summary["total_anomalies"] == 0

    @pytest.mark.asyncio
    async def test_string_sensitivity(self, trained_profile: BehavioralProfile) -> None:
        monitor = Monitor(
            profile=trained_profile,
            sensitivity="paranoid",
        )
        assert monitor.sensitivity == Sensitivity.PARANOID

    @pytest.mark.asyncio
    async def test_custom_policy(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        monitor = Monitor(
            profile=trained_profile,
            response_policy={
                "CRITICAL": "alert",
                "HIGH": "log",
            },
        )
        monitor.start()

        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="database_query")],
        )
        events = await monitor.analyze(trace)
        blocked = [
            e for e in events if e.action_taken and e.action_taken.value == "block"
        ]
        assert len(blocked) == 0

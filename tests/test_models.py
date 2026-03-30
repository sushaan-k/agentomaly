"""Tests for core data models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    LLMCall,
    ResponseAction,
    Sensitivity,
    SensitivityThresholds,
    Severity,
    ToolCall,
    ToolStats,
    VolumeStats,
)


class TestSeverity:
    def test_values(self) -> None:
        assert Severity.LOW.value == "LOW"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.HIGH.value == "HIGH"
        assert Severity.CRITICAL.value == "CRITICAL"


class TestResponseAction:
    def test_values(self) -> None:
        assert ResponseAction.LOG.value == "log"
        assert ResponseAction.ALERT.value == "alert"
        assert ResponseAction.QUARANTINE.value == "quarantine"
        assert ResponseAction.BLOCK.value == "block"


class TestSensitivity:
    def test_values(self) -> None:
        assert Sensitivity.LOW.value == "low"
        assert Sensitivity.PARANOID.value == "paranoid"


class TestSensitivityThresholds:
    def test_get_threshold(self) -> None:
        thresholds = SensitivityThresholds()
        assert thresholds.get_threshold(Sensitivity.LOW) == 4.0
        assert thresholds.get_threshold(Sensitivity.MEDIUM) == 3.0
        assert thresholds.get_threshold(Sensitivity.HIGH) == 2.0
        assert thresholds.get_threshold(Sensitivity.PARANOID) == 1.5


class TestLLMCall:
    def test_defaults(self) -> None:
        call = LLMCall()
        assert call.model == ""
        assert call.total_tokens == 0
        assert call.call_id != ""

    def test_with_values(self) -> None:
        call = LLMCall(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert call.model == "gpt-4"
        assert call.total_tokens == 150


class TestToolCall:
    def test_creation(self) -> None:
        tc = ToolCall(tool_name="search_kb", arguments={"query": "test"})
        assert tc.tool_name == "search_kb"
        assert tc.arguments == {"query": "test"}
        assert tc.success is True

    def test_failed_call(self) -> None:
        tc = ToolCall(tool_name="search_kb", success=False)
        assert tc.success is False


class TestAgentTrace:
    def test_duration(self) -> None:
        started = datetime.now(UTC)
        ended = started + timedelta(seconds=5)
        trace = AgentTrace(
            agent_type="test",
            started_at=started,
            ended_at=ended,
        )
        assert abs(trace.duration_ms - 5000.0) < 1.0

    def test_duration_no_end(self) -> None:
        trace = AgentTrace(agent_type="test")
        assert trace.duration_ms == 0.0

    def test_total_tokens(self) -> None:
        trace = AgentTrace(
            agent_type="test",
            llm_calls=[
                LLMCall(total_tokens=100),
                LLMCall(total_tokens=200),
            ],
        )
        assert trace.total_tokens == 300

    def test_tool_names(self) -> None:
        trace = AgentTrace(
            agent_type="test",
            tool_calls=[
                ToolCall(tool_name="search"),
                ToolCall(tool_name="respond"),
            ],
        )
        assert trace.tool_names == ["search", "respond"]

    def test_action_sequence(self) -> None:
        now = datetime.now(UTC)
        trace = AgentTrace(
            agent_type="test",
            llm_calls=[
                LLMCall(timestamp=now),
            ],
            tool_calls=[
                ToolCall(
                    tool_name="search",
                    timestamp=now + timedelta(seconds=1),
                ),
            ],
        )
        assert trace.action_sequence == ["__llm_call__", "search"]

    def test_auto_generated_trace_id(self) -> None:
        t1 = AgentTrace(agent_type="test")
        t2 = AgentTrace(agent_type="test")
        assert t1.trace_id != t2.trace_id


class TestAnomalyEvent:
    def test_creation(self) -> None:
        event = AnomalyEvent(
            trace_id="abc123",
            agent_type="test",
            detector_type=DetectorType.TOOL_USAGE,
            severity=Severity.CRITICAL,
            title="Test anomaly",
            description="Something bad happened",
            score=0.95,
        )
        assert event.severity == Severity.CRITICAL
        assert event.score == 0.95
        assert event.event_id != ""

    def test_score_validation(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnomalyEvent(
                trace_id="abc",
                agent_type="test",
                detector_type=DetectorType.TOOL_USAGE,
                severity=Severity.LOW,
                title="test",
                description="test",
                score=1.5,
            )


class TestToolStats:
    def test_creation(self) -> None:
        stats = ToolStats(
            tool_name="search",
            usage_frequency=0.95,
            avg_calls_per_trace=2.3,
        )
        assert stats.tool_name == "search"
        assert stats.usage_frequency == 0.95


class TestVolumeStats:
    def test_defaults(self) -> None:
        vs = VolumeStats()
        assert vs.llm_calls_mean == 0.0
        assert vs.tool_calls_std == 0.0

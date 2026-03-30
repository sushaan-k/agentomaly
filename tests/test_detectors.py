"""Tests for all anomaly detectors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from spectra.detectors.content_anomaly import ContentAnomalyDetector
from spectra.detectors.injection import InjectionDetector
from spectra.detectors.sequence_anomaly import SequenceAnomalyDetector
from spectra.detectors.tool_anomaly import ToolAnomalyDetector
from spectra.detectors.volume_anomaly import VolumeAnomalyDetector
from spectra.models import (
    AgentTrace,
    DetectorType,
    LLMCall,
    Sensitivity,
    Severity,
    ToolCall,
)
from spectra.profiler.profile import BehavioralProfile


class TestToolAnomalyDetector:
    def test_detects_novel_tool(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ToolAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="database_query")],
        )
        events = detector.analyze(trace, trained_profile)
        novel_events = [e for e in events if "Never-seen" in e.title]
        assert len(novel_events) == 1
        assert novel_events[0].severity == Severity.CRITICAL
        assert novel_events[0].score == 1.0

    def test_no_anomaly_for_known_tools(
        self,
        trained_profile: BehavioralProfile,
        normal_trace: AgentTrace,
    ) -> None:
        detector = ToolAnomalyDetector()
        events = detector.analyze(normal_trace, trained_profile)
        novel_events = [e for e in events if "Never-seen" in e.title]
        assert len(novel_events) == 0

    def test_frequency_anomaly(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ToolAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="search_kb") for _ in range(50)],
        )
        events = detector.analyze(trace, trained_profile)
        freq_events = [e for e in events if "frequency" in e.title.lower()]
        assert len(freq_events) >= 1

    def test_argument_anomaly(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ToolAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    arguments={"query": "test", "admin_override": True},
                )
            ],
        )
        events = detector.analyze(trace, trained_profile)
        arg_events = [e for e in events if "argument" in e.title.lower()]
        assert len(arg_events) >= 1

    def test_detector_type(self) -> None:
        detector = ToolAnomalyDetector()
        assert detector.detector_type == DetectorType.TOOL_USAGE


class TestSequenceAnomalyDetector:
    def test_detects_novel_transition(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        now = datetime.now(UTC)
        detector = SequenceAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now,
                ),
                ToolCall(
                    tool_name="database_query",
                    timestamp=now + timedelta(seconds=1),
                ),
                ToolCall(
                    tool_name="delete_record",
                    timestamp=now + timedelta(seconds=2),
                ),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        novel_events = [e for e in events if "Novel" in e.title]
        assert len(novel_events) >= 1
        assert novel_events[0].severity == Severity.CRITICAL

    def test_detects_loop(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        now = datetime.now(UTC)
        detector = SequenceAnomalyDetector(loop_threshold=3)
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=i),
                )
                for i in range(10)
            ],
        )
        events = detector.analyze(trace, trained_profile)
        loop_events = [e for e in events if "loop" in e.title.lower()]
        assert len(loop_events) >= 1

    def test_no_anomaly_normal_sequence(
        self,
        trained_profile: BehavioralProfile,
        normal_trace: AgentTrace,
    ) -> None:
        detector = SequenceAnomalyDetector(sensitivity=Sensitivity.LOW)
        events = detector.analyze(normal_trace, trained_profile)
        novel_events = [e for e in events if "Novel" in e.title]
        assert len(novel_events) == 0

    def test_empty_trace(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = SequenceAnomalyDetector()
        trace = AgentTrace(agent_type="test-agent")
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0


class TestVolumeAnomalyDetector:
    def test_detects_high_volume(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        started = datetime.now(UTC)
        trace = AgentTrace(
            agent_type="test-agent",
            started_at=started,
            ended_at=started + timedelta(hours=2),
            llm_calls=[LLMCall(total_tokens=5000) for _ in range(30)],
            tool_calls=[ToolCall(tool_name="search_kb") for _ in range(30)],
        )
        events = detector.analyze(trace, trained_profile)
        assert len(events) >= 1
        assert any(e.detector_type == DetectorType.VOLUME for e in events)

    def test_no_anomaly_normal_volume(
        self,
        trained_profile: BehavioralProfile,
        normal_trace: AgentTrace,
    ) -> None:
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.LOW)
        events = detector.analyze(normal_trace, trained_profile)
        volume_events = [e for e in events if e.detector_type == DetectorType.VOLUME]
        assert len(volume_events) == 0

    def test_paranoid_sensitivity_catches_more(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        started = datetime.now(UTC)
        trace = AgentTrace(
            agent_type="test-agent",
            started_at=started,
            ended_at=started + timedelta(minutes=10),
            llm_calls=[LLMCall(total_tokens=3000) for _ in range(6)],
            tool_calls=[ToolCall(tool_name="search_kb") for _ in range(6)],
        )
        low_detector = VolumeAnomalyDetector(sensitivity=Sensitivity.LOW)
        paranoid_detector = VolumeAnomalyDetector(sensitivity=Sensitivity.PARANOID)

        low_events = low_detector.analyze(trace, trained_profile)
        paranoid_events = paranoid_detector.analyze(trace, trained_profile)
        assert len(paranoid_events) >= len(low_events)


class TestContentAnomalyDetector:
    def test_detects_code_in_output(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ContentAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            output="Here is some code: ```python\nprint('hello')\n```",
        )
        events = detector.analyze(trace, trained_profile)
        code_events = [e for e in events if "code" in e.title.lower()]
        assert len(code_events) >= 1

    def test_detects_url_in_output(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ContentAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            output="Visit https://evil.example.com/phishing for details.",
        )
        events = detector.analyze(trace, trained_profile)
        url_events = [e for e in events if "URL" in e.title]
        assert len(url_events) >= 1

    def test_detects_length_anomaly(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ContentAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        trace = AgentTrace(
            agent_type="test-agent",
            output="x" * 50000,
        )
        events = detector.analyze(trace, trained_profile)
        length_events = [e for e in events if "length" in e.title.lower()]
        assert len(length_events) >= 1

    def test_no_anomaly_empty_output(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = ContentAnomalyDetector()
        trace = AgentTrace(agent_type="test-agent", output="")
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0


class TestInjectionDetector:
    def test_detects_behavioral_shift(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        now = datetime.now(UTC)
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now,
                ),
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=1),
                ),
                ToolCall(
                    tool_name="respond",
                    timestamp=now + timedelta(seconds=2),
                ),
                ToolCall(
                    tool_name="database_query",
                    timestamp=now + timedelta(seconds=3),
                ),
                ToolCall(
                    tool_name="delete_record",
                    timestamp=now + timedelta(seconds=4),
                ),
                ToolCall(
                    tool_name="send_email",
                    result_summary="Sent admin credentials to external email",
                    timestamp=now + timedelta(seconds=5),
                ),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        assert len(events) >= 1
        assert any(
            "injection" in e.title.lower() or "Suspicious" in e.title for e in events
        )

    def test_no_injection_normal_trace(
        self,
        trained_profile: BehavioralProfile,
        normal_trace: AgentTrace,
    ) -> None:
        detector = InjectionDetector()
        events = detector.analyze(normal_trace, trained_profile)
        injection_events = [
            e for e in events if e.detector_type == DetectorType.INJECTION
        ]
        assert len(injection_events) == 0

    def test_short_trace_no_false_positive(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="search_kb")],
        )
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0

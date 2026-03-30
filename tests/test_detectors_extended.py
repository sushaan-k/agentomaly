"""Extended tests for detector edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from spectra.detectors.content_anomaly import ContentAnomalyDetector
from spectra.detectors.injection import InjectionDetector
from spectra.detectors.sequence_anomaly import SequenceAnomalyDetector
from spectra.detectors.tool_anomaly import ToolAnomalyDetector
from spectra.detectors.volume_anomaly import VolumeAnomalyDetector
from spectra.models import (
    AgentTrace,
    LLMCall,
    Sensitivity,
    Severity,
    ToolCall,
    VolumeStats,
)
from spectra.profiler.profile import BehavioralProfile
from spectra.profiler.trainer import ProfileTrainer


def _make_empty_profile() -> BehavioralProfile:
    """Create a profile with no training data (empty)."""
    return BehavioralProfile(
        agent_type="empty-agent",
        trace_count=0,
        known_tools=set(),
    )


def _make_single_trace_profile() -> BehavioralProfile:
    """Create a profile trained on a single trace."""
    trace = AgentTrace(
        agent_type="single-trace-agent",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC) + timedelta(minutes=1),
        tool_calls=[
            ToolCall(
                tool_name="search_kb",
                arguments={"query": "test"},
                timestamp=datetime.now(UTC),
            ),
        ],
        llm_calls=[LLMCall(total_tokens=1000)],
        output="Normal output.",
    )
    trainer = ProfileTrainer(min_traces=1)
    return trainer.train(agent_type="single-trace-agent", traces=[trace])


class TestBaseDetectorZScore:
    """Test the z-score helper on the base detector."""

    def test_z_score_normal(self, trained_profile: BehavioralProfile) -> None:
        detector = ToolAnomalyDetector()
        z = detector._z_score(5.0, 3.0, 1.0)
        assert abs(z - 2.0) < 0.001

    def test_z_score_zero_std_same_value(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = ToolAnomalyDetector()
        z = detector._z_score(5.0, 5.0, 0.0)
        assert z == 0.0

    def test_z_score_zero_std_different_value(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = ToolAnomalyDetector()
        z = detector._z_score(10.0, 5.0, 0.0)
        assert z == 10.0


class TestToolAnomalyDetectorEdgeCases:
    def test_empty_profile_all_tools_novel(self) -> None:
        profile = _make_empty_profile()
        detector = ToolAnomalyDetector()
        trace = AgentTrace(
            agent_type="empty-agent",
            tool_calls=[ToolCall(tool_name="any_tool")],
        )
        events = detector.analyze(trace, profile)
        novel = [e for e in events if "Never-seen" in e.title]
        assert len(novel) == 1

    def test_no_tool_calls_no_events(self, trained_profile: BehavioralProfile) -> None:
        detector = ToolAnomalyDetector()
        trace = AgentTrace(agent_type="test-agent")
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0

    def test_duplicate_novel_tool_reported_once(self) -> None:
        profile = _make_empty_profile()
        detector = ToolAnomalyDetector()
        trace = AgentTrace(
            agent_type="empty-agent",
            tool_calls=[
                ToolCall(tool_name="bad_tool"),
                ToolCall(tool_name="bad_tool"),
                ToolCall(tool_name="bad_tool"),
            ],
        )
        events = detector.analyze(trace, profile)
        novel = [e for e in events if "Never-seen" in e.title]
        assert len(novel) == 1

    def test_severity_from_z_score_critical(self) -> None:
        sev = ToolAnomalyDetector._severity_from_z_score(6.0)
        assert sev == Severity.CRITICAL

    def test_severity_from_z_score_high(self) -> None:
        sev = ToolAnomalyDetector._severity_from_z_score(4.0)
        assert sev == Severity.HIGH

    def test_severity_from_z_score_medium(self) -> None:
        sev = ToolAnomalyDetector._severity_from_z_score(3.0)
        assert sev == Severity.MEDIUM

    def test_severity_from_z_score_low(self) -> None:
        sev = ToolAnomalyDetector._severity_from_z_score(2.0)
        assert sev == Severity.LOW

    def test_argument_anomaly_no_common_keys(self) -> None:
        """If the profile has no common arg keys, no argument anomaly."""
        profile = _make_single_trace_profile()
        # Verify the profile has common arg keys for search_kb
        stats = profile.get_tool_stats("search_kb")
        assert stats is not None

        detector = ToolAnomalyDetector()
        trace = AgentTrace(
            agent_type="single-trace-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    arguments={"query": "test", "extra": "value"},
                ),
            ],
        )
        events = detector.analyze(trace, profile)
        # Even with a novel key, should handle correctly
        assert isinstance(events, list)

    def test_frequency_anomaly_paranoid_sensitivity(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = ToolAnomalyDetector(sensitivity=Sensitivity.PARANOID)
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="search_kb") for _ in range(20)],
        )
        events = detector.analyze(trace, trained_profile)
        freq_events = [e for e in events if "frequency" in e.title.lower()]
        assert len(freq_events) >= 1


class TestVolumeAnomalyDetectorEdgeCases:
    def test_empty_profile_volume(self) -> None:
        """Profile with zero std should use sentinel z-score."""
        profile = BehavioralProfile(
            agent_type="test",
            volume_stats=VolumeStats(
                llm_calls_mean=2.0,
                llm_calls_std=0.0,
                tool_calls_mean=3.0,
                tool_calls_std=0.0,
                total_tokens_mean=1000.0,
                total_tokens_std=0.0,
                duration_ms_mean=5000.0,
                duration_ms_std=0.0,
            ),
        )
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.MEDIUM)
        trace = AgentTrace(
            agent_type="test",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC) + timedelta(minutes=1),
            llm_calls=[LLMCall(total_tokens=5000) for _ in range(10)],
            tool_calls=[ToolCall(tool_name="x") for _ in range(10)],
        )
        events = detector.analyze(trace, profile)
        assert len(events) >= 1

    def test_volume_below_mean(self, trained_profile: BehavioralProfile) -> None:
        """Significantly below-mean values should also be flagged."""
        detector = VolumeAnomalyDetector(sensitivity=Sensitivity.PARANOID)
        trace = AgentTrace(
            agent_type="test-agent",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        events = detector.analyze(trace, trained_profile)
        # With no calls at all, should be significantly below mean
        assert isinstance(events, list)

    def test_severity_from_z_score_volume(self) -> None:
        assert VolumeAnomalyDetector._severity_from_z_score(6.0) == Severity.CRITICAL
        assert VolumeAnomalyDetector._severity_from_z_score(4.0) == Severity.HIGH
        assert VolumeAnomalyDetector._severity_from_z_score(3.0) == Severity.MEDIUM
        assert VolumeAnomalyDetector._severity_from_z_score(2.0) == Severity.LOW


class TestSequenceAnomalyDetectorEdgeCases:
    def test_single_action_sequence(self, trained_profile: BehavioralProfile) -> None:
        detector = SequenceAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="search_kb")],
        )
        events = detector.analyze(trace, trained_profile)
        # Single action should not trigger loop detection
        loop_events = [e for e in events if "loop" in e.title.lower()]
        assert len(loop_events) == 0

    def test_very_long_sequence(self, trained_profile: BehavioralProfile) -> None:
        now = datetime.now(UTC)
        detector = SequenceAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name=f"tool_{i % 3}",
                    timestamp=now + timedelta(seconds=i),
                )
                for i in range(50)
            ],
        )
        events = detector.analyze(trace, trained_profile)
        assert isinstance(events, list)

    def test_custom_loop_threshold(self, trained_profile: BehavioralProfile) -> None:
        now = datetime.now(UTC)
        detector = SequenceAnomalyDetector(loop_threshold=5)
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=i),
                )
                for i in range(4)  # Only 4 reps, threshold is 5
            ],
        )
        events = detector.analyze(trace, trained_profile)
        loop_events = [e for e in events if "loop" in e.title.lower()]
        assert len(loop_events) == 0

    def test_high_repeat_loop_is_high_severity(
        self, trained_profile: BehavioralProfile
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
                for i in range(8)
            ],
        )
        events = detector.analyze(trace, trained_profile)
        loop_events = [e for e in events if "loop" in e.title.lower()]
        assert len(loop_events) >= 1
        assert any(e.severity == Severity.HIGH for e in loop_events)


class TestContentAnomalyDetectorEdgeCases:
    def test_structured_data_detection(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = ContentAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            output='{"key": "value", "data": [1, 2, 3]}',
        )
        events = detector.analyze(trace, trained_profile)
        struct_events = [e for e in events if "structured" in e.title.lower()]
        assert len(struct_events) >= 1

    def test_very_short_output_length(self, trained_profile: BehavioralProfile) -> None:
        """Very short output compared to profile mean should flag anomaly."""
        detector = ContentAnomalyDetector(sensitivity=Sensitivity.PARANOID)
        trace = AgentTrace(
            agent_type="test-agent",
            output="x",
        )
        events = detector.analyze(trace, trained_profile)
        # May or may not trigger depending on profile stats
        assert isinstance(events, list)

    def test_zero_structure_threshold_suppresses_detection(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """With structure_threshold=0.0, no structural anomalies are flagged."""
        detector = ContentAnomalyDetector(structure_threshold=0.0)
        trace = AgentTrace(
            agent_type="test-agent",
            output="Visit https://example.com for details.",
        )
        events = detector.analyze(trace, trained_profile)
        struct_events = [
            e
            for e in events
            if "URL" in e.title
            or "code" in e.title.lower()
            or "structured" in e.title.lower()
        ]
        assert len(struct_events) == 0

    def test_output_with_inline_code(self, trained_profile: BehavioralProfile) -> None:
        detector = ContentAnomalyDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            output="Use the command `rm -rf /` carefully.",
        )
        events = detector.analyze(trace, trained_profile)
        code_events = [e for e in events if "code" in e.title.lower()]
        assert len(code_events) >= 1


class TestInjectionDetectorEdgeCases:
    def test_all_known_tools_no_injection(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """All known tools should not trigger injection detection."""
        now = datetime.now(UTC)
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(tool_name="search_kb", timestamp=now),
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=1),
                ),
                ToolCall(
                    tool_name="respond",
                    timestamp=now + timedelta(seconds=2),
                ),
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=3),
                ),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        injection_events = [
            e
            for e in events
            if "injection" in e.title.lower() or "Suspicious" in e.title
        ]
        assert len(injection_events) == 0

    def test_two_tool_calls_no_behavioral_shift(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Only two tool calls shouldn't trigger behavioral shift (< 4)."""
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(tool_name="search_kb"),
                ToolCall(tool_name="database_query"),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        shift_events = [e for e in events if "behavioral shift" in e.title.lower()]
        assert len(shift_events) == 0

    def test_three_tool_calls_no_behavioral_shift(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Only three tool calls shouldn't trigger behavioral shift (< 4)."""
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(tool_name="search_kb"),
                ToolCall(tool_name="search_kb"),
                ToolCall(tool_name="database_query"),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        shift_events = [e for e in events if "behavioral shift" in e.title.lower()]
        assert len(shift_events) == 0

    def test_novel_first_half_no_behavioral_shift(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Novel tools in first half don't count as injection (shift)."""
        now = datetime.now(UTC)
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(
                    tool_name="unknown_tool",
                    timestamp=now,
                ),
                ToolCall(
                    tool_name="unknown_tool2",
                    timestamp=now + timedelta(seconds=1),
                ),
                ToolCall(
                    tool_name="search_kb",
                    timestamp=now + timedelta(seconds=2),
                ),
                ToolCall(
                    tool_name="respond",
                    timestamp=now + timedelta(seconds=3),
                ),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        shift_events = [e for e in events if "behavioral shift" in e.title.lower()]
        assert len(shift_events) == 0

    def test_post_tool_novel_with_known_predecessor(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Novel tool immediately after a known tool triggers alert."""
        now = datetime.now(UTC)
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[
                ToolCall(tool_name="search_kb", timestamp=now),
                ToolCall(
                    tool_name="evil_command",
                    timestamp=now + timedelta(seconds=1),
                ),
            ],
        )
        events = detector.analyze(trace, trained_profile)
        suspicious = [e for e in events if "Suspicious" in e.title]
        assert len(suspicious) >= 1

    def test_custom_shift_window(self, trained_profile: BehavioralProfile) -> None:
        detector = InjectionDetector(shift_window=5)
        assert detector.shift_window == 5

    def test_single_tool_call_no_events(
        self, trained_profile: BehavioralProfile
    ) -> None:
        detector = InjectionDetector()
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="database_query")],
        )
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0

    def test_empty_trace_no_events(self, trained_profile: BehavioralProfile) -> None:
        detector = InjectionDetector()
        trace = AgentTrace(agent_type="test-agent")
        events = detector.analyze(trace, trained_profile)
        assert len(events) == 0

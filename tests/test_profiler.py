"""Tests for the behavioral profiler and trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectra.exceptions import InsufficientTraceError
from spectra.models import AgentTrace
from spectra.profiler.profile import BehavioralProfile
from spectra.profiler.trainer import ProfileTrainer


class TestProfileTrainer:
    def test_train_success(self, training_traces: list[AgentTrace]) -> None:
        trainer = ProfileTrainer(min_traces=100)
        profile = trainer.train(agent_type="test-agent", traces=training_traces)

        assert profile.agent_type == "test-agent"
        assert profile.trace_count == len(training_traces)
        assert "search_kb" in profile.known_tools
        assert "respond" in profile.known_tools

    def test_train_insufficient_traces(self) -> None:
        trainer = ProfileTrainer(min_traces=100)
        traces = [AgentTrace(agent_type="test") for _ in range(10)]
        with pytest.raises(InsufficientTraceError) as exc_info:
            trainer.train(agent_type="test", traces=traces)
        assert exc_info.value.required == 100
        assert exc_info.value.provided == 10

    def test_tool_stats_computed(self, trained_profile: BehavioralProfile) -> None:
        stats = trained_profile.get_tool_stats("search_kb")
        assert stats is not None
        assert stats.usage_frequency > 0.5
        assert stats.avg_calls_per_trace > 0.0

    def test_volume_stats_computed(self, trained_profile: BehavioralProfile) -> None:
        vs = trained_profile.volume_stats
        assert vs.llm_calls_mean > 0.0
        assert vs.tool_calls_mean > 0.0
        assert vs.total_tokens_mean > 0.0

    def test_markov_chain_trained(self, trained_profile: BehavioralProfile) -> None:
        chain = trained_profile.markov_chain
        assert len(chain.known_states) > 0
        prob = chain.transition_probability("search_kb", "respond")
        assert prob > 0.0

    def test_content_stats_computed(self, trained_profile: BehavioralProfile) -> None:
        cs = trained_profile.content_stats
        assert cs.avg_output_length > 0.0

    def test_known_tool_check(self, trained_profile: BehavioralProfile) -> None:
        assert trained_profile.is_known_tool("search_kb")
        assert not trained_profile.is_known_tool("database_query")

    def test_custom_min_traces(self) -> None:
        trainer = ProfileTrainer(min_traces=5)
        traces = [AgentTrace(agent_type="test") for _ in range(5)]
        profile = trainer.train(agent_type="test", traces=traces)
        assert profile.trace_count == 5

    def test_late_tool_introduced_counts_prior_zero_usage(self) -> None:
        traces = [
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(agent_type="test", tool_calls=[]),
            AgentTrace(
                agent_type="test",
                tool_calls=[{"tool_name": "late_tool"}],
            ),
        ]
        trainer = ProfileTrainer(min_traces=1)
        profile = trainer.train(agent_type="test", traces=traces)
        stats = profile.get_tool_stats("late_tool")
        assert stats is not None
        assert stats.avg_calls_per_trace < 0.1


class TestBehavioralProfile:
    def test_save_and_load(self, trained_profile: BehavioralProfile) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            trained_profile.save(path)

            loaded = BehavioralProfile.load(path)
            assert loaded.agent_type == trained_profile.agent_type
            assert loaded.trace_count == trained_profile.trace_count
            assert loaded.known_tools == trained_profile.known_tools

    def test_get_tool_stats_unknown(self, trained_profile: BehavioralProfile) -> None:
        assert trained_profile.get_tool_stats("nonexistent") is None

    def test_serialization_roundtrip(self, trained_profile: BehavioralProfile) -> None:
        data = trained_profile.model_dump(mode="json")
        restored = BehavioralProfile.model_validate(data)
        assert restored.agent_type == trained_profile.agent_type
        assert len(restored.known_tools) == len(trained_profile.known_tools)

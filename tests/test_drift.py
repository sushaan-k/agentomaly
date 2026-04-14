"""Tests for profile comparison and drift detection."""

from __future__ import annotations

import random

from spectra.drift import compare
from spectra.profiler.profile import BehavioralProfile
from spectra.profiler.trainer import ProfileTrainer
from tests.conftest import _make_trace, _make_training_traces


class TestProfileComparison:
    def test_identical_profiles_no_drift(
        self, trained_profile: BehavioralProfile
    ) -> None:
        """Comparing a profile to itself should yield zero drift."""
        result = compare(trained_profile, trained_profile)
        assert result["new_tools"] == []
        assert result["removed_tools"] == []
        assert result["frequency_drift"] == {}
        assert result["markov_divergence"] == 0.0
        assert result["drift_score"] == 0.0

    def test_detects_new_tools(self) -> None:
        """Adding a tool to the second profile should show up as new."""
        random.seed(100)
        traces_a = _make_training_traces(count=120, agent_type="test-agent")

        # Add traces with a new tool
        random.seed(200)
        traces_b = _make_training_traces(count=100, agent_type="test-agent")
        for _ in range(20):
            traces_b.append(
                _make_trace(
                    agent_type="test-agent",
                    tools=["search_kb", "new_tool", "respond"],
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile_a = trainer.train("test-agent", traces_a)
        profile_b = trainer.train("test-agent", traces_b)

        result = compare(profile_a, profile_b)
        assert "new_tool" in result["new_tools"]

    def test_detects_removed_tools(self) -> None:
        """Removing a tool from the second profile should be flagged."""
        random.seed(300)
        traces_a = _make_training_traces(count=120, agent_type="test-agent")

        # Build traces_b without 'send_email' by using only patterns
        # that don't include it
        random.seed(400)
        traces_b = []
        for _ in range(120):
            pattern = random.choice(
                [
                    ["search_kb", "search_kb", "respond"],
                    ["search_kb", "respond"],
                    ["search_kb", "create_ticket", "respond"],
                ]
            )
            traces_b.append(_make_trace(agent_type="test-agent", tools=pattern))

        trainer = ProfileTrainer(min_traces=100)
        profile_a = trainer.train("test-agent", traces_a)
        profile_b = trainer.train("test-agent", traces_b)

        result = compare(profile_a, profile_b)
        assert "send_email" in result["removed_tools"]

    def test_frequency_drift_nonzero_for_changed_profiles(self) -> None:
        """Profiles with different tool usage rates should show frequency drift."""
        random.seed(500)
        # Profile A: mostly search_kb, respond
        traces_a = []
        for _ in range(120):
            traces_a.append(
                _make_trace(
                    agent_type="test-agent",
                    tools=["search_kb", "respond"],
                )
            )

        # Profile B: heavy search_kb usage
        random.seed(600)
        traces_b = []
        for _ in range(120):
            traces_b.append(
                _make_trace(
                    agent_type="test-agent",
                    tools=["search_kb"] * 5 + ["respond"],
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile_a = trainer.train("test-agent", traces_a)
        profile_b = trainer.train("test-agent", traces_b)

        result = compare(profile_a, profile_b)
        assert len(result["frequency_drift"]) > 0
        assert result["drift_score"] > 0.0

    def test_markov_divergence_increases_with_different_sequences(self) -> None:
        """Profiles with very different action sequences should have higher divergence."""
        random.seed(700)
        traces_a = []
        for _ in range(120):
            traces_a.append(
                _make_trace(
                    agent_type="test-agent",
                    tools=["search_kb", "respond"],
                )
            )

        random.seed(800)
        traces_b = []
        for _ in range(120):
            traces_b.append(
                _make_trace(
                    agent_type="test-agent",
                    tools=["create_ticket", "send_email", "respond"],
                )
            )

        trainer = ProfileTrainer(min_traces=100)
        profile_a = trainer.train("test-agent", traces_a)
        profile_b = trainer.train("test-agent", traces_b)

        result = compare(profile_a, profile_b)
        assert result["markov_divergence"] > 0.0

    def test_empty_profiles_no_crash(self) -> None:
        """Comparing two empty profiles should not raise."""
        profile_a = BehavioralProfile(agent_type="empty-a")
        profile_b = BehavioralProfile(agent_type="empty-b")
        result = compare(profile_a, profile_b)
        assert result["drift_score"] == 0.0

    def test_drift_score_bounded(self, trained_profile: BehavioralProfile) -> None:
        """Drift score should always be in [0, 1]."""
        empty = BehavioralProfile(agent_type="empty")
        result = compare(trained_profile, empty)
        assert 0.0 <= result["drift_score"] <= 1.0

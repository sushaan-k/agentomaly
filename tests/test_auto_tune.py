"""Tests for detector sensitivity auto-tuning."""

from __future__ import annotations

import random

from spectra.models import SensitivityThresholds
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile
from tests.conftest import _make_training_traces


class TestAutoTune:
    def test_auto_tune_lowers_fpr(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        """Auto-tuning on normal traces should yield a low false-positive rate."""
        random.seed(123)
        normal_traces = _make_training_traces(count=120, agent_type="test-agent")

        monitor = Monitor(
            profile=trained_profile,
            sensitivity="high",
        )
        result = monitor.auto_tune(
            traces=normal_traces,
            target_false_positive_rate=0.10,
        )

        assert "z_threshold" in result
        assert "achieved_fpr" in result
        assert result["achieved_fpr"] <= 0.10
        assert result["z_threshold"] >= 1.0

    def test_auto_tune_applies_thresholds(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        """After auto-tuning, detectors should use the new threshold."""
        random.seed(456)
        normal_traces = _make_training_traces(count=120, agent_type="test-agent")

        monitor = Monitor(
            profile=trained_profile,
            sensitivity="medium",
        )
        result = monitor.auto_tune(
            traces=normal_traces,
            target_false_positive_rate=0.05,
        )

        chosen_z = result["z_threshold"]
        for detector in monitor._detectors:
            if hasattr(detector, "thresholds"):
                assert detector.thresholds.medium == chosen_z

    def test_auto_tune_empty_traces(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        """Auto-tuning with no traces returns safe defaults."""
        monitor = Monitor(profile=trained_profile)
        result = monitor.auto_tune(traces=[], target_false_positive_rate=0.05)
        assert result["z_threshold"] == 3.0
        assert result["achieved_fpr"] == 0.0

    def test_auto_tune_strict_target(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        """A very strict target (0%) should push the threshold high."""
        random.seed(789)
        normal_traces = _make_training_traces(count=120, agent_type="test-agent")

        monitor = Monitor(
            profile=trained_profile,
            sensitivity="medium",
        )
        result = monitor.auto_tune(
            traces=normal_traces,
            target_false_positive_rate=0.0,
        )

        assert result["achieved_fpr"] == 0.0
        # With 0% target, threshold should be pushed higher
        assert result["z_threshold"] >= 3.0

    def test_apply_threshold_helper(
        self,
        trained_profile: BehavioralProfile,
    ) -> None:
        """_apply_threshold should update all detectors with thresholds attr."""
        monitor = Monitor(profile=trained_profile)
        monitor._apply_threshold(5.5)

        for detector in monitor._detectors:
            if hasattr(detector, "thresholds"):
                assert isinstance(detector.thresholds, SensitivityThresholds)
                assert detector.thresholds.low == 5.5
                assert detector.thresholds.medium == 5.5

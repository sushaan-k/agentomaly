"""Action sequence anomaly detector.

Uses the learned Markov chain model to detect novel, improbable, or
looping action sequences that deviate from normal agent behavior.
"""

from __future__ import annotations

import logging

from spectra.detectors.base import BaseDetector
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    Sensitivity,
    SensitivityThresholds,
    Severity,
)
from spectra.profiler.profile import BehavioralProfile

logger = logging.getLogger(__name__)


class SequenceAnomalyDetector(BaseDetector):
    """Detects anomalies in the sequence of agent actions.

    Uses the profile's Markov chain to identify:
    1. Novel transitions never observed in training.
    2. Sequences with unusually low probability.
    3. Action loops suggesting the agent is stuck.

    Args:
        sensitivity: Detection sensitivity level.
        thresholds: Z-score thresholds for probability scoring.
        loop_threshold: Number of consecutive repetitions to flag as a loop.
    """

    detector_type = DetectorType.SEQUENCE

    def __init__(
        self,
        sensitivity: Sensitivity = Sensitivity.MEDIUM,
        thresholds: SensitivityThresholds | None = None,
        loop_threshold: int = 3,
    ) -> None:
        super().__init__(sensitivity)
        self.thresholds = thresholds or SensitivityThresholds()
        self.loop_threshold = loop_threshold

    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Analyze action sequences in a trace for anomalies.

        Args:
            trace: The agent execution trace to analyze.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for detected sequence anomalies.
        """
        events: list[AnomalyEvent] = []
        sequence = trace.action_sequence

        if not sequence:
            return events

        events.extend(self._check_novel_transitions(trace, profile, sequence))
        events.extend(self._check_low_probability(trace, profile, sequence))
        events.extend(self._check_loops(trace, profile, sequence))

        return events

    def _check_novel_transitions(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
        sequence: list[str],
    ) -> list[AnomalyEvent]:
        """Detect transitions never observed in training data.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.
            sequence: The action sequence to check.

        Returns:
            List of CRITICAL anomaly events for novel transitions.
        """
        events: list[AnomalyEvent] = []
        novel = profile.markov_chain.has_novel_transition(sequence)

        if novel:
            novel_descriptions = [f"{a} -> {b}" for a, b in novel]
            events.append(
                AnomalyEvent(
                    trace_id=trace.trace_id,
                    agent_type=trace.agent_type,
                    detector_type=self.detector_type,
                    severity=Severity.CRITICAL,
                    title="Novel action sequence detected",
                    description=(
                        f"Agent performed {len(novel)} never-before-seen "
                        f"action transition(s): "
                        f"{', '.join(novel_descriptions[:5])}."
                    ),
                    score=1.0,
                    details={
                        "novel_transitions": [{"from": a, "to": b} for a, b in novel],
                        "full_sequence": sequence,
                    },
                )
            )

        return events

    def _check_low_probability(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
        sequence: list[str],
    ) -> list[AnomalyEvent]:
        """Flag sequences with unusually low probability.

        A very negative log-probability indicates the sequence is highly
        unlikely under the learned model.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.
            sequence: The action sequence to score.

        Returns:
            List of anomaly events for improbable sequences.
        """
        events: list[AnomalyEvent] = []
        log_prob = profile.markov_chain.sequence_log_probability(sequence)

        normalized_log_prob = log_prob / max(len(sequence), 1)
        anomaly_threshold = -5.0 + self.thresholds.get_threshold(self.sensitivity)

        if normalized_log_prob < anomaly_threshold:
            score = min(abs(normalized_log_prob) / 20.0, 1.0)
            severity = Severity.HIGH if score > 0.7 else Severity.MEDIUM
            events.append(
                AnomalyEvent(
                    trace_id=trace.trace_id,
                    agent_type=trace.agent_type,
                    detector_type=self.detector_type,
                    severity=severity,
                    title="Improbable action sequence",
                    description=(
                        f"Action sequence has unusually low probability "
                        f"(normalized log-prob: {normalized_log_prob:.2f}). "
                        f"Sequence: {' -> '.join(sequence[:10])}"
                        f"{'...' if len(sequence) > 10 else ''}."
                    ),
                    score=score,
                    details={
                        "log_probability": log_prob,
                        "normalized_log_probability": normalized_log_prob,
                        "sequence_length": len(sequence),
                        "full_sequence": sequence,
                    },
                )
            )

        return events

    def _check_loops(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
        sequence: list[str],
    ) -> list[AnomalyEvent]:
        """Detect repeated action patterns suggesting the agent is stuck.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.
            sequence: The action sequence to check.

        Returns:
            List of anomaly events for detected loops.
        """
        events: list[AnomalyEvent] = []
        loops = profile.markov_chain.detect_loops(sequence, self.loop_threshold)

        for action, repeat_count in loops:
            severity = Severity.HIGH if repeat_count >= 5 else Severity.MEDIUM
            score = min(repeat_count / 10.0, 1.0)
            events.append(
                AnomalyEvent(
                    trace_id=trace.trace_id,
                    agent_type=trace.agent_type,
                    detector_type=self.detector_type,
                    severity=severity,
                    title=f"Action loop detected: {action}",
                    description=(
                        f"Action '{action}' was repeated {repeat_count} "
                        f"times consecutively, suggesting the agent may "
                        f"be stuck in a loop."
                    ),
                    score=score,
                    details={
                        "action": action,
                        "repeat_count": repeat_count,
                        "loop_threshold": self.loop_threshold,
                        "full_sequence": sequence,
                    },
                )
            )

        return events

"""Content anomaly detector.

Analyzes agent output to detect structural deviations from normal patterns,
such as unexpected code blocks, URLs, or dramatic changes in output length.
"""

from __future__ import annotations

import logging
import re

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

_URL_PATTERN = re.compile(r"https?://\S+")
_CODE_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")
_STRUCTURED_PATTERN = re.compile(r"\{[\s\S]*?\}|\[[\s\S]*?\]")


class ContentAnomalyDetector(BaseDetector):
    """Detects anomalies in agent output content.

    Checks for:
    1. Unexpected structural elements (code, URLs, structured data).
    2. Dramatic changes in output length.

    Args:
        sensitivity: Detection sensitivity level.
        thresholds: Z-score thresholds for length anomalies.
        structure_threshold: Frequency below which a structural element
            is considered unexpected (default 0.05 = seen in <5% of traces).
    """

    detector_type = DetectorType.CONTENT

    def __init__(
        self,
        sensitivity: Sensitivity = Sensitivity.MEDIUM,
        thresholds: SensitivityThresholds | None = None,
        structure_threshold: float = 0.05,
    ) -> None:
        super().__init__(sensitivity)
        self.thresholds = thresholds or SensitivityThresholds()
        self.structure_threshold = structure_threshold

    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Analyze agent output content for anomalies.

        Args:
            trace: The agent execution trace to analyze.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for content anomalies.
        """
        events: list[AnomalyEvent] = []

        if not trace.output:
            return events

        events.extend(self._check_structure_anomalies(trace, profile))
        events.extend(self._check_length_anomaly(trace, profile))

        return events

    def _check_structure_anomalies(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Flag unexpected structural elements in the output.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for unexpected structural elements.
        """
        events: list[AnomalyEvent] = []
        output = trace.output
        cs = profile.content_stats

        checks: list[tuple[str, re.Pattern[str], float]] = [
            ("code blocks", _CODE_PATTERN, cs.contains_code_frequency),
            ("URLs", _URL_PATTERN, cs.contains_urls_frequency),
            (
                "structured data",
                _STRUCTURED_PATTERN,
                cs.contains_structured_data_frequency,
            ),
        ]

        for label, pattern, historical_freq in checks:
            if pattern.search(output) and historical_freq < self.structure_threshold:
                score = 1.0 - historical_freq
                events.append(
                    AnomalyEvent(
                        trace_id=trace.trace_id,
                        agent_type=trace.agent_type,
                        detector_type=self.detector_type,
                        severity=Severity.MEDIUM,
                        title=f"Unexpected output content: {label}",
                        description=(
                            f"Agent output contains {label}, which has only "
                            f"been observed in {historical_freq:.1%} of "
                            f"historical traces."
                        ),
                        score=score,
                        details={
                            "content_type": label,
                            "historical_frequency": historical_freq,
                            "threshold": self.structure_threshold,
                        },
                    )
                )

        return events

    def _check_length_anomaly(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Flag dramatic changes in output length.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for abnormal output length.
        """
        events: list[AnomalyEvent] = []
        cs = profile.content_stats
        output_length = float(len(trace.output))
        threshold = self.thresholds.get_threshold(self.sensitivity)

        z = self._z_score(output_length, cs.avg_output_length, cs.std_output_length)
        if z >= threshold:
            direction = "longer" if output_length > cs.avg_output_length else "shorter"
            score = min(z / 10.0, 1.0)
            severity = Severity.HIGH if z >= 5.0 else Severity.MEDIUM

            events.append(
                AnomalyEvent(
                    trace_id=trace.trace_id,
                    agent_type=trace.agent_type,
                    detector_type=self.detector_type,
                    severity=severity,
                    title="Output length anomaly",
                    description=(
                        f"Output length ({int(output_length)} chars) is "
                        f"significantly {direction} than normal "
                        f"({cs.avg_output_length:.0f} +/- "
                        f"{cs.std_output_length:.0f}). "
                        f"Z-score: {z:.1f}."
                    ),
                    score=score,
                    details={
                        "actual_length": int(output_length),
                        "expected_mean": cs.avg_output_length,
                        "expected_std": cs.std_output_length,
                        "z_score": z,
                        "direction": direction,
                    },
                )
            )

        return events

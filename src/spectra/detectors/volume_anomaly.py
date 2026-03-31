"""Volume and duration anomaly detector.

Uses z-score analysis to flag agent executions with abnormal numbers of
LLM calls, tool calls, token consumption, or wall-clock duration.
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


class VolumeAnomalyDetector(BaseDetector):
    """Detects anomalies in execution volume and duration metrics.

    Compares LLM call count, tool call count, total token usage, and
    wall-clock duration against the profile's statistical baseline
    using z-score analysis.

    Args:
        sensitivity: Detection sensitivity level.
        thresholds: Z-score thresholds per sensitivity level.
    """

    detector_type = DetectorType.VOLUME

    def __init__(
        self,
        sensitivity: Sensitivity = Sensitivity.MEDIUM,
        thresholds: SensitivityThresholds | None = None,
    ) -> None:
        super().__init__(sensitivity)
        self.thresholds = thresholds or SensitivityThresholds()

    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Analyze volume metrics in a trace for anomalies.

        Args:
            trace: The agent execution trace to analyze.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for any volume/duration anomalies.
        """
        events: list[AnomalyEvent] = []
        threshold = self.thresholds.get_threshold(self.sensitivity)
        vs = profile.volume_stats

        tool_count = float(len(trace.tool_calls))
        # Skip the tool-call-count check when both the trace and the
        # profile have zero tool calls -- there is no deviation to report.
        include_tool_count = not (
            tool_count == 0.0
            and vs.tool_calls_mean == 0.0
            and vs.tool_calls_std == 0.0
        )

        checks: list[tuple[str, float, float, float, str]] = [
            (
                "LLM call count",
                float(len(trace.llm_calls)),
                vs.llm_calls_mean,
                vs.llm_calls_std,
                "llm_calls",
            ),
        ]
        if include_tool_count:
            checks.append(
                (
                    "Tool call count",
                    tool_count,
                    vs.tool_calls_mean,
                    vs.tool_calls_std,
                    "tool_calls",
                ),
            )
        checks.extend([
            (
                "Total tokens",
                float(trace.total_tokens),
                vs.total_tokens_mean,
                vs.total_tokens_std,
                "total_tokens",
            ),
            (
                "Duration",
                trace.duration_ms,
                vs.duration_ms_mean,
                vs.duration_ms_std,
                "duration_ms",
            ),
        ])

        for label, actual, mean, std, metric_key in checks:
            z = self._z_score(actual, mean, std)
            if z >= threshold:
                severity = self._severity_from_z_score(z)
                score = min(z / 10.0, 1.0)
                direction = "above" if actual > mean else "below"

                events.append(
                    AnomalyEvent(
                        trace_id=trace.trace_id,
                        agent_type=trace.agent_type,
                        detector_type=self.detector_type,
                        severity=severity,
                        title=f"Volume anomaly: {label}",
                        description=(
                            f"{label} is {actual:.0f}, which is {z:.1f} "
                            f"standard deviations {direction} the mean of "
                            f"{mean:.1f} (+/- {std:.1f})."
                        ),
                        score=score,
                        details={
                            "metric": metric_key,
                            "actual_value": actual,
                            "expected_mean": mean,
                            "expected_std": std,
                            "z_score": z,
                            "direction": direction,
                        },
                    )
                )

        return events

    @staticmethod
    def _severity_from_z_score(z: float) -> Severity:
        """Map a z-score to a severity level.

        Args:
            z: Absolute z-score.

        Returns:
            Severity based on the magnitude of deviation.
        """
        if z >= 5.0:
            return Severity.CRITICAL
        if z >= 3.5:
            return Severity.HIGH
        if z >= 2.5:
            return Severity.MEDIUM
        return Severity.LOW

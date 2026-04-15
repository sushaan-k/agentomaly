"""Tool usage anomaly detector.

Detects when an agent uses tools in ways that deviate from its learned
behavioral profile: never-before-seen tools, unusual frequency, and
unexpected argument patterns.
"""

from __future__ import annotations

import logging
from collections import Counter

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


class ToolAnomalyDetector(BaseDetector):
    """Detects anomalies in tool usage patterns.

    Checks for three types of anomalies:
    1. Never-seen tools: the agent calls a tool not observed in training.
    2. Frequency anomaly: a tool is called significantly more/less than normal.
    3. Argument anomaly: tool arguments contain keys not seen in training.

    Args:
        sensitivity: Detection sensitivity level.
        thresholds: Z-score thresholds for frequency anomalies.
    """

    detector_type = DetectorType.TOOL_USAGE

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
        """Analyze tool usage in a trace for anomalies.

        Args:
            trace: The agent execution trace to analyze.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for any detected tool usage anomalies.
        """
        events: list[AnomalyEvent] = []
        if not trace.tool_calls:
            return events
        events.extend(self._check_novel_tools(trace, profile))
        events.extend(self._check_frequency_anomalies(trace, profile))
        events.extend(self._check_argument_anomalies(trace, profile))
        return events

    def _check_novel_tools(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Flag tools that the agent has never used before.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of CRITICAL anomaly events for never-seen tools.
        """
        events: list[AnomalyEvent] = []
        seen_in_trace: set[str] = set()

        for tc in trace.tool_calls:
            if tc.tool_name in seen_in_trace:
                continue
            seen_in_trace.add(tc.tool_name)

            if not profile.is_known_tool(tc.tool_name):
                events.append(
                    AnomalyEvent(
                        trace_id=trace.trace_id,
                        agent_type=trace.agent_type,
                        detector_type=self.detector_type,
                        severity=Severity.CRITICAL,
                        title=f"Never-seen tool: {tc.tool_name}",
                        description=(
                            f"Agent '{trace.agent_type}' called tool "
                            f"'{tc.tool_name}' which has never been observed "
                            f"in {profile.trace_count} historical traces."
                        ),
                        score=1.0,
                        details={
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "known_tools": sorted(profile.known_tools),
                        },
                    )
                )

        return events

    def _check_frequency_anomalies(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Flag tools called significantly more or less than normal.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for frequency deviations.
        """
        events: list[AnomalyEvent] = []
        tool_counts = Counter(tc.tool_name for tc in trace.tool_calls)
        threshold = self.thresholds.get_threshold(self.sensitivity)

        for tool_name, count in tool_counts.items():
            stats = profile.get_tool_stats(tool_name)
            if stats is None:
                continue

            z = self._z_score(
                count, stats.avg_calls_per_trace, stats.std_calls_per_trace
            )
            if z >= threshold:
                severity = self._severity_from_z_score(z)
                score = min(z / 10.0, 1.0)
                events.append(
                    AnomalyEvent(
                        trace_id=trace.trace_id,
                        agent_type=trace.agent_type,
                        detector_type=self.detector_type,
                        severity=severity,
                        title=f"Tool frequency anomaly: {tool_name}",
                        description=(
                            f"Tool '{tool_name}' called {count} times. "
                            f"Normal: {stats.avg_calls_per_trace:.1f} "
                            f"+/- {stats.std_calls_per_trace:.1f} "
                            f"(z-score: {z:.1f})."
                        ),
                        score=score,
                        details={
                            "tool_name": tool_name,
                            "actual_count": count,
                            "expected_mean": stats.avg_calls_per_trace,
                            "expected_std": stats.std_calls_per_trace,
                            "z_score": z,
                        },
                    )
                )

        return events

    def _check_argument_anomalies(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Flag tool calls with unusual argument patterns.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for unexpected argument keys.
        """
        events: list[AnomalyEvent] = []

        for tc in trace.tool_calls:
            stats = profile.get_tool_stats(tc.tool_name)
            if stats is None or not stats.common_arg_keys:
                continue

            arg_keys = set(tc.arguments.keys())
            common = set(stats.common_arg_keys)
            novel_keys = arg_keys - common

            if novel_keys:
                score = min(len(novel_keys) / max(len(arg_keys), 1), 1.0)
                events.append(
                    AnomalyEvent(
                        trace_id=trace.trace_id,
                        agent_type=trace.agent_type,
                        detector_type=self.detector_type,
                        severity=Severity.MEDIUM,
                        title=f"Unusual tool arguments: {tc.tool_name}",
                        description=(
                            f"Tool '{tc.tool_name}' called with unexpected "
                            f"argument keys: {sorted(novel_keys)}. "
                            f"Common keys: {sorted(common)}."
                        ),
                        score=score,
                        details={
                            "tool_name": tc.tool_name,
                            "novel_arg_keys": sorted(novel_keys),
                            "common_arg_keys": sorted(common),
                            "all_arg_keys": sorted(arg_keys),
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
            Severity level based on the magnitude of the deviation.
        """
        if z >= 5.0:
            return Severity.CRITICAL
        if z >= 3.5:
            return Severity.HIGH
        if z >= 2.5:
            return Severity.MEDIUM
        return Severity.LOW

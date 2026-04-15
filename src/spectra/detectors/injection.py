"""Prompt injection detection via behavioral analysis.

Detects indirect prompt injection by monitoring for sudden behavioral
shifts that correlate with the ingestion of external content. Rather than
trying to identify malicious text in inputs, this detector identifies the
*effect* of a successful injection: abrupt changes in tool usage, action
patterns, and output characteristics.
"""

from __future__ import annotations

import logging

from spectra.detectors.base import BaseDetector
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    DetectorType,
    Sensitivity,
    Severity,
)
from spectra.profiler.profile import BehavioralProfile

logger = logging.getLogger(__name__)


class InjectionDetector(BaseDetector):
    """Detects prompt injection via behavioral shift analysis.

    Monitors for:
    1. Behavioral shift after external content ingestion (tool results).
    2. Sudden appearance of never-seen tools immediately after tool results.
    3. Pattern of normal behavior followed by anomalous behavior (split).

    Args:
        sensitivity: Detection sensitivity level.
        shift_window: Number of actions to consider as "before" and "after"
            a potential injection point.
    """

    detector_type = DetectorType.INJECTION

    def __init__(
        self,
        sensitivity: Sensitivity = Sensitivity.MEDIUM,
        shift_window: int = 3,
    ) -> None:
        super().__init__(sensitivity)
        self.shift_window = shift_window

    def analyze(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Analyze a trace for signs of prompt injection.

        Args:
            trace: The agent execution trace to analyze.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events if injection is suspected.
        """
        events: list[AnomalyEvent] = []
        if not trace.tool_calls:
            return events
        events.extend(self._check_behavioral_shift(trace, profile))
        events.extend(self._check_post_tool_novel_actions(trace, profile))
        return events

    def _check_behavioral_shift(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Detect a split between normal and anomalous behavior segments.

        Divides the tool call sequence into first-half and second-half,
        and checks whether the second half introduces novel tools while
        the first half was normal.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events if a behavioral shift is detected.
        """
        events: list[AnomalyEvent] = []
        tool_calls = trace.tool_calls

        window = self.shift_window
        if len(tool_calls) < max(window * 2, 2):
            return events

        first_half_tools = {tc.tool_name for tc in tool_calls[:window]}
        second_half_tools = {tc.tool_name for tc in tool_calls[-window:]}

        first_half_known = all(profile.is_known_tool(t) for t in first_half_tools)
        second_half_novel = {
            t for t in second_half_tools if not profile.is_known_tool(t)
        }

        if first_half_known and second_half_novel:
            events.append(
                AnomalyEvent(
                    trace_id=trace.trace_id,
                    agent_type=trace.agent_type,
                    detector_type=self.detector_type,
                    severity=Severity.CRITICAL,
                    title="Possible prompt injection: behavioral shift detected",
                    description=(
                        f"Agent behavior shifted mid-execution. First "
                        f"{window} actions used only known tools, but "
                        f"subsequent actions introduced novel tools: "
                        f"{sorted(second_half_novel)}. This pattern is "
                        f"consistent with indirect prompt injection."
                    ),
                    score=0.95,
                    details={
                        "first_half_tools": sorted(first_half_tools),
                        "second_half_tools": sorted(second_half_tools),
                        "novel_tools_in_second_half": sorted(second_half_novel),
                        "window_size": window,
                        "total_tool_calls": len(tool_calls),
                    },
                )
            )

        return events

    def _check_post_tool_novel_actions(
        self,
        trace: AgentTrace,
        profile: BehavioralProfile,
    ) -> list[AnomalyEvent]:
        """Detect novel actions that appear immediately after tool results.

        Correlates the appearance of never-seen tools or unusual action
        patterns with prior tool call results, which may contain injected
        instructions.

        Args:
            trace: The agent execution trace.
            profile: The learned behavioral profile.

        Returns:
            List of anomaly events for post-tool behavioral anomalies.
        """
        events: list[AnomalyEvent] = []
        tool_calls = trace.tool_calls

        if len(tool_calls) < 2:
            return events

        for i in range(1, len(tool_calls)):
            prev_tc = tool_calls[i - 1]
            curr_tc = tool_calls[i]

            if profile.is_known_tool(prev_tc.tool_name) and not profile.is_known_tool(
                curr_tc.tool_name
            ):
                transition_prob = profile.markov_chain.transition_probability(
                    prev_tc.tool_name, curr_tc.tool_name
                )

                if transition_prob == 0.0:
                    events.append(
                        AnomalyEvent(
                            trace_id=trace.trace_id,
                            agent_type=trace.agent_type,
                            detector_type=self.detector_type,
                            severity=Severity.HIGH,
                            title=(f"Suspicious tool call after '{prev_tc.tool_name}'"),
                            description=(
                                f"Novel tool '{curr_tc.tool_name}' was "
                                f"called immediately after "
                                f"'{prev_tc.tool_name}'. The transition "
                                f"'{prev_tc.tool_name}' -> "
                                f"'{curr_tc.tool_name}' was never observed "
                                f"in training data. The result from "
                                f"'{prev_tc.tool_name}' may have contained "
                                f"injected instructions."
                            ),
                            score=0.85,
                            details={
                                "preceding_tool": prev_tc.tool_name,
                                "novel_tool": curr_tc.tool_name,
                                "novel_tool_args": curr_tc.arguments,
                                "preceding_tool_result": prev_tc.result_summary[:200],
                                "transition_probability": transition_prob,
                            },
                        )
                    )

        return events

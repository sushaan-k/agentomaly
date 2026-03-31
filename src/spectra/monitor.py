"""Main monitoring runtime.

The Monitor is the central orchestrator that connects the behavioral
profile, anomaly detectors, and response policy into a single cohesive
runtime for real-time agent monitoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from spectra.detectors.base import BaseDetector
from spectra.detectors.content_anomaly import ContentAnomalyDetector
from spectra.detectors.injection import InjectionDetector
from spectra.detectors.sequence_anomaly import SequenceAnomalyDetector
from spectra.detectors.tool_anomaly import ToolAnomalyDetector
from spectra.detectors.volume_anomaly import VolumeAnomalyDetector
from spectra.exceptions import MonitorNotRunningError
from spectra.models import (
    AgentTrace,
    AnomalyEvent,
    ResponseAction,
    Sensitivity,
    Severity,
)
from spectra.profiler.profile import BehavioralProfile
from spectra.response.alerter import AlertChannel, LogChannel
from spectra.response.blocker import TaskBlocker
from spectra.response.policy import ResponsePolicy
from spectra.trend import Trend, TrendTracker

logger = logging.getLogger(__name__)


class Monitor:
    """Real-time behavioral anomaly monitor for AI agents.

    The Monitor is the primary user-facing class. It accepts a behavioral
    profile, runs all configured detectors against incoming traces, and
    executes the response policy for any detected anomalies.

    Args:
        profile: The trained behavioral profile to monitor against.
        sensitivity: Detection sensitivity preset.
        response_policy: Mapping of severity levels to response actions.
            Accepts both enum types and string shorthands.
        alert_channels: List of channels for delivering anomaly alerts.
        blocker: Custom task blocker for block/quarantine actions.
        detectors: Custom list of detectors. If None, all built-in
            detectors are enabled.

    Example::

        from spectra import Monitor
        from spectra.profiler import BehavioralProfile

        profile = BehavioralProfile.load("profile.json")
        monitor = Monitor(
            profile=profile,
            sensitivity="medium",
            response_policy={"CRITICAL": "block", "HIGH": "alert"},
        )
        monitor.start()
        events = await monitor.analyze(trace)
    """

    def __init__(
        self,
        profile: BehavioralProfile,
        sensitivity: Sensitivity | str = Sensitivity.MEDIUM,
        response_policy: dict[Severity, ResponseAction] | dict[str, str] | None = None,
        alert_channels: list[AlertChannel] | None = None,
        blocker: TaskBlocker | None = None,
        detectors: list[BaseDetector] | None = None,
    ) -> None:
        self.profile = profile
        self.sensitivity = (
            Sensitivity(sensitivity) if isinstance(sensitivity, str) else sensitivity
        )
        self._running = False
        self._event_log: list[AnomalyEvent] = []
        self._trend_tracker = TrendTracker()

        if detectors is not None:
            self._detectors = detectors
        else:
            self._detectors = self._default_detectors()

        self._policy = ResponsePolicy(
            policy=response_policy,
            alert_channels=alert_channels or [LogChannel()],
            blocker=blocker,
        )

    def _default_detectors(self) -> list[BaseDetector]:
        """Create the default set of all built-in detectors.

        Returns:
            List of detector instances configured with the monitor's
            sensitivity level.
        """
        return [
            ToolAnomalyDetector(sensitivity=self.sensitivity),
            SequenceAnomalyDetector(sensitivity=self.sensitivity),
            VolumeAnomalyDetector(sensitivity=self.sensitivity),
            ContentAnomalyDetector(sensitivity=self.sensitivity),
            InjectionDetector(sensitivity=self.sensitivity),
        ]

    def start(self) -> None:
        """Start the monitor.

        After calling start(), the monitor is ready to analyze traces.
        """
        self._running = True
        logger.info(
            "Monitor started",
            extra={
                "agent_type": self.profile.agent_type,
                "sensitivity": self.sensitivity.value,
                "detectors": [type(d).__name__ for d in self._detectors],
            },
        )

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False
        logger.info("Monitor stopped")

    @property
    def is_running(self) -> bool:
        """Whether the monitor is currently active."""
        return self._running

    async def analyze(self, trace: AgentTrace) -> list[AnomalyEvent]:
        """Analyze an agent trace for behavioral anomalies.

        Runs all configured detectors against the trace and executes
        the response policy for each detected anomaly.

        Args:
            trace: The agent execution trace to analyze.

        Returns:
            List of anomaly events detected in this trace.

        Raises:
            MonitorNotRunningError: If the monitor has not been started.
        """
        if not self._running:
            raise MonitorNotRunningError(
                "Monitor must be started before analyzing traces. Call monitor.start()."
            )

        all_events: list[AnomalyEvent] = []

        for detector in self._detectors:
            try:
                events = detector.analyze(trace, self.profile)
                all_events.extend(events)
            except Exception:
                logger.exception(
                    "Detector failed",
                    extra={"detector": type(detector).__name__},
                )

        for event in all_events:
            await self._policy.handle(event)

        self._event_log.extend(all_events)
        self._trend_tracker.record_many(all_events)

        if all_events:
            logger.info(
                "Anomalies detected",
                extra={
                    "trace_id": trace.trace_id,
                    "anomaly_count": len(all_events),
                    "severities": [e.severity.value for e in all_events],
                },
            )

        return all_events

    @property
    def event_log(self) -> list[AnomalyEvent]:
        """All anomaly events detected since the monitor was created."""
        return list(self._event_log)

    def clear_event_log(self) -> None:
        """Clear the anomaly event log."""
        self._event_log.clear()

    def get_trend(self) -> Trend:
        """Return the current anomaly severity trend.

        Analyzes the rolling window of recent anomaly events to determine
        whether severity is escalating, stable, or de-escalating.

        Returns:
            The current :class:`~spectra.trend.Trend` direction.
        """
        return self._trend_tracker.get_trend()

    def auto_tune(
        self,
        traces: list[AgentTrace],
        target_false_positive_rate: float = 0.05,
    ) -> dict[str, float]:
        """Automatically adjust detector z-score thresholds.

        Analyzes a set of known-good traces and calibrates the z-score
        threshold so that the proportion of traces that produce anomaly
        events (false positives) is at or below the target rate.

        The method performs a binary search over z-score values, applying
        each candidate threshold to detectors that support the
        ``thresholds`` attribute, and returns the chosen threshold.

        Args:
            traces: Collection of traces assumed to be normal behavior.
            target_false_positive_rate: Desired maximum fraction of normal
                traces that should trigger anomalies (default 0.05 = 5%).

        Returns:
            Dictionary with ``"z_threshold"`` (the selected value) and
            ``"achieved_fpr"`` (the measured false positive rate).
        """
        if not traces:
            return {"z_threshold": 3.0, "achieved_fpr": 0.0}

        lo, hi = 1.0, 8.0
        best_threshold = hi
        best_fpr = 0.0

        for _ in range(20):
            mid = (lo + hi) / 2.0
            self._apply_threshold(mid)
            fp_count = sum(
                1
                for trace in traces
                if any(
                    d.analyze(trace, self.profile)
                    for d in self._detectors
                )
            )
            fpr = fp_count / len(traces)

            if fpr <= target_false_positive_rate:
                best_threshold = mid
                best_fpr = fpr
                hi = mid
            else:
                lo = mid

        self._apply_threshold(best_threshold)

        logger.info(
            "Auto-tune complete",
            extra={
                "z_threshold": best_threshold,
                "achieved_fpr": best_fpr,
                "target_fpr": target_false_positive_rate,
                "num_traces": len(traces),
            },
        )
        return {"z_threshold": best_threshold, "achieved_fpr": best_fpr}

    def _apply_threshold(self, z_threshold: float) -> None:
        """Set the z-score threshold on all detectors that support it.

        Args:
            z_threshold: The z-score value to apply to the monitor's
                active sensitivity level on each detector's thresholds.
        """
        for detector in self._detectors:
            if hasattr(detector, "thresholds"):
                updated = detector.thresholds.model_copy()
                setattr(updated, self.sensitivity.value, z_threshold)
                detector.thresholds = updated

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the monitor's current state.

        Returns:
            Dictionary with monitor status and anomaly statistics.
        """
        severity_counts: dict[str, int] = {}
        for event in self._event_log:
            key = event.severity.value
            severity_counts[key] = severity_counts.get(key, 0) + 1

        return {
            "running": self._running,
            "agent_type": self.profile.agent_type,
            "sensitivity": self.sensitivity.value,
            "total_anomalies": len(self._event_log),
            "severity_counts": severity_counts,
            "detectors": [type(d).__name__ for d in self._detectors],
        }

    def to_jsonl(self, path: str | Path) -> int:
        """Export all recorded anomaly events to a JSON Lines file.

        Each line in the output file is a self-contained JSON object
        representing one :class:`~spectra.models.AnomalyEvent`, suitable
        for ingestion by log aggregation pipelines (e.g. Elastic, Splunk,
        Datadog).

        Args:
            path: Destination file path.  Parent directories are created
                automatically if they do not exist.

        Returns:
            Number of events written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with path.open("w", encoding="utf-8") as fh:
            for event in self._event_log:
                line = event.model_dump(mode="json")
                fh.write(json.dumps(line, default=str) + "\n")
                count += 1

        logger.info(
            "Exported anomaly events to JSONL",
            extra={"path": str(path), "event_count": count},
        )
        return count

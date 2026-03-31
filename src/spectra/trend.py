"""Anomaly severity trend tracking.

Records anomaly events over time and computes a rolling severity trend
to determine whether anomalous behavior is escalating, stable, or
de-escalating.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from enum import StrEnum

from spectra.models import AnomalyEvent, Severity

logger = logging.getLogger(__name__)

_SEVERITY_SCORES: dict[Severity, float] = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.0,
    Severity.HIGH: 3.0,
    Severity.CRITICAL: 4.0,
}


class Trend(StrEnum):
    """Direction of severity trend over time."""

    ESCALATING = "escalating"
    STABLE = "stable"
    DE_ESCALATING = "de-escalating"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendTracker:
    """Tracks anomaly severity over a rolling window and reports trends.

    The tracker records severity-scored events and computes whether the
    overall anomaly severity is escalating, remaining stable, or
    de-escalating by comparing the mean severity of the first and second
    halves of the window.

    Args:
        window_size: Maximum number of events to retain in the rolling
            window.  Must be at least 4 to compute a meaningful trend.
        escalation_threshold: Minimum difference in mean severity between
            the two halves to classify as escalating or de-escalating.
            Defaults to 0.5.
    """

    def __init__(
        self,
        window_size: int = 20,
        escalation_threshold: float = 0.5,
    ) -> None:
        if window_size < 4:
            raise ValueError("window_size must be at least 4")
        self._window: deque[tuple[datetime, float]] = deque(maxlen=window_size)
        self._window_size = window_size
        self._escalation_threshold = escalation_threshold

    @property
    def window_size(self) -> int:
        """Maximum number of events retained in the rolling window."""
        return self._window_size

    def record(self, event: AnomalyEvent) -> None:
        """Record an anomaly event in the rolling window.

        Args:
            event: The anomaly event to record.
        """
        score = _SEVERITY_SCORES.get(event.severity, 1.0)
        self._window.append((event.timestamp, score))

    def record_many(self, events: list[AnomalyEvent]) -> None:
        """Record multiple anomaly events.

        Args:
            events: Anomaly events to record, in chronological order.
        """
        for event in events:
            self.record(event)

    def get_trend(self) -> Trend:
        """Compute the current severity trend.

        Splits the rolling window into two halves and compares their mean
        severity scores.  If the second half is higher by more than the
        escalation threshold the trend is *escalating*; if lower by more
        than the threshold it is *de-escalating*; otherwise *stable*.

        Returns:
            The current trend direction, or ``INSUFFICIENT_DATA`` if fewer
            than 4 events have been recorded.
        """
        if len(self._window) < 4:
            return Trend.INSUFFICIENT_DATA

        scores = [score for _, score in self._window]
        mid = len(scores) // 2
        first_half_mean = sum(scores[:mid]) / mid
        second_half_mean = sum(scores[mid:]) / (len(scores) - mid)

        diff = second_half_mean - first_half_mean

        if diff >= self._escalation_threshold:
            return Trend.ESCALATING
        if diff <= -self._escalation_threshold:
            return Trend.DE_ESCALATING
        return Trend.STABLE

    def current_mean_severity(self) -> float | None:
        """Return the mean severity score in the current window.

        Returns:
            Mean severity score, or ``None`` if no events recorded.
        """
        if not self._window:
            return None
        scores = [score for _, score in self._window]
        return sum(scores) / len(scores)

    def clear(self) -> None:
        """Clear all recorded events from the window."""
        self._window.clear()

    def snapshot(self) -> dict[str, object]:
        """Return a summary of the current trend state.

        Returns:
            Dictionary with trend direction, event count, and mean severity.
        """
        return {
            "trend": self.get_trend().value,
            "event_count": len(self._window),
            "window_size": self._window_size,
            "mean_severity": self.current_mean_severity(),
        }

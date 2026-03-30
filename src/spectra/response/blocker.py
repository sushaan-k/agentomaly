"""Task blocking and quarantine mechanisms.

Provides the TaskBlocker which can terminate or pause agent tasks
when critical anomalies are detected.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from spectra.models import AnomalyEvent, ResponseAction

logger = logging.getLogger(__name__)

BlockCallback = Callable[[AnomalyEvent], Coroutine[Any, Any, None]]


class TaskBlocker:
    """Manages task blocking and quarantine in response to anomalies.

    The TaskBlocker maintains a registry of active tasks and can terminate
    or quarantine them when directed by the response policy. Users can
    register custom callbacks to define what "blocking" means for their
    specific agent infrastructure.

    Args:
        on_block: Async callback invoked when a task is blocked.
        on_quarantine: Async callback invoked when a task is quarantined.
    """

    def __init__(
        self,
        on_block: BlockCallback | None = None,
        on_quarantine: BlockCallback | None = None,
    ) -> None:
        self._on_block = on_block or self._default_block
        self._on_quarantine = on_quarantine or self._default_quarantine
        self._blocked_traces: set[str] = set()
        self._quarantined_traces: set[str] = set()
        self._quarantine_events: dict[str, asyncio.Event] = {}

    async def execute(self, action: ResponseAction, event: AnomalyEvent) -> None:
        """Execute a response action for an anomaly event.

        Args:
            action: The response action to take.
            event: The anomaly event that triggered this action.
        """
        if action == ResponseAction.BLOCK:
            await self._block(event)
        elif action == ResponseAction.QUARANTINE:
            await self._quarantine(event)

    async def _block(self, event: AnomalyEvent) -> None:
        """Terminate the agent task associated with this anomaly.

        Args:
            event: The anomaly event that triggered the block.
        """
        self._blocked_traces.add(event.trace_id)
        logger.warning(
            "Task BLOCKED",
            extra={
                "trace_id": event.trace_id,
                "reason": event.title,
                "severity": event.severity.value,
            },
        )
        await self._on_block(event)

    async def _quarantine(self, event: AnomalyEvent) -> None:
        """Pause the agent task for human review.

        Args:
            event: The anomaly event that triggered the quarantine.
        """
        self._quarantined_traces.add(event.trace_id)
        release_event = asyncio.Event()
        self._quarantine_events[event.trace_id] = release_event
        logger.warning(
            "Task QUARANTINED",
            extra={
                "trace_id": event.trace_id,
                "reason": event.title,
                "severity": event.severity.value,
            },
        )
        await self._on_quarantine(event)
        await release_event.wait()

    def release_quarantine(self, trace_id: str) -> bool:
        """Release a quarantined task, allowing it to resume.

        Args:
            trace_id: The trace ID of the quarantined task.

        Returns:
            True if the task was quarantined and has been released,
            False if the trace_id was not found in quarantine.
        """
        if trace_id in self._quarantine_events:
            self._quarantine_events[trace_id].set()
            self._quarantined_traces.discard(trace_id)
            del self._quarantine_events[trace_id]
            logger.info("Task released from quarantine", extra={"trace_id": trace_id})
            return True
        return False

    def is_blocked(self, trace_id: str) -> bool:
        """Check if a task has been blocked.

        Args:
            trace_id: The trace ID to check.

        Returns:
            True if the task is blocked.
        """
        return trace_id in self._blocked_traces

    def is_quarantined(self, trace_id: str) -> bool:
        """Check if a task is quarantined.

        Args:
            trace_id: The trace ID to check.

        Returns:
            True if the task is quarantined.
        """
        return trace_id in self._quarantined_traces

    @staticmethod
    async def _default_block(event: AnomalyEvent) -> None:
        """Default block handler that logs the block action.

        Args:
            event: The anomaly event.
        """
        logger.critical(
            "BLOCKED: %s (trace=%s, severity=%s)",
            event.title,
            event.trace_id,
            event.severity.value,
        )

    @staticmethod
    async def _default_quarantine(event: AnomalyEvent) -> None:
        """Default quarantine handler that logs the quarantine action.

        Args:
            event: The anomaly event.
        """
        logger.warning(
            "QUARANTINED: %s (trace=%s, severity=%s)",
            event.title,
            event.trace_id,
            event.severity.value,
        )

"""Response policy engine.

Maps anomaly severity levels to response actions and coordinates the
execution of those actions through alert channels and the task blocker.
"""

from __future__ import annotations

import logging

from spectra.models import AnomalyEvent, ResponseAction, Severity
from spectra.response.alerter import AlertChannel, LogChannel
from spectra.response.blocker import TaskBlocker

logger = logging.getLogger(__name__)

DEFAULT_POLICY: dict[Severity, ResponseAction] = {
    Severity.LOW: ResponseAction.LOG,
    Severity.MEDIUM: ResponseAction.ALERT,
    Severity.HIGH: ResponseAction.QUARANTINE,
    Severity.CRITICAL: ResponseAction.BLOCK,
}


class ResponsePolicy:
    """Maps anomaly events to response actions and executes them.

    The response policy determines what happens when an anomaly is detected.
    It coordinates alerting, blocking, and quarantine based on configurable
    severity-to-action mappings.

    Args:
        policy: Mapping from severity levels to response actions.
        alert_channels: List of alert channels for delivering notifications.
        blocker: Task blocker for block/quarantine actions.
    """

    def __init__(
        self,
        policy: dict[Severity, ResponseAction] | dict[str, str] | None = None,
        alert_channels: list[AlertChannel] | None = None,
        blocker: TaskBlocker | None = None,
    ) -> None:
        self._policy = self._normalize_policy(policy or DEFAULT_POLICY)
        self._channels = alert_channels or [LogChannel()]
        self._blocker = blocker or TaskBlocker()

    async def handle(self, event: AnomalyEvent) -> ResponseAction:
        """Determine and execute the appropriate response for an anomaly.

        Args:
            event: The anomaly event to handle.

        Returns:
            The response action that was taken.
        """
        action = self._policy.get(event.severity, ResponseAction.LOG)
        event.action_taken = action

        logger.info(
            "Handling anomaly",
            extra={
                "severity": event.severity.value,
                "action": action.value,
                "title": event.title,
                "trace_id": event.trace_id,
            },
        )

        if action in (
            ResponseAction.ALERT,
            ResponseAction.QUARANTINE,
            ResponseAction.BLOCK,
        ):
            await self._send_alerts(event)

        if action in (ResponseAction.BLOCK, ResponseAction.QUARANTINE):
            await self._blocker.execute(action, event)

        return action

    async def _send_alerts(self, event: AnomalyEvent) -> None:
        """Deliver the anomaly event to all configured alert channels.

        Failures in individual channels are logged but do not prevent
        other channels from receiving the alert.

        Args:
            event: The anomaly event to deliver.
        """
        for channel in self._channels:
            try:
                await channel.send(event)
            except Exception:
                logger.exception(
                    "Alert channel failed",
                    extra={"channel": type(channel).__name__},
                )

    @staticmethod
    def _normalize_policy(
        policy: dict[Severity, ResponseAction] | dict[str, str],
    ) -> dict[Severity, ResponseAction]:
        """Normalize a policy dict to use proper enum types.

        Accepts both ``{Severity.HIGH: ResponseAction.BLOCK}`` and
        ``{"HIGH": "block"}`` formats for user convenience.

        Args:
            policy: Raw policy mapping.

        Returns:
            Normalized mapping with enum keys and values.
        """
        normalized: dict[Severity, ResponseAction] = {}
        for key, value in policy.items():
            sev = Severity(key) if isinstance(key, str) else key
            act = ResponseAction(value) if isinstance(value, str) else value
            normalized[sev] = act
        return normalized

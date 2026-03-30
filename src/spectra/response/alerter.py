"""Alert channels for delivering anomaly notifications.

Provides an abstract AlertChannel interface and concrete implementations
for logging, webhooks, Slack, and PagerDuty. Includes rate-limiting
support to prevent alert floods.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

import httpx

from spectra.exceptions import AlertChannelError
from spectra.models import AnomalyEvent

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS: float = 60.0


class AlertChannel(abc.ABC):
    """Abstract base class for alert delivery channels."""

    @abc.abstractmethod
    async def send(self, event: AnomalyEvent) -> None:
        """Send an anomaly alert through this channel.

        Args:
            event: The anomaly event to report.

        Raises:
            AlertChannelError: If delivery fails.
        """


class RateLimitedChannel(AlertChannel):
    """Wrapper that rate-limits an inner AlertChannel.

    Prevents sending more than one alert per ``cooldown_seconds`` for
    each detector type. Suppressed alerts are logged at DEBUG level.

    Args:
        channel: The inner alert channel to delegate to.
        cooldown_seconds: Minimum seconds between alerts for the same
            detector type. Defaults to 60.
    """

    def __init__(
        self,
        channel: AlertChannel,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._channel = channel
        self._cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, float] = {}

    async def send(self, event: AnomalyEvent) -> None:
        """Send the alert if the cooldown has elapsed for its detector type.

        Args:
            event: The anomaly event to deliver.
        """
        now = time.monotonic()
        key = event.detector_type.value
        last = self._last_sent.get(key, 0.0)

        if now - last < self._cooldown_seconds:
            logger.debug(
                "Alert suppressed (cooldown)",
                extra={
                    "detector_type": key,
                    "cooldown_seconds": self._cooldown_seconds,
                },
            )
            return

        await self._channel.send(event)
        self._last_sent[key] = now


class LogChannel(AlertChannel):
    """Alert channel that writes anomaly events to the Python logger.

    Useful for local development and as a fallback channel.

    Args:
        logger_name: Name of the logger to write to.
    """

    def __init__(self, logger_name: str = "spectra.alerts") -> None:
        self._logger = logging.getLogger(logger_name)

    async def send(self, event: AnomalyEvent) -> None:
        """Log the anomaly event at the appropriate severity level.

        Args:
            event: The anomaly event to log.
        """
        level_map = {
            "LOW": logging.INFO,
            "MEDIUM": logging.WARNING,
            "HIGH": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(event.severity.value, logging.WARNING)
        self._logger.log(
            level,
            "[%s] %s — %s (score=%.2f, trace=%s)",
            event.severity.value,
            event.title,
            event.description,
            event.score,
            event.trace_id,
        )


class WebhookChannel(AlertChannel):
    """Alert channel that delivers anomaly events via HTTP POST webhook.

    Sends a JSON payload to the configured URL containing the full
    anomaly event data.

    Args:
        url: The webhook endpoint URL.
        headers: Optional additional HTTP headers.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    async def send(self, event: AnomalyEvent) -> None:
        """POST the anomaly event as JSON to the webhook URL.

        Args:
            event: The anomaly event to deliver.

        Raises:
            AlertChannelError: If the HTTP request fails.
        """
        payload = event.model_dump(mode="json")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    headers={"Content-Type": "application/json", **self.headers},
                    timeout=self.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AlertChannelError(
                f"Webhook delivery failed to {self.url}: {exc}"
            ) from exc

        logger.debug("Webhook alert sent", extra={"url": self.url})


class SlackWebhook(AlertChannel):
    """Alert channel that delivers anomaly events to a Slack channel.

    Formats the anomaly event into a Slack-compatible message with
    rich formatting.

    Args:
        webhook_url: Slack incoming webhook URL.
        timeout: Request timeout in seconds.
    """

    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send(self, event: AnomalyEvent) -> None:
        """Send a formatted alert to Slack via incoming webhook.

        Args:
            event: The anomaly event to deliver.

        Raises:
            AlertChannelError: If the Slack API request fails.
        """
        severity_emoji = {
            "LOW": ":information_source:",
            "MEDIUM": ":warning:",
            "HIGH": ":rotating_light:",
            "CRITICAL": ":fire:",
        }
        emoji = severity_emoji.get(event.severity.value, ":question:")
        payload = self._build_slack_payload(event, emoji)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AlertChannelError(f"Slack webhook delivery failed: {exc}") from exc

        logger.debug("Slack alert sent")

    @staticmethod
    def _build_slack_payload(event: AnomalyEvent, emoji: str) -> dict[str, Any]:
        """Build a Slack Block Kit message payload.

        Args:
            event: The anomaly event.
            emoji: Severity-appropriate emoji.

        Returns:
            Slack-compatible JSON payload.
        """
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {event.severity.value} ANOMALY",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*{event.title}*\n"
                            f"Agent: `{event.agent_type}` | "
                            f"Detector: `{event.detector_type.value}` | "
                            f"Score: `{event.score:.2f}`"
                        ),
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": event.description,
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"Trace: `{event.trace_id}` | "
                                f"Time: `{event.timestamp.isoformat()}`"
                            ),
                        }
                    ],
                },
            ]
        }


class PagerDutyChannel(AlertChannel):
    """Alert channel that delivers anomaly events to PagerDuty.

    Sends a PagerDuty Events API v2 trigger event with the anomaly
    metadata attached as custom details.

    Args:
        routing_key: PagerDuty Events API integration key.
        source: Originating service or system name shown in PagerDuty.
        timeout: Request timeout in seconds.
        endpoint: PagerDuty Events API endpoint URL.
    """

    DEFAULT_ENDPOINT = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        source: str = "spectra",
        timeout: float = 10.0,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self.routing_key = routing_key
        self.source = source
        self.timeout = timeout
        self.endpoint = endpoint

    async def send(self, event: AnomalyEvent) -> None:
        """Send an anomaly event to PagerDuty as a trigger event."""
        payload = self._build_pagerduty_payload(event)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AlertChannelError(
                f"PagerDuty delivery failed to {self.endpoint}: {exc}"
            ) from exc

        logger.debug(
            "PagerDuty alert sent",
            extra={"endpoint": self.endpoint, "trace_id": event.trace_id},
        )

    def _build_pagerduty_payload(self, event: AnomalyEvent) -> dict[str, Any]:
        """Build a PagerDuty Events API v2 trigger payload."""
        return {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": event.event_id,
            "payload": {
                "summary": event.title,
                "source": self.source,
                "severity": self._severity_to_pagerduty(event.severity.value),
                "component": event.detector_type.value,
                "group": event.agent_type,
                "class": "spectra-anomaly",
                "custom_details": {
                    "event_id": event.event_id,
                    "trace_id": event.trace_id,
                    "timestamp": event.timestamp.isoformat(),
                    "agent_type": event.agent_type,
                    "detector_type": event.detector_type.value,
                    "severity": event.severity.value,
                    "score": event.score,
                    "title": event.title,
                    "description": event.description,
                    "action_taken": (
                        event.action_taken.value if event.action_taken else None
                    ),
                    "details": event.details,
                    "metadata": event.metadata,
                },
            },
        }

    @staticmethod
    def _severity_to_pagerduty(severity: str) -> str:
        """Map spectra severities to PagerDuty severities."""
        return {
            "LOW": "info",
            "MEDIUM": "warning",
            "HIGH": "error",
            "CRITICAL": "critical",
        }.get(severity, "warning")


__all__ = [
    "AlertChannel",
    "LogChannel",
    "PagerDutyChannel",
    "RateLimitedChannel",
    "SlackWebhook",
    "WebhookChannel",
]

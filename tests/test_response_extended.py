"""Extended tests for response engine: alerter, blocker, policy edge cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from spectra.exceptions import AlertChannelError
from spectra.models import (
    AnomalyEvent,
    DetectorType,
    ResponseAction,
    Severity,
)
from spectra.response.alerter import (
    LogChannel,
    PagerDutyChannel,
    SlackWebhook,
    WebhookChannel,
)
from spectra.response.blocker import TaskBlocker
from spectra.response.policy import ResponsePolicy


def _make_event(
    severity: Severity = Severity.CRITICAL,
    trace_id: str = "test-trace-001",
) -> AnomalyEvent:
    return AnomalyEvent(
        trace_id=trace_id,
        agent_type="test-agent",
        detector_type=DetectorType.TOOL_USAGE,
        severity=severity,
        title="Test anomaly",
        description="A test anomaly event.",
        score=0.9,
    )


class TestWebhookChannelSend:
    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        event = _make_event()
        channel = WebhookChannel(url="https://example.com/hook")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await channel.send(event)
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_raises(self) -> None:
        event = _make_event()
        channel = WebhookChannel(url="https://example.com/hook")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(AlertChannelError, match="Webhook delivery failed"):
                await channel.send(event)

    @pytest.mark.asyncio
    async def test_custom_headers(self) -> None:
        event = _make_event()
        channel = WebhookChannel(
            url="https://example.com/hook",
            headers={"Authorization": "Bearer token123"},
            timeout=5.0,
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await channel.send(event)
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer token123"
            assert call_kwargs[1]["timeout"] == 5.0


class TestSlackWebhookSend:
    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        event = _make_event()
        channel = SlackWebhook(webhook_url="https://hooks.slack.com/services/XXX")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await channel.send(event)
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_raises(self) -> None:
        event = _make_event()
        channel = SlackWebhook(webhook_url="https://hooks.slack.com/services/XXX")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Slack error"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(
                AlertChannelError, match="Slack webhook delivery failed"
            ):
                await channel.send(event)

    def test_build_payload_all_severities(self) -> None:
        for severity in Severity:
            event = _make_event(severity=severity)
            emoji_map = {
                "LOW": ":information_source:",
                "MEDIUM": ":warning:",
                "HIGH": ":rotating_light:",
                "CRITICAL": ":fire:",
            }
            emoji = emoji_map[severity.value]
            payload = SlackWebhook._build_slack_payload(event, emoji)
            assert "blocks" in payload
            header_text = payload["blocks"][0]["text"]["text"]
            assert severity.value in header_text


class TestPagerDutyChannelSend:
    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        event = _make_event()
        channel = PagerDutyChannel(
            routing_key="PD_ROUTING_KEY",
            source="spectra",
            timeout=7.5,
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await channel.send(event)
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args[1]
            assert call_kwargs["timeout"] == 7.5
            assert call_kwargs["json"]["routing_key"] == "PD_ROUTING_KEY"
            assert call_kwargs["json"]["payload"]["source"] == "spectra"

    @pytest.mark.asyncio
    async def test_send_failure_raises(self) -> None:
        event = _make_event()
        channel = PagerDutyChannel(routing_key="PD_ROUTING_KEY")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("PD error"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(AlertChannelError, match="PagerDuty delivery failed"):
                await channel.send(event)

    def test_build_payload_maps_severity(self) -> None:
        channel = PagerDutyChannel(routing_key="PD_ROUTING_KEY")
        for severity, expected in (
            (Severity.LOW, "info"),
            (Severity.MEDIUM, "warning"),
            (Severity.HIGH, "error"),
            (Severity.CRITICAL, "critical"),
        ):
            event = _make_event(severity=severity)
            payload = channel._build_pagerduty_payload(event)
            assert payload["payload"]["severity"] == expected


class TestLogChannelExtended:
    @pytest.mark.asyncio
    async def test_send_all_severities(self) -> None:
        channel = LogChannel()
        for severity in Severity:
            event = _make_event(severity=severity)
            await channel.send(event)

    @pytest.mark.asyncio
    async def test_custom_logger_name(self) -> None:
        channel = LogChannel(logger_name="test.alerts")
        event = _make_event()
        await channel.send(event)


class TestTaskBlockerExtended:
    @pytest.mark.asyncio
    async def test_default_block_callback(self) -> None:
        """Default block callback just logs, doesn't crash."""
        blocker = TaskBlocker()
        event = _make_event()
        await blocker.execute(ResponseAction.BLOCK, event)
        assert blocker.is_blocked("test-trace-001")

    @pytest.mark.asyncio
    async def test_default_quarantine_callback(self) -> None:
        """Default quarantine callback just logs, doesn't crash."""
        import asyncio

        blocker = TaskBlocker()
        event = _make_event()
        task = asyncio.create_task(blocker.execute(ResponseAction.QUARANTINE, event))
        await asyncio.sleep(0)
        assert blocker.is_quarantined("test-trace-001")
        assert blocker.release_quarantine("test-trace-001") is True
        await task

    @pytest.mark.asyncio
    async def test_execute_log_action_noop(self) -> None:
        """LOG and ALERT actions are no-ops for the blocker."""
        blocker = TaskBlocker()
        event = _make_event()
        await blocker.execute(ResponseAction.LOG, event)
        assert not blocker.is_blocked("test-trace-001")
        assert not blocker.is_quarantined("test-trace-001")

    @pytest.mark.asyncio
    async def test_is_blocked_unknown_trace(self) -> None:
        blocker = TaskBlocker()
        assert not blocker.is_blocked("unknown")

    @pytest.mark.asyncio
    async def test_is_quarantined_unknown_trace(self) -> None:
        blocker = TaskBlocker()
        assert not blocker.is_quarantined("unknown")

    @pytest.mark.asyncio
    async def test_multiple_blocks(self) -> None:
        blocker = TaskBlocker()
        for i in range(3):
            event = _make_event(trace_id=f"trace-{i}")
            await blocker.execute(ResponseAction.BLOCK, event)
        assert blocker.is_blocked("trace-0")
        assert blocker.is_blocked("trace-1")
        assert blocker.is_blocked("trace-2")


class TestResponsePolicyExtended:
    @pytest.mark.asyncio
    async def test_alert_action_sends_to_channels(self) -> None:
        mock_channel = AsyncMock()
        policy = ResponsePolicy(
            policy={Severity.MEDIUM: ResponseAction.ALERT},
            alert_channels=[mock_channel],
        )
        event = _make_event(severity=Severity.MEDIUM)
        action = await policy.handle(event)
        assert action == ResponseAction.ALERT
        mock_channel.send.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_channel_failure_does_not_crash(self) -> None:
        """If an alert channel fails, other channels still work."""
        failing_channel = AsyncMock()
        failing_channel.send = AsyncMock(side_effect=Exception("channel error"))
        working_channel = AsyncMock()

        policy = ResponsePolicy(
            policy={Severity.HIGH: ResponseAction.ALERT},
            alert_channels=[failing_channel, working_channel],
        )
        event = _make_event(severity=Severity.HIGH)
        action = await policy.handle(event)
        assert action == ResponseAction.ALERT
        working_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_block_action_triggers_blocker_and_alerts(self) -> None:
        mock_channel = AsyncMock()
        blocked: list[str] = []

        async def on_block(event: AnomalyEvent) -> None:
            blocked.append(event.trace_id)

        blocker = TaskBlocker(on_block=on_block)
        policy = ResponsePolicy(
            policy={Severity.CRITICAL: ResponseAction.BLOCK},
            alert_channels=[mock_channel],
            blocker=blocker,
        )
        event = _make_event(severity=Severity.CRITICAL)
        action = await policy.handle(event)
        assert action == ResponseAction.BLOCK
        assert "test-trace-001" in blocked
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_quarantine_action_triggers_blocker(self) -> None:
        import asyncio

        quarantined: list[str] = []

        async def on_quarantine(event: AnomalyEvent) -> None:
            quarantined.append(event.trace_id)

        blocker = TaskBlocker(on_quarantine=on_quarantine)
        policy = ResponsePolicy(
            policy={Severity.HIGH: ResponseAction.QUARANTINE},
            blocker=blocker,
        )
        event = _make_event(severity=Severity.HIGH)
        task = asyncio.create_task(policy.handle(event))
        await asyncio.sleep(0)
        assert blocker.is_quarantined("test-trace-001")
        assert blocker.release_quarantine("test-trace-001") is True
        action = await task
        assert action == ResponseAction.QUARANTINE
        assert "test-trace-001" in quarantined

    @pytest.mark.asyncio
    async def test_unknown_severity_defaults_to_log(self) -> None:
        """Severity not in policy defaults to LOG."""
        policy = ResponsePolicy(
            policy={Severity.CRITICAL: ResponseAction.BLOCK},
        )
        event = _make_event(severity=Severity.LOW)
        action = await policy.handle(event)
        assert action == ResponseAction.LOG

    @pytest.mark.asyncio
    async def test_log_action_no_alerts_sent(self) -> None:
        mock_channel = AsyncMock()
        policy = ResponsePolicy(
            policy={Severity.LOW: ResponseAction.LOG},
            alert_channels=[mock_channel],
        )
        event = _make_event(severity=Severity.LOW)
        action = await policy.handle(event)
        assert action == ResponseAction.LOG
        mock_channel.send.assert_not_called()

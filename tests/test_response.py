"""Tests for the response engine (policy, alerter, blocker)."""

from __future__ import annotations

import pytest

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


class TestLogChannel:
    @pytest.mark.asyncio
    async def test_send_logs(self) -> None:
        channel = LogChannel()
        event = _make_event()
        await channel.send(event)


class TestWebhookChannel:
    @pytest.mark.asyncio
    async def test_init(self) -> None:
        channel = WebhookChannel(
            url="https://example.com/hook",
            headers={"Authorization": "Bearer test"},
        )
        assert channel.url == "https://example.com/hook"
        assert channel.timeout == 10.0


class TestSlackWebhook:
    def test_build_payload(self) -> None:
        event = _make_event()
        payload = SlackWebhook._build_slack_payload(event, ":fire:")
        assert "blocks" in payload
        assert len(payload["blocks"]) == 4
        assert payload["blocks"][0]["type"] == "header"


class TestPagerDutyChannel:
    def test_build_payload(self) -> None:
        event = _make_event()
        channel = PagerDutyChannel(routing_key="routing-key-123")
        payload = channel._build_pagerduty_payload(event)
        assert payload["routing_key"] == "routing-key-123"
        assert payload["event_action"] == "trigger"
        assert payload["payload"]["summary"] == event.title
        assert payload["payload"]["severity"] == "critical"
        assert payload["payload"]["custom_details"]["trace_id"] == event.trace_id


class TestTaskBlocker:
    @pytest.mark.asyncio
    async def test_block(self) -> None:
        blocked: list[str] = []

        async def on_block(event: AnomalyEvent) -> None:
            blocked.append(event.trace_id)

        blocker = TaskBlocker(on_block=on_block)
        event = _make_event()
        await blocker.execute(ResponseAction.BLOCK, event)
        assert "test-trace-001" in blocked
        assert blocker.is_blocked("test-trace-001")

    @pytest.mark.asyncio
    async def test_quarantine_and_release(self) -> None:
        import asyncio

        quarantined: list[str] = []

        async def on_quarantine(event: AnomalyEvent) -> None:
            quarantined.append(event.trace_id)

        blocker = TaskBlocker(on_quarantine=on_quarantine)
        event = _make_event()
        task = asyncio.create_task(blocker.execute(ResponseAction.QUARANTINE, event))
        await asyncio.sleep(0)
        assert blocker.is_quarantined("test-trace-001")

        released = blocker.release_quarantine("test-trace-001")
        assert released is True
        await task
        assert not blocker.is_quarantined("test-trace-001")

    @pytest.mark.asyncio
    async def test_release_unknown_trace(self) -> None:
        blocker = TaskBlocker()
        assert blocker.release_quarantine("nonexistent") is False


class TestResponsePolicy:
    @pytest.mark.asyncio
    async def test_default_policy(self) -> None:
        policy = ResponsePolicy()
        event = _make_event(severity=Severity.LOW)
        action = await policy.handle(event)
        assert action == ResponseAction.LOG

    @pytest.mark.asyncio
    async def test_custom_policy(self) -> None:
        policy = ResponsePolicy(
            policy={"CRITICAL": "block", "HIGH": "alert"},
        )
        event = _make_event(severity=Severity.CRITICAL)
        action = await policy.handle(event)
        assert action == ResponseAction.BLOCK

    @pytest.mark.asyncio
    async def test_policy_sets_action_on_event(self) -> None:
        policy = ResponsePolicy()
        event = _make_event(severity=Severity.MEDIUM)
        await policy.handle(event)
        assert event.action_taken == ResponseAction.ALERT

    @pytest.mark.asyncio
    async def test_string_based_policy(self) -> None:
        policy = ResponsePolicy(
            policy={"LOW": "log", "MEDIUM": "alert"},
        )
        event = _make_event(severity=Severity.MEDIUM)
        action = await policy.handle(event)
        assert action == ResponseAction.ALERT

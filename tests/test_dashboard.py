"""Tests for the FastAPI dashboard endpoints."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from spectra.dashboard.app import create_app, register_monitor
from spectra.models import (
    AgentTrace,
    LLMCall,
    ToolCall,
)
from spectra.monitor import Monitor
from spectra.profiler.profile import BehavioralProfile
from spectra.profiler.trainer import ProfileTrainer


def _make_profile() -> BehavioralProfile:
    """Build a trained profile for dashboard tests."""
    random.seed(42)
    traces = []
    for _ in range(120):
        started = datetime.now(UTC)
        traces.append(
            AgentTrace(
                agent_type="test-agent",
                started_at=started,
                ended_at=started + timedelta(minutes=2),
                tool_calls=[
                    ToolCall(tool_name="search_kb"),
                    ToolCall(tool_name="respond"),
                ],
                llm_calls=[LLMCall(total_tokens=1000)],
                output="Standard response.",
            )
        )
    trainer = ProfileTrainer(min_traces=100)
    return trainer.train(agent_type="test-agent", traces=traces)


class TestDashboardNoMonitor:
    """Tests for dashboard when no monitor is registered."""

    @pytest.fixture
    def app(self):
        return create_app(monitor=None)

    @pytest.mark.asyncio
    async def test_health(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_status_no_monitor(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/status")
            assert response.status_code == 200
            assert response.json() == {"error": "No monitor registered"}

    @pytest.mark.asyncio
    async def test_events_no_monitor(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events")
            assert response.status_code == 200
            assert response.json() == []

    @pytest.mark.asyncio
    async def test_profile_no_monitor(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/profile")
            assert response.status_code == 200
            assert response.json() == {"error": "No monitor registered"}


class TestDashboardWithMonitor:
    """Tests for dashboard with a running monitor."""

    @pytest.fixture
    def monitor(self) -> Monitor:
        profile = _make_profile()
        mon = Monitor(
            profile=profile,
            sensitivity="medium",
            response_policy={
                "LOW": "log",
                "MEDIUM": "alert",
                "HIGH": "alert",
                "CRITICAL": "alert",
            },
        )
        mon.start()
        return mon

    @pytest.fixture
    def app(self, monitor):
        return create_app(monitor=monitor)

    @pytest.mark.asyncio
    async def test_status_with_monitor(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is True
            assert data["agent_type"] == "test-agent"
            assert data["sensitivity"] == "medium"

    @pytest.mark.asyncio
    async def test_profile_with_monitor(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/profile")
            assert response.status_code == 200
            data = response.json()
            assert data["agent_type"] == "test-agent"
            assert data["trace_count"] == 120
            assert "search_kb" in data["known_tools"]
            assert "volume_stats" in data
            assert "content_stats" in data

    @pytest.mark.asyncio
    async def test_events_empty(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events")
            assert response.status_code == 200
            assert response.json() == []

    @pytest.mark.asyncio
    async def test_events_after_analysis(self, app, monitor) -> None:
        # Create an anomalous trace
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="evil_tool")],
        )
        await monitor.analyze(trace)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events")
            assert response.status_code == 200
            events = response.json()
            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_events_severity_filter(self, app, monitor) -> None:
        # Create an anomalous trace that produces CRITICAL events
        trace = AgentTrace(
            agent_type="test-agent",
            tool_calls=[ToolCall(tool_name="evil_tool")],
        )
        await monitor.analyze(trace)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events?severity=CRITICAL")
            assert response.status_code == 200
            events = response.json()
            assert all(e["severity"] == "CRITICAL" for e in events)

    @pytest.mark.asyncio
    async def test_events_limit(self, app, monitor) -> None:
        # Create multiple anomalous traces
        for _ in range(5):
            trace = AgentTrace(
                agent_type="test-agent",
                tool_calls=[ToolCall(tool_name="evil_tool")],
            )
            await monitor.analyze(trace)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events?limit=2")
            assert response.status_code == 200
            events = response.json()
            assert len(events) <= 2

    @pytest.mark.asyncio
    async def test_events_invalid_severity_returns_400(self, app, monitor) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/events?severity=bogus")
            assert response.status_code == 400


class TestRegisterMonitor:
    def test_register_monitor(self) -> None:
        profile = _make_profile()
        mon = Monitor(profile=profile)
        register_monitor(mon)
        # Verification: the function should not raise

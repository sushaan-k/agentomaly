"""FastAPI web dashboard for spectra monitoring.

Provides a lightweight REST API and dashboard for viewing anomaly events,
monitor status, and agent behavioral profiles.
"""

from __future__ import annotations

import logging
from typing import Any

from spectra.models import Severity
from spectra.monitor import Monitor

logger = logging.getLogger(__name__)

_monitor_instance: Monitor | None = None


def create_app(
    monitor: Monitor | None = None,
    api_key: str | None = None,
) -> Any:
    """Create the FastAPI application for the spectra dashboard.

    Args:
        monitor: The Monitor instance to expose via the dashboard.
            If None, endpoints will return empty data until a monitor
            is registered.
        api_key: Optional API key for authentication. When set, all API
            endpoints require the key as a Bearer token in the
            Authorization header or as a ``key`` query parameter.
            The ``/health`` endpoint is always public.

    Returns:
        A FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the dashboard. "
            "Install with: pip install spectra-ai[dashboard]"
        ) from exc

    global _monitor_instance
    _monitor_instance = monitor

    app = FastAPI(
        title="spectra Dashboard",
        description="Runtime behavioral anomaly detection for AI agents",
        version="0.1.0",
    )

    if api_key is not None:

        @app.middleware("http")
        async def _auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
            if request.url.path == "/health":
                return await call_next(request)

            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer ") and auth_header[7:] == api_key:
                return await call_next(request)

            query_key = request.query_params.get("key")
            if query_key == api_key:
                return await call_next(request)

            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        if _monitor_instance is None:
            return {"error": "No monitor registered"}
        return _monitor_instance.summary()

    @app.get("/api/events")
    async def events(
        severity: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if _monitor_instance is None:
            return []
        event_log = _monitor_instance.event_log
        if severity:
            try:
                sev = Severity(severity.upper())
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid severity value"
                ) from exc
            event_log = [e for e in event_log if e.severity == sev]
        event_log = event_log[-limit:]
        return [e.model_dump(mode="json") for e in event_log]

    @app.get("/api/profile")
    async def profile() -> dict[str, Any]:
        if _monitor_instance is None:
            return {"error": "No monitor registered"}
        return {
            "agent_type": _monitor_instance.profile.agent_type,
            "trace_count": _monitor_instance.profile.trace_count,
            "known_tools": sorted(_monitor_instance.profile.known_tools),
            "volume_stats": _monitor_instance.profile.volume_stats.model_dump(),
            "content_stats": _monitor_instance.profile.content_stats.model_dump(),
        }

    return app


def register_monitor(monitor: Monitor) -> None:
    """Register a Monitor instance with the global dashboard.

    Args:
        monitor: The Monitor instance to expose.
    """
    global _monitor_instance
    _monitor_instance = monitor

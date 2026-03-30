"""Decorator-based instrumentation for AI agents.

Provides the ``@spectra.trace`` decorator that wraps agent functions
to automatically capture execution traces.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ParamSpec, TypeVar

from spectra.models import AgentTrace, LLMCall, ToolCall

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_MAX_COMPLETED_TRACES = 1000

_active_traces: dict[str, AgentTrace] = {}
_completed_traces: OrderedDict[str, AgentTrace] = OrderedDict()


def trace(
    agent_type: str,
    task_id: str = "",
    user_id: str = "",
    session_id: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that instruments an agent function for behavioral tracing.

    Wraps the decorated function to automatically create an ``AgentTrace``
    that captures timing, tool calls, and LLM calls. The trace is stored
    in a module-level registry and can be retrieved for analysis.

    Supports both sync and async functions.

    Args:
        agent_type: Identifier for the agent type (e.g., "customer-support").
        task_id: Optional task identifier.
        user_id: Optional user identifier.
        session_id: Optional session identifier.

    Returns:
        Decorator function.

    Example::

        @spectra.trace(agent_type="customer-support")
        async def handle_request(message: str) -> str:
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                trace_obj = AgentTrace(
                    agent_type=agent_type,
                    task_id=task_id,
                    user_id=user_id,
                    session_id=session_id,
                    started_at=datetime.now(UTC),
                )
                _active_traces[trace_obj.trace_id] = trace_obj

                token = _current_trace.set(trace_obj)
                start = time.monotonic()

                try:
                    result: R = await fn(*args, **kwargs)
                    trace_obj.success = True
                    trace_obj.output = _stringify_output(result)
                    return result
                except Exception:
                    trace_obj.success = False
                    raise
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000.0
                    trace_obj.ended_at = datetime.now(UTC)
                    _current_trace.reset(token)

                    _active_traces.pop(trace_obj.trace_id, None)
                    _completed_traces[trace_obj.trace_id] = trace_obj
                    if len(_completed_traces) > _MAX_COMPLETED_TRACES:
                        _completed_traces.popitem(last=False)

                    logger.info(
                        "Trace completed",
                        extra={
                            "trace_id": trace_obj.trace_id,
                            "agent_type": agent_type,
                            "duration_ms": elapsed_ms,
                            "tool_calls": len(trace_obj.tool_calls),
                            "llm_calls": len(trace_obj.llm_calls),
                            "success": trace_obj.success,
                        },
                    )

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                trace_obj = AgentTrace(
                    agent_type=agent_type,
                    task_id=task_id,
                    user_id=user_id,
                    session_id=session_id,
                    started_at=datetime.now(UTC),
                )
                _active_traces[trace_obj.trace_id] = trace_obj

                token = _current_trace.set(trace_obj)
                start = time.monotonic()

                try:
                    result = fn(*args, **kwargs)
                    trace_obj.success = True
                    trace_obj.output = _stringify_output(result)
                    return result
                except Exception:
                    trace_obj.success = False
                    raise
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000.0
                    trace_obj.ended_at = datetime.now(UTC)
                    _current_trace.reset(token)

                    _active_traces.pop(trace_obj.trace_id, None)
                    _completed_traces[trace_obj.trace_id] = trace_obj
                    if len(_completed_traces) > _MAX_COMPLETED_TRACES:
                        _completed_traces.popitem(last=False)

                    logger.info(
                        "Trace completed",
                        extra={
                            "trace_id": trace_obj.trace_id,
                            "agent_type": agent_type,
                            "duration_ms": elapsed_ms,
                            "tool_calls": len(trace_obj.tool_calls),
                            "llm_calls": len(trace_obj.llm_calls),
                            "success": trace_obj.success,
                        },
                    )

            return sync_wrapper

    return decorator


_current_trace: contextvars.ContextVar[AgentTrace | None] = contextvars.ContextVar(
    "spectra_current_trace", default=None
)


def _stringify_output(result: object) -> str:
    """Store a compact string representation of the agent output."""
    if result is None:
        return ""
    return str(result)


def get_current_trace() -> AgentTrace | None:
    """Retrieve the currently active trace for this execution context.

    Returns:
        The active AgentTrace, or None if no trace is active.
    """
    return _current_trace.get()


def record_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    result_summary: str = "",
    success: bool = True,
    duration_ms: float = 0.0,
) -> None:
    """Record a tool call on the currently active trace.

    This function is a convenience for manual instrumentation. Call it
    inside a ``@spectra.trace``-decorated function to record tool usage.

    Args:
        tool_name: Name of the tool that was called.
        arguments: Arguments passed to the tool.
        result_summary: Brief summary of the tool's result.
        success: Whether the tool call succeeded.
        duration_ms: Duration of the tool call in milliseconds.
    """
    current = _current_trace.get()
    if current is None:
        logger.warning("record_tool_call called outside of a traced context")
        return

    current.tool_calls.append(
        ToolCall(
            tool_name=tool_name,
            arguments=arguments or {},
            result_summary=result_summary,
            success=success,
            duration_ms=duration_ms,
        )
    )


def record_llm_call(
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    duration_ms: float = 0.0,
    prompt_summary: str = "",
    response_summary: str = "",
) -> None:
    """Record an LLM call on the currently active trace.

    Args:
        model: The model identifier (e.g., "gpt-4", "claude-3-opus").
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        total_tokens: Total tokens (prompt + completion).
        duration_ms: Duration of the LLM call in milliseconds.
        prompt_summary: Brief summary of the prompt.
        response_summary: Brief summary of the response.
    """
    current = _current_trace.get()
    if current is None:
        logger.warning("record_llm_call called outside of a traced context")
        return

    current.llm_calls.append(
        LLMCall(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            duration_ms=duration_ms,
            prompt_summary=prompt_summary,
            response_summary=response_summary,
        )
    )


def get_trace(trace_id: str) -> AgentTrace | None:
    """Retrieve a trace by its ID from the active or completed trace registry.

    Args:
        trace_id: The trace ID to look up.

    Returns:
        The AgentTrace if found, None otherwise.
    """
    return _active_traces.get(trace_id) or _completed_traces.get(trace_id)


def clear_traces() -> None:
    """Clear all traces from the active and completed trace registries.

    Useful for testing and cleanup.
    """
    _active_traces.clear()
    _completed_traces.clear()

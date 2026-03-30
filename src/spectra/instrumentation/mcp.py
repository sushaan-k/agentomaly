"""MCP (Model Context Protocol) middleware for spectra.

Provides middleware that intercepts MCP tool calls and records them
into spectra traces for behavioral analysis.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from spectra.instrumentation.decorator import get_current_trace
from spectra.models import ToolCall

logger = logging.getLogger(__name__)


class MCPMiddleware:
    """Middleware that captures MCP tool invocations as spectra trace events.

    Wraps MCP server tool handlers to automatically record each tool call
    (name, arguments, result, duration) on the active spectra trace.

    Args:
        agent_type: Identifier for the agent type being monitored.

    Example::

        from spectra.instrumentation.mcp import MCPMiddleware

        middleware = MCPMiddleware(agent_type="customer-support")

        @middleware.wrap_tool("search_kb")
        async def search_kb(query: str) -> str:
            ...
    """

    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type

    def wrap_tool(
        self,
        tool_name: str,
    ) -> Callable[
        [Callable[..., Awaitable[Any]]],
        Callable[..., Awaitable[Any]],
    ]:
        """Decorator that wraps an MCP tool handler with tracing.

        Args:
            tool_name: The name of the MCP tool being wrapped.

        Returns:
            Decorator that adds tracing to the tool handler.
        """

        def decorator(
            fn: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = get_current_trace()
                start = time.monotonic()
                success = True
                result_summary = ""

                try:
                    result = await fn(*args, **kwargs)
                    result_summary = str(result)[:200] if result is not None else ""
                    return result
                except Exception as exc:
                    success = False
                    result_summary = f"Error: {exc}"
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000.0
                    if trace is not None:
                        arguments = dict(kwargs)
                        if args:
                            arguments["_args"] = list(args)
                        trace.tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                arguments=arguments,
                                result_summary=result_summary,
                                success=success,
                                duration_ms=duration_ms,
                            )
                        )
                    logger.debug(
                        "MCP tool call recorded",
                        extra={
                            "tool_name": tool_name,
                            "success": success,
                            "duration_ms": duration_ms,
                        },
                    )

            return wrapper

        return decorator

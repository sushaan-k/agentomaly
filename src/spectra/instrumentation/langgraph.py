"""LangGraph callback integration for spectra.

Provides a callback handler that captures LangGraph node executions,
tool calls, and LLM invocations as spectra trace events.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from spectra.instrumentation.decorator import get_current_trace
from spectra.models import LLMCall, ToolCall

logger = logging.getLogger(__name__)


class LangGraphCallback:
    """Callback handler for LangGraph that records events to spectra traces.

    Attach this to a LangGraph ``StateGraph`` to automatically capture
    tool calls and LLM invocations for behavioral analysis.

    Args:
        agent_type: Identifier for the agent type being monitored.

    Example::

        from spectra.instrumentation.langgraph import LangGraphCallback

        callback = LangGraphCallback(agent_type="research-agent")

        # Use in your LangGraph workflow
        callback.on_tool_start("search", {"query": "latest news"})
        # ... tool executes ...
        callback.on_tool_end("search", result="Found 5 results")
    """

    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type
        self._start_times: dict[str, list[float]] = defaultdict(list)

    def on_tool_start(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Record the start of a tool invocation.

        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.
        """
        self._start_times[tool_name].append(time.monotonic())
        logger.debug("Tool started", extra={"tool_name": tool_name})

    def on_tool_end(
        self,
        tool_name: str,
        result: str = "",
        success: bool = True,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Record the completion of a tool invocation.

        Args:
            tool_name: Name of the tool that completed.
            result: Brief summary of the tool's result.
            success: Whether the tool call succeeded.
            arguments: Arguments that were passed to the tool.
        """
        starts = self._start_times.get(tool_name, [])
        start = starts.pop() if starts else None
        if not starts:
            self._start_times.pop(tool_name, None)
        duration_ms = (time.monotonic() - start) * 1000.0 if start else 0.0

        trace = get_current_trace()
        if trace is not None:
            trace.tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments or {},
                    result_summary=result[:200],
                    success=success,
                    duration_ms=duration_ms,
                )
            )
        logger.debug(
            "Tool completed",
            extra={
                "tool_name": tool_name,
                "success": success,
                "duration_ms": duration_ms,
            },
        )

    def on_llm_start(
        self,
        model: str = "",
        prompt_summary: str = "",
    ) -> None:
        """Record the start of an LLM invocation.

        Args:
            model: The model identifier.
            prompt_summary: Brief summary of the prompt.
        """
        self._start_times[f"__llm__{model}"].append(time.monotonic())
        logger.debug("LLM call started", extra={"model": model})

    def on_llm_end(
        self,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_summary: str = "",
    ) -> None:
        """Record the completion of an LLM invocation.

        Args:
            model: The model identifier.
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens generated.
            response_summary: Brief summary of the response.
        """
        key = f"__llm__{model}"
        starts = self._start_times.get(key, [])
        start = starts.pop() if starts else None
        if not starts:
            self._start_times.pop(key, None)
        duration_ms = (time.monotonic() - start) * 1000.0 if start else 0.0

        trace = get_current_trace()
        if trace is not None:
            trace.llm_calls.append(
                LLMCall(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    duration_ms=duration_ms,
                    response_summary=response_summary[:200],
                )
            )
        logger.debug(
            "LLM call completed",
            extra={
                "model": model,
                "total_tokens": prompt_tokens + completion_tokens,
                "duration_ms": duration_ms,
            },
        )

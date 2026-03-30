"""Tests for the instrumentation layer."""

from __future__ import annotations

import pytest

import spectra
from spectra.instrumentation.decorator import (
    clear_traces,
    get_current_trace,
    record_llm_call,
    record_tool_call,
)
from spectra.instrumentation.langgraph import LangGraphCallback


class TestDecorator:
    @pytest.mark.asyncio
    async def test_async_trace(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="test-agent", task_id="t1")
        async def my_agent(msg: str) -> str:
            record_tool_call(tool_name="search", arguments={"q": msg})
            record_llm_call(model="gpt-4", total_tokens=100)
            return f"Response to {msg}"

        result = await my_agent("hello")
        assert result == "Response to hello"

    @pytest.mark.asyncio
    async def test_trace_captures_tool_calls(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="test-agent")
        async def my_agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            record_tool_call(tool_name="search", arguments={"q": "test"})
            record_tool_call(tool_name="respond")
            return "done"

        await my_agent()

    def test_sync_trace(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="test-agent")
        def my_sync_agent() -> str:
            record_tool_call(tool_name="search")
            return "ok"

        result = my_sync_agent()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_trace_handles_exception(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="test-agent")
        async def failing_agent() -> None:
            raise ValueError("intentional error")

        with pytest.raises(ValueError, match="intentional"):
            await failing_agent()

    def test_record_outside_trace(self) -> None:
        record_tool_call(tool_name="search")


class TestLangGraphCallback:
    def test_tool_recording(self) -> None:
        cb = LangGraphCallback(agent_type="test")
        cb.on_tool_start("search", {"query": "test"})
        cb.on_tool_end("search", result="found it", arguments={"query": "test"})

    def test_llm_recording(self) -> None:
        cb = LangGraphCallback(agent_type="test")
        cb.on_llm_start(model="gpt-4")
        cb.on_llm_end(
            model="gpt-4",
            prompt_tokens=50,
            completion_tokens=30,
        )

"""Extended tests for instrumentation: decorator edge cases, LangGraph lifecycle."""

from __future__ import annotations

import pytest

import spectra
from spectra.instrumentation.decorator import (
    _active_traces,
    clear_traces,
    get_current_trace,
    get_trace,
    record_llm_call,
    record_tool_call,
)
from spectra.instrumentation.langgraph import LangGraphCallback


class TestDecoratorEdgeCases:
    def test_sync_trace_records_tool_calls(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="sync-agent", task_id="t1")
        def agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            record_tool_call(tool_name="search", arguments={"q": "test"})
            record_llm_call(model="gpt-4", total_tokens=100)
            assert len(trace.tool_calls) == 1
            assert len(trace.llm_calls) == 1
            return "done"

        result = agent()
        assert result == "done"

    def test_sync_trace_exception_marks_failure(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="sync-agent")
        def failing_agent() -> None:
            raise ValueError("sync error")

        with pytest.raises(ValueError, match="sync error"):
            failing_agent()

    @pytest.mark.asyncio
    async def test_async_trace_failure_sets_success_false(self) -> None:
        clear_traces()

        @spectra.trace(agent_type="test-agent")
        async def failing_agent() -> None:
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await failing_agent()

    @pytest.mark.asyncio
    async def test_nested_traces(self) -> None:
        """Nested decorated functions each have their own trace context."""
        clear_traces()

        @spectra.trace(agent_type="inner-agent")
        async def inner_agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            assert trace.agent_type == "inner-agent"
            record_tool_call(tool_name="inner_tool")
            return "inner"

        @spectra.trace(agent_type="outer-agent")
        async def outer_agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            assert trace.agent_type == "outer-agent"
            record_tool_call(tool_name="outer_tool")
            result = await inner_agent()
            return f"outer-{result}"

        result = await outer_agent()
        assert result == "outer-inner"

    def test_record_tool_call_outside_trace_no_error(self) -> None:
        """record_tool_call outside a trace logs a warning but doesn't crash."""
        clear_traces()
        record_tool_call(tool_name="orphan")

    def test_record_llm_call_outside_trace_no_error(self) -> None:
        """record_llm_call outside a trace logs a warning but doesn't crash."""
        clear_traces()
        record_llm_call(model="gpt-4", total_tokens=100)

    @pytest.mark.asyncio
    async def test_get_trace_by_id(self) -> None:
        clear_traces()

        captured_id: list[str] = []

        @spectra.trace(agent_type="test-agent")
        async def agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            captured_id.append(trace.trace_id)
            return "ok"

        await agent()

        retrieved = get_trace(captured_id[0])
        assert retrieved is not None
        assert retrieved.trace_id == captured_id[0]

    def test_get_trace_unknown_id(self) -> None:
        assert get_trace("nonexistent-id") is None

    def test_clear_traces(self) -> None:
        clear_traces()
        assert len(_active_traces) == 0

    @pytest.mark.asyncio
    async def test_trace_preserves_function_name(self) -> None:
        """Decorated functions preserve their __name__."""

        @spectra.trace(agent_type="test-agent")
        async def my_specific_agent() -> str:
            return "ok"

        assert my_specific_agent.__name__ == "my_specific_agent"

    def test_sync_trace_preserves_function_name(self) -> None:
        @spectra.trace(agent_type="test-agent")
        def my_sync_agent() -> str:
            return "ok"

        assert my_sync_agent.__name__ == "my_sync_agent"

    @pytest.mark.asyncio
    async def test_record_llm_call_token_sum(self) -> None:
        """total_tokens defaults to prompt_tokens + completion_tokens."""
        clear_traces()

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            record_llm_call(
                model="gpt-4",
                prompt_tokens=50,
                completion_tokens=30,
            )
            trace = get_current_trace()
            assert trace is not None
            # total_tokens should be prompt + completion when total is 0
            assert trace.llm_calls[0].total_tokens == 80

        await agent()

    @pytest.mark.asyncio
    async def test_trace_with_all_params(self) -> None:
        clear_traces()

        @spectra.trace(
            agent_type="full-agent",
            task_id="task-123",
            user_id="user-456",
            session_id="sess-789",
        )
        async def agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            assert trace.agent_type == "full-agent"
            assert trace.task_id == "task-123"
            assert trace.user_id == "user-456"
            assert trace.session_id == "sess-789"
            return "ok"

        await agent()

    @pytest.mark.asyncio
    async def test_trace_records_output(self) -> None:
        clear_traces()
        captured_id: list[str] = []

        @spectra.trace(agent_type="test-agent")
        async def agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            captured_id.append(trace.trace_id)
            return "final output"

        await agent()
        trace = get_trace(captured_id[0])
        assert trace is not None
        assert trace.output == "final output"

    def test_sync_trace_with_all_params(self) -> None:
        clear_traces()

        @spectra.trace(
            agent_type="full-sync",
            task_id="task-s",
            user_id="user-s",
            session_id="sess-s",
        )
        def agent() -> str:
            trace = get_current_trace()
            assert trace is not None
            assert trace.task_id == "task-s"
            return "ok"

        result = agent()
        assert result == "ok"


class TestLangGraphCallbackExtended:
    def test_tool_end_without_start(self) -> None:
        """on_tool_end without prior on_tool_start should not crash."""
        cb = LangGraphCallback(agent_type="test")
        cb.on_tool_end("search", result="found it")

    def test_llm_end_without_start(self) -> None:
        """on_llm_end without prior on_llm_start should not crash."""
        cb = LangGraphCallback(agent_type="test")
        cb.on_llm_end(model="gpt-4", prompt_tokens=10, completion_tokens=5)

    @pytest.mark.asyncio
    async def test_tool_lifecycle_with_trace(self) -> None:
        """Tool start/end events are recorded on the active trace."""
        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_tool_start("search", {"query": "test"})
            cb.on_tool_end("search", result="found", arguments={"query": "test"})
            assert len(trace.tool_calls) == 1
            assert trace.tool_calls[0].tool_name == "search"
            assert trace.tool_calls[0].success is True

        await agent()

    @pytest.mark.asyncio
    async def test_llm_lifecycle_with_trace(self) -> None:
        """LLM start/end events are recorded on the active trace."""
        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_llm_start(model="claude-3", prompt_summary="hello")
            cb.on_llm_end(
                model="claude-3",
                prompt_tokens=50,
                completion_tokens=30,
                response_summary="world",
            )
            assert len(trace.llm_calls) == 1
            assert trace.llm_calls[0].model == "claude-3"
            assert trace.llm_calls[0].total_tokens == 80

        await agent()

    @pytest.mark.asyncio
    async def test_tool_failure_recording(self) -> None:
        """Failed tool calls are recorded correctly."""
        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_tool_start("search", {"query": "test"})
            cb.on_tool_end("search", result="error", success=False)
            assert trace.tool_calls[0].success is False

        await agent()

    @pytest.mark.asyncio
    async def test_event_ordering(self) -> None:
        """Events are recorded in the order they occur."""
        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_tool_start("search", {"query": "q1"})
            cb.on_tool_end(
                "search",
                result="r1",
                arguments={"query": "q1"},
            )
            cb.on_llm_start(model="gpt-4")
            cb.on_llm_end(model="gpt-4", prompt_tokens=10, completion_tokens=5)
            cb.on_tool_start("respond", {})
            cb.on_tool_end("respond", result="done")
            assert len(trace.tool_calls) == 2
            assert len(trace.llm_calls) == 1
            assert trace.tool_calls[0].tool_name == "search"
            assert trace.tool_calls[1].tool_name == "respond"

        await agent()

    @pytest.mark.asyncio
    async def test_overlapping_same_tool_calls_preserve_both_durations(self) -> None:
        import asyncio

        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_tool_start("search", {"query": "a"})
            await asyncio.sleep(0.001)
            cb.on_tool_start("search", {"query": "b"})
            await asyncio.sleep(0.001)
            cb.on_tool_end("search", result="b", arguments={"query": "b"})
            cb.on_tool_end("search", result="a", arguments={"query": "a"})
            assert len(trace.tool_calls) == 2
            assert all(call.duration_ms >= 0.0 for call in trace.tool_calls)

        await agent()

    def test_long_response_summary_truncated(self) -> None:
        """Response summaries longer than 200 chars are truncated."""
        clear_traces()
        cb = LangGraphCallback(agent_type="test")

        @spectra.trace(agent_type="test-agent")
        def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            cb.on_tool_start("verbose_tool", {})
            cb.on_tool_end("verbose_tool", result="x" * 500, arguments={})
            assert len(trace.tool_calls[0].result_summary) <= 200

            cb.on_llm_start(model="gpt-4")
            cb.on_llm_end(
                model="gpt-4",
                prompt_tokens=10,
                completion_tokens=5,
                response_summary="y" * 500,
            )
            assert len(trace.llm_calls[0].response_summary) <= 200

        agent()

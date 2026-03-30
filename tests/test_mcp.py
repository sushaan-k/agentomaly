"""Tests for MCP middleware instrumentation."""

from __future__ import annotations

import pytest

import spectra
from spectra.instrumentation.decorator import clear_traces, get_current_trace
from spectra.instrumentation.mcp import MCPMiddleware


class TestMCPMiddleware:
    def test_init(self) -> None:
        mw = MCPMiddleware(agent_type="test-agent")
        assert mw.agent_type == "test-agent"

    @pytest.mark.asyncio
    async def test_wrap_tool_records_call(self) -> None:
        """Wrapped tool calls are recorded on the active trace."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("search_kb")
        async def search_kb(query: str = "") -> str:
            return f"Results for {query}"

        @spectra.trace(agent_type="test-agent")
        async def agent() -> str:
            result = await search_kb(query="hello")
            return result

        result = await agent()
        assert result == "Results for hello"

    @pytest.mark.asyncio
    async def test_wrap_tool_success_recorded(self) -> None:
        """Tool call is recorded with success=True on the trace."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("lookup")
        async def lookup(key: str = "") -> str:
            return "found"

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            await lookup(key="x")
            assert len(trace.tool_calls) == 1
            assert trace.tool_calls[0].tool_name == "lookup"
            assert trace.tool_calls[0].success is True
            assert trace.tool_calls[0].duration_ms >= 0.0

        await agent()

    @pytest.mark.asyncio
    async def test_wrap_tool_error_recorded(self) -> None:
        """Tool call errors are recorded with success=False."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("failing_tool")
        async def failing_tool() -> str:
            raise RuntimeError("tool failed")

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            with pytest.raises(RuntimeError, match="tool failed"):
                await failing_tool()
            assert len(trace.tool_calls) == 1
            assert trace.tool_calls[0].success is False
            assert "Error:" in trace.tool_calls[0].result_summary

        await agent()

    @pytest.mark.asyncio
    async def test_wrap_tool_outside_trace(self) -> None:
        """Tool calls outside a traced context are logged but not recorded."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("standalone")
        async def standalone() -> str:
            return "ok"

        # Should not raise even without active trace
        result = await standalone()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_wrap_tool_none_result(self) -> None:
        """Tool that returns None is handled correctly."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("void_tool")
        async def void_tool() -> None:
            pass

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            await void_tool()
            assert len(trace.tool_calls) == 1
            assert trace.tool_calls[0].result_summary == ""

        await agent()

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self) -> None:
        """Multiple concurrent tool calls are all recorded."""
        import asyncio

        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("tool_a")
        async def tool_a() -> str:
            await asyncio.sleep(0.01)
            return "a"

        @mw.wrap_tool("tool_b")
        async def tool_b() -> str:
            await asyncio.sleep(0.01)
            return "b"

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            await asyncio.gather(tool_a(), tool_b())
            assert len(trace.tool_calls) == 2
            tool_names = {tc.tool_name for tc in trace.tool_calls}
            assert tool_names == {"tool_a", "tool_b"}

        await agent()

    @pytest.mark.asyncio
    async def test_wrap_tool_long_result_truncated(self) -> None:
        """Tool results longer than 200 chars are truncated."""
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("verbose")
        async def verbose() -> str:
            return "x" * 500

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            await verbose()
            assert len(trace.tool_calls[0].result_summary) <= 200

        await agent()

    @pytest.mark.asyncio
    async def test_wrap_tool_preserves_positional_arguments(self) -> None:
        clear_traces()
        mw = MCPMiddleware(agent_type="test-agent")

        @mw.wrap_tool("search_kb")
        async def search_kb(query: str) -> str:
            return query.upper()

        @spectra.trace(agent_type="test-agent")
        async def agent() -> None:
            trace = get_current_trace()
            assert trace is not None
            await search_kb("hello")
            assert trace.tool_calls[0].arguments["_args"] == ["hello"]

        await agent()

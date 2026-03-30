"""Example: integrating spectra with a LangGraph agent.

Demonstrates how to use the LangGraphCallback to capture tool calls
and LLM invocations from a LangGraph workflow, then analyze them
for anomalies.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta

from spectra import Monitor
from spectra.instrumentation.langgraph import LangGraphCallback
from spectra.models import AgentTrace, LLMCall, ToolCall
from spectra.profiler.trainer import ProfileTrainer


def simulate_langgraph_workflow(
    callback: LangGraphCallback,
) -> AgentTrace:
    """Simulate a LangGraph workflow with spectra callback instrumentation.

    In a real LangGraph application, you would attach the callback to
    the graph's execution lifecycle. This simulation shows the callback
    API without requiring LangGraph as a dependency.
    """
    # Simulate: LLM decides to search
    callback.on_llm_start(model="claude-3-opus")
    callback.on_llm_end(
        model="claude-3-opus",
        prompt_tokens=500,
        completion_tokens=100,
        response_summary="I'll search the knowledge base for that.",
    )

    # Simulate: Tool call to search
    callback.on_tool_start("search_kb", {"query": "refund policy"})
    callback.on_tool_end(
        "search_kb",
        result="Found: Refunds are available within 30 days...",
        arguments={"query": "refund policy"},
    )

    # Simulate: LLM generates response
    callback.on_llm_start(model="claude-3-opus")
    callback.on_llm_end(
        model="claude-3-opus",
        prompt_tokens=800,
        completion_tokens=200,
        response_summary="Based on our policy, refunds are available...",
    )

    # Build a trace manually for analysis (in production, the decorator
    # or middleware would handle this automatically)
    now = datetime.now(UTC)
    return AgentTrace(
        agent_type="research-agent",
        task_id="task-lg-001",
        started_at=now,
        ended_at=now + timedelta(minutes=1),
        tool_calls=[
            ToolCall(tool_name="search_kb", arguments={"query": "refund policy"}),
        ],
        llm_calls=[
            LLMCall(model="claude-3-opus", total_tokens=600),
            LLMCall(model="claude-3-opus", total_tokens=1000),
        ],
        output="Based on our policy, refunds are available within 30 days.",
    )


async def main() -> None:
    """Run the LangGraph integration example."""
    random.seed(42)

    # Generate training data
    training_traces: list[AgentTrace] = []
    for _ in range(120):
        started = datetime.now(UTC)
        training_traces.append(
            AgentTrace(
                agent_type="research-agent",
                started_at=started,
                ended_at=started + timedelta(minutes=random.uniform(1, 3)),
                tool_calls=[
                    ToolCall(
                        tool_name="search_kb",
                        arguments={"query": "customer question"},
                    ),
                ],
                llm_calls=[
                    LLMCall(model="claude-3-opus", total_tokens=random.randint(800, 2000)),
                    LLMCall(model="claude-3-opus", total_tokens=random.randint(800, 2000)),
                ],
                output="Standard helpful response. " * random.randint(1, 3),
            )
        )

    # Train profile
    trainer = ProfileTrainer(min_traces=100)
    profile = trainer.train(agent_type="research-agent", traces=training_traces)
    print(f"Trained profile with {profile.trace_count} traces.")
    print(f"Known tools: {sorted(profile.known_tools)}")

    # Create callback and simulate workflow
    callback = LangGraphCallback(agent_type="research-agent")
    trace = simulate_langgraph_workflow(callback)

    # Analyze with monitor
    monitor = Monitor(profile=profile, sensitivity="medium")
    monitor.start()
    events = await monitor.analyze(trace)

    print(f"\nAnalyzed LangGraph trace: {len(events)} anomalies detected.")
    for event in events:
        print(f"  [{event.severity.value}] {event.title}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Offline demo for spectra."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from spectra import Monitor
from spectra.models import AgentTrace, LLMCall, ToolCall
from spectra.profiler.trainer import ProfileTrainer


def build_trace(offset: int, tools: list[str], tokens: int) -> AgentTrace:
    started_at = datetime.now(UTC) - timedelta(minutes=offset)
    return AgentTrace(
        agent_type="support-agent",
        task_id=f"trace-{offset}",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=20),
        tool_calls=[
            ToolCall(
                tool_name=name, timestamp=started_at + timedelta(seconds=index * 3)
            )
            for index, name in enumerate(tools)
        ],
        llm_calls=[LLMCall(model="demo-model", total_tokens=tokens)],
        output="Customer issue resolved.",
    )


async def main() -> None:
    training = [
        build_trace(i, ["search_kb", "respond"], 900 + (i % 40)) for i in range(120)
    ]
    profile = ProfileTrainer(min_traces=100).train(
        agent_type="support-agent",
        traces=training,
    )

    monitor = Monitor(
        profile=profile,
        sensitivity="medium",
        response_policy={
            "LOW": "log",
            "MEDIUM": "alert",
            "HIGH": "alert",
            "CRITICAL": "alert",
        },
    )
    monitor.start()
    suspicious = build_trace(999, ["shell_exec", "export_data", "respond"], 5400)
    events = await monitor.analyze(suspicious)

    print("spectra demo")
    print(f"trained tools: {sorted(profile.known_tools)}")
    print(f"anomalies detected: {len(events)}")
    if events:
        print(f"top severity: {events[0].severity.value}")
        print(f"first detector: {events[0].detector_type.value}")


if __name__ == "__main__":
    asyncio.run(main())

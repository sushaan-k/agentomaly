"""Core data models for the spectra anomaly detection framework.

Pydantic schemas for agent behaviors, execution traces, anomaly events,
and behavioral profiles.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """Anomaly severity levels, from least to most severe."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResponseAction(StrEnum):
    """Actions the response engine can take when an anomaly is detected."""

    LOG = "log"
    ALERT = "alert"
    QUARANTINE = "quarantine"
    BLOCK = "block"


class Sensitivity(StrEnum):
    """Detection sensitivity presets controlling threshold aggressiveness."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PARANOID = "paranoid"


class DetectorType(StrEnum):
    """Types of anomaly detectors available in spectra."""

    TOOL_USAGE = "tool_usage"
    SEQUENCE = "sequence"
    VOLUME = "volume"
    CONTENT = "content"
    INJECTION = "injection"


# ---------------------------------------------------------------------------
# Trace models -- capture what the agent actually did
# ---------------------------------------------------------------------------


class LLMCall(BaseModel):
    """Record of a single LLM invocation within an agent task."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    prompt_summary: str = ""
    response_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Record of a single tool invocation within an agent task."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateTransition(BaseModel):
    """Record of a state change in the agent."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    from_state: str
    to_state: str
    trigger: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Complete execution trace for a single agent task.

    This is the fundamental unit of data that spectra operates on.
    Each trace captures everything an agent did during one task execution.
    """

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str
    task_id: str = ""
    user_id: str = ""
    session_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    llm_calls: list[LLMCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    output: str = ""
    output_metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total wall-clock duration of the trace in milliseconds."""
        if self.ended_at is None:
            return 0.0
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000.0

    @property
    def total_tokens(self) -> int:
        """Sum of tokens across all LLM calls."""
        return sum(call.total_tokens for call in self.llm_calls)

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names invoked during this trace."""
        return [tc.tool_name for tc in self.tool_calls]

    @property
    def action_sequence(self) -> list[str]:
        """Chronological sequence of all actions (LLM calls + tool calls).

        Returns action names sorted by timestamp. LLM calls are represented
        as ``__llm_call__`` tokens.
        """
        events: list[tuple[datetime, str]] = []
        for llm_call in self.llm_calls:
            events.append((llm_call.timestamp, "__llm_call__"))
        for tool_call in self.tool_calls:
            events.append((tool_call.timestamp, tool_call.tool_name))
        events.sort(key=lambda e: e[0])
        return [name for _, name in events]


# ---------------------------------------------------------------------------
# Anomaly event models
# ---------------------------------------------------------------------------


class AnomalyEvent(BaseModel):
    """An anomaly detected by one of spectra's detectors.

    Anomaly events are the primary output of the detection engine.
    They describe what was anomalous, how severe it is, and provide
    context for investigation.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str
    agent_type: str
    detector_type: DetectorType
    severity: Severity
    title: str
    description: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Anomaly score between 0 (normal) and 1 (highly anomalous)",
    )
    details: dict[str, Any] = Field(default_factory=dict)
    action_taken: ResponseAction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile statistics models
# ---------------------------------------------------------------------------


class ToolStats(BaseModel):
    """Statistical profile for a single tool's usage patterns."""

    tool_name: str
    usage_frequency: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of traces that use this tool",
    )
    avg_calls_per_trace: float = 0.0
    std_calls_per_trace: float = 0.0
    max_calls_per_trace: int = 0
    avg_duration_ms: float = 0.0
    common_arg_keys: list[str] = Field(default_factory=list)


class VolumeStats(BaseModel):
    """Statistical profile for volume and duration metrics."""

    llm_calls_mean: float = 0.0
    llm_calls_std: float = 0.0
    tool_calls_mean: float = 0.0
    tool_calls_std: float = 0.0
    total_tokens_mean: float = 0.0
    total_tokens_std: float = 0.0
    duration_ms_mean: float = 0.0
    duration_ms_std: float = 0.0


class ContentStats(BaseModel):
    """Statistical profile for output content characteristics."""

    avg_output_length: float = 0.0
    std_output_length: float = 0.0
    contains_code_frequency: float = 0.0
    contains_urls_frequency: float = 0.0
    contains_structured_data_frequency: float = 0.0


class SensitivityThresholds(BaseModel):
    """Z-score thresholds for each sensitivity level."""

    low: float = 4.0
    medium: float = 3.0
    high: float = 2.0
    paranoid: float = 1.5

    def get_threshold(self, sensitivity: Sensitivity) -> float:
        """Return the z-score threshold for the given sensitivity level."""
        result: float = getattr(self, sensitivity.value)
        return result

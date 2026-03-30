"""Behavioral profile definition.

A BehavioralProfile encapsulates the learned normal behavior of an agent
type, including tool usage patterns, action sequences, volume statistics,
and content characteristics.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spectra.models import ContentStats, ToolStats, VolumeStats
from spectra.profiler.markov import MarkovChain

logger = logging.getLogger(__name__)


class BehavioralProfile(BaseModel):
    """Learned behavioral profile for an agent type.

    This is the core artifact produced by training. It contains all
    statistical summaries and models needed to detect anomalies at runtime.

    Attributes:
        agent_type: Identifier for the agent type this profile covers.
        created_at: When this profile was created.
        trace_count: Number of traces used to build this profile.
        tool_stats: Per-tool usage statistics.
        known_tools: Complete set of tools observed during training.
        volume_stats: Aggregate volume and duration statistics.
        content_stats: Output content characteristics.
        markov_chain: Transition model for action sequences.
        metadata: Arbitrary extra metadata.
    """

    agent_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_count: int = 0
    tool_stats: dict[str, ToolStats] = Field(default_factory=dict)
    known_tools: set[str] = Field(default_factory=set)
    volume_stats: VolumeStats = Field(default_factory=VolumeStats)
    content_stats: ContentStats = Field(default_factory=ContentStats)
    markov_chain: MarkovChain = Field(default_factory=MarkovChain)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_known_tool(self, tool_name: str) -> bool:
        """Check whether a tool was observed during training.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool was seen in training data.
        """
        return tool_name in self.known_tools

    def get_tool_stats(self, tool_name: str) -> ToolStats | None:
        """Retrieve usage statistics for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            ToolStats if the tool was observed, None otherwise.
        """
        return self.tool_stats.get(tool_name)

    def save(self, path: str | Path) -> None:
        """Serialize the profile to a JSON file.

        Args:
            path: File path to write to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Profile saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: str | Path) -> BehavioralProfile:
        """Deserialize a profile from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            The loaded BehavioralProfile.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        profile = cls.model_validate(data)
        logger.info(
            "Profile loaded",
            extra={"path": str(path), "agent_type": profile.agent_type},
        )
        return profile

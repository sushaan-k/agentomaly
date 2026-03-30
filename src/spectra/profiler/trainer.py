"""Profile trainer: builds a BehavioralProfile from historical traces.

Computes statistical summaries over a collection of AgentTrace objects
to establish the baseline of normal behavior for an agent type.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np

from spectra.exceptions import InsufficientTraceError
from spectra.models import AgentTrace, ContentStats, ToolStats, VolumeStats
from spectra.profiler.markov import MarkovChain
from spectra.profiler.profile import BehavioralProfile

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+")
_CODE_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")
_STRUCTURED_PATTERN = re.compile(r"\{[\s\S]*?\}|\[[\s\S]*?\]")


class ProfileTrainer:
    """Trains a BehavioralProfile from a set of agent execution traces.

    Analyzes tool usage, action sequences, volume metrics, and output
    content to build a statistical baseline of normal behavior.

    Args:
        min_traces: Minimum number of traces required for training.
    """

    def __init__(self, min_traces: int = 100) -> None:
        self.min_traces = min_traces

    def train(
        self,
        agent_type: str,
        traces: list[AgentTrace],
    ) -> BehavioralProfile:
        """Train a behavioral profile from historical traces.

        Args:
            agent_type: Identifier for the agent type.
            traces: Collection of execution traces to learn from.

        Returns:
            A fully trained BehavioralProfile.

        Raises:
            InsufficientTraceError: If fewer than ``min_traces`` are provided.
        """
        if len(traces) < self.min_traces:
            raise InsufficientTraceError(required=self.min_traces, provided=len(traces))

        logger.info(
            "Training behavioral profile",
            extra={"agent_type": agent_type, "trace_count": len(traces)},
        )

        tool_stats, known_tools = self._compute_tool_stats(traces)
        volume_stats = self._compute_volume_stats(traces)
        content_stats = self._compute_content_stats(traces)
        markov_chain = self._train_markov_chain(traces)

        profile = BehavioralProfile(
            agent_type=agent_type,
            trace_count=len(traces),
            tool_stats=tool_stats,
            known_tools=known_tools,
            volume_stats=volume_stats,
            content_stats=content_stats,
            markov_chain=markov_chain,
        )

        logger.info(
            "Profile training complete",
            extra={
                "agent_type": agent_type,
                "known_tools": len(known_tools),
                "trace_count": len(traces),
            },
        )
        return profile

    def _compute_tool_stats(
        self, traces: list[AgentTrace]
    ) -> tuple[dict[str, ToolStats], set[str]]:
        """Compute per-tool usage statistics across all traces.

        Args:
            traces: Collection of execution traces.

        Returns:
            Tuple of (tool_name -> ToolStats mapping, set of all known tools).
        """
        tool_usage_counts: dict[str, list[int]] = defaultdict(list)
        tool_durations: dict[str, list[float]] = defaultdict(list)
        tool_arg_keys: dict[str, list[set[str]]] = defaultdict(list)
        all_tools = {tc.tool_name for trace in traces for tc in trace.tool_calls}
        per_trace_counts: list[dict[str, int]] = []

        for trace in traces:
            trace_tool_counts: dict[str, int] = defaultdict(int)
            for tc in trace.tool_calls:
                trace_tool_counts[tc.tool_name] += 1
                tool_durations[tc.tool_name].append(tc.duration_ms)
                tool_arg_keys[tc.tool_name].append(set(tc.arguments.keys()))
            per_trace_counts.append(trace_tool_counts)

        for tool_name in all_tools:
            for trace_tool_counts in per_trace_counts:
                tool_usage_counts[tool_name].append(trace_tool_counts.get(tool_name, 0))

        num_traces = len(traces)
        stats: dict[str, ToolStats] = {}

        for tool_name in all_tools:
            counts = tool_usage_counts[tool_name]
            counts_array = np.array(counts, dtype=np.float64)
            nonzero_count = int(np.count_nonzero(counts_array))

            all_arg_sets = tool_arg_keys.get(tool_name, [])
            common_keys: list[str] = []
            if all_arg_sets:
                key_counts: dict[str, int] = defaultdict(int)
                for arg_set in all_arg_sets:
                    for key in arg_set:
                        key_counts[key] += 1
                common_keys = sorted(key_counts)

            durations = tool_durations.get(tool_name, [])
            avg_duration = float(np.mean(durations)) if durations else 0.0

            stats[tool_name] = ToolStats(
                tool_name=tool_name,
                usage_frequency=nonzero_count / num_traces if num_traces > 0 else 0.0,
                avg_calls_per_trace=(
                    float(np.mean(counts_array)) if len(counts_array) > 0 else 0.0
                ),
                std_calls_per_trace=(
                    float(np.std(counts_array)) if len(counts_array) > 0 else 0.0
                ),
                max_calls_per_trace=(
                    int(np.max(counts_array)) if len(counts_array) > 0 else 0
                ),
                avg_duration_ms=avg_duration,
                common_arg_keys=common_keys,
            )

        return stats, all_tools

    def _compute_volume_stats(self, traces: list[AgentTrace]) -> VolumeStats:
        """Compute aggregate volume and duration statistics.

        Args:
            traces: Collection of execution traces.

        Returns:
            VolumeStats summarizing normal volume metrics.
        """
        llm_counts = np.array([len(t.llm_calls) for t in traces], dtype=np.float64)
        tool_counts = np.array([len(t.tool_calls) for t in traces], dtype=np.float64)
        token_counts = np.array([t.total_tokens for t in traces], dtype=np.float64)
        durations = np.array([t.duration_ms for t in traces], dtype=np.float64)

        return VolumeStats(
            llm_calls_mean=float(np.mean(llm_counts)),
            llm_calls_std=float(np.std(llm_counts)),
            tool_calls_mean=float(np.mean(tool_counts)),
            tool_calls_std=float(np.std(tool_counts)),
            total_tokens_mean=float(np.mean(token_counts)),
            total_tokens_std=float(np.std(token_counts)),
            duration_ms_mean=float(np.mean(durations)),
            duration_ms_std=float(np.std(durations)),
        )

    def _compute_content_stats(self, traces: list[AgentTrace]) -> ContentStats:
        """Compute output content characteristics.

        Args:
            traces: Collection of execution traces.

        Returns:
            ContentStats summarizing output patterns.
        """
        output_lengths: list[float] = []
        code_count = 0
        url_count = 0
        structured_count = 0

        for trace in traces:
            output = trace.output
            output_lengths.append(float(len(output)))

            if _CODE_PATTERN.search(output):
                code_count += 1
            if _URL_PATTERN.search(output):
                url_count += 1
            if _STRUCTURED_PATTERN.search(output):
                structured_count += 1

        num_traces = len(traces)
        lengths_array = np.array(output_lengths, dtype=np.float64)

        return ContentStats(
            avg_output_length=float(np.mean(lengths_array)) if num_traces > 0 else 0.0,
            std_output_length=float(np.std(lengths_array)) if num_traces > 0 else 0.0,
            contains_code_frequency=code_count / num_traces if num_traces > 0 else 0.0,
            contains_urls_frequency=url_count / num_traces if num_traces > 0 else 0.0,
            contains_structured_data_frequency=(
                structured_count / num_traces if num_traces > 0 else 0.0
            ),
        )

    def _train_markov_chain(self, traces: list[AgentTrace]) -> MarkovChain:
        """Train a Markov chain on action sequences from all traces.

        Args:
            traces: Collection of execution traces.

        Returns:
            A trained MarkovChain.
        """
        sequences = [trace.action_sequence for trace in traces]
        chain = MarkovChain()
        chain.fit(sequences)
        return chain

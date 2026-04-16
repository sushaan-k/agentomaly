"""Profile comparison and behavioral drift detection.

Compares two behavioral profiles to measure how much an agent's behavior
has drifted over time.  Computes drift across tool usage, call
frequencies, and Markov chain transition distributions.
"""

from __future__ import annotations

import logging
import math

from spectra.profiler.profile import BehavioralProfile

logger = logging.getLogger(__name__)


def compare(
    profile_a: BehavioralProfile,
    profile_b: BehavioralProfile,
) -> dict[str, object]:
    """Compute behavioral drift between two profiles.

    Analyzes three dimensions of change:

    * **Tool drift** -- tools that were added or removed between the two
      profiles.
    * **Frequency drift** -- per-tool change in mean calls-per-trace.
    * **Markov divergence** -- symmetric KL divergence between the two
      Markov chain transition distributions.

    Args:
        profile_a: The baseline (older) profile.
        profile_b: The comparison (newer) profile.

    Returns:
        Dictionary containing:
        - ``new_tools``: tools present in *b* but not *a*.
        - ``removed_tools``: tools present in *a* but not *b*.
        - ``frequency_drift``: per-tool absolute change in mean usage.
        - ``markov_divergence``: symmetric KL divergence (float).
        - ``drift_score``: composite 0-1 score summarising overall drift.
    """
    new_tools = sorted(profile_b.known_tools - profile_a.known_tools)
    removed_tools = sorted(profile_a.known_tools - profile_b.known_tools)

    frequency_drift = _frequency_drift(profile_a, profile_b)
    markov_div = _markov_divergence(profile_a, profile_b)
    drift_score = _composite_drift_score(
        new_tools, removed_tools, frequency_drift, markov_div
    )

    return {
        "new_tools": new_tools,
        "removed_tools": removed_tools,
        "frequency_drift": frequency_drift,
        "markov_divergence": markov_div,
        "drift_score": drift_score,
    }


def _frequency_drift(
    profile_a: BehavioralProfile,
    profile_b: BehavioralProfile,
) -> dict[str, float]:
    """Compute per-tool absolute change in mean calls-per-trace.

    Args:
        profile_a: Baseline profile.
        profile_b: Comparison profile.

    Returns:
        Mapping of tool name to absolute frequency difference.
    """
    all_tools = profile_a.known_tools | profile_b.known_tools
    drift: dict[str, float] = {}

    for tool in sorted(all_tools):
        mean_a = (
            profile_a.tool_stats[tool].avg_calls_per_trace
            if tool in profile_a.tool_stats
            else 0.0
        )
        mean_b = (
            profile_b.tool_stats[tool].avg_calls_per_trace
            if tool in profile_b.tool_stats
            else 0.0
        )
        diff = abs(mean_b - mean_a)
        if diff > 0.0:
            drift[tool] = round(diff, 4)

    return drift


def _markov_divergence(
    profile_a: BehavioralProfile,
    profile_b: BehavioralProfile,
) -> float:
    """Compute symmetric KL divergence between two Markov chains.

    Uses the average of KL(P||Q) and KL(Q||P) where P and Q are the
    row-normalised transition distributions of each chain.  A small
    smoothing constant avoids log(0).

    Args:
        profile_a: Baseline profile.
        profile_b: Comparison profile.

    Returns:
        Symmetric KL divergence (non-negative float).
    """
    probs_a = profile_a.markov_chain.transition_probs
    probs_b = profile_b.markov_chain.transition_probs

    all_states = set(probs_a.keys()) | set(probs_b.keys())
    if not all_states:
        return 0.0

    all_next_states: set[str] = set()
    for d in [probs_a, probs_b]:
        for transitions in d.values():
            all_next_states.update(transitions.keys())

    smoothing = 1e-10
    kl_ab = 0.0
    kl_ba = 0.0
    count = 0

    for state in all_states:
        row_a = probs_a.get(state, {})
        row_b = probs_b.get(state, {})
        next_states = set(row_a.keys()) | set(row_b.keys())

        for ns in next_states:
            p = row_a.get(ns, 0.0) + smoothing
            q = row_b.get(ns, 0.0) + smoothing
            kl_ab += p * math.log(p / q)
            kl_ba += q * math.log(q / p)
            count += 1

    if count == 0:
        return 0.0

    return round((kl_ab + kl_ba) / 2.0, 6)


def _composite_drift_score(
    new_tools: list[str],
    removed_tools: list[str],
    frequency_drift: dict[str, float],
    markov_divergence: float,
) -> float:
    """Combine drift signals into a single 0-1 score.

    Weights:
    * Tool set change (new + removed tools): 40 %
    * Frequency drift (sum of absolute deltas, capped): 30 %
    * Markov divergence (log-scaled, capped): 30 %

    Args:
        new_tools: Tools added in the newer profile.
        removed_tools: Tools removed in the newer profile.
        frequency_drift: Per-tool frequency differences.
        markov_divergence: Symmetric KL divergence value.

    Returns:
        Composite score in [0.0, 1.0].
    """
    tool_change = len(new_tools) + len(removed_tools)
    tool_component = min(tool_change / 5.0, 1.0) * 0.4

    freq_sum = sum(frequency_drift.values())
    freq_component = min(freq_sum / 5.0, 1.0) * 0.3

    markov_component = min(markov_divergence / 2.0, 1.0) * 0.3

    return round(tool_component + freq_component + markov_component, 4)

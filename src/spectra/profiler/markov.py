"""First-order Markov chain model for action sequence analysis.

Learns transition probabilities from observed action sequences and scores
novel sequences against the learned distribution.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_START_TOKEN = "__START__"
_END_TOKEN = "__END__"


class MarkovChain(BaseModel):
    """First-order Markov chain over action sequences.

    Learns transition probabilities P(action_j | action_i) from training
    data and provides methods to score new sequences.

    Attributes:
        transition_counts: Raw count of transitions from state A to state B.
        transition_probs: Normalized transition probabilities.
        known_states: Set of all states observed during training.
    """

    transition_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    transition_probs: dict[str, dict[str, float]] = Field(default_factory=dict)
    known_states: set[str] = Field(default_factory=set)

    def fit(self, sequences: list[list[str]]) -> MarkovChain:
        """Train the Markov chain from a collection of action sequences.

        Args:
            sequences: List of action sequences, where each sequence is an
                ordered list of action names.

        Returns:
            Self, for method chaining.
        """
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        states: set[str] = set()

        for seq in sequences:
            if not seq:
                continue
            padded = [_START_TOKEN, *seq, _END_TOKEN]
            for current, next_state in zip(padded[:-1], padded[1:]):
                counts[current][next_state] += 1
                states.add(current)
                states.add(next_state)

        self.transition_counts = {k: dict(v) for k, v in counts.items()}
        self.known_states = states

        probs: dict[str, dict[str, float]] = {}
        for state, transitions in counts.items():
            total = sum(transitions.values())
            if total > 0:
                probs[state] = {s: c / total for s, c in transitions.items()}
        self.transition_probs = probs

        logger.info(
            "Markov chain trained",
            extra={
                "num_sequences": len(sequences),
                "num_states": len(states),
            },
        )
        return self

    def transition_probability(self, from_state: str, to_state: str) -> float:
        """Return the probability of transitioning from one state to another.

        Args:
            from_state: The current state.
            to_state: The next state.

        Returns:
            Transition probability in [0, 1]. Returns 0.0 for unseen
            transitions.
        """
        return self.transition_probs.get(from_state, {}).get(to_state, 0.0)

    def sequence_log_probability(self, sequence: list[str]) -> float:
        """Compute the log-probability of an entire action sequence.

        Uses additive smoothing (Laplace) to avoid -inf for unseen
        transitions.

        Args:
            sequence: Ordered list of action names.

        Returns:
            Log-probability of the sequence. More negative values indicate
            more unusual sequences.
        """
        if not sequence:
            return 0.0

        padded = [_START_TOKEN, *sequence, _END_TOKEN]
        log_prob = 0.0
        smoothing = 1e-10
        num_states = max(len(self.known_states), 1)

        for current, next_state in zip(padded[:-1], padded[1:]):
            prob = self.transition_probability(current, next_state)
            smoothed = (prob + smoothing) / (1.0 + smoothing * num_states)
            log_prob += math.log(smoothed)

        return log_prob

    def has_novel_transition(self, sequence: list[str]) -> list[tuple[str, str]]:
        """Identify transitions in a sequence that were never observed.

        Args:
            sequence: Ordered list of action names.

        Returns:
            List of (from_state, to_state) tuples representing transitions
            not present in training data.
        """
        if not sequence:
            return []

        novel: list[tuple[str, str]] = []
        padded = [_START_TOKEN, *sequence, _END_TOKEN]

        for current, next_state in zip(padded[:-1], padded[1:]):
            if self.transition_probability(current, next_state) == 0.0:
                novel.append((current, next_state))

        return novel

    def detect_loops(
        self, sequence: list[str], max_repeat: int = 3
    ) -> list[tuple[str, int]]:
        """Detect repeated action patterns that suggest the agent is looping.

        Args:
            sequence: Ordered list of action names.
            max_repeat: Number of consecutive repetitions to flag as a loop.

        Returns:
            List of (action, repeat_count) tuples for detected loops.
        """
        if not sequence:
            return []

        loops: list[tuple[str, int]] = []
        current_action = sequence[0]
        count = 1

        for action in sequence[1:]:
            if action == current_action:
                count += 1
            else:
                if count >= max_repeat:
                    loops.append((current_action, count))
                current_action = action
                count = 1

        if count >= max_repeat:
            loops.append((current_action, count))

        return loops

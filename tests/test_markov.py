"""Tests for the Markov chain model."""

from __future__ import annotations

from spectra.profiler.markov import MarkovChain


class TestMarkovChain:
    def test_fit_basic(self) -> None:
        chain = MarkovChain()
        sequences = [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["a", "c"],
        ]
        chain.fit(sequences)
        assert "a" in chain.known_states
        assert "b" in chain.known_states
        assert "c" in chain.known_states

    def test_transition_probability(self) -> None:
        chain = MarkovChain()
        sequences = [
            ["a", "b"],
            ["a", "b"],
            ["a", "c"],
        ]
        chain.fit(sequences)
        prob_ab = chain.transition_probability("a", "b")
        prob_ac = chain.transition_probability("a", "c")
        assert prob_ab > prob_ac
        assert abs(prob_ab + prob_ac - 1.0) < 0.01

    def test_unseen_transition_is_zero(self) -> None:
        chain = MarkovChain()
        chain.fit([["a", "b"]])
        assert chain.transition_probability("b", "a") == 0.0

    def test_sequence_log_probability(self) -> None:
        chain = MarkovChain()
        sequences = [["a", "b", "c"]] * 10
        chain.fit(sequences)

        normal_logp = chain.sequence_log_probability(["a", "b", "c"])
        weird_logp = chain.sequence_log_probability(["c", "a", "b"])
        assert normal_logp > weird_logp

    def test_empty_sequence(self) -> None:
        chain = MarkovChain()
        chain.fit([["a"]])
        assert chain.sequence_log_probability([]) == 0.0

    def test_has_novel_transition(self) -> None:
        chain = MarkovChain()
        chain.fit([["a", "b", "c"]])

        novel = chain.has_novel_transition(["a", "b", "c"])
        assert len(novel) == 0

        novel = chain.has_novel_transition(["c", "a"])
        assert len(novel) > 0

    def test_detect_loops(self) -> None:
        chain = MarkovChain()
        chain.fit([["a", "b"]])

        loops = chain.detect_loops(["a", "a", "a", "a", "b"], max_repeat=3)
        assert len(loops) == 1
        assert loops[0] == ("a", 4)

    def test_detect_loops_no_loop(self) -> None:
        chain = MarkovChain()
        chain.fit([["a", "b"]])

        loops = chain.detect_loops(["a", "b", "a", "b"], max_repeat=3)
        assert len(loops) == 0

    def test_empty_sequences(self) -> None:
        chain = MarkovChain()
        chain.fit([])
        assert chain.transition_probability("a", "b") == 0.0
        assert chain.has_novel_transition([]) == []

    def test_fit_returns_self(self) -> None:
        chain = MarkovChain()
        result = chain.fit([["a", "b"]])
        assert result is chain

from __future__ import annotations

import importlib.util
import unittest

from metamath_generator.model import Node
from neural_prover.environment import ProofState, Tactic, Transition
from neural_prover.hybrid import HybridActions


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def chain_state(name: str | None) -> ProofState:
    return ProofState(
        hypotheses=(),
        goals=() if name is None else (Node("|-", (Node(name),)),),
        d_constraints=frozenset(),
        variable_types=(),
    )


class ChainEnvironment:
    """Deterministic three-step fixture isolating the MCTS return calculation."""

    def __init__(self) -> None:
        self.states = tuple(chain_state(name) for name in ("A", "B", "C", None))

    def actions(self, state: ProofState) -> HybridActions:
        index = self.states.index(state)
        return HybridActions((Tactic.create(f"step-{index}"),), ())

    def apply(self, state: ProofState, tactic: Tactic) -> Transition:
        index = self.states.index(state)
        assert tactic == Tactic.create(f"step-{index}")
        after = self.states[index + 1]
        return Transition(state, tactic, after, after.goals)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for MCTS")
class MCTSReturnTests(unittest.TestCase):
    def make_search(self, step_penalty: float = 0.01):
        from neural_prover.mcts import MCTSConfig, ProofMCTS
        from neural_prover.search import UniformPolicy

        environment = ChainEnvironment()
        search = ProofMCTS(
            environment,
            environment,
            UniformPolicy(),
            MCTSConfig(
                simulations=4,
                max_depth=3,
                step_penalty=step_penalty,
                dirichlet_fraction=0.0,
            ),
        )
        return environment, search

    def test_success_values_charge_remaining_proof_steps(self) -> None:
        environment, search = self.make_search()
        result = search.prove(environment.states[0])

        self.assertTrue(result.search.solved)
        self.assertEqual(len(result.search.actions), 3)
        self.assertEqual(len(result.experiences), 3)
        for experience, expected in zip(result.experiences, (0.97, 0.98, 0.99)):
            self.assertAlmostEqual(experience.value_target, expected)
            self.assertTrue(experience.value_target_valid)
            self.assertTrue(experience.policy_target_valid)

    def test_same_suffix_has_same_values_from_different_roots(self) -> None:
        environment, search = self.make_search()
        full = search.prove(environment.states[0])
        suffix = search.prove(environment.states[1])
        full_values = {item.state: item.value_target for item in full.experiences}

        self.assertTrue(suffix.search.solved)
        self.assertEqual(len(suffix.experiences), 2)
        for experience in suffix.experiences:
            self.assertEqual(experience.value_target, full_values[experience.state])

    def test_success_values_are_clamped_at_zero_like_backup(self) -> None:
        environment, search = self.make_search(step_penalty=0.6)
        result = search.prove(environment.states[0])

        self.assertTrue(result.search.solved)
        self.assertEqual(len(result.experiences), 3)
        for experience, expected in zip(result.experiences, (0.0, 0.0, 0.4)):
            self.assertAlmostEqual(experience.value_target, expected)
            self.assertTrue(experience.value_target_valid)


if __name__ == "__main__":
    unittest.main()

"""Local cuts must agree across sequential replay, HTPS and certificates."""

import importlib.util
from pathlib import Path
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed")

from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import parse
from neural_prover.certificate import compile_certificate, verify_certificate
from neural_prover.environment import (
    InvalidTactic,
    LEMMA_BINDING,
    LEMMA_COMMIT_OP,
    LEMMA_RELEASE_OP,
    PROPOSE_LEMMA_RULE,
    ProofState,
    Tactic,
    parse_tactic_tokens,
)
from neural_prover.external import (
    discover_metamath_executable,
    verify_certificate_external,
)
from neural_prover.lemma import (
    LemmaActionGenerator,
    LemmaBackwardEnvironment,
    propose_lemma,
)
from neural_prover.search import HeuristicPolicy, RankedTactic, SearchResult
from neural_prover.tokenizer import MetamathTokenizer
from htps_prover.hypergraph import HTPSConfig, HyperTreeProofSearch


ROOT = Path(__file__).resolve().parents[1]
PEANO = ROOT / "formal" / "peano.mm"


def implication(left, right):
    return Node("implies", (left, right))


def assertion(formula):
    return Node("|-", (formula,))


class LemmaScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = parse(PEANO)
        cls.tokenizer = MetamathTokenizer.from_database(cls.database)

    def setUp(self):
        self.environment = LemmaBackwardEnvironment(self.database)
        self.p, self.q = Node("p"), Node("q")
        self.a = implication(self.p, implication(self.q, self.p))
        self.b = implication(self.p, self.a)
        self.lemma = assertion(implication(self.a, self.b))
        self.target = Theorem(
            "scoped-cut-target", [], assertion(self.b),
            variable_types={"p": "wff", "q": "wff"},
        )

    def certify(self, target, transitions):
        self.assertTrue(transitions[-1].after.solved)
        result = SearchResult(
            True, len(transitions), len(transitions) + 1,
            tuple(t.tactic for t in transitions), tuple(transitions),
            transitions[-1].after,
        )
        certificate = compile_certificate(target, result, self.database)
        verify_certificate(certificate, self.database)
        return certificate

    def test_cut_does_not_leak_to_an_existing_sibling(self):
        state = ProofState.from_theorem(self.target)
        actions = (
            Tactic.create("ax-mp", {"phi": self.a, "psi": self.b}),
            propose_lemma(self.lemma),
            Tactic.create("ax-1", {"phi": self.a, "psi": self.p}),
            Tactic.create("ax-1", {"phi": self.p, "psi": self.q}),
        )
        transitions = []
        for action in actions:
            transition = self.environment.apply(state, action)
            transitions.append(transition)
            state = transition.after
        self.assertEqual(state.current_goal, self.lemma)
        self.assertNotIn(self.lemma, state.hypotheses)
        # Previously this false assumption closed the queue and produced a
        # solved search whose certificate requested an unavailable hypothesis.
        with self.assertRaises(InvalidTactic):
            self.environment.apply(state, Tactic.create("<ASSUMPTION>"))
        transitions.append(self.environment.apply(state, actions[2]))
        self.certify(self.target, transitions)

    def test_nested_cuts_restore_scopes_and_serialize_every_state(self):
        inner = assertion(implication(self.q, implication(self.p, self.q)))
        later = assertion(implication(self.p, implication(self.a, self.p)))
        actions = (
            Tactic.create("ax-mp", {"phi": self.a, "psi": self.b}),
            propose_lemma(self.lemma),
            propose_lemma(inner),
            Tactic.create("ax-1", {"phi": self.q, "psi": self.p}),
            Tactic.create("ax-1", {"phi": self.a, "psi": self.p}),
            propose_lemma(later),
            Tactic.create("ax-1", {"phi": self.p, "psi": self.a}),
            Tactic.create("ax-1", {"phi": self.p, "psi": self.q}),
            Tactic.create("ax-1", {"phi": self.a, "psi": self.p}),
        )
        state = ProofState.from_theorem(self.target)
        transitions = []
        for index, action in enumerate(actions):
            before_tokens = self.tokenizer.proof_state_tokens(state)
            restored = self.tokenizer.proof_state_from_tokens(before_tokens, self.database)
            canonical = self.tokenizer.proof_state_variables(state)
            action_tokens = (
                self.tokenizer.lemma_tactic_tokens(
                    action.substitution_dict()[LEMMA_BINDING], canonical,
                )
                if action.rule == PROPOSE_LEMMA_RULE else
                self.tokenizer.assertion_tactic_tokens(
                    self.environment.assertions[action.rule],
                    action.substitution_dict(), canonical,
                )
            )
            restored_action = parse_tactic_tokens(
                action_tokens, restored, self.tokenizer, self.database,
                environment=self.environment,
            )
            restored_after = self.environment.apply(restored, restored_action).after
            transition = self.environment.apply(state, action)
            transitions.append(transition)
            state = transition.after
            tokens = self.tokenizer.proof_state_tokens(state)
            self.assertEqual(self.tokenizer.proof_state_tokens(restored_after), tokens)
            self.tokenizer.encode(tokens)
            self.assertNotIn(LEMMA_COMMIT_OP, tokens)
            self.assertNotIn(LEMMA_RELEASE_OP, tokens)
            decoded = self.tokenizer.proof_state_from_tokens(tokens, self.database)
            self.assertEqual(self.tokenizer.proof_state_tokens(decoded), tokens)
            if index == 3:
                self.assertEqual(state.hypotheses, (inner,))
            elif index == 4:
                self.assertEqual(state.hypotheses, (self.lemma,))
            elif index == 6:
                self.assertEqual(state.hypotheses, (self.lemma, later))
            elif index >= 7:
                self.assertFalse(state.hypotheses)
        self.certify(self.target, transitions)

    def test_local_lemma_remains_available_to_all_continuation_children(self):
        lemma = assertion(self.a)
        target = Theorem(
            "cut-continuation-children",
            [
                Hypothesis("h1", assertion(implication(self.a, self.p))),
                Hypothesis("h2", assertion(implication(
                    self.a, implication(self.p, self.q),
                ))),
            ],
            assertion(self.q), variable_types={"p": "wff", "q": "wff"},
        )
        actions = (
            propose_lemma(lemma),
            Tactic.create("ax-1", {"phi": self.p, "psi": self.q}),
            Tactic.create("ax-mp", {"phi": self.p, "psi": self.q}),
            Tactic.create("ax-mp", {"phi": self.a, "psi": self.p}),
            Tactic.create("ax-mp", {
                "phi": self.a, "psi": implication(self.p, self.q),
            }),
        )
        initial = ProofState.from_theorem(target)
        state = initial
        transitions = []
        for index, action in enumerate(actions):
            transition = self.environment.apply(state, action)
            transitions.append(transition)
            state = transition.after
            if index in (1, 2, 3):
                self.assertIn(lemma, state.hypotheses)
        self.assertEqual(state.hypotheses, initial.hypotheses)
        self.certify(target, transitions)

    def sibling_search(self):
        formula = self.a
        lemma = assertion(formula)
        target = Theorem(
            "sibling-cuts",
            [
                Hypothesis("h1", assertion(implication(formula, self.p))),
                Hypothesis("h2", assertion(implication(
                    formula, implication(self.p, self.q),
                ))),
            ],
            assertion(self.q), variable_types={"p": "wff", "q": "wff"},
        )
        initial = ProofState.from_theorem(target)
        environment = self.environment
        root_tactic = Tactic.create("ax-mp", {"phi": self.p, "psi": self.q})

        class PickLegalRoute:
            def rank(self, state, tactics):
                if state == initial:
                    chosen = root_tactic
                elif state.current_goal == lemma:
                    chosen = Tactic.create("ax-1", {
                        "phi": formula.args[0], "psi": formula.args[1].args[0],
                    })
                elif lemma in state.hypotheses:
                    chosen = Tactic.create("ax-mp", {
                        "phi": formula, "psi": state.current_goal.args[0],
                    })
                else:
                    chosen = propose_lemma(lemma)
                # Use the production generator, not an invented cut action.
                if chosen not in tactics:
                    raise AssertionError("required tactic missing from legal candidates")
                return [RankedTactic(chosen, 0.0, 0.9)]

        result, _ = HyperTreeProofSearch(
            environment, LemmaActionGenerator(environment), PickLegalRoute(),
            HTPSConfig(
                simulations=30, expansion_budget=30, branching=1,
                parallel_selections=1,
            ),
        ).prove(initial)
        self.assertTrue(result.search.solved)
        self.assertEqual(result.search.actions.count(propose_lemma(lemma)), 2)
        self.assertEqual(result.search.final_state.hypotheses, initial.hypotheses)
        return self.certify(target, result.search.transitions)

    def test_htps_siblings_can_each_prove_and_use_the_same_local_lemma(self):
        self.sibling_search()

    def test_scoped_htps_certificate_passes_official_metamath(self):
        executable = discover_metamath_executable()
        if executable is None:
            bundled = ROOT / "outputs" / "tools" / "metamath" / "metamath.exe"
            executable = discover_metamath_executable(bundled)
        if executable is None:
            self.skipTest("official Metamath executable is not configured")
        result = verify_certificate_external(
            self.sibling_search(), PEANO, executable=executable,
        )
        self.assertTrue(result.passed, result.output_tail)

    def test_scope_markers_are_not_counted_as_heuristic_obligations(self):
        initial = ProofState.from_theorem(self.target)
        action = Tactic.create("ax-1", {"phi": self.a, "psi": self.p})
        continuation = assertion(self.a)
        plain = ProofState(
            (), (self.lemma, continuation), initial.d_constraints,
            initial.variable_types,
        )
        scoped = self.environment.apply(
            ProofState((), (continuation,), initial.d_constraints, initial.variable_types),
            propose_lemma(self.lemma),
        ).after
        policy = HeuristicPolicy(self.environment)
        # Both leave exactly one directly provable formula after this action.
        self.assertEqual(policy.rank(plain, [action]), policy.rank(scoped, [action]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import parse
from neural_prover.certificate import compile_certificate, verify_certificate
from neural_prover.environment import BackwardEnvironment, ProofState, Tactic
from neural_prover.hybrid import HybridActionGenerator, HybridActions
from neural_prover.lemma import LemmaBackwardEnvironment
if torch is not None:
    from neural_prover.mcts import MCTSConfig, ProofMCTS
    from neural_prover.model import ProofTransformer, ProofTransformerConfig
    from neural_prover.rl import ReplayBuffer, ReplayExample
    from neural_prover.search import RankedTactic, TransformerPolicy, UniformPolicy
    from htps_prover.hypergraph import HTPSConfig, HyperEdge, HyperTreeProofSearch
from neural_prover.tokenizer import MetamathTokenizer


ROOT = Path(__file__).resolve().parents[1]


def implies(left, right):
    return Node("implies", (left, right))


def assertion(formula):
    return Node("|-", (formula,))


class _OrderedPolicy:
    def rank(self, state, tactics):
        return [
            RankedTactic(tactic, -10.0 * index, 0.9)
            for index, tactic in enumerate(tactics)
        ]


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SearchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.database = parse(ROOT / "formal" / "peano.mm")

    def _cycle_search(
        self, *, escape=True, parallel=1, escape_at_root=False,
        simulations=100, expansion_budget=100,
    ):
        p, q, x, y = map(Node, ("p", "q", "x", "y"))
        lemma = implies(x, implies(y, x))
        target = Theorem(
            "cycle_boundary",
            [
                Hypothesis("h1", assertion(implies(q, p))),
                Hypothesis("h2", assertion(implies(p, q))),
                Hypothesis("h3", assertion(implies(lemma, p if escape_at_root else q))),
            ],
            assertion(p),
            variable_types={name: "wff" for name in ("p", "q", "x", "y")},
        )
        environment = LemmaBackwardEnvironment(self.database)

        class Actions:
            def actions(self, state):
                goal = state.current_goal.args[0]
                if goal == p:
                    tactics = [Tactic.create("ax-mp", {"phi": q})]
                    if escape and escape_at_root:
                        tactics.append(Tactic.create("ax-mp", {"phi": lemma}))
                elif goal == q:
                    tactics = [Tactic.create("ax-mp", {"phi": p})]
                    if escape and not escape_at_root:
                        tactics.append(Tactic.create("ax-mp", {"phi": lemma}))
                else:
                    tactics = [Tactic.create("ax-1")]
                return HybridActions(tuple(tactics), ())

        searcher = HyperTreeProofSearch(
            environment,
            Actions(),
            _OrderedPolicy(),
            HTPSConfig(
                simulations=simulations, expansion_budget=expansion_budget,
                parallel_selections=parallel,
            ),
        )
        result, metrics = searcher.prove(ProofState.from_theorem(target))
        return target, searcher, result, metrics

    def test_htps_backtracks_from_cycle_to_valid_metamathematical_proof(self):
        for parallel in (1, 4):
            with self.subTest(parallel=parallel):
                theorem, searcher, result, metrics = self._cycle_search(parallel=parallel)
                self.assertTrue(result.search.solved)
                self.assertGreater(metrics.cycles_rejected, 0)
                self.assertEqual(metrics.expansions, 3)
                verify_certificate(
                    compile_certificate(theorem, result.search, self.database),
                    self.database,
                )
                self.assertTrue(all(
                    edge.virtual_visits == 0
                    for node in searcher.nodes for edge in node.edges
                ))

    def test_htps_path_cycle_does_not_invalidate_shared_edge(self):
        theorem, searcher, result, metrics = self._cycle_search(escape_at_root=True)
        self.assertTrue(result.search.solved)
        self.assertGreater(metrics.cycles_rejected, 0)
        shared = next(node for node in searcher.nodes if node.state.current_goal == assertion(Node("q")))
        self.assertTrue(shared.solved)
        self.assertFalse(shared.edges[0].invalid)
        verify_certificate(compile_certificate(theorem, result.search, self.database), self.database)

    def test_htps_pure_cycle_stops_without_spinning_or_global_invalidation(self):
        _, searcher, result, metrics = self._cycle_search(escape=False)
        self.assertFalse(result.search.solved)
        self.assertLess(metrics.simulations, 100)
        self.assertEqual(metrics.expansions, 2)
        self.assertTrue(all(
            not edge.invalid and edge.virtual_visits == 0
            for node in searcher.nodes for edge in node.edges
        ))

    def _dense_cycle_selection(self, size, *, root_escape):
        # Exercise the graph algorithm independently of candidate caps.  Each
        # node has an OR edge to every other node; only the root may escape.
        environment = LemmaBackwardEnvironment(self.database)
        searcher = HyperTreeProofSearch(
            environment, None, _OrderedPolicy(),
            HTPSConfig(parallel_selections=1),
        )
        for index in range(size + int(root_escape)):
            state = ProofState(
                (), (assertion(Node(f"p{index}")),), frozenset(), (),
            )
            searcher._intern(state)
        for node_id in range(size):
            node = searcher.nodes[node_id]
            node.expanded = True
            node.edges = [
                HyperEdge(Tactic.create("cycle"), 1.0, 0.9, (child,))
                for child in range(size) if child != node_id
            ]
        if root_escape:
            searcher.nodes[0].edges.append(
                HyperEdge(Tactic.create("escape"), 1e-9, 0.1, (size,))
            )
        selected, frontier = [], []
        productive = searcher._select_hypertree(
            0, frozenset(), selected, frontier,
        )
        return searcher, productive, selected, frontier

    def test_dense_closed_cycle_is_filtered_without_enumerating_simple_paths(self):
        for size in (5, 8, 12):
            with self.subTest(size=size):
                searcher, productive, selected, frontier = self._dense_cycle_selection(
                    size, root_escape=False,
                )
                self.assertFalse(productive)
                self.assertEqual(selected, [])
                self.assertEqual(frontier, [])
                self.assertLessEqual(sum(node.visits for node in searcher.nodes), size)
                self.assertLessEqual(searcher._cycles_rejected, size * size)
                self.assertTrue(all(
                    not edge.invalid and edge.virtual_visits == 0
                    for node in searcher.nodes for edge in node.edges
                ))

    def test_dense_cycle_cannot_use_an_ancestor_escape_to_claim_productivity(self):
        for size in (5, 8, 12):
            with self.subTest(size=size):
                searcher, productive, selected, frontier = self._dense_cycle_selection(
                    size, root_escape=True,
                )
                self.assertTrue(productive)
                self.assertEqual(frontier, [size])
                self.assertEqual(len(selected), 1)
                self.assertEqual(searcher.nodes[0].edges[selected[0][1]].tactic.rule, "escape")
                self.assertLessEqual(sum(node.visits for node in searcher.nodes), size)
                self.assertLessEqual(searcher._cycles_rejected, size * size)
                self.assertFalse(any(
                    edge.invalid for node in searcher.nodes for edge in node.edges
                ))
                searcher._backup(selected)
                self.assertTrue(all(
                    edge.virtual_visits == 0
                    for node in searcher.nodes for edge in node.edges
                ))

    def test_htps_keeps_hard_search_budgets(self):
        for limits in ({"simulations": 1}, {"expansion_budget": 1}):
            with self.subTest(limits=limits):
                _, _, result, metrics = self._cycle_search(**limits)
                self.assertFalse(result.search.solved)
                self.assertEqual(metrics.simulations, 1)
                self.assertEqual(metrics.expansions, 1)

    def test_neural_length_filter_is_unknown_and_cannot_create_negative_replay(self):
        database = parse(ROOT / "formal" / "peano-number-theory.mm")
        tokenizer = MetamathTokenizer.from_database(database)
        environment = BackwardEnvironment(database)
        target = Theorem(
            "reflexivity", [], assertion(Node("=", (Node("a"), Node("a")))),
            variable_types={"a": "term"},
        )
        initial = ProofState.from_theorem(target)
        generator = HybridActionGenerator(environment)
        model = ProofTransformer(ProofTransformerConfig(
            vocab_size=len(tokenizer.tokens), pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
            d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout=0, max_state_tokens=512, max_action_tokens=5,
        ))
        policy = TransformerPolicy(model, tokenizer, environment, "cpu")
        candidates = generator.actions(initial)
        self.assertTrue(any(environment.apply(initial, t).after.solved for t in candidates.all))
        self.assertEqual(policy.rank(initial, candidates.all), [])
        result = ProofMCTS(environment, generator, policy, MCTSConfig(simulations=20)).prove(initial)
        self.assertFalse(result.search.solved)
        self.assertEqual(result.outcome, "budget_exhausted")
        self.assertFalse(any(item.value_target_valid for item in result.experiences))
        replay = ReplayBuffer()
        self.assertEqual(replay.add_mcts(result, tokenizer, environment), 0)

    def test_bounded_empty_candidates_and_branching_limits_are_not_dead_ends(self):
        target = Theorem("identity", [], assertion(implies(Node("p"), implies(Node("q"), Node("p")))), variable_types={"p": "wff", "q": "wff"})
        state = ProofState.from_theorem(target)
        environment = BackwardEnvironment(self.database)

        class EmptyBounded:
            def actions(self, state):
                return HybridActions((), ())

        for generator, branching in ((EmptyBounded(), 48), (HybridActionGenerator(environment), 0)):
            result = ProofMCTS(environment, generator, UniformPolicy(), MCTSConfig(simulations=20, branching=branching)).prove(state)
            self.assertEqual(result.outcome, "budget_exhausted")
            self.assertFalse(any(item.value_target_valid for item in result.experiences))

    def test_symbolically_exhaustive_empty_actions_keep_negative_supervision(self):
        environment = BackwardEnvironment(self.database)
        environment.assertions.clear()
        target = Theorem("no_rules", [], assertion(Node("p")), variable_types={"p": "wff"})

        class ExhaustiveEmpty:
            def actions(self, state):
                assert not environment.assertions and not state.hypotheses
                return HybridActions((), (), exhaustive=True)

        result = ProofMCTS(environment, ExhaustiveEmpty(), UniformPolicy(), MCTSConfig(simulations=20)).prove(ProofState.from_theorem(target))
        self.assertEqual(result.outcome, "proven_dead_end")
        self.assertEqual(len(result.experiences), 1)
        self.assertTrue(result.experiences[0].value_target_valid)
        self.assertEqual(result.experiences[0].value_target, 0.0)
        tokenizer = MetamathTokenizer.from_database(self.database)
        replay = ReplayBuffer()
        self.assertEqual(replay.add_mcts(result, tokenizer, environment), 1)
        self.assertEqual(replay.examples[0].value_weight, 1.0)

    def test_repeated_filtered_child_does_not_label_ancestors_or_overwrite_replay(self):
        p, q = Node("p"), Node("q")
        lemma = implies(q, implies(p, q))
        target = Theorem("ancestor", [Hypothesis("h", assertion(implies(lemma, p)))], assertion(p), variable_types={"p": "wff", "q": "wff"})
        initial = ProofState.from_theorem(target)
        environment = BackwardEnvironment(self.database)

        class Actions:
            def actions(self, state):
                tactic = Tactic.create("ax-mp", {"phi": lemma}) if state == initial else Tactic.create("ax-1")
                return HybridActions((tactic,), (), exhaustive=True)

        class FilteredChild:
            def rank(self, state, tactics):
                return [RankedTactic(tactics[0], 0.0, 0.5)] if state == initial else []

        searcher = ProofMCTS(environment, Actions(), FilteredChild(), MCTSConfig(simulations=20, dirichlet_fraction=0))
        result = searcher.prove(initial)
        self.assertGreater(searcher._expanded_nodes[1].visits, 2)
        self.assertFalse(any(item.value_target_valid for item in result.experiences))
        tokenizer = MetamathTokenizer.from_database(self.database)
        ids = tuple(tokenizer.encode(tokenizer.proof_state_tokens(initial)))
        replay = ReplayBuffer()
        positive = ReplayExample(ids, (), (), 1.0, policy_weight=0.0, value_weight=1.0)
        replay.examples.append(positive)
        replay._by_state[ids] = 0
        self.assertEqual(replay.add_mcts(result, tokenizer, environment), 0)
        self.assertEqual(replay.examples, [positive])


if __name__ == "__main__":
    unittest.main()

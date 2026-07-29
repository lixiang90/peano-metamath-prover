from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import parse
from neural_prover.benchmark import (
    BenchmarkBuildConfig,
    build_benchmarks,
    load_benchmarks,
)
from neural_prover.certificate import (
    compile_certificate,
    verify_certificate,
)
from neural_prover.environment import (
    BackwardEnvironment,
    InvalidTactic,
    ProofState,
    Tactic,
)
from neural_prover.tokenizer import MetamathTokenizer

ROOT = Path(__file__).resolve().parents[1]
PEANO_NT = ROOT / "formal" / "peano-number-theory.mm"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class NeuralTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO_NT)
        cls.tokenizer = MetamathTokenizer.from_database(cls.database)

    def test_formal_tokenizer_is_atomic_and_lossless(self) -> None:
        theorem = self.database.logical_assertions["eq-sym"]
        tokens = self.tokenizer.state_tokens(theorem)
        ids = self.tokenizer.encode(tokens)
        self.assertEqual(self.tokenizer.decode(ids), tokens)
        self.assertIn("<STATE>", tokens)
        self.assertIn("<GOAL>", tokens)
        self.assertIn("implies", tokens)
        self.assertNotIn(self.tokenizer.token_to_id["<UNK>"], ids)

    def test_state_tokens_round_trip_to_typed_theorem(self) -> None:
        theorem = self.database.logical_assertions["eq-sym"]
        tokens = self.tokenizer.state_tokens(theorem)
        decoded = self.tokenizer.theorem_from_state_tokens(
            tokens, self.database
        )
        self.assertEqual(
            self.tokenizer.state_tokens(decoded),
            tokens,
        )

    def test_decoded_distinct_pairs_are_normalized(self) -> None:
        theorem = self.database.logical_assertions["alpha_1"]
        decoded = self.tokenizer.theorem_from_state_tokens(
            self.tokenizer.state_tokens(theorem),
            self.database,
        )
        self.assertTrue(decoded.d_constraints)
        self.assertTrue(all(
            left < right
            for left, right in decoded.d_constraints
        ))

    def test_symbolic_environment_closes_reflexivity(self) -> None:
        target = Theorem(
            "refl0",
            [],
            Node("|-", (Node("=", (Node("0"), Node("0"))),)),
        )
        environment = BackwardEnvironment(self.database)
        state = ProofState.from_theorem(target)
        tactics = environment.enumerate_tactics(
            state,
            include_derived=False,
        )
        reflexivity = next(
            tactic for tactic in tactics if tactic.rule == "eq-refl"
        )
        self.assertTrue(environment.apply(state, reflexivity).after.solved)

    def test_rule_variables_are_standardized_apart(self) -> None:
        target = Theorem(
            "right_zero",
            [],
            Node("|-", (
                Node("=", (
                    Node("x"),
                    Node("+", (Node("x"), Node("0"))),
                )),
            )),
            variable_types={"x": "var"},
        )
        environment = BackwardEnvironment(self.database)
        transition = environment.apply(
            ProofState.from_theorem(target),
            Tactic.create("pa_ax3", {"x": Node("x")}),
        )
        self.assertTrue(transition.after.solved)

    def test_var_rule_rejects_term_substitution(self) -> None:
        target = Theorem(
            "ill_typed_right_zero",
            [],
            Node("|-", (
                Node("=", (
                    Node("t"),
                    Node("+", (Node("t"), Node("0"))),
                )),
            )),
            variable_types={"t": "term"},
        )
        environment = BackwardEnvironment(self.database)
        with self.assertRaises(InvalidTactic):
            environment.apply(
                ProofState.from_theorem(target),
                Tactic.create("pa_ax3", {"x": Node("t")}),
            )

    def test_benchmark_has_separate_famous_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            build_benchmarks(
                PEANO_NT,
                path,
                BenchmarkBuildConfig(
                    seeds=(101,),
                    steps_per_seed=50,
                    synthetic_per_difficulty=1,
                    max_proof_depth=3,
                ),
            )
            cases = load_benchmarks(path)
            famous = [case for case in cases if case.origin == "famous"]
            curated = [
                case for case in cases if case.origin == "curated"
            ]
            self.assertEqual(len(famous), 4)
            self.assertEqual(len(curated), 6)
            self.assertTrue(all(
                case.difficulty == "frontier" for case in famous
            ))
            self.assertEqual(
                sum(case.expected_status == "open conjecture"
                    for case in famous),
                2,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not visible in this sandbox")
class TorchNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO_NT)

    def test_transformer_forward_shapes(self) -> None:
        import torch

        from neural_prover.model import (
            ProofTransformer,
            ProofTransformerConfig,
        )

        config = ProofTransformerConfig(
            vocab_size=64,
            pad_id=0,
            bos_id=2,
            eos_id=3,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            max_state_tokens=16,
            max_action_tokens=12,
        )
        model = ProofTransformer(config)
        logits, value = model(
            torch.randint(4, 64, (2, 10)),
            torch.randint(4, 64, (2, 7)),
        )
        self.assertEqual(tuple(logits.shape), (2, 7, 64))
        self.assertEqual(tuple(value.shape), (2,))

    def test_hybrid_mcts_emits_verified_certificate(self) -> None:
        from neural_prover.hybrid import HybridActionGenerator
        from neural_prover.mcts import MCTSConfig, ProofMCTS
        from neural_prover.search import HeuristicPolicy

        target = Theorem(
            "symmetry",
            [
                Hypothesis(
                    "h",
                    Node("|-", (Node("=", (Node("a"), Node("b"))),)),
                )
            ],
            Node("|-", (Node("=", (Node("b"), Node("a"))),)),
            variable_types={"a": "term", "b": "term"},
        )
        environment = BackwardEnvironment(self.database)
        result = ProofMCTS(
            environment,
            HybridActionGenerator(environment),
            HeuristicPolicy(environment),
            MCTSConfig(simulations=30, max_depth=6),
        ).prove(ProofState.from_theorem(target))
        self.assertTrue(result.search.solved)
        certificate = compile_certificate(
            target,
            result.search,
            self.database,
        )
        verify_certificate(certificate, self.database)


if __name__ == "__main__":
    unittest.main()

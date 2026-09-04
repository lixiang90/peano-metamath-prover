from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import parse
from neural_prover.certificate import compile_certificate, verify_certificate
from neural_prover.environment import ProofState, Tactic
from neural_prover.hybrid import HybridActions
from neural_prover.lemma import LemmaBackwardEnvironment, propose_lemma
from neural_prover.search import HeuristicPolicy

from htps_prover.forward import ForwardDAGConfig, build_forward_dag_dataset
from htps_prover.hypergraph import (
    HTPSConfig,
    HyperTreeProofSearch,
    materialize_search_config,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
PEANO = ROOT / "formal" / "peano.mm"


class _FixedActions:
    def __init__(self, environment, initial, lemma, cut) -> None:
        self.environment = environment
        self.initial = initial
        self.lemma = lemma
        self.cut = cut

    def actions(self, state):
        if state == self.initial:
            return HybridActions((self.cut,), ())
        tactics = self.environment.enumerate_tactics(
            state, include_derived=True
        )
        if state.current_goal == self.lemma:
            selected = next(
                tactic for tactic in tactics
                if tactic.rule == "ax-1"
                and self.environment.apply(state, tactic).after.solved
            )
        else:
            selected = next(
                tactic for tactic in tactics
                if tactic.rule == "ax-mp"
                and self.environment.apply(state, tactic).after.solved
            )
        return HybridActions((selected,), ())


class HTPSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO)

    def test_guarded_lemma_hypertree_replays_and_certifies(self) -> None:
        p = Node("p")
        q = Node("q")
        target = Theorem(
            "htps-lemma-target",
            [Hypothesis("hp", Node("|-", (p,)))],
            Node("|-", (Node("implies", (q, p)),)),
            variable_types={"p": "wff", "q": "wff"},
        )
        lemma = Node("|-", (
            Node("implies", (p, Node("implies", (q, p)))),
        ))
        initial = ProofState.from_theorem(target)
        environment = LemmaBackwardEnvironment(self.database)
        cut = propose_lemma(lemma)
        searcher = HyperTreeProofSearch(
            environment,
            _FixedActions(environment, initial, lemma, cut),
            HeuristicPolicy(environment),
            HTPSConfig(
                simulations=12,
                expansion_budget=12,
                branching=2,
                max_frontier=4,
            ),
        )
        result, metrics = searcher.prove(initial)
        self.assertTrue(result.search.solved)
        self.assertEqual(result.search.actions[0], cut)
        self.assertEqual(metrics.guarded_edges, 1)
        certificate = compile_certificate(
            target, result.search, self.database, name="htps_lemma_cert"
        )
        verify_certificate(certificate, self.database)

    def test_ordinary_and_edge_expands_both_frontier_goals(self) -> None:
        p = Node("p")
        q = Node("q")
        r = Node("r")
        phi = Node("implies", (p, Node("implies", (q, p))))
        target_formula = Node("implies", (r, phi))
        target = Theorem(
            "htps-and-target",
            [],
            Node("|-", (target_formula,)),
            variable_types={"p": "wff", "q": "wff", "r": "wff"},
        )
        initial = ProofState.from_theorem(target)
        environment = LemmaBackwardEnvironment(self.database)
        root_tactic = Tactic.create(
            "ax-mp", {"phi": phi, "psi": target_formula}
        )

        class FixedAndActions:
            def actions(inner_self, state):
                if state == initial:
                    return HybridActions((root_tactic,), ())
                return HybridActions((Tactic.create("ax-1"),), ())

        result, metrics = HyperTreeProofSearch(
            environment,
            FixedAndActions(),
            HeuristicPolicy(environment),
            HTPSConfig(
                simulations=8,
                expansion_budget=8,
                branching=2,
                max_frontier=4,
            ),
        ).prove(initial)
        self.assertTrue(result.search.solved)
        self.assertEqual(result.search.actions[0], root_tactic)
        self.assertEqual(len(result.search.actions), 3)
        self.assertGreaterEqual(metrics.maximum_frontier, 2)
        certificate = compile_certificate(
            target, result.search, self.database, name="htps_and_cert"
        )
        verify_certificate(certificate, self.database)

    def test_forward_dag_export_is_independently_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_forward_dag_dataset(
                PEANO,
                directory,
                ForwardDAGConfig(
                    seeds=(7,),
                    steps_per_seed=25,
                    max_proof_depth=5,
                    verify_deepest_per_seed=1,
                ),
            )
            output = Path(directory)
            self.assertGreater(manifest["proof_dag_nodes"], 0)
            self.assertGreater(
                manifest["independent_verification"][
                    "replayed_generated_declarations"
                ],
                0,
            )
            self.assertTrue((output / "policy_train.jsonl").exists())
            self.assertTrue((output / "lemma_train.jsonl").exists())

    def test_randomized_search_config_is_bounded_and_reproducible(self) -> None:
        base = HTPSConfig(
            simulations=100,
            expansion_budget=200,
            branching=20,
            randomize=True,
            seed=13,
        )
        first = materialize_search_config(base, 4)
        again = materialize_search_config(base, 4)
        other = materialize_search_config(base, 5)
        self.assertEqual(first, again)
        self.assertNotEqual(first.seed, other.seed)
        self.assertFalse(first.randomize)
        self.assertLessEqual(first.simulations, base.simulations)
        self.assertGreaterEqual(first.simulations, base.simulations // 2)
        self.assertLessEqual(first.expansion_budget, base.expansion_budget)
        self.assertGreaterEqual(
            first.expansion_budget, base.expansion_budget // 2
        )
        self.assertLessEqual(first.branching, base.branching)
        self.assertGreaterEqual(first.branching, base.branching // 2)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class HTPSTrainingTests(unittest.TestCase):
    def test_forward_sft_and_soft_replay_train_latent_checkpoint(self) -> None:
        import json

        from neural_prover.latent_model import (
            LatentProofTransformer,
            LatentReasoningConfig,
        )
        from neural_prover.model import ProofTransformerConfig
        from neural_prover.rl import ReplayBuffer, ReplayExample
        from neural_prover.tokenizer import MetamathTokenizer

        from htps_prover.training import (
            ReplayTrainConfig,
            SupervisedTrainConfig,
            train_latent_from_replay,
            train_supervised_latent,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            build_forward_dag_dataset(
                PEANO,
                data,
                ForwardDAGConfig(
                    seeds=(7,),
                    steps_per_seed=25,
                    max_proof_depth=5,
                    verify_deepest_per_seed=0,
                ),
            )
            tokenizer = MetamathTokenizer.load(data / "tokenizer.json")
            model = LatentProofTransformer(
                ProofTransformerConfig(
                    vocab_size=len(tokenizer),
                    pad_id=tokenizer.pad_id,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    d_model=16,
                    nhead=4,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    dim_feedforward=32,
                    max_state_tokens=384,
                    max_action_tokens=256,
                ),
                LatentReasoningConfig(
                    max_thought_steps=2,
                    min_thought_steps=1,
                    lemma_token_id=tokenizer.token_to_id[
                        "<PROPOSE_LEMMA>"
                    ],
                ),
            )
            initial = root / "initial.pt"
            supervised = root / "supervised.pt"
            online = root / "online.pt"
            model.save_checkpoint(initial)
            sft = train_supervised_latent(
                initial,
                data / "tokenizer.json",
                data / "policy_train.jsonl",
                data / "lemma_train.jsonl",
                supervised,
                SupervisedTrainConfig(
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    max_examples=2,
                ),
            )
            self.assertEqual(sft["examples"], 2)
            policy = json.loads(
                (data / "policy_train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )
            lemma_lines = (data / "lemma_train.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            lemma = json.loads(lemma_lines[0])
            replay = ReplayBuffer()
            replay.examples.append(ReplayExample(
                tuple(policy["state_ids"]),
                (
                    tuple(policy["action_ids"]),
                    tuple(lemma["lemma_action_ids"]),
                ),
                (0.8, 0.2),
                0.75,
                1.0,
                1.0,
            ))
            trained = train_latent_from_replay(
                supervised,
                replay,
                online,
                ReplayTrainConfig(
                    epochs=1,
                    gradient_accumulation=1,
                    device="cpu",
                ),
            )
            self.assertEqual(trained["examples"], 1)
            restored, payload = LatentProofTransformer.load_checkpoint(online)
            self.assertTrue(restored.candidate_head_trained)
            self.assertIn("htps_online", payload["metadata"])


if __name__ == "__main__":
    unittest.main()

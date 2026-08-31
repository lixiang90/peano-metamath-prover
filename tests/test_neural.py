from __future__ import annotations

import gzip
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.definitions import load_definition_catalog
from metamath_generator.parser import parse
from metamath_generator.unification import substitute_simultaneous
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
    parse_tactic_tokens,
)
from neural_prover.lemma import (
    LemmaBackwardEnvironment,
    lemma_from_tactic,
    propose_lemma,
)
from neural_prover.tokenizer import MetamathTokenizer

ROOT = Path(__file__).resolve().parents[1]
PEANO_NT = ROOT / "formal" / "peano-number-theory.mm"
PA_PLUS = ROOT / "formal" / "peano-pa-plus.mm"
PA_PLUS_CATALOG = ROOT / "formal" / "pa-plus-definitions.json"
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

    def test_pa_plus_context_and_bounded_inference_are_formal(self) -> None:
        database = parse(PA_PLUS)
        catalog = load_definition_catalog(PA_PLUS_CATALOG)
        context = MetamathTokenizer.pa_plus_context_from_database(
            database,
            catalog.definition_names,
            catalog.statement_names,
            bounded_nat_max=2,
        )
        tokenizer = MetamathTokenizer.from_database(
            database, pa_plus_context=context
        )
        statement = database.syntax_statements[
            "factorial-zero-statement"
        ]
        target = Theorem(
            "factorial-zero-target",
            [],
            Node("|-", statement.conclusion.args),
        )
        tokens = tokenizer.state_tokens(target)
        self.assertIn("<PA_PLUS_CONTEXT>", tokens)
        self.assertIn("factorial", tokens)
        self.assertIn("factorial-zero-statement", tokens)
        self.assertIn("<GROUND_TERMS>", tokens)
        self.assertNotIn(tokenizer.token_to_id["<UNK>"], tokenizer.encode(tokens))

        environment = BackwardEnvironment(database)
        environment.configure_from_tokenizer(tokenizer)
        state = ProofState.from_theorem(target)
        candidates = environment._local_candidate_expressions(
            state, "term", 12
        )
        self.assertEqual(
            candidates[:3],
            [
                Node("0"),
                Node("S", (Node("0"),)),
                Node("S", (Node("S", (Node("0"),)),)),
            ],
        )

        bridge = environment.assertions["gen_df_factorial_fold"]
        self.assertIsNotNone(bridge.proof)
        self.assertTrue(bridge.proof.source_labels)

    def test_pa_plus_bridge_macro_inlines_to_source_certificate(self) -> None:
        from neural_prover.search import SearchResult

        database = parse(PA_PLUS)
        catalog = load_definition_catalog(PA_PLUS_CATALOG)
        context = MetamathTokenizer.pa_plus_context_from_database(
            database,
            catalog.definition_names,
            catalog.statement_names,
            bounded_nat_max=2,
        )
        tokenizer = MetamathTokenizer.from_database(
            database, pa_plus_context=context
        )
        environment = BackwardEnvironment(database)
        environment.configure_from_tokenizer(tokenizer)
        bridge = environment.assertions["gen_df_positive_fold"]
        variable = next(iter(bridge.variable_types))
        substitution = {variable: Node("S", (Node("0"),))}
        target = Theorem(
            "positive-fold-target",
            [],
            substitute_simultaneous(bridge.conclusion, substitution),
        )
        before = ProofState.from_theorem(target)
        tactic = Tactic.create(bridge.name, substitution)
        transition = environment.apply(before, tactic)
        self.assertTrue(transition.after.solved)
        result = SearchResult(
            True,
            1,
            2,
            (tactic,),
            (transition,),
            transition.after,
        )
        certificate = compile_certificate(target, result, database)
        verify_certificate(certificate, database)
        self.assertNotIn(bridge.name, certificate.proof.source_labels)
        self.assertIn("df-positive", certificate.proof.source_labels)

    def test_pa_plus_upgrade_preserves_old_token_ids(self) -> None:
        database = parse(PA_PLUS)
        catalog = load_definition_catalog(PA_PLUS_CATALOG)
        upgraded = self.tokenizer.upgraded_for_pa_plus(
            database,
            catalog.definition_names,
            catalog.statement_names,
        )
        self.assertEqual(
            upgraded.tokens[:len(self.tokenizer)],
            self.tokenizer.tokens,
        )
        self.assertIn("<PA_PLUS_CONTEXT>", upgraded.token_to_id)
        self.assertIn("gen_df_factorial_unfold", upgraded.token_to_id)

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

    def test_typed_rule_variable_action_round_trip(self) -> None:
        theorem = self.database.logical_assertions["eq-sym"]
        canonical = self.tokenizer.canonical_variables(theorem)
        order = [
            floating.expr.args[0].op for floating in theorem.floating
        ] or list(theorem.variable_types)
        substitution = {
            variable: Node(variable) for variable in order
        }
        tokens = self.tokenizer.tactic_tokens(
            theorem.name,
            substitution,
            canonical,
            variable_order=order,
            rule_variable_types=theorem.variable_types,
        )
        self.assertTrue(any(
            token.startswith("<V:term:") for token in tokens
        ))
        restored = parse_tactic_tokens(
            tokens, theorem, self.tokenizer, self.database
        )
        self.assertEqual(restored, Tactic.create(theorem.name, substitution))

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

    def test_environment_excludes_target_equivalent_rule_and_counts_work(self) -> None:
        target = Theorem(
            "refl0",
            [],
            Node("|-", (Node("=", (Node("0"), Node("0"))),)),
        )
        environment = BackwardEnvironment(
            self.database,
            excluded_assertions=("eq-refl",),
        )
        state = ProofState.from_theorem(target)
        tactics = environment.enumerate_tactics(
            state,
            include_derived=False,
        )
        self.assertNotIn("eq-refl", environment.assertions)
        self.assertTrue(all(tactic.rule != "eq-refl" for tactic in tactics))
        self.assertEqual(environment.metrics.tactic_enumerations, 1)
        self.assertGreaterEqual(environment.metrics.candidates_returned, 0)

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
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "peano-proof-benchmark-v2")
            self.assertTrue(all(
                "reference_rule" not in record
                for record in payload["cases"]
            ))
            foundational = [
                case for case in cases if case.origin == "foundational"
            ]
            self.assertTrue(all(
                case.score_group == "sanity"
                and not case.research_eligible
                for case in foundational
            ))
            self.assertTrue(all(
                case.score_group == "frontier"
                and not case.research_eligible
                for case in famous
            ))

    def test_external_verifier_result_is_fail_closed_when_unconfigured(self) -> None:
        from neural_prover.external import verify_certificate_external

        with patch(
            "neural_prover.external.discover_metamath_executable",
            return_value=None,
        ):
            certificate = Theorem(
                "external-smoke",
                [],
                Node("|-", (Node("=", (Node("0"), Node("0"))),)),
            )
            result = verify_certificate_external(
                certificate,
                PEANO_NT,
            )
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "not_configured")

    def test_mcts_defaults_require_leakage_eligible_research_cases(self) -> None:
        from neural_prover.evaluate import MCTSEvaluationConfig

        config = MCTSEvaluationConfig()
        self.assertEqual(config.score_groups, ("research",))
        self.assertTrue(config.require_research_eligible)

    def test_lemma_vocabulary_upgrade_preserves_legacy_ids(self) -> None:
        controls = {
            "<PROPOSE_LEMMA>", "<LEMMA>", "<END_LEMMA>"
        }
        current = MetamathTokenizer.from_database(self.database)
        legacy_tokens = [
            token for token in current.tokens if token not in controls
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-tokenizer.json"
            path.write_text(json.dumps({
                "format": "peano-metamath-tokenizer-v1",
                "config": {"max_variables_per_type": 32, "strict": True},
                "tokens": legacy_tokens,
            }), encoding="utf-8")
            legacy = MetamathTokenizer.load(path)
        self.assertFalse(legacy.supports_lemma_actions)
        upgraded = legacy.upgraded_for_lemma_actions()
        self.assertTrue(upgraded.supports_lemma_actions)
        self.assertEqual(len(upgraded), len(legacy) + 3)
        self.assertTrue(all(
            upgraded.token_to_id[token] == index
            for index, token in enumerate(legacy.tokens)
        ))

    def test_lemma_action_token_round_trip(self) -> None:
        target = Theorem(
            "lemma-token-target",
            [],
            Node("|-", (Node("p"),)),
            variable_types={"p": "wff", "q": "wff"},
        )
        lemma = Node("|-", (
            Node("implies", (Node("q"), Node("p"))),
        ))
        tokenizer = MetamathTokenizer.from_database(self.database)
        canonical = tokenizer.canonical_variables(target)
        tokens = tokenizer.lemma_tactic_tokens(lemma, canonical)
        restored = parse_tactic_tokens(
            tokens, target, tokenizer, self.database
        )
        self.assertEqual(lemma_from_tactic(restored), lemma)

    def test_intermediate_lemma_is_proved_before_activation(self) -> None:
        p = Node("p")
        q = Node("q")
        target = Theorem(
            "lemma-cut-target",
            [Hypothesis("hp", Node("|-", (p,)))],
            Node("|-", (Node("implies", (q, p)),)),
            variable_types={"p": "wff", "q": "wff"},
        )
        lemma = Node("|-", (
            Node("implies", (
                p,
                Node("implies", (q, p)),
            )),
        ))
        environment = LemmaBackwardEnvironment(self.database)
        initial = ProofState.from_theorem(target)
        cut = propose_lemma(lemma)
        cut_transition = environment.apply(initial, cut)
        self.assertEqual(cut_transition.after.current_goal, lemma)
        self.assertNotIn(lemma, cut_transition.after.hypotheses)

        ax1 = next(
            tactic
            for tactic in environment.enumerate_tactics(
                cut_transition.after, include_derived=False
            )
            if tactic.rule == "ax-1"
            and environment.apply(
                cut_transition.after, tactic
            ).after.current_goal == target.conclusion
        )
        lemma_transition = environment.apply(
            cut_transition.after, ax1
        )
        self.assertIn(lemma, lemma_transition.after.hypotheses)
        self.assertEqual(
            lemma_transition.after.current_goal, target.conclusion
        )
        axmp = next(
            tactic
            for tactic in environment.enumerate_tactics(
                lemma_transition.after, include_derived=True
            )
            if tactic.rule == "ax-mp"
            and environment.apply(
                lemma_transition.after, tactic
            ).after.solved
        )
        final_transition = environment.apply(
            lemma_transition.after, axmp
        )

        from neural_prover.search import SearchResult

        transitions = (
            cut_transition, lemma_transition, final_transition
        )
        result = SearchResult(
            True,
            3,
            4,
            tuple(item.tactic for item in transitions),
            transitions,
            final_transition.after,
        )
        certificate = compile_certificate(
            target, result, self.database, name="lemma_cut"
        )
        verify_certificate(certificate, self.database)
        self.assertNotIn("<PROPOSE_LEMMA>", certificate.proof.source_labels)

    def test_mcts_closes_a_decomposed_lemma_proof(self) -> None:
        from neural_prover.hybrid import HybridActions
        from neural_prover.lemma import (
            LemmaActionGenerator,
            LemmaGeneratorConfig,
        )
        from neural_prover.mcts import MCTSConfig, ProofMCTS
        from neural_prover.rl import ReplayBuffer
        from neural_prover.search import HeuristicPolicy

        p = Node("p")
        q = Node("q")
        target = Theorem(
            "lemma-mcts-target",
            [Hypothesis("hp", Node("|-", (p,)))],
            Node("|-", (Node("implies", (q, p)),)),
            variable_types={"p": "wff", "q": "wff"},
        )
        lemma = Node("|-", (
            Node("implies", (
                p,
                Node("implies", (q, p)),
            )),
        ))
        environment = LemmaBackwardEnvironment(self.database)
        initial = ProofState.from_theorem(target)
        cut = propose_lemma(lemma)
        generated = LemmaActionGenerator(
            environment,
            LemmaGeneratorConfig(max_lemma_candidates=12),
        ).actions(initial)
        self.assertIn(cut, generated.constructions)

        class FixedDecompositionGenerator:
            def actions(inner_self, state):
                if state == initial:
                    return HybridActions((cut,), ())
                tactics = environment.enumerate_tactics(
                    state, include_derived=True
                )
                if state.current_goal == lemma:
                    selected = next(
                        tactic for tactic in tactics
                        if tactic.rule == "ax-1"
                        and environment.apply(
                            state, tactic
                        ).after.current_goal == target.conclusion
                    )
                else:
                    selected = next(
                        tactic for tactic in tactics
                        if tactic.rule == "ax-mp"
                        and environment.apply(state, tactic).after.solved
                    )
                return HybridActions((selected,), ())

        result = ProofMCTS(
            environment,
            FixedDecompositionGenerator(),
            HeuristicPolicy(environment),
            MCTSConfig(
                simulations=8,
                max_depth=5,
                branching=2,
                dirichlet_fraction=0.0,
            ),
        ).prove(initial)
        self.assertTrue(result.search.solved)
        self.assertEqual(result.search.actions[0], cut)
        certificate = compile_certificate(
            target, result.search, self.database, name="lemma_mcts"
        )
        verify_certificate(certificate, self.database)

        tokenizer = MetamathTokenizer.from_database(self.database)
        replay = ReplayBuffer()
        self.assertGreater(replay.add_mcts(
            result, tokenizer, environment
        ), 0)
        lemma_token_id = tokenizer.token_to_id["<PROPOSE_LEMMA>"]
        self.assertTrue(any(
            lemma_token_id in action_ids
            for example in replay.examples
            for action_ids in example.action_ids
        ))


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

    def test_continuous_latent_reasoning_and_checkpoint_upgrade(self) -> None:
        import torch

        from neural_prover.latent_model import (
            LatentProofTransformer,
            LatentReasoningConfig,
        )
        from neural_prover.model import (
            ProofTransformer,
            ProofTransformerConfig,
        )
        from neural_prover.latent_train import latent_policy_value_loss

        controls = {
            "<PROPOSE_LEMMA>", "<LEMMA>", "<END_LEMMA>"
        }
        full = MetamathTokenizer.from_database(self.database)
        legacy = MetamathTokenizer(
            [token for token in full.tokens if token not in controls],
            preserve_token_order=True,
        )
        upgraded = legacy.upgraded_for_lemma_actions()
        config = ProofTransformerConfig(
            vocab_size=len(legacy),
            pad_id=legacy.pad_id,
            bos_id=legacy.bos_id,
            eos_id=legacy.eos_id,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            max_state_tokens=16,
            max_action_tokens=12,
        )
        base = ProofTransformer(config)
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.pt"
            latent_path = Path(directory) / "latent.pt"
            base.save_checkpoint(base_path)
            latent, upgrade = LatentProofTransformer.from_base_checkpoint(
                base_path,
                upgraded,
                LatentReasoningConfig(
                    max_thought_steps=4,
                    min_thought_steps=2,
                    halt_threshold=0.8,
                ),
            )
            self.assertEqual(
                upgrade["vocabulary_expansion"]["old_size"],
                len(legacy),
            )
            self.assertTrue(torch.equal(
                latent.token_embedding.weight[:len(legacy)],
                base.token_embedding.weight,
            ))
            with torch.no_grad():
                latent.halt_head.weight.zero_()
                latent.halt_head.bias.fill_(20.0)
            latent.eval()
            state = torch.randint(4, len(legacy), (1, 8))
            candidates = torch.randint(4, len(legacy), (3, 7))
            candidates[0, 1] = upgraded.token_to_id[
                "<PROPOSE_LEMMA>"
            ]
            scores, value = latent.score_candidates(state, candidates)
            self.assertEqual(tuple(scores.shape), (3,))
            self.assertEqual(tuple(value.shape), (1,))
            self.assertEqual(latent.last_reasoning_steps, 2)
            self.assertEqual(latent.reasoning_metrics()[
                "reasoning_mode"
            ], "continuous_latent")
            self.assertEqual(latent.ponder_loss().ndim, 0)
            latent.train()
            train_states = torch.randint(4, len(legacy), (2, 8))
            train_actions = torch.randint(4, len(legacy), (2, 7))
            loss, parts = latent_policy_value_loss(
                latent,
                train_states,
                train_actions,
                torch.tensor([1.0, 0.0]),
            )
            loss.backward()
            self.assertIsNotNone(latent.halt_head.weight.grad)
            self.assertGreaterEqual(
                float(parts["ponder_loss"].detach()), 0.0
            )
            latent.save_checkpoint(latent_path)
            restored, _ = LatentProofTransformer.load_checkpoint(
                latent_path
            )
        self.assertEqual(
            restored.config.vocab_size, len(upgraded)
        )
        self.assertEqual(
            restored.reasoning_config.max_thought_steps, 4
        )

    def test_structured_candidate_head_scores_and_round_trips(self) -> None:
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
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            max_state_tokens=16,
            max_action_tokens=12,
        )
        model = ProofTransformer(config)
        logits, value = model.score_candidates(
            torch.randint(4, 64, (1, 8)),
            torch.randint(4, 64, (3, 7)),
        )
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(value.shape), (1,))
        matrix, batch_value = model.score_candidate_matrix(
            torch.randint(4, 64, (3, 8)),
            torch.randint(4, 64, (3, 7)),
        )
        self.assertEqual(tuple(matrix.shape), (3, 3))
        self.assertEqual(tuple(batch_value.shape), (3,))
        old_embedding = model.token_embedding.weight.detach().clone()
        expansion = model.resize_vocabulary(71)
        self.assertEqual(expansion["added"], 7)
        self.assertTrue(torch.equal(
            model.token_embedding.weight[:64], old_embedding
        ))
        model.candidate_head_trained = True
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            model.save_checkpoint(checkpoint)
            restored, payload = ProofTransformer.load_checkpoint(checkpoint)
        self.assertTrue(restored.candidate_head_trained)
        self.assertEqual(
            payload["metadata"]["candidate_policy"]["mode"],
            "finite-kernel-candidate-index",
        )

    def test_alpha_zero_replay_trains_candidate_head(self) -> None:
        from neural_prover.model import (
            ProofTransformer,
            ProofTransformerConfig,
        )
        from neural_prover.rl import (
            ReinforcementConfig,
            ReplayBuffer,
            ReplayExample,
            reinforce_model,
        )

        config = ProofTransformerConfig(
            vocab_size=32,
            pad_id=0,
            bos_id=2,
            eos_id=3,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            max_state_tokens=12,
            max_action_tokens=10,
        )
        replay = ReplayBuffer()
        replay.examples.append(ReplayExample(
            state_ids=(2, 4, 5, 3),
            action_ids=((2, 6, 3), (2, 7, 3)),
            policy_target=(0.75, 0.25),
            value_target=1.0,
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.pt"
            updated = root / "updated.pt"
            ProofTransformer(config).save_checkpoint(initial)
            reinforce_model(
                initial,
                replay,
                updated,
                ReinforcementConfig(
                    epochs=1,
                    learning_rate=1e-3,
                    gradient_accumulation=1,
                    device="cpu",
                ),
            )
            restored, _ = ProofTransformer.load_checkpoint(updated)
        self.assertTrue(restored.candidate_head_trained)

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
        self.assertEqual(result.outcome, "certifiable_solution")
        self.assertTrue(any(
            experience.policy_target_valid
            and experience.value_target_valid
            for experience in result.experiences
        ))
        certificate = compile_certificate(
            target,
            result.search,
            self.database,
        )
        verify_certificate(certificate, self.database)
        from neural_prover.external import (
            discover_metamath_executable,
            verify_certificate_external,
        )
        executable = discover_metamath_executable()
        if executable is not None:
            external = verify_certificate_external(
                certificate,
                PEANO_NT,
                executable=executable,
            )
            self.assertTrue(external.passed, external.output_tail)

    def test_scale_resume_matches_uninterrupted_training(self) -> None:
        import torch

        from neural_prover.model import ProofTransformer
        from neural_prover.scale_train import (
            ScaleTrainingConfig,
            train_scale_model,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            tokenizer = MetamathTokenizer.from_database(self.database)
            tokenizer.save(corpus / "tokenizer.json")
            state = [
                tokenizer.bos_id,
                tokenizer.token_to_id["<STATE>"],
                tokenizer.token_to_id["<NO_HYP>"],
                tokenizer.token_to_id["<END_STATE>"],
                tokenizer.eos_id,
            ]
            action = [
                tokenizer.bos_id,
                tokenizer.token_to_id["<ACTION>"],
                tokenizer.token_to_id["<ASSUMPTION>"],
                tokenizer.token_to_id["<END_ACTION>"],
                tokenizer.eos_id,
            ]
            shards: dict[str, list[dict]] = {}
            for split in ("train", "validation", "test"):
                split_directory = corpus / split
                split_directory.mkdir()
                shard = split_directory / "part-00000.jsonl.gz"
                with gzip.open(
                    shard, "wt", encoding="utf-8"
                ) as stream:
                    for index in range(8):
                        record = {
                            "id": f"{split}-{index}",
                            "state": [
                                *state[:-1],
                                tokenizer.token_to_id[
                                    "implies"
                                    if index % 2 else "="
                                ],
                                state[-1],
                            ],
                            "action": action,
                            "value": (index + 1) / 10,
                        }
                        stream.write(json.dumps(record) + "\n")
                shards[split] = [{
                    "path": str(shard.relative_to(corpus)),
                    "records": 8,
                    "bytes": shard.stat().st_size,
                }]
            (corpus / "manifest.json").write_text(
                json.dumps({
                    "configuration": {
                        "max_state_tokens": 8,
                        "max_action_tokens": 8,
                    },
                    "shards": shards,
                }),
                encoding="utf-8",
            )
            common = dict(
                max_steps=4,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                warmup_steps=1,
                device="cpu",
                d_model=16,
                nhead=4,
                encoder_layers=1,
                decoder_layers=1,
                dim_feedforward=32,
                dropout=0.2,
                gradient_checkpointing=False,
                initial_context_tokens=8,
                context_warmup_steps=1,
                shuffle_buffer=3,
                checkpoint_every=1,
                validation_batches=1,
                require_long_context_step=False,
                long_context_threshold=2,
                enforce_scale_parameter_range=False,
            )
            continuous_output = root / "continuous"
            continuous = train_scale_model(
                corpus,
                continuous_output,
                ScaleTrainingConfig(**common),
            )
            resumed_output = root / "resumed"
            paused = train_scale_model(
                corpus,
                resumed_output,
                ScaleTrainingConfig(**common, run_steps=2),
            )
            self.assertEqual(paused["status"], "paused")
            resumed = train_scale_model(
                corpus,
                resumed_output,
                ScaleTrainingConfig(**common),
                resume_from=resumed_output / "latest.pt",
            )
            uninterrupted_model, _ = (
                ProofTransformer.load_checkpoint(
                    continuous_output / "final.pt"
                )
            )
            resumed_model, _ = ProofTransformer.load_checkpoint(
                resumed_output / "final.pt"
            )
            for name, expected in uninterrupted_model.state_dict().items():
                self.assertTrue(
                    torch.equal(expected, resumed_model.state_dict()[name]),
                    name,
                )
            self.assertEqual(resumed["train_examples_seen"], 4)
            self.assertEqual(len(resumed["history"]), 4)
            self.assertEqual(
                [item["loss"] for item in continuous["history"]],
                [item["loss"] for item in resumed["history"]],
            )

    def test_scale_evaluation_metadata_is_json_safe(self) -> None:
        import torch

        from neural_prover.scale_train import (
            _checkpoint_report_metadata,
        )

        report = _checkpoint_report_metadata({
            "parameter_count": 123,
            "training_state": {
                "step": 7,
                "train_examples_seen": 28,
                "rng_state": {"torch": torch.get_rng_state()},
                "history": [{"loss": 1.0}],
            },
        })
        json.dumps(report)
        self.assertEqual(report["training_state"]["step"], 7)
        self.assertNotIn("rng_state", report["training_state"])
        self.assertNotIn("history", report["training_state"])


if __name__ == "__main__":
    unittest.main()

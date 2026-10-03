from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CandidateEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from metamath_generator.model import Hypothesis, Node, Theorem
        from metamath_generator.parser import parse
        from neural_prover.environment import BackwardEnvironment, ProofState, Tactic
        from neural_prover.tokenizer import MetamathTokenizer

        torch.set_num_threads(1)
        cls.database = parse(ROOT / "formal/peano.mm")
        cls.environment = BackwardEnvironment(cls.database)
        cls.tokenizer = MetamathTokenizer.from_database(cls.database)
        x = Node("=", (Node("x"), Node("x")))
        y = Node("=", (Node("y"), Node("y")))
        xy = Node("implies", (x, y))
        yx = Node("implies", (y, x))
        cls.state = ProofState.from_theorem(Theorem(
            "candidate-order", [Hypothesis("h", Node("|-", (xy,)))],
            Node("|-", (x,)), variable_types={"x": "var", "y": "var"},
        ))
        cls.tactics = [
            Tactic.create("ax-mp", {"phi": premise, "psi": x})
            for premise in (xy, yx)
        ]
        canonical = cls.tokenizer.proof_state_variables(cls.state)
        cls.sequences = [
            cls.tokenizer.assertion_tactic_tokens(
                cls.environment.assertions[tactic.rule],
                tactic.substitution_dict(), canonical,
            )
            for tactic in cls.tactics
        ]
        cls.candidates = torch.tensor([
            cls.tokenizer.encode(tokens) for tokens in cls.sequences
        ])
        cls.states = torch.tensor([
            cls.tokenizer.encode(cls.tokenizer.proof_state_tokens(cls.state))
        ])

    def make_model(self, *, latent=False):
        import torch
        from neural_prover.model import ProofTransformer, ProofTransformerConfig

        torch.manual_seed(11)
        config = ProofTransformerConfig(
            len(self.tokenizer), self.tokenizer.pad_id,
            self.tokenizer.bos_id, self.tokenizer.eos_id,
            d_model=16, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=32, dropout=0,
            max_state_tokens=256, max_action_tokens=128,
        )
        if latent:
            from neural_prover.latent_model import (
                LatentProofTransformer, LatentReasoningConfig,
            )
            return LatentProofTransformer(
                config, LatentReasoningConfig(max_thought_steps=2),
            ).eval()
        return ProofTransformer(config).eval()

    def test_distinct_legal_actions_with_same_multiset_have_distinct_scores(self):
        import torch

        self.assertEqual(Counter(self.sequences[0]), Counter(self.sequences[1]))
        after = [self.environment.apply(self.state, tactic).after
                 for tactic in self.tactics]
        self.assertNotEqual(after[0], after[1])
        model = self.make_model().double()
        scores, _ = model.score_candidates(self.states, self.candidates)
        self.assertGreater(abs(float((scores[0] - scores[1]).detach())), 1e-10)
        loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
        loss.backward()
        self.assertGreater(float(model.candidate_encoder.weight_ih_l0.grad.norm()), 0)

    def test_single_state_and_matrix_share_ordered_padding_invariant_encoder(self):
        import torch
        from torch.nn import functional as F

        model = self.make_model()
        with torch.no_grad():
            single, value = model.score_candidates(self.states, self.candidates)
            matrix, values = model.score_candidate_matrix(
                self.states.repeat(2, 1), self.candidates,
            )
            padded, _ = model.score_candidates(
                self.states,
                F.pad(self.candidates, (0, 5), value=self.tokenizer.pad_id),
            )
        torch.testing.assert_close(matrix, single.unsqueeze(0).expand(2, -1))
        torch.testing.assert_close(values, value.expand(2))
        torch.testing.assert_close(padded, single)

    def test_candidate_policy_can_learn_to_separate_reordered_bindings(self):
        import torch

        model = self.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        target = torch.tensor([1])
        for _ in range(50):
            optimizer.zero_grad()
            scores, _ = model.score_candidates(self.states, self.candidates)
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
        # The old mean-pooled encoder has identical logits for these actions
        # for every set of weights, so it cannot improve beyond log(2).
        self.assertLess(float(loss.detach()), 0.1)

    def test_variable_length_candidate_training_under_cpu_bfloat16(self):
        import torch

        model = self.make_model()
        actions = self.candidates.clone()
        actions[0, -4:] = self.tokenizer.pad_id
        with torch.no_grad():
            batch = model._encode_candidate_actions(actions)
            short = model._encode_candidate_actions(actions[:1, :-4])
        torch.testing.assert_close(batch[:1], short)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores, _ = model.score_candidate_matrix(self.states.repeat(2, 1), actions)
            loss = torch.nn.functional.cross_entropy(scores, torch.arange(2))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(
            model.candidate_encoder.weight_ih_l0.grad,
        ).all()))

    def test_base_and_latent_new_checkpoint_roundtrip(self):
        import torch
        from neural_prover.model import CANDIDATE_ENCODER_VERSION

        for latent in (False, True):
            with self.subTest(latent=latent), tempfile.TemporaryDirectory() as directory:
                model = self.make_model(latent=latent)
                model.candidate_head_trained = True
                path = Path(directory) / "model.pt"
                model.save_checkpoint(path)
                restored, payload = type(model).load_checkpoint(path)
                restored.eval()
                self.assertTrue(restored.candidate_head_trained)
                self.assertEqual(payload["metadata"]["candidate_policy"]["encoder"],
                                 CANDIDATE_ENCODER_VERSION)
                with torch.no_grad():
                    before, _ = model.score_candidates(self.states, self.candidates)
                    after, _ = restored.score_candidates(self.states, self.candidates)
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_legacy_checkpoints_fall_back_to_autoregressive_scoring(self):
        import torch
        from neural_prover.search import TransformerPolicy

        for latent, with_old_head in ((False, False), (False, True), (True, True)):
            with self.subTest(latent=latent, old_head=with_old_head), \
                    tempfile.TemporaryDirectory() as directory:
                original = self.make_model(latent=latent)
                original.candidate_head_trained = True
                path = Path(directory) / "legacy.pt"
                original.save_checkpoint(path)
                payload = torch.load(path, weights_only=False)
                payload["metadata"]["candidate_policy"].pop("encoder")
                payload["model_state"] = {
                    key: value for key, value in payload["model_state"].items()
                    if not key.startswith(("candidate_encoder.", "candidate_projection."))
                    and (with_old_head or not key.startswith("candidate_"))
                }
                torch.save(payload, path)
                restored, migrated = type(original).load_checkpoint(path)
                self.assertFalse(restored.candidate_head_trained)
                self.assertTrue(migrated["metadata"]["candidate_policy_migration"][
                    "retraining_required"
                ])
                policy = TransformerPolicy(
                    restored, self.tokenizer, self.environment, "cpu",
                )
                self.assertEqual(policy.scoring_mode, "autoregressive_bootstrap")
                with patch.object(restored, "score_candidates_from_memory",
                                  side_effect=AssertionError("random head used")):
                    ranked = policy.rank(self.state, self.tactics)
                self.assertEqual(len(ranked), 2)
                self.assertTrue(all(torch.isfinite(torch.tensor(item.log_probability))
                                    for item in ranked))
                # Migration must not change the pretrained autoregressive policy.
                with torch.no_grad():
                    before, _ = original(self.states, self.candidates[:1, :-1])
                    after, _ = restored(self.states, self.candidates[:1, :-1])
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_versioned_checkpoint_rejects_missing_encoder_weights(self):
        import torch

        model = self.make_model()
        model.candidate_head_trained = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pt"
            model.save_checkpoint(path)
            payload = torch.load(path, weights_only=False)
            del payload["model_state"]["candidate_encoder.weight_ih_l0"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checkpoint state is incompatible"):
                type(model).load_checkpoint(path)

    def test_legacy_base_upgrade_does_not_reenable_untrained_candidate_head(self):
        import torch
        from neural_prover.latent_model import LatentProofTransformer

        model = self.make_model()
        model.candidate_head_trained = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model.save_checkpoint(path)
            payload = torch.load(path, weights_only=False)
            payload["metadata"]["candidate_policy"].pop("encoder")
            payload["model_state"] = {
                key: value for key, value in payload["model_state"].items()
                if not key.startswith(("candidate_encoder.", "candidate_projection."))
            }
            torch.save(payload, path)
            tokenizer = self.tokenizer.upgraded_for_lemma_actions()
            upgraded, _ = LatentProofTransformer.from_base_checkpoint(path, tokenizer)
        self.assertFalse(upgraded.candidate_head_trained)


if __name__ == "__main__":
    unittest.main()

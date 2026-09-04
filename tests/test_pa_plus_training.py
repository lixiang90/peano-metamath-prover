from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from htps_prover.data import audit_forward_dataset
from htps_prover.forward import ForwardDAGConfig, build_forward_dag_dataset
from metamath_generator.parser import parse
from neural_prover.data_contract import tokenizer_fingerprint
from neural_prover.tokenizer import MetamathTokenizer

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "formal/peano-pa-plus.mm"
CATALOG = ROOT / "formal/pa-plus-definitions.json"


class PAPlusForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = ForwardDAGConfig(
            seeds=(7, 11), steps_per_seed=8, max_proof_depth=8, max_ast_depth=64,
            max_variables=24, max_state_tokens=1024, max_action_tokens=512,
            definition_catalog=str(CATALOG), bootstrap_definitions=True,
            definition_coverage_weight=3, bounded_nat_max=2,
            ground_instances_per_predicate=1, target_guidance_weight=4,
            max_definition_only_search_per_predicate=1, verify_deepest_per_seed=1,
        )
        cls.manifest = build_forward_dag_dataset(PA, cls.root, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_pa_data_contains_bridges_instances_lemmas_and_disjoint_groups(self):
        manifest = self.manifest
        self.assertEqual(manifest["generation_kinds"]["policy"]["definition_bridge"], 136)
        self.assertEqual(manifest["generation_kinds"]["policy"]["bounded_instance"], 136)
        self.assertGreater(manifest["filtered"]["duplicate_policy"], 0)
        self.assertEqual(manifest["action_audit"]["invalid_actions"], 0)
        self.assertGreater(manifest["independent_verification"]["replayed_generated_declarations"], 0)
        self.assertEqual(manifest["environment"]["definition_bridges"], 136)
        groups = {}
        for split in ("train", "validation", "test"):
            for kind in ("policy", "lemma"):
                lines = (self.root / f"{kind}_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines)
                for line in lines:
                    record = json.loads(line)
                    self.assertEqual(groups.setdefault(record["split_group_sha256"], split), split)
                    self.assertIn("definition_support", record)
                    self.assertEqual(record["tokenizer_sha256"], manifest["tokenizer_sha256"])

    def test_actions_replay_with_another_hash_seed(self):
        env = dict(os.environ, PYTHONHASHSEED="211", PYTHONPATH=str(ROOT / "src"))
        command = (
            "from htps_prover.data import audit_forward_dataset; "
            f"r=audit_forward_dataset({str(PA)!r},{str(self.root)!r}); "
            "assert r['invalid_actions']==0, r['failures']; print(r['valid_actions'])"
        )
        result = subprocess.run([sys.executable, "-c", command], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout), self.manifest["action_audit"]["valid_actions"])

    def test_invalid_generation_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "catalog"):
            replace(self.config, definition_catalog=None).validate()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            replace(self.config, test_fraction=-0.1).validate()

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_sft_consumes_pa_and_lemma_records_and_rejects_wrong_ids(self):
        from htps_prover.training import FreshModelConfig, SupervisedTrainConfig, initialize_fresh_latent_model, train_supervised_latent
        from neural_prover.latent_model import LatentProofTransformer

        initial, trained = self.root / "initial.pt", self.root / "trained.pt"
        tokenizer_path = self.root / "tokenizer.json"
        initialize_fresh_latent_model(tokenizer_path, initial, FreshModelConfig(
            d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, max_state_tokens=1024, max_action_tokens=512,
            max_thought_steps=2,
        ))
        original_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
        result = train_supervised_latent(initial, tokenizer_path, self.root / "policy_train.jsonl",
            self.root / "lemma_train.jsonl", trained,
            SupervisedTrainConfig(epochs=1, batch_size=4, max_examples=24, device="cpu", candidate_loss_weight=0.25))
        self.assertGreater(result["generation_kinds"]["bounded_instance"], 0)
        self.assertGreater(result["lemma_examples"], 0)
        self.assertTrue(result["balanced_sampling"])
        self.assertTrue(all(torch.isfinite(torch.tensor(result["history"][0][key])) for key in ("loss", "candidate", "ponder")))
        model, payload = LatentProofTransformer.load_checkpoint(trained)
        before, _ = LatentProofTransformer.load_checkpoint(initial)
        self.assertFalse(torch.equal(before.token_embedding.weight, model.token_embedding.weight))
        self.assertTrue(model.candidate_head_trained)
        self.assertEqual(payload["metadata"]["tokenizer_sha256"], result["tokenizer_sha256"])
        self.assertEqual(hashlib.sha256(initial.read_bytes()).hexdigest(), original_hash)
        tokenizer = MetamathTokenizer.load(tokenizer_path)
        swapped = list(tokenizer.tokens)
        swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
        wrong = MetamathTokenizer(swapped, tokenizer.config, preserve_token_order=True, pa_plus_context=tokenizer.pa_plus_context)
        wrong.save(self.root / "wrong-tokenizer.json")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            train_supervised_latent(initial, self.root / "wrong-tokenizer.json", self.root / "policy_train.jsonl", None,
                self.root / "wrong.pt", SupervisedTrainConfig(epochs=1))
        record = json.loads((self.root / "policy_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        record["action_ids"][0] = tokenizer.pad_id
        with patch("htps_prover.training._load_records", return_value=[record]):
            with self.assertRaisesRegex(ValueError, "token/ID"):
                train_supervised_latent(initial, tokenizer_path, "unused", None, self.root / "corrupt.pt")

        from htps_prover.pipeline import evaluate_htps
        from htps_prover.hypergraph import HTPSConfig
        test_records = [json.loads(line) for line in (self.root / "policy_test.jsonl").read_text(encoding="utf-8").splitlines()]
        bounded = min((r for r in test_records if r["generation_kind"] == "bounded_instance"), key=lambda r: len(r["state_tokens"]))
        target_path = self.root / "bounded-evaluation.jsonl"
        target_path.write_text(json.dumps(bounded) + "\n", encoding="utf-8")
        verifier = os.environ.get("METAMATH_EXECUTABLE")
        local = ROOT / "outputs/tools/metamath/metamath.exe"
        if not verifier and local.is_file():
            verifier = str(local)
        evaluation = evaluate_htps(trained, tokenizer_path, target_path, PA, self.root / "bounded-evaluation.json",
            limit=1, device_name="cpu", search_config=HTPSConfig(simulations=8, expansion_budget=16, branching=24),
            external_verifier=verifier, require_external_verification=bool(verifier))
        self.assertEqual(evaluation["certified"], 1, evaluation["episodes"])
        self.assertEqual(evaluation["episodes"][0]["environment"]["definition_bridges"], 136)


@unittest.skipIf(torch is None, "PyTorch required")
class PAPlusFineTuneTests(unittest.TestCase):
    def test_upgrade_generate_with_fixed_ids_and_fine_tune_base_checkpoint(self):
        from neural_prover.cli import main
        from neural_prover.data import CorpusBuildConfig, build_corpus
        from neural_prover.model import ProofTransformer, ProofTransformerConfig
        from neural_prover.train import TrainingConfig, train_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = MetamathTokenizer.from_database(parse(ROOT / "formal/peano.mm"))
            tokenizer.save(root / "base-tokenizer.json")
            model = ProofTransformer(ProofTransformerConfig(vocab_size=len(tokenizer),
                pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                dim_feedforward=32, max_state_tokens=1024, max_action_tokens=512))
            model.save_checkpoint(root / "base.pt", metadata={"tokenizer_sha256": tokenizer_fingerprint(tokenizer)})
            base_hash = hashlib.sha256((root / "base.pt").read_bytes()).hexdigest()
            main(["upgrade-pa-plus", str(root / "base.pt"), str(root / "base-tokenizer.json"),
                str(PA), str(CATALOG), str(root / "upgraded.pt"), str(root / "upgraded-tokenizer.json")])
            build_corpus(PA, root / "corpus", CorpusBuildConfig(seeds=(7,), steps_per_seed=1,
                max_proof_depth=8, max_ast_depth=64, max_variables=24, max_state_tokens=1024,
                max_action_tokens=512, definition_catalog=str(CATALOG), bootstrap_definitions=True,
                bounded_nat_max=2, ground_instances_per_predicate=1, base_tokenizer=str(root / "upgraded-tokenizer.json")))
            current = MetamathTokenizer.load(root / "corpus/tokenizer.json")
            self.assertEqual(current.tokens[:len(tokenizer)], tokenizer.tokens)
            self.assertEqual(tokenizer_fingerprint(current), tokenizer_fingerprint(MetamathTokenizer.load(root / "upgraded-tokenizer.json")))
            # Keep the training regression cheap; generation above covers the full catalog.
            for split in ("train", "validation"):
                path = root / f"corpus/{split}.jsonl"
                path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:4]) + "\n", encoding="utf-8")
            config = TrainingConfig(epochs=1, batch_size=2, device="cpu", learning_rate=1e-5,
                checkpoint=str(root / "upgraded.pt"), checkpoint_tokenizer=str(root / "upgraded-tokenizer.json"))
            result = train_model(root / "corpus", root / "fine-tuned", config)
            restored, payload = ProofTransformer.load_checkpoint(root / "fine-tuned/final.pt")
            upgraded, _ = ProofTransformer.load_checkpoint(root / "upgraded.pt")
            self.assertFalse(torch.equal(upgraded.token_embedding.weight, restored.token_embedding.weight))
            self.assertEqual(restored.config.d_model, 16)  # Not TrainingConfig's fresh-model default.
            self.assertTrue(result["initialization"]["optimizer_restarted"])
            self.assertEqual(payload["metadata"]["tokenizer_sha256"], tokenizer_fingerprint(current))
            self.assertEqual(hashlib.sha256((root / "base.pt").read_bytes()).hexdigest(), base_hash)
            with self.assertRaisesRegex(ValueError, "differs from corpus"):
                train_model(root / "corpus", root / "mismatched", replace(config, checkpoint_tokenizer=str(root / "base-tokenizer.json")))


if __name__ == "__main__":
    unittest.main()

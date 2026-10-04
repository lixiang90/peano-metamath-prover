from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from metamath_generator.definitions import load_definition_catalog
from metamath_generator.model import Theorem
from metamath_generator.parser import parse
from neural_prover.audit import audit_corpus_actions, audit_scale_corpus
from neural_prover.benchmark import BenchmarkCase
from neural_prover.certificate import compile_certificate
from neural_prover.data import CorpusBuildConfig, build_corpus, load_examples
from neural_prover.environment import BackwardEnvironment, InvalidTactic, ProofState, Tactic, parse_tactic_tokens
from neural_prover.external import ExternalVerificationResult, discover_metamath_executable
from neural_prover.search import SearchResult
from neural_prover.tokenizer import MetamathTokenizer
from neural_prover.verification import verification_record

ROOT = Path(__file__).resolve().parents[1]
PA_PLUS = ROOT / "formal/peano-pa-plus.mm"
CATALOG = ROOT / "formal/pa-plus-definitions.json"
EXTERNAL_VERIFIER = os.environ.get("METAMATH_EXECUTABLE") or (
    str(ROOT / "outputs/tools/metamath/metamath.exe")
    if (ROOT / "outputs/tools/metamath/metamath.exe").is_file() else None
)

try:
    import torch
except ImportError:
    torch = None


class CertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = parse(PA_PLUS)
        catalog = load_definition_catalog(CATALOG)
        cls.tokenizer = MetamathTokenizer.from_database(
            cls.database,
            pa_plus_context=MetamathTokenizer.pa_plus_context_from_database(
                cls.database, catalog.definition_names, catalog.statement_names,
                bounded_nat_max=2,
            ),
        )

    def environment(self, **kwargs):
        env = BackwardEnvironment(self.database, **kwargs)
        env.configure_from_tokenizer(self.tokenizer)
        return env

    def certificate(self, bridge_name="gen_df_positive_fold"):
        env = self.environment()
        bridge = env.assertions[bridge_name]
        target = Theorem(
            "gate_target", [], bridge.conclusion,
            variable_types=dict(bridge.variable_types),
            d_constraints=set(bridge.d_constraints),
        )
        tactic = Tactic.create(bridge_name)
        transition = env.apply(ProofState.from_theorem(target), tactic)
        self.assertTrue(transition.after.solved)
        result = SearchResult(True, 1, 2, (tactic,), (transition,), transition.after)
        return compile_certificate(target, result, self.database)

    def test_guidance_layers_and_reconfiguration_are_independent(self):
        no_hints = self.tokenizer.with_target_hints(False)
        self.assertEqual(no_hints.tokens, self.tokenizer.tokens)
        self.assertEqual(no_hints.pa_plus_context.max_target_hints, 0)
        statement = self.database.syntax_statements["factorial-zero-statement"]
        from metamath_generator.model import Node
        target = Theorem("factorial_goal", [], Node("|-", statement.conclusion.args))
        self.assertIn("factorial-zero-statement", self.tokenizer.state_tokens(target))
        self.assertNotIn("factorial-zero-statement", no_hints.state_tokens(target))
        env = self.environment()
        original = env.configuration_record()
        env.configure_from_tokenizer(no_hints)
        self.assertEqual(original, env.configuration_record())
        env.enumerate_tactics(ProofState.from_theorem(target), max_tactics=8)
        env.configure_from_tokenizer(no_hints, target_guidance=False)
        self.assertFalse(env._tactic_cache)
        off = env.configuration_record()
        self.assertEqual(off["definition_bridges"], 136)
        self.assertEqual(off["bounded_nat_max"], 2)
        self.assertEqual(off["target_formulas"], 0)
        self.assertNotEqual(original["fingerprint"], off["fingerprint"])
        env.configure_from_tokenizer(self.tokenizer)
        self.assertEqual(original, env.configuration_record())

    def test_excluded_rules_cannot_return_through_bridges(self):
        env = self.environment(excluded_assertions=("df-positive",))
        self.assertNotIn("gen_df_positive_fold", env.assertions)
        self.assertNotIn("gen_df_positive_unfold", env.assertions)
        self.assertEqual(env.configuration_record()["definition_bridges"], 134)
        env = self.environment(excluded_assertions=("gen_df_positive_fold",))
        self.assertNotIn("gen_df_positive_fold", env.assertions)
        env = self.environment(excluded_assertions=("ax-mp",))
        self.assertEqual(env.configuration_record()["definition_bridges"], 0)

    def test_corpus_and_shard_audit_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            corpus = directory / "corpus"
            build_corpus(PA_PLUS, corpus, CorpusBuildConfig(
                generation_mode="random",
                seeds=(7,), steps_per_seed=0,
                definition_catalog=str(CATALOG), bootstrap_definitions=True,
                bounded_nat_max=2, ground_instances_per_predicate=1,
                max_ast_depth=64, max_variables=24,
                max_state_tokens=1024, max_action_tokens=512,
            ))
            report = audit_corpus_actions(PA_PLUS, corpus)
            self.assertEqual(report["invalid_actions"], 0, report["failures"][:3])
            self.assertEqual(report["valid_actions"], 399)
            subprocess.run(
                [sys.executable, "-m", "neural_prover", "audit-corpus",
                 str(PA_PLUS), str(corpus)],
                check=True, capture_output=True, text=True, timeout=60,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "193"},
                cwd=ROOT,
            )
            samples = [
                item for split in ("train", "validation", "test")
                for item in load_examples(corpus / f"{split}.jsonl")
                if item.generation_kind == "bounded_instance"
            ]
            shard_dir = directory / "sharded"
            shard_dir.mkdir()
            tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
            tokenizer.save(shard_dir / "tokenizer.json")
            with gzip.open(shard_dir / "sample.jsonl.gz", "wt", encoding="utf-8") as stream:
                for item in samples:
                    stream.write(json.dumps({
                        "id": item.example_id, "state": item.state_ids,
                        "action": item.action_ids, "proof_depth": item.proof_depth,
                    }) + "\n")
            (shard_dir / "manifest.json").write_text(json.dumps({
                "counts": {"train": len(samples), "validation": 0, "test": 0},
                "shards": {
                    "train": [{"path": "sample.jsonl.gz"}], "validation": [], "test": [],
                },
            }), encoding="utf-8")
            audited = audit_scale_corpus(PA_PLUS, shard_dir, sample_size=136)
            self.assertEqual(audited["valid_actions"], 136, audited["failures"][:3])
            payload = json.loads((corpus / "tokenizer.json").read_text(encoding="utf-8"))
            del payload["pa_plus_context"]["bridge_variable_order"]
            (corpus / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")
            legacy = MetamathTokenizer.load(corpus / "legacy.json")
            theorem = legacy.theorem_from_state_tokens(samples[0].state_tokens, self.database)
            with self.assertRaisesRegex(InvalidTactic, "regenerate corpus"):
                parse_tactic_tokens(samples[0].action_tokens, theorem, legacy,
                                    self.database, environment=self.environment())

    def test_external_gate_is_fail_closed_for_all_nonpassing_statuses(self):
        cert = self.certificate()
        for status in ("not_configured", "failed", "error", "passed"):
            with self.subTest(status=status), patch(
                "neural_prover.verification.verify_certificate_external",
                return_value=ExternalVerificationResult(status, None, None, None, 0, ""),
            ):
                passed, record = verification_record(
                    cert, self.database, PA_PLUS, require_external=True
                )
                self.assertEqual(passed, status == "passed")
                self.assertTrue(record["internal_verified"])
        with patch("neural_prover.verification.verify_certificate_external", side_effect=OSError("timeout")):
            passed, record = verification_record(cert, self.database, PA_PLUS, require_external=True)
            self.assertFalse(passed)
            self.assertEqual(record["external_verification"]["status"], "error")

    def test_explicit_missing_verifier_cannot_fall_back(self):
        with patch.dict(os.environ, {"METAMATH_EXECUTABLE": sys.executable}):
            self.assertIsNone(discover_metamath_executable("missing-verifier-193.exe"))

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_htps_cli_propagates_gate_and_guidance_options(self):
        from htps_prover.cli import main
        for command, function in (
            ("evaluate", "evaluate_htps"),
            ("collect", "collect_htps_replay"),
            ("closed-loop", "run_closed_loop"),
        ):
            with self.subTest(command=command), patch(
                f"htps_prover.cli.{function}", return_value={}
            ) as called:
                main([
                    command, "model", "tokenizer", "policy", "database", "output",
                    "--require-external-verification", "--external-verifier", "verifier",
                    "--external-timeout-seconds", "7", "--no-model-target-hints",
                    "--no-inference-target-guidance",
                ])
                if command == "evaluate":
                    values = called.call_args.kwargs
                else:
                    config = called.call_args.args[-1]
                    if command == "closed-loop":
                        config = config.collection
                    from dataclasses import asdict
                    values = asdict(config)
                self.assertTrue(values["require_external_verification"])
                self.assertEqual(values["external_timeout_seconds"], 7)
                self.assertFalse(values["model_target_hints"])
                self.assertFalse(values["inference_target_guidance"])

    @unittest.skipUnless(EXTERNAL_VERIFIER, "set METAMATH_EXECUTABLE for official Metamath tests")
    def test_real_external_bridge_certificates(self):
        for bridge in ("gen_df_positive_fold", "gen_df_gcdrel_unfold", "gen_df_qadd_fold"):
            with self.subTest(bridge=bridge):
                cert = self.certificate(bridge)
                passed, record = verification_record(
                    cert, self.database, PA_PLUS, require_external=True,
                    external_verifier=EXTERNAL_VERIFIER,
                )
                self.assertTrue(passed, record)
                self.assertNotIn(bridge, cert.proof.source_labels)

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_four_policy_evaluation_uses_identical_pa_plus_environment(self):
        from neural_prover.evaluate import MCTSEvaluationConfig, evaluate_mcts_checkpoint
        from neural_prover.model import ProofTransformer, ProofTransformerConfig

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.tokenizer.save(root / "tokenizer.json")
            model = ProofTransformer(ProofTransformerConfig(
                len(self.tokenizer), self.tokenizer.pad_id, self.tokenizer.bos_id,
                self.tokenizer.eos_id, d_model=16, nhead=2, num_encoder_layers=1,
                num_decoder_layers=1, dim_feedforward=32,
                max_state_tokens=1024, max_action_tokens=512,
            ))
            model.save_checkpoint(root / "model.pt")
            source = self.database.logical_assertions["ax-1"]
            case = BenchmarkCase(
                "fairness", "fairness", "synthetic", "easy", "provable", (),
                source.conclusion.to_prefix(), tuple(source.d_constraints),
                tuple(source.variable_types.items()),
            )
            (root / "benchmark.json").write_text(json.dumps({
                "format": "peano-proof-benchmark-v2", "cases": [case.to_record()],
            }), encoding="utf-8")
            for guidance in (True, False):
                report = evaluate_mcts_checkpoint(
                    root / "model.pt", root, root / "benchmark.json", PA_PLUS,
                    root / "result.json", MCTSEvaluationConfig(
                        device="cpu", simulations=1, branching=4, max_search_depth=2,
                        seeds=(7,), cases_per_difficulty=1,
                        model_target_hints=False, inference_target_guidance=guidance,
                    ),
                )
                environments = [item["environment"] for item in report["cases"]]
                self.assertEqual(len(environments), 4)
                self.assertTrue(all(item == environments[0] for item in environments))
                self.assertEqual(environments[0]["definition_bridges"], 136)
                self.assertEqual(environments[0]["inference_target_guidance"], guidance)

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_htps_external_rejection_is_not_successful_replay(self):
        from htps_prover.hypergraph import HTPSConfig
        from htps_prover.pipeline import evaluate_htps
        from htps_prover.training import ReplayCollectionConfig, collect_htps_replay
        from neural_prover.latent_model import LatentProofTransformer, LatentReasoningConfig
        from neural_prover.model import ProofTransformerConfig

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.tokenizer.save(root / "tokenizer.json")
            model = LatentProofTransformer(ProofTransformerConfig(
                len(self.tokenizer), self.tokenizer.pad_id, self.tokenizer.bos_id,
                self.tokenizer.eos_id, d_model=16, nhead=2, num_encoder_layers=1,
                num_decoder_layers=1, dim_feedforward=32,
                max_state_tokens=1024, max_action_tokens=512,
            ), LatentReasoningConfig(max_thought_steps=1))
            model.save_checkpoint(root / "model.pt")
            source = self.database.logical_assertions["ax-1"]
            (root / "policy.jsonl").write_text(json.dumps({
                "theorem_name": "gate_target", "state_tokens": self.tokenizer.state_tokens(source),
            }) + "\n", encoding="utf-8")
            search = HTPSConfig(simulations=2, expansion_budget=2, branching=4)
            failed = ExternalVerificationResult("error", None, None, None, 0, "timeout")
            with patch("neural_prover.verification.verify_certificate_external", return_value=failed):
                result = evaluate_htps(
                    root / "model.pt", root / "tokenizer.json", root / "policy.jsonl",
                    PA_PLUS, root / "eval.json", search_config=search, device_name="cpu",
                    require_external_verification=True,
                )
                self.assertEqual(result["certified"], 0)
                self.assertTrue(result["episodes"][0]["verification"]["internal_verified"])
                replay = collect_htps_replay(
                    root / "model.pt", root / "tokenizer.json", root / "policy.jsonl",
                    PA_PLUS, root / "replay.jsonl", ReplayCollectionConfig(
                        examples=1, device="cpu", htps=search, require_external_verification=True,
                    ),
                )
                self.assertEqual(replay["rejected_certificates"], 1)
                self.assertEqual(replay["replay_examples"], 0)
            if EXTERNAL_VERIFIER:
                result = evaluate_htps(
                    root / "model.pt", root / "tokenizer.json", root / "policy.jsonl",
                    PA_PLUS, root / "eval-external.json", search_config=search,
                    device_name="cpu", require_external_verification=True,
                    external_verifier=EXTERNAL_VERIFIER,
                )
                self.assertEqual(result["certified"], 1, result["episodes"])
                self.assertEqual(
                    result["episodes"][0]["verification"]["certification_level"],
                    "internal+external",
                )


if __name__ == "__main__":
    unittest.main()

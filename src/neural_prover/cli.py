from __future__ import annotations

import argparse
import json
from pathlib import Path

from metamath_generator.parser import parse

from .benchmark import (
    BenchmarkBuildConfig,
    build_benchmarks,
    load_benchmarks,
)
from .certificate import (
    compile_certificate,
    export_certificate,
    verify_certificate,
)
from .data import CorpusBuildConfig, build_corpus
from .environment import BackwardEnvironment, ProofState
from .hybrid import HybridActionGenerator
from .tokenizer import MetamathTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transformer-guided neural-symbolic Metamath prover"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser(
        "build-corpus",
        help="generate a leak-controlled state/action corpus",
    )
    corpus.add_argument("database")
    corpus.add_argument("output")
    corpus.add_argument("--seeds", default="7,11,19,23")
    corpus.add_argument("--steps-per-seed", type=int, default=5_000)
    corpus.add_argument("--max-proof-depth", type=int, default=7)
    corpus.add_argument("--max-ast-depth", type=int, default=32)
    corpus.add_argument("--max-hypotheses", type=int, default=8)
    corpus.add_argument("--max-variables", type=int, default=16)
    corpus.add_argument(
        "--full-discharge-probability", type=float, default=0.72
    )
    corpus.add_argument(
        "--closed-parent-probability", type=float, default=0.78
    )
    corpus.add_argument(
        "--max-proof-states-per-conclusion", type=int, default=3
    )
    corpus.add_argument("--depth-parent-bias", type=float, default=0.0)
    corpus.add_argument("--max-state-tokens", type=int, default=256)
    corpus.add_argument("--max-action-tokens", type=int, default=192)
    corpus.add_argument("--definition-catalog")
    corpus.add_argument("--base-tokenizer")
    corpus.add_argument("--bootstrap-definitions", action="store_true")
    corpus.add_argument(
        "--definition-coverage-weight", type=float, default=0.0
    )
    corpus.add_argument("--bounded-nat-max", type=int, default=-1)
    corpus.add_argument(
        "--ground-instances-per-predicate", type=int, default=0
    )
    corpus.add_argument(
        "--target-guidance-weight", type=float, default=0.0
    )
    corpus.add_argument(
        "--max-definition-only-search-per-predicate",
        type=int,
        default=-1,
    )
    corpus.add_argument("--max-target-hints", type=int, default=3)

    benchmark = commands.add_parser(
        "build-benchmark",
        help="build synthetic, foundational, and famous evaluations",
    )
    benchmark.add_argument("database")
    benchmark.add_argument("output")
    benchmark.add_argument("--seeds", default="101,103,107")
    benchmark.add_argument("--steps-per-seed", type=int, default=5_000)
    benchmark.add_argument("--max-proof-depth", type=int, default=7)
    benchmark.add_argument("--max-ast-depth", type=int, default=64)
    benchmark.add_argument("--max-hypotheses", type=int, default=12)
    benchmark.add_argument("--max-variables", type=int, default=24)
    benchmark.add_argument(
        "--full-discharge-probability", type=float, default=0.8
    )
    benchmark.add_argument(
        "--closed-parent-probability", type=float, default=0.45
    )
    benchmark.add_argument("--depth-parent-bias", type=float, default=0.0)
    benchmark.add_argument(
        "--synthetic-per-difficulty",
        type=int,
        default=20,
    )
    benchmark.add_argument(
        "--training-corpus",
        action="append",
        default=[],
        help="training corpus directory to exclude from research targets",
    )
    benchmark.add_argument(
        "--reference-output",
        help="private sidecar for reference rule/depth metadata",
    )

    train = commands.add_parser(
        "train",
        help="supervised policy/value Transformer training",
    )
    train.add_argument("corpus")
    train.add_argument("output")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--checkpoint", help="fine-tune a base checkpoint with a new optimizer")
    train.add_argument("--checkpoint-tokenizer", help="explicit tokenizer belonging to --checkpoint")
    train.add_argument(
        "--candidate-loss-weight", type=float, default=0.25
    )
    train.add_argument(
        "--no-pa-plus-balanced-sampling", action="store_true"
    )

    upgrade_pa = commands.add_parser(
        "upgrade-pa-plus",
        help="append PA+ vocabulary/context to an existing checkpoint",
    )
    upgrade_pa.add_argument("checkpoint")
    upgrade_pa.add_argument("tokenizer")
    upgrade_pa.add_argument("database")
    upgrade_pa.add_argument("definition_catalog")
    upgrade_pa.add_argument("output_checkpoint")
    upgrade_pa.add_argument("output_tokenizer")
    upgrade_pa.add_argument("--bounded-nat-max", type=int, default=2)
    upgrade_pa.add_argument("--max-target-hints", type=int, default=3)

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate policy and certified proof search",
    )
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("corpus")
    evaluate.add_argument("benchmark")
    evaluate.add_argument("database")
    evaluate.add_argument("output")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--easy-sims", type=int, default=100)
    evaluate.add_argument("--medium-sims", type=int, default=300)
    evaluate.add_argument("--hard-sims", type=int, default=1_000)
    evaluate.add_argument("--frontier-sims", type=int, default=100)
    evaluate.add_argument("--external-verifier")
    evaluate.add_argument(
        "--require-external-verification", action="store_true"
    )

    evaluate_mcts = commands.add_parser(
        "evaluate-mcts",
        help="evaluate hybrid PUCT/MCTS with certificate replay",
    )
    evaluate_mcts.add_argument("checkpoint")
    evaluate_mcts.add_argument("corpus")
    evaluate_mcts.add_argument("benchmark")
    evaluate_mcts.add_argument("database")
    evaluate_mcts.add_argument("output")
    evaluate_mcts.add_argument("--device", default="auto")
    evaluate_mcts.add_argument("--simulations", type=int, default=60)
    evaluate_mcts.add_argument("--max-depth", type=int, default=16)
    evaluate_mcts.add_argument("--branching", type=int, default=32)
    evaluate_mcts.add_argument(
        "--cases-per-difficulty", type=int, default=5
    )
    evaluate_mcts.add_argument("--seeds", default="17,19,23")
    evaluate_mcts.add_argument(
        "--policies",
        default="uniform,heuristic,neural,hybrid",
    )
    evaluate_mcts.add_argument("--external-verifier")
    evaluate_mcts.add_argument(
        "--require-external-verification", action="store_true"
    )
    evaluate_mcts.add_argument(
        "--include-frontier", action="store_true"
    )
    evaluate_mcts.add_argument(
        "--origins",
        default="",
        help="comma-separated: synthetic,curated,foundational,famous",
    )

    prove = commands.add_parser(
        "prove",
        help="prove one benchmark case with hybrid PUCT/MCTS",
    )
    prove.add_argument("checkpoint")
    prove.add_argument("corpus")
    prove.add_argument("benchmark")
    prove.add_argument("case_id")
    prove.add_argument("database")
    prove.add_argument("--certificate")
    prove.add_argument("--device", default="auto")
    prove.add_argument("--simulations", type=int, default=800)
    prove.add_argument("--max-depth", type=int, default=24)
    prove.add_argument("--branching", type=int, default=48)
    prove.add_argument("--seed", type=int, default=7)
    prove.add_argument(
        "--policy",
        choices=("uniform", "heuristic", "neural", "hybrid"),
        default="hybrid",
    )
    prove.add_argument("--external-verifier")
    prove.add_argument(
        "--require-external-verification", action="store_true"
    )

    init_latent = commands.add_parser(
        "init-latent",
        help="upgrade a checkpoint with lemma tokens and latent thought",
    )
    init_latent.add_argument("base_checkpoint")
    init_latent.add_argument("base_tokenizer")
    init_latent.add_argument("output_checkpoint")
    init_latent.add_argument("output_tokenizer")
    init_latent.add_argument("--max-thought-steps", type=int, default=6)
    init_latent.add_argument("--min-thought-steps", type=int, default=1)
    init_latent.add_argument("--halt-threshold", type=float, default=0.85)
    init_latent.add_argument("--ponder-cost", type=float, default=1e-3)

    prove_decomposed = commands.add_parser(
        "prove-decomposed",
        help="prove one case with latent thought and intermediate lemmas",
    )
    prove_decomposed.add_argument("checkpoint")
    prove_decomposed.add_argument("tokenizer")
    prove_decomposed.add_argument("benchmark")
    prove_decomposed.add_argument("case_id")
    prove_decomposed.add_argument("database")
    prove_decomposed.add_argument("--certificate")
    prove_decomposed.add_argument("--device", default="auto")
    prove_decomposed.add_argument("--simulations", type=int, default=100)
    prove_decomposed.add_argument("--max-depth", type=int, default=20)
    prove_decomposed.add_argument("--branching", type=int, default=24)
    prove_decomposed.add_argument("--seed", type=int, default=7)
    prove_decomposed.add_argument("--external-verifier")
    prove_decomposed.add_argument(
        "--require-external-verification", action="store_true"
    )

    reinforce = commands.add_parser(
        "reinforce",
        help="update a checkpoint from a saved MCTS replay buffer",
    )
    reinforce.add_argument("checkpoint")
    reinforce.add_argument("replay")
    reinforce.add_argument("output")
    reinforce.add_argument("--epochs", type=int, default=3)
    reinforce.add_argument("--learning-rate", type=float, default=1e-5)
    reinforce.add_argument("--device", default="auto")

    collect = commands.add_parser(
        "collect-replay",
        help="run hybrid PUCT/MCTS on training states",
    )
    collect.add_argument("checkpoint")
    collect.add_argument("corpus")
    collect.add_argument("database")
    collect.add_argument("output")
    collect.add_argument("--examples", type=int, default=32)
    collect.add_argument("--simulations", type=int, default=100)
    collect.add_argument("--max-depth", type=int, default=16)
    collect.add_argument("--branching", type=int, default=32)
    collect.add_argument("--device", default="auto")

    audit = commands.add_parser(
        "audit-corpus",
        help="replay every corpus action through the typed kernel",
    )
    audit.add_argument("database")
    audit.add_argument("corpus")
    audit.add_argument("--output")

    scale_audit = commands.add_parser(
        "audit-scale-corpus",
        help="replay a deterministic scale-corpus sample through the kernel",
    )
    scale_audit.add_argument("database")
    scale_audit.add_argument("corpus")
    scale_audit.add_argument("--output")
    scale_audit.add_argument("--sample-size", type=int, default=10_000)
    scale_audit.add_argument("--seed", type=int, default=20260729)

    scale_corpus = commands.add_parser(
        "build-scale-corpus",
        help="build/resume compact sharded million-record data",
    )
    scale_corpus.add_argument("database")
    scale_corpus.add_argument("base_corpus")
    scale_corpus.add_argument("output")
    scale_corpus.add_argument("--train-examples", type=int, default=990_000)
    scale_corpus.add_argument(
        "--validation-examples", type=int, default=5_000
    )
    scale_corpus.add_argument("--test-examples", type=int, default=5_000)
    scale_corpus.add_argument("--shard-size", type=int, default=20_000)
    scale_corpus.add_argument("--max-state-tokens", type=int, default=2304)
    scale_corpus.add_argument("--max-action-tokens", type=int, default=2304)
    scale_corpus.add_argument("--seed", type=int, default=20260729)
    scale_corpus.add_argument("--workers", type=int, default=8)
    scale_corpus.add_argument(
        "--kernel-validation-interval", type=int, default=1000
    )
    scale_corpus.add_argument("--max-new-records", type=int, default=0)
    scale_corpus.add_argument(
        "--no-depth-balancing",
        action="store_true",
        help="sample base templates by frequency instead of proof depth",
    )

    scale_train = commands.add_parser(
        "train-scale",
        help="train/resume the ~100M, 2304-context scale model",
    )
    scale_train.add_argument("corpus")
    scale_train.add_argument("output")
    scale_train.add_argument("--resume")
    scale_train.add_argument("--max-steps", type=int, default=100)
    scale_train.add_argument(
        "--run-steps",
        type=int,
        default=0,
        help=(
            "stop after this many steps in the current invocation while "
            "preserving the max-steps schedule"
        ),
    )
    scale_train.add_argument("--micro-batch-size", type=int, default=1)
    scale_train.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    scale_train.add_argument("--learning-rate", type=float, default=1e-4)
    scale_train.add_argument("--candidate-loss-weight", type=float, default=0.0)
    scale_train.add_argument("--warmup-steps", type=int, default=10)
    scale_train.add_argument("--device", default="auto")
    scale_train.add_argument("--d-model", type=int, default=768)
    scale_train.add_argument("--nhead", type=int, default=12)
    scale_train.add_argument("--encoder-layers", type=int, default=6)
    scale_train.add_argument("--decoder-layers", type=int, default=6)
    scale_train.add_argument("--dim-feedforward", type=int, default=3072)
    scale_train.add_argument(
        "--initial-context-tokens", type=int, default=512
    )
    scale_train.add_argument(
        "--context-warmup-steps", type=int, default=50
    )
    scale_train.add_argument("--checkpoint-every", type=int, default=25)
    scale_train.add_argument("--validation-batches", type=int, default=4)
    scale_train.add_argument(
        "--no-long-context-step", action="store_true"
    )

    scale_evaluate = commands.add_parser(
        "evaluate-scale",
        help="teacher-force a deterministic held-out scale sample",
    )
    scale_evaluate.add_argument("checkpoint")
    scale_evaluate.add_argument("corpus")
    scale_evaluate.add_argument("output")
    scale_evaluate.add_argument("--split", default="test")
    scale_evaluate.add_argument("--examples", type=int, default=256)
    scale_evaluate.add_argument("--device", default="auto")
    scale_evaluate.add_argument("--seed", type=int, default=20260801)
    for command in (evaluate, evaluate_mcts, prove):
        command.add_argument("--no-model-target-hints", action="store_true")
        command.add_argument("--no-inference-target-guidance", action="store_true")
        command.add_argument("--external-timeout-seconds", type=float, default=60.0)
    return parser


def _seeds(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-corpus":
        manifest = build_corpus(
            args.database,
            args.output,
            CorpusBuildConfig(
                seeds=_seeds(args.seeds),
                steps_per_seed=args.steps_per_seed,
                max_proof_depth=args.max_proof_depth,
                max_ast_depth=args.max_ast_depth,
                max_hypotheses=args.max_hypotheses,
                max_variables=args.max_variables,
                full_discharge_probability=
                    args.full_discharge_probability,
                closed_parent_probability=
                    args.closed_parent_probability,
                max_proof_states_per_conclusion=
                    args.max_proof_states_per_conclusion,
                depth_parent_bias=args.depth_parent_bias,
                max_state_tokens=args.max_state_tokens,
                max_action_tokens=args.max_action_tokens,
                definition_catalog=args.definition_catalog,
                bootstrap_definitions=args.bootstrap_definitions,
                definition_coverage_weight=
                    args.definition_coverage_weight,
                bounded_nat_max=args.bounded_nat_max,
                ground_instances_per_predicate=
                    args.ground_instances_per_predicate,
                target_guidance_weight=args.target_guidance_weight,
                max_definition_only_search_per_predicate=(
                    args.max_definition_only_search_per_predicate
                ),
                max_target_hints=args.max_target_hints,
                base_tokenizer=args.base_tokenizer,
            ),
        )
        print(json.dumps(manifest["counts"], ensure_ascii=False))
        return 0
    if args.command == "build-benchmark":
        manifest = build_benchmarks(
            args.database,
            args.output,
            BenchmarkBuildConfig(
                seeds=_seeds(args.seeds),
                steps_per_seed=args.steps_per_seed,
                max_proof_depth=args.max_proof_depth,
                max_ast_depth=args.max_ast_depth,
                max_hypotheses=args.max_hypotheses,
                max_variables=args.max_variables,
                full_discharge_probability=
                    args.full_discharge_probability,
                closed_parent_probability=
                    args.closed_parent_probability,
                depth_parent_bias=args.depth_parent_bias,
                synthetic_per_difficulty=
                    args.synthetic_per_difficulty,
                training_corpora=tuple(args.training_corpus),
                reference_output=args.reference_output,
            ),
        )
        print(json.dumps(manifest["counts"], ensure_ascii=False))
        return 0
    if args.command == "train":
        from .train import TrainingConfig, train_model

        summary = train_model(
            args.corpus,
            args.output,
            TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                d_model=args.d_model,
                encoder_layers=args.layers,
                decoder_layers=args.layers,
                candidate_loss_weight=args.candidate_loss_weight,
                pa_plus_balanced_sampling=
                    not args.no_pa_plus_balanced_sampling,
                checkpoint=args.checkpoint,
                checkpoint_tokenizer=args.checkpoint_tokenizer,
            ),
        )
        print(json.dumps({
            "device": summary["device"],
            "best_validation_loss": summary["best_validation_loss"],
            "best_checkpoint": summary["best_checkpoint"],
        }, ensure_ascii=False))
        return 0
    if args.command == "upgrade-pa-plus":
        if Path(args.checkpoint).resolve() == Path(args.output_checkpoint).resolve() or Path(args.tokenizer).resolve() == Path(args.output_tokenizer).resolve():
            raise ValueError("choose new output paths for the upgraded checkpoint and tokenizer")
        import torch

        from metamath_generator.definitions import load_definition_catalog

        from .latent_model import LatentProofTransformer
        from .model import ProofTransformer

        database = parse(args.database)
        catalog = load_definition_catalog(args.definition_catalog)
        tokenizer = MetamathTokenizer.load(args.tokenizer)
        upgraded = tokenizer.upgraded_for_pa_plus(
            database,
            catalog.definition_names,
            catalog.statement_names,
            bounded_nat_max=args.bounded_nat_max,
            max_target_hints=args.max_target_hints,
        )
        payload = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_format = payload.get("format")
        if checkpoint_format == "peano-proof-transformer-v1":
            model, original = ProofTransformer.load_checkpoint(args.checkpoint)
        elif checkpoint_format == "peano-latent-proof-transformer-v1":
            model, original = LatentProofTransformer.load_checkpoint(
                args.checkpoint
            )
        else:
            raise SystemExit(
                f"unsupported checkpoint format: {checkpoint_format}"
            )
        from .data_contract import tokenizer_fingerprint, validate_checkpoint_tokenizer
        validate_checkpoint_tokenizer(model, original, tokenizer)
        expansion = model.resize_vocabulary(len(upgraded))
        checkpoint_path = Path(args.output_checkpoint)
        tokenizer_path = Path(args.output_tokenizer)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        upgraded.save(tokenizer_path)
        metadata = dict(original.get("metadata", {}))
        metadata["pa_plus_vocabulary"] = expansion
        metadata["tokenizer_sha256"] = tokenizer_fingerprint(upgraded)
        model.save_checkpoint(checkpoint_path, metadata=metadata)
        print(json.dumps({
            "checkpoint": str(checkpoint_path.resolve()),
            "tokenizer": str(tokenizer_path.resolve()),
            "vocabulary_expansion": expansion,
        }, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        from .evaluate import EvaluationConfig, evaluate_checkpoint

        report = evaluate_checkpoint(
            args.checkpoint,
            args.corpus,
            args.benchmark,
            args.database,
            args.output,
            EvaluationConfig(
                model_target_hints=not args.no_model_target_hints,
                inference_target_guidance=not args.no_inference_target_guidance,
                external_timeout_seconds=args.external_timeout_seconds,
                device=args.device,
                easy_simulations=args.easy_sims,
                medium_simulations=args.medium_sims,
                hard_simulations=args.hard_sims,
                frontier_simulations=args.frontier_sims,
                external_verifier=args.external_verifier,
                require_external_verification=
                    args.require_external_verification,
            ),
        )
        print(json.dumps({
            "supervised_test": report["supervised_test"],
            "search": [
                {
                    "policy": result["policy"],
                    "certified": result["certified_total"],
                    "total": result["total"],
                }
                for result in report["search"]
            ],
        }, ensure_ascii=False))
        return 0
    if args.command == "evaluate-mcts":
        from .evaluate import (
            MCTSEvaluationConfig,
            evaluate_mcts_checkpoint,
        )

        report = evaluate_mcts_checkpoint(
            args.checkpoint,
            args.corpus,
            args.benchmark,
            args.database,
            args.output,
            MCTSEvaluationConfig(
                model_target_hints=not args.no_model_target_hints,
                inference_target_guidance=not args.no_inference_target_guidance,
                external_timeout_seconds=args.external_timeout_seconds,
                device=args.device,
                simulations=args.simulations,
                max_search_depth=args.max_depth,
                branching=args.branching,
                cases_per_difficulty=args.cases_per_difficulty,
                include_frontier=args.include_frontier,
                origins=tuple(
                    item.strip()
                    for item in args.origins.split(",")
                    if item.strip()
                ),
                seeds=_seeds(args.seeds),
                policy_modes=tuple(
                    item.strip()
                    for item in args.policies.split(",")
                    if item.strip()
                ),
                external_verifier=args.external_verifier,
                require_external_verification=
                    args.require_external_verification,
            ),
        )
        print(json.dumps({
            "certified": report["certified_total"],
            "total": report["total"],
            "aggregate": report["aggregate"],
            "aggregate_origin": report["aggregate_origin"],
            "elapsed_seconds": report["elapsed_seconds"],
        }, ensure_ascii=False))
        return 0
    if args.command == "reinforce":
        from .rl import (
            ReinforcementConfig,
            ReplayBuffer,
            reinforce_model,
        )

        summary = reinforce_model(
            args.checkpoint,
            ReplayBuffer.load(args.replay),
            args.output,
            ReinforcementConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "collect-replay":
        from .rl import (
            ReplayCollectionConfig,
            collect_replay_from_corpus,
        )

        summary = collect_replay_from_corpus(
            args.checkpoint,
            args.corpus,
            args.database,
            args.output,
            ReplayCollectionConfig(
                examples=args.examples,
                simulations=args.simulations,
                max_depth=args.max_depth,
                branching=args.branching,
                device=args.device,
            ),
        )
        print(json.dumps({
            "attempted": summary["attempted"],
            "certified": summary["certified"],
            "replay_examples": summary["replay_examples"],
            "device": summary["device"],
        }, ensure_ascii=False))
        return 0
    if args.command == "audit-corpus":
        from .audit import audit_corpus_actions

        report = audit_corpus_actions(
            args.database,
            args.corpus,
            args.output,
        )
        print(json.dumps({
            "valid_actions": report["valid_actions"],
            "invalid_actions": report["invalid_actions"],
            "counts": report["counts"],
        }, ensure_ascii=False))
        return int(report["invalid_actions"] != 0)
    if args.command == "audit-scale-corpus":
        from .audit import audit_scale_corpus

        report = audit_scale_corpus(
            args.database,
            args.corpus,
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(json.dumps({
            "sample_audited": report["sample_audited"],
            "valid_actions": report["valid_actions"],
            "invalid_actions": report["invalid_actions"],
            "duplicate_ids_in_sample":
                report["duplicate_ids_in_sample"],
        }, ensure_ascii=False))
        return int(report["invalid_actions"] != 0)
    if args.command == "build-scale-corpus":
        from .scale_data import ScaleCorpusConfig, build_scale_corpus

        manifest = build_scale_corpus(
            args.database,
            args.base_corpus,
            args.output,
            ScaleCorpusConfig(
                train_examples=args.train_examples,
                validation_examples=args.validation_examples,
                test_examples=args.test_examples,
                shard_size=args.shard_size,
                max_state_tokens=args.max_state_tokens,
                max_action_tokens=args.max_action_tokens,
                seed=args.seed,
                workers=args.workers,
                kernel_validation_interval=
                    args.kernel_validation_interval,
                max_new_records=args.max_new_records,
                depth_balanced=not args.no_depth_balancing,
            ),
        )
        print(json.dumps({
            "counts": manifest["counts"],
            "total": manifest["total"],
            "complete": manifest["complete"],
            "maximum_proof_depth":
                manifest.get("maximum_proof_depth", 0),
            "proof_depth_histogram":
                manifest.get("proof_depth_histogram", {}),
        }, ensure_ascii=False))
        return 0
    if args.command == "train-scale":
        from .scale_train import (
            ScaleTrainingConfig,
            train_scale_model,
        )

        summary = train_scale_model(
            args.corpus,
            args.output,
            ScaleTrainingConfig(
                max_steps=args.max_steps,
                run_steps=args.run_steps,
                micro_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=
                    args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                candidate_loss_weight=args.candidate_loss_weight,
                warmup_steps=args.warmup_steps,
                device=args.device,
                d_model=args.d_model,
                nhead=args.nhead,
                encoder_layers=args.encoder_layers,
                decoder_layers=args.decoder_layers,
                dim_feedforward=args.dim_feedforward,
                initial_context_tokens=args.initial_context_tokens,
                context_warmup_steps=args.context_warmup_steps,
                checkpoint_every=args.checkpoint_every,
                validation_batches=args.validation_batches,
                require_long_context_step=
                    not args.no_long_context_step,
            ),
            resume_from=args.resume,
        )
        result = {
            "status": summary["status"],
            "device": summary["device"],
            "parameter_count": summary["parameter_count"],
            "completed_steps": summary["completed_steps"],
        }
        if summary["status"] == "complete":
            result.update({
                "peak_cuda_memory_bytes":
                    summary["peak_cuda_memory_bytes"],
                "validation": summary["validation"],
                "long_context_training_exercised":
                    summary["long_context_training_exercised"],
                "final_checkpoint": summary["final_checkpoint"],
            })
        else:
            result["resume_checkpoint"] = summary["resume_checkpoint"]
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "evaluate-scale":
        from .scale_train import evaluate_scale_checkpoint

        report = evaluate_scale_checkpoint(
            args.checkpoint,
            args.corpus,
            args.output,
            split=args.split,
            examples=args.examples,
            device_name=args.device,
            seed=args.seed,
        )
        print(json.dumps({
            "split": report["split"],
            "examples": report["examples"],
            "metrics": report["metrics"],
            "examples_per_second": report["examples_per_second"],
            "peak_cuda_memory_bytes":
                report["peak_cuda_memory_bytes"],
        }, ensure_ascii=False))
        return 0
    if args.command == "init-latent":
        from .decomposed import initialize_latent_checkpoint
        from .latent_model import LatentReasoningConfig

        summary = initialize_latent_checkpoint(
            args.base_checkpoint,
            args.base_tokenizer,
            args.output_checkpoint,
            args.output_tokenizer,
            LatentReasoningConfig(
                max_thought_steps=args.max_thought_steps,
                min_thought_steps=args.min_thought_steps,
                halt_threshold=args.halt_threshold,
                ponder_cost=args.ponder_cost,
            ),
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "prove-decomposed":
        from .data_contract import validate_checkpoint_tokenizer
        from .decomposed import (
            DecomposedProver,
            DecomposedProverConfig,
        )
        from .external import verify_certificate_external
        from .latent_model import LatentProofTransformer

        database = parse(args.database)
        cases = {
            case.case_id: case
            for case in load_benchmarks(args.benchmark)
        }
        if args.case_id not in cases:
            raise SystemExit(f"unknown benchmark case {args.case_id}")
        case = cases[args.case_id]
        target = case.theorem(database)
        tokenizer = MetamathTokenizer.load(args.tokenizer)
        model, checkpoint_payload = LatentProofTransformer.load_checkpoint(
            args.checkpoint,
            map_location=args.device if args.device != "auto" else "cpu",
        )
        validate_checkpoint_tokenizer(model, checkpoint_payload, tokenizer)
        prover = DecomposedProver(
            database,
            model,
            tokenizer,
            DecomposedProverConfig(
                simulations=args.simulations,
                max_depth=args.max_depth,
                branching=args.branching,
                seed=args.seed,
                device=args.device,
            ),
            excluded_assertions=case.excluded_labels,
        )
        result = prover.prove(target)
        payload = {
            "case_id": args.case_id,
            **prover.metrics(result),
            "actions": [
                tactic.rule for tactic in result.search.actions
            ],
            "internal_verified": False,
            "external_verification": {"status": "not_run"},
            "certified": False,
        }
        if result.search.solved:
            certificate = compile_certificate(
                target,
                result.search,
                database,
                name=f"decomposed_{args.case_id}",
            )
            verify_certificate(certificate, database)
            external = verify_certificate_external(
                certificate,
                args.database,
                executable=args.external_verifier,
            )
            payload["internal_verified"] = True
            payload["external_verification"] = external.to_record()
            payload["certified"] = (
                external.passed
                if args.require_external_verification else True
            )
            if args.certificate:
                export_certificate(certificate, args.certificate)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "prove":
        import torch

        from .data_contract import validate_checkpoint_tokenizer
        from .mcts import MCTSConfig, ProofMCTS
        from .model import ProofTransformer
        from .search import (
            HeuristicPolicy,
            HybridPolicy,
            TransformerPolicy,
            UniformPolicy,
        )

        database = parse(args.database)
        cases = {
            case.case_id: case
            for case in load_benchmarks(args.benchmark)
        }
        if args.case_id not in cases:
            raise SystemExit(f"unknown benchmark case {args.case_id}")
        case = cases[args.case_id]
        target = case.theorem(database)
        tokenizer = MetamathTokenizer.load(
            Path(args.corpus) / "tokenizer.json"
        )
        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else ("cpu" if args.device == "auto" else args.device)
        )
        model, checkpoint_payload = ProofTransformer.load_checkpoint(
            args.checkpoint,
            map_location=device,
        )
        validate_checkpoint_tokenizer(model, checkpoint_payload, tokenizer)
        environment = BackwardEnvironment(
            database, excluded_assertions=case.excluded_labels
        )
        environment.configure_from_tokenizer(
            tokenizer, target_guidance=not args.no_inference_target_guidance
        )
        neural = TransformerPolicy(
            model,
            tokenizer.with_target_hints(not args.no_model_target_hints),
            environment,
            device,
            configure_environment=False,
        )
        if args.policy == "uniform":
            policy = UniformPolicy()
        elif args.policy == "heuristic":
            policy = HeuristicPolicy(environment)
        elif args.policy == "neural":
            policy = neural
        else:
            policy = HybridPolicy(environment, neural)
        result = ProofMCTS(
            environment,
            HybridActionGenerator(environment),
            policy,
            MCTSConfig(
                simulations=args.simulations,
                max_depth=args.max_depth,
                branching=args.branching,
                seed=args.seed,
            ),
        ).prove(ProofState.from_theorem(target))
        payload = {
            "case_id": args.case_id,
            "policy": args.policy,
            "environment": environment.configuration_record(),
            "solved": result.search.solved,
            "simulations": result.search.simulations,
            "actions": [
                tactic.rule for tactic in result.search.actions
            ],
        }
        if result.search.solved:
            certificate = compile_certificate(
                target,
                result.search,
                database,
                name=f"mcts_{args.case_id}",
            )
            verify_certificate(certificate, database)
            from .external import verify_certificate_external

            external = verify_certificate_external(
                certificate,
                args.database,
                executable=args.external_verifier,
                timeout_seconds=args.external_timeout_seconds,
            )
            payload["internal_verified"] = True
            payload["external_verification"] = external.to_record()
            payload["certified"] = (
                external.passed
                if args.require_external_verification else True
            )
            if args.certificate:
                export_certificate(certificate, args.certificate)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

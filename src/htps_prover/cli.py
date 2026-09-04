from __future__ import annotations

import argparse
import json

from neural_prover.decomposed import initialize_latent_checkpoint
from neural_prover.rl import ReplayBuffer

from .forward import ForwardDAGConfig, build_forward_dag_dataset
from .data import audit_forward_dataset
from .hypergraph import HTPSConfig
from .pipeline import ClosedLoopConfig, evaluate_htps, run_closed_loop
from .training import (
    FreshModelConfig,
    ReplayCollectionConfig,
    ReplayTrainConfig,
    SupervisedTrainConfig,
    collect_htps_replay,
    initialize_fresh_latent_model,
    train_latent_from_replay,
    train_supervised_latent,
)


def _print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _search_config(args) -> HTPSConfig:
    return HTPSConfig(
        simulations=args.simulations,
        expansion_budget=args.expansions,
        branching=args.branching,
        max_frontier=args.max_frontier,
        parallel_selections=args.parallel_selections,
        randomize=args.randomize_search,
        seed=args.seed,
    )


def _add_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--expansions", type=int, default=1000)
    parser.add_argument("--branching", type=int, default=24)
    parser.add_argument("--max-frontier", type=int, default=8)
    parser.add_argument("--parallel-selections", type=int, default=4)
    parser.add_argument(
        "--randomize-search",
        action="store_true",
        help="deterministically sample HTPS effort and PUCT settings per goal",
    )
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peano-htps",
        description="Verified HTPS data/training/inference pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="build certified forward DAG data")
    generate.add_argument("database")
    generate.add_argument("output")
    generate.add_argument("--steps", type=int, default=5000)
    generate.add_argument("--seeds", default="7,11,19,23")
    generate.add_argument("--max-proof-depth", type=int, default=12)
    generate.add_argument("--max-ast-depth", type=int, default=40)
    generate.add_argument("--max-variables", type=int, default=16)
    generate.add_argument("--max-state-tokens", type=int, default=384)
    generate.add_argument("--max-action-tokens", type=int, default=256)
    generate.add_argument("--definition-catalog")
    generate.add_argument("--bootstrap-definitions", action="store_true")
    generate.add_argument("--definition-coverage-weight", type=float, default=0.0)
    generate.add_argument("--bounded-nat-max", type=int, default=-1)
    generate.add_argument("--ground-instances-per-predicate", type=int, default=0)
    generate.add_argument("--target-guidance-weight", type=float, default=0.0)
    generate.add_argument("--max-definition-only-search-per-predicate", type=int, default=-1)
    generate.add_argument("--max-target-hints", type=int, default=3)
    generate.add_argument(
        "--base-tokenizer",
        help="preserve all token IDs from an existing checkpoint tokenizer",
    )

    initialize = sub.add_parser("init-latent", help="upgrade a base checkpoint")
    initialize.add_argument("base_checkpoint")
    initialize.add_argument("base_tokenizer")
    initialize.add_argument("output_checkpoint")
    initialize.add_argument("output_tokenizer")

    audit = sub.add_parser("audit-data", help="replay every serialized policy/lemma action")
    audit.add_argument("database")
    audit.add_argument("directory")
    audit.add_argument("--output")

    fresh = sub.add_parser(
        "init-model", help="create a fresh latent+lemma checkpoint"
    )
    fresh.add_argument("tokenizer")
    fresh.add_argument("output_checkpoint")
    fresh.add_argument("--d-model", type=int, default=768)
    fresh.add_argument("--heads", type=int, default=12)
    fresh.add_argument("--encoder-layers", type=int, default=6)
    fresh.add_argument("--decoder-layers", type=int, default=6)
    fresh.add_argument("--ffn", type=int, default=3072)
    fresh.add_argument("--max-state-tokens", type=int, default=2304)
    fresh.add_argument("--max-action-tokens", type=int, default=2304)
    fresh.add_argument("--thought-steps", type=int, default=6)
    fresh.add_argument("--min-thought-steps", type=int, default=1)
    fresh.add_argument("--seed", type=int, default=7)

    sft = sub.add_parser("train-supervised", help="train on policy and lemma DAG data")
    sft.add_argument("checkpoint")
    sft.add_argument("tokenizer")
    sft.add_argument("policy")
    sft.add_argument("output_checkpoint")
    sft.add_argument("--lemmas")
    sft.add_argument("--epochs", type=int, default=1)
    sft.add_argument("--batch-size", type=int, default=8)
    sft.add_argument("--device", default="auto")
    sft.add_argument("--max-examples", type=int)
    sft.add_argument("--learning-rate", type=float, default=2e-5)
    sft.add_argument("--no-pa-plus-balanced-sampling", action="store_true")
    sft.add_argument("--candidate-loss-weight", type=float, default=0.0)

    collect = sub.add_parser("collect", help="collect verified HTPS replay")
    collect.add_argument("checkpoint")
    collect.add_argument("tokenizer")
    collect.add_argument("policy_corpus")
    collect.add_argument("database")
    collect.add_argument("output")
    collect.add_argument("--examples", type=int, default=32)
    collect.add_argument("--device", default="auto")
    _add_search(collect)

    reinforce = sub.add_parser("train-replay", help="train latent policy and soft critic")
    reinforce.add_argument("checkpoint")
    reinforce.add_argument("replay")
    reinforce.add_argument("output_checkpoint")
    reinforce.add_argument("--epochs", type=int, default=2)
    reinforce.add_argument("--device", default="auto")

    evaluate = sub.add_parser("evaluate", help="certificate-only HTPS evaluation")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("tokenizer")
    evaluate.add_argument("policy_corpus")
    evaluate.add_argument("database")
    evaluate.add_argument("output")
    evaluate.add_argument("--limit", type=int, default=32)
    evaluate.add_argument("--device", default="auto")
    _add_search(evaluate)

    loop = sub.add_parser("closed-loop", help="run synchronous actor/trainer iterations")
    loop.add_argument("checkpoint")
    loop.add_argument("tokenizer")
    loop.add_argument("policy_corpus")
    loop.add_argument("database")
    loop.add_argument("output")
    loop.add_argument("--iterations", type=int, default=2)
    loop.add_argument("--examples", type=int, default=32)
    loop.add_argument("--epochs", type=int, default=2)
    loop.add_argument("--device", default="auto")
    _add_search(loop)
    for command in (collect, evaluate, loop):
        command.add_argument("--external-verifier")
        command.add_argument("--require-external-verification", action="store_true")
        command.add_argument("--external-timeout-seconds", type=float, default=60.0)
        command.add_argument("--no-model-target-hints", action="store_true")
        command.add_argument("--no-inference-target-guidance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        _print(build_forward_dag_dataset(
            args.database,
            args.output,
            ForwardDAGConfig(
                seeds=seeds,
                steps_per_seed=args.steps,
                max_proof_depth=args.max_proof_depth,
                max_ast_depth=args.max_ast_depth,
                max_variables=args.max_variables,
                max_state_tokens=args.max_state_tokens,
                max_action_tokens=args.max_action_tokens,
                definition_catalog=args.definition_catalog,
                bootstrap_definitions=args.bootstrap_definitions,
                definition_coverage_weight=args.definition_coverage_weight,
                bounded_nat_max=args.bounded_nat_max,
                ground_instances_per_predicate=args.ground_instances_per_predicate,
                target_guidance_weight=args.target_guidance_weight,
                max_definition_only_search_per_predicate=args.max_definition_only_search_per_predicate,
                max_target_hints=args.max_target_hints,
            ),
            base_tokenizer_path=args.base_tokenizer,
        ))
    elif args.command == "audit-data":
        result = audit_forward_dataset(args.database, args.directory, args.output)
        _print(result)
        if result["invalid_actions"]:
            raise SystemExit(1)
    elif args.command == "init-latent":
        _print(initialize_latent_checkpoint(
            args.base_checkpoint,
            args.base_tokenizer,
            args.output_checkpoint,
            args.output_tokenizer,
        ))
    elif args.command == "init-model":
        _print(initialize_fresh_latent_model(
            args.tokenizer,
            args.output_checkpoint,
            FreshModelConfig(
                d_model=args.d_model,
                nhead=args.heads,
                num_encoder_layers=args.encoder_layers,
                num_decoder_layers=args.decoder_layers,
                dim_feedforward=args.ffn,
                max_state_tokens=args.max_state_tokens,
                max_action_tokens=args.max_action_tokens,
                max_thought_steps=args.thought_steps,
                min_thought_steps=args.min_thought_steps,
                seed=args.seed,
            ),
        ))
    elif args.command == "train-supervised":
        _print(train_supervised_latent(
            args.checkpoint,
            args.tokenizer,
            args.policy,
            args.lemmas,
            args.output_checkpoint,
            SupervisedTrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                max_examples=args.max_examples,
                learning_rate=args.learning_rate,
                pa_plus_balanced_sampling=not args.no_pa_plus_balanced_sampling,
                candidate_loss_weight=args.candidate_loss_weight,
            ),
        ))
    elif args.command == "collect":
        _print(collect_htps_replay(
            args.checkpoint,
            args.tokenizer,
            args.policy_corpus,
            args.database,
            args.output,
            ReplayCollectionConfig(
                external_verifier=args.external_verifier,
                require_external_verification=args.require_external_verification,
                external_timeout_seconds=args.external_timeout_seconds,
                model_target_hints=not args.no_model_target_hints,
                inference_target_guidance=not args.no_inference_target_guidance,
                examples=args.examples,
                seed=args.seed,
                device=args.device,
                htps=_search_config(args),
            ),
        ))
    elif args.command == "train-replay":
        replay = ReplayBuffer.load(args.replay)
        _print(train_latent_from_replay(
            args.checkpoint,
            replay,
            args.output_checkpoint,
            ReplayTrainConfig(epochs=args.epochs, device=args.device),
        ))
    elif args.command == "evaluate":
        _print(evaluate_htps(
            args.checkpoint,
            args.tokenizer,
            args.policy_corpus,
            args.database,
            args.output,
            limit=args.limit,
            device_name=args.device,
            search_config=_search_config(args),
            external_verifier=args.external_verifier,
            require_external_verification=args.require_external_verification,
            external_timeout_seconds=args.external_timeout_seconds,
            model_target_hints=not args.no_model_target_hints,
            inference_target_guidance=not args.no_inference_target_guidance,
        ))
    elif args.command == "closed-loop":
        _print(run_closed_loop(
            args.checkpoint,
            args.tokenizer,
            args.policy_corpus,
            args.database,
            args.output,
            ClosedLoopConfig(
                iterations=args.iterations,
                collection=ReplayCollectionConfig(
                    external_verifier=args.external_verifier,
                    require_external_verification=args.require_external_verification,
                    external_timeout_seconds=args.external_timeout_seconds,
                    model_target_hints=not args.no_model_target_hints,
                    inference_target_guidance=not args.no_inference_target_guidance,
                    examples=args.examples,
                    seed=args.seed,
                    device=args.device,
                    htps=_search_config(args),
                ),
                training=ReplayTrainConfig(
                    epochs=args.epochs, device=args.device
                ),
            ),
        ))


if __name__ == "__main__":
    main()

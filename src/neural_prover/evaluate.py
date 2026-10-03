from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from metamath_generator.parser import parse

from .benchmark import BenchmarkCase, load_benchmarks
from .certificate import CertificateError, compile_certificate
from .data import load_examples
from .data_contract import validate_checkpoint_tokenizer
from .environment import BackwardEnvironment, ProofState
from .verification import verification_record
from .hybrid import HybridActionGenerator
from .mcts import MCTSConfig, ProofMCTS
from .model import ProofTransformer
from .scale_train import _checkpoint_report_metadata
from .search import (
    HeuristicPolicy,
    HybridPolicy,
    NeuralBestFirstSearch,
    TransformerPolicy,
    UniformPolicy,
)
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    model_target_hints: bool = True
    inference_target_guidance: bool = True
    device: str = "auto"
    easy_simulations: int = 100
    medium_simulations: int = 300
    hard_simulations: int = 1_000
    frontier_simulations: int = 100
    max_search_depth: int = 16
    branching: int = 32
    evaluate_heuristic: bool = True
    evaluate_neural: bool = True
    evaluate_uniform: bool = True
    external_verifier: str | None = None
    require_external_verification: bool = False
    external_timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class MCTSEvaluationConfig:
    model_target_hints: bool = True
    inference_target_guidance: bool = True
    device: str = "auto"
    simulations: int = 60
    max_search_depth: int = 16
    branching: int = 32
    cases_per_difficulty: int = 5
    include_frontier: bool = False
    origins: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (17, 19, 23)
    selection_seed: int = 20260822
    score_groups: tuple[str, ...] = ("research",)
    require_research_eligible: bool = True
    external_verifier: str | None = None
    require_external_verification: bool = False
    external_timeout_seconds: float = 60.0
    policy_modes: tuple[str, ...] = (
        "uniform", "heuristic", "neural", "hybrid"
    )


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate_supervised(
    model: ProofTransformer,
    tokenizer: MetamathTokenizer,
    test_path: str | Path,
    device: torch.device,
    *,
    model_target_hints: bool = True,
) -> dict:
    examples = load_examples(test_path)
    if not examples:
        return {
            "examples": 0,
            "policy_loss": None,
            "perplexity": None,
            "token_accuracy": None,
            "exact_action_accuracy": None,
            "value_mse": None,
        }
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact = 0
    value_error = 0.0
    model.eval()
    for example in examples:
        state_ids = example.state_ids
        if not model_target_hints:
            tokens = list(example.state_tokens)
            if "<TARGET_HINTS>" in tokens:
                start = tokens.index("<TARGET_HINTS>") + 1
                end = tokens.index("<END_TARGET_HINTS>", start)
                del tokens[start:end]
                state_ids = tokenizer.encode(tokens)
        state = torch.tensor(
            [state_ids],
            dtype=torch.long,
            device=device,
        )
        action = torch.tensor(
            [example.action_ids],
            dtype=torch.long,
            device=device,
        )
        logits, value = model(state, action[:, :-1])
        target = action[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="sum",
        )
        predictions = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_tokens += target.numel()
        correct_tokens += int((predictions == target).sum().item())
        exact += int(bool(torch.all(predictions == target).item()))
        value_error += (
            float(value.item()) - example.value_target
        ) ** 2
    policy_loss = total_loss / max(1, total_tokens)
    return {
        "examples": len(examples),
        "policy_loss": policy_loss,
        "perplexity": math.exp(min(20.0, policy_loss)),
        "token_accuracy": correct_tokens / max(1, total_tokens),
        "exact_action_accuracy": exact / len(examples),
        "value_mse": value_error / len(examples),
        "model_target_hints": model_target_hints,
    }


def _budget(case: BenchmarkCase, config: EvaluationConfig) -> int:
    return {
        "easy": config.easy_simulations,
        "medium": config.medium_simulations,
        "hard": config.hard_simulations,
        "frontier": config.frontier_simulations,
    }[case.difficulty]


def _wilson(successes: int, total: int) -> dict[str, float | None]:
    if total <= 0:
        return {"rate": None, "low": None, "high": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "rate": rate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def _verification_record(
    certificate,
    database,
    database_path: str | Path,
    *,
    external_verifier: str | None,
    require_external: bool,
    timeout_seconds: float,
) -> tuple[bool, dict]:
    return verification_record(
        certificate, database, database_path,
        external_verifier=external_verifier,
        require_external=require_external,
        timeout_seconds=timeout_seconds,
    )


def _run_policy(
    name: str,
    policy_factory,
    cases: list[BenchmarkCase],
    database,
    database_path: str | Path,
    config: EvaluationConfig,
    tokenizer: MetamathTokenizer | None = None,
) -> dict:
    records: list[dict] = []
    for case in cases:
        environment = BackwardEnvironment(
            database, excluded_assertions=case.excluded_labels
        )
        if tokenizer is not None:
            environment.configure_from_tokenizer(
                tokenizer, target_guidance=config.inference_target_guidance
            )
        policy = policy_factory(environment)
        target = case.theorem(database)
        search = NeuralBestFirstSearch(
            environment,
            policy,
            max_depth=config.max_search_depth,
            branching=config.branching,
        )
        started = time.perf_counter()
        error: str | None = None
        certified = False
        verification = {
            "internal_verified": False,
            "external_verification": {"status": "not_run"},
            "certification_level": "none",
        }
        try:
            result = search.prove(
                ProofState.from_theorem(target),
                simulations=_budget(case, config),
            )
            if result.solved:
                certificate = compile_certificate(
                    target,
                    result,
                    database,
                    name=f"eval_{case.case_id}",
                )
                certified, verification = _verification_record(
                    certificate,
                    database,
                    database_path,
                    external_verifier=config.external_verifier,
                    require_external=config.require_external_verification,
                    timeout_seconds=config.external_timeout_seconds,
                )
        except (ValueError, RuntimeError, CertificateError) as exc:
            result = None
            error = str(exc)
        elapsed = time.perf_counter() - started
        records.append({
            "case_id": case.case_id,
            "title": case.title,
            "origin": case.origin,
            "difficulty": case.difficulty,
            "reference_proof_depth": case.proof_depth,
            "expected_status": case.expected_status,
            "score_group": case.score_group,
            "research_eligible": case.research_eligible,
            "solved": bool(result and result.solved),
            "certified": certified,
            "simulations": result.simulations if result else 0,
            "nodes": result.nodes if result else 0,
            "actions": (
                [tactic.rule for tactic in result.actions]
                if result else []
            ),
            "remaining_goals": (
                len(result.final_state.goals) if result else None
            ),
            "elapsed_seconds": elapsed,
            "verification": verification,
            "environment": environment.configuration_record(),
            "compute": {
                "environment": environment.metrics.to_record(),
                "policy": (
                    policy.metrics() if hasattr(policy, "metrics") else {}
                ),
            },
            "error": error,
        })
    aggregate: dict[str, dict] = {}
    for dimension in ("difficulty", "origin"):
        values = sorted({record[dimension] for record in records})
        aggregate[dimension] = {}
        for value in values:
            subset = [
                record for record in records
                if record[dimension] == value
            ]
            aggregate[dimension][value] = {
                "cases": len(subset),
                "solved": sum(record["certified"] for record in subset),
                "solve_rate": sum(
                    record["certified"] for record in subset
                ) / len(subset),
                "wilson_95": _wilson(
                    sum(record["certified"] for record in subset),
                    len(subset),
                ),
            }
    research = [record for record in records if record["research_eligible"]]
    return {
        "policy": name,
        "cases": records,
        "aggregate": aggregate,
        "certified_total": sum(record["certified"] for record in records),
        "total": len(records),
        "research": {
            "certified": sum(record["certified"] for record in research),
            "total": len(research),
            "wilson_95": _wilson(
                sum(record["certified"] for record in research),
                len(research),
            ),
        },
    }


def evaluate_checkpoint(
    checkpoint: str | Path,
    corpus_directory: str | Path,
    benchmark_path: str | Path,
    database_path: str | Path,
    destination: str | Path,
    config: EvaluationConfig | None = None,
) -> dict:
    cfg = config or EvaluationConfig()
    device = _device(cfg.device)
    tokenizer = MetamathTokenizer.load(
        Path(corpus_directory) / "tokenizer.json"
    )
    model, payload = ProofTransformer.load_checkpoint(
        checkpoint,
        map_location=device,
    )
    validate_checkpoint_tokenizer(model, payload, tokenizer)
    model.to(device).eval()
    supervised = evaluate_supervised(
        model,
        tokenizer,
        Path(corpus_directory) / "test.jsonl",
        device,
        model_target_hints=cfg.model_target_hints,
    )
    database = parse(database_path)
    cases = load_benchmarks(benchmark_path)
    policies: list[dict] = []
    if cfg.evaluate_uniform:
        policies.append(_run_policy(
            "uniform",
            lambda environment: UniformPolicy(),
            cases,
            database,
            database_path,
            cfg,
            tokenizer,
        ))
    if cfg.evaluate_heuristic:
        policies.append(_run_policy(
            "heuristic",
            lambda environment: HeuristicPolicy(environment),
            cases,
            database,
            database_path,
            cfg,
            tokenizer,
        ))
    if cfg.evaluate_neural:
        policies.append(_run_policy(
            "transformer",
            lambda environment: TransformerPolicy(
                model, tokenizer.with_target_hints(cfg.model_target_hints),
                environment, device, configure_environment=False,
            ),
            cases,
            database,
            database_path,
            cfg,
            tokenizer,
        ))
    report = {
        "format": "peano-proof-evaluation-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_metadata": _checkpoint_report_metadata(
            payload.get("metadata", {})
        ),
        "device": str(device),
        "configuration": asdict(cfg),
        "supervised_test": supervised,
        "search": policies,
        "soundness_gate": {
            "internal_required": True,
            "external_required": cfg.require_external_verification,
            "external_verifier": cfg.external_verifier,
        },
        "frontier_policy": (
            "famous statements are never training labels; open conjectures "
            "and theorems without a library proof are reported separately"
        ),
    }
    Path(destination).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def evaluate_mcts_checkpoint(
    checkpoint: str | Path,
    corpus_directory: str | Path,
    benchmark_path: str | Path,
    database_path: str | Path,
    destination: str | Path,
    config: MCTSEvaluationConfig | None = None,
) -> dict:
    """Evaluate the AlphaZero/AlphaGeometry-style hybrid search."""

    cfg = config or MCTSEvaluationConfig()
    device = _device(cfg.device)
    tokenizer = MetamathTokenizer.load(
        Path(corpus_directory) / "tokenizer.json"
    )
    model, payload = ProofTransformer.load_checkpoint(
        checkpoint,
        map_location=device,
    )
    validate_checkpoint_tokenizer(model, payload, tokenizer)
    model.to(device).eval()
    database = parse(database_path)
    all_cases = load_benchmarks(benchmark_path)
    if cfg.score_groups:
        all_cases = [
            case for case in all_cases
            if case.score_group in cfg.score_groups
        ]
    if cfg.require_research_eligible:
        all_cases = [
            case for case in all_cases
            if case.score_group != "research" or case.research_eligible
        ]
    if cfg.origins:
        all_cases = [
            case for case in all_cases
            if case.origin in cfg.origins
        ]
    cases: list[BenchmarkCase] = []
    difficulties = ["easy", "medium", "hard"]
    if cfg.include_frontier:
        difficulties.append("frontier")
    for difficulty in difficulties:
        subset = [
            case for case in all_cases
            if case.difficulty == difficulty
        ]
        random.Random(
            f"{cfg.selection_seed}:{difficulty}"
        ).shuffle(subset)
        cases.extend(
            subset[:cfg.cases_per_difficulty]
            if cfg.cases_per_difficulty > 0 else subset
        )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = destination_path.with_suffix(
        destination_path.suffix + ".progress.jsonl"
    )
    total_attempts = len(cases) * len(cfg.seeds) * len(cfg.policy_modes)
    progress_path.write_text(
        json.dumps({
            "format": "peano-mcts-evaluation-progress-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "benchmark": str(Path(benchmark_path).resolve()),
            "configuration": asdict(cfg),
            "total_attempts": total_attempts,
            "case_ids": [case.case_id for case in cases],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records: list[dict] = []
    for case in cases:
        for seed in cfg.seeds:
            for policy_mode in cfg.policy_modes:
                environment = BackwardEnvironment(
                    database, excluded_assertions=case.excluded_labels
                )
                environment.configure_from_tokenizer(
                    tokenizer, target_guidance=cfg.inference_target_guidance
                )
                neural = None
                if policy_mode == "uniform":
                    policy = UniformPolicy()
                elif policy_mode == "heuristic":
                    policy = HeuristicPolicy(environment)
                elif policy_mode in {"neural", "hybrid"}:
                    neural = TransformerPolicy(
                        model, tokenizer.with_target_hints(cfg.model_target_hints),
                        environment, device, configure_environment=False,
                    )
                    policy = (
                        neural if policy_mode == "neural"
                        else HybridPolicy(environment, neural)
                    )
                else:
                    raise ValueError(f"unknown policy mode {policy_mode!r}")
                prover = ProofMCTS(
                    environment,
                    HybridActionGenerator(environment),
                    policy,
                    MCTSConfig(
                        simulations=cfg.simulations,
                        max_depth=cfg.max_search_depth,
                        branching=cfg.branching,
                        seed=seed,
                    ),
                )
                target = case.theorem(database)
                started = time.perf_counter()
                certified = False
                error = None
                verification = {
                    "internal_verified": False,
                    "external_verification": {"status": "not_run"},
                    "certification_level": "none",
                }
                try:
                    result = prover.prove(ProofState.from_theorem(target))
                    if result.search.solved:
                        certificate = compile_certificate(
                            target,
                            result.search,
                            database,
                            name=(
                                f"mcts_eval_{policy_mode}_"
                                f"{seed}_{case.case_id}"
                            ),
                        )
                        certified, verification = _verification_record(
                            certificate,
                            database,
                            database_path,
                            external_verifier=cfg.external_verifier,
                            require_external=
                                cfg.require_external_verification,
                            timeout_seconds=cfg.external_timeout_seconds,
                        )
                except (ValueError, RuntimeError, CertificateError) as exc:
                    result = None
                    error = str(exc)
                record = {
                    "case_id": case.case_id,
                    "origin": case.origin,
                    "difficulty": case.difficulty,
                    "reference_proof_depth": case.proof_depth,
                    "score_group": case.score_group,
                    "research_eligible": case.research_eligible,
                    "seed": seed,
                    "policy": policy_mode,
                    "certified": certified,
                    "simulations": (
                        result.search.simulations if result else 0
                    ),
                    "nodes": result.search.nodes if result else 0,
                    "outcome": result.outcome if result else "error",
                    "elapsed_seconds": time.perf_counter() - started,
                    "verification": verification,
                    "environment": environment.configuration_record(),
                    "compute": {
                        "environment": environment.metrics.to_record(),
                        "policy": neural.metrics() if neural else {},
                    },
                    "error": error,
                }
                records.append(record)
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                print(
                    "mcts-eval "
                    f"{len(records)}/{total_attempts} "
                    f"case={case.case_id} seed={seed} "
                    f"policy={policy_mode} "
                    f"certified={int(certified)} "
                    f"seconds={record['elapsed_seconds']:.3f}",
                    flush=True,
                )
    aggregate = {}
    for policy_mode in cfg.policy_modes:
        aggregate[policy_mode] = {}
        for difficulty in difficulties:
            subset = [
                record for record in records
                if record["difficulty"] == difficulty
                and record["policy"] == policy_mode
            ]
            if subset:
                aggregate[policy_mode][difficulty] = {
                    "attempts": len(subset),
                    "certified": sum(
                        record["certified"] for record in subset
                    ),
                    "wilson_95": _wilson(
                        sum(record["certified"] for record in subset),
                        len(subset),
                    ),
                }
    aggregate_origin = {}
    aggregate_depth = {}
    for policy_mode in cfg.policy_modes:
        aggregate_origin[policy_mode] = {}
        for origin in sorted({record["origin"] for record in records}):
            subset = [
                record for record in records
                if record["origin"] == origin
                and record["policy"] == policy_mode
            ]
            if subset:
                aggregate_origin[policy_mode][origin] = {
                    "attempts": len(subset),
                    "certified": sum(
                        record["certified"] for record in subset
                    ),
                }
        aggregate_depth[policy_mode] = {}
        for depth in sorted({
            record["reference_proof_depth"] for record in records
        }):
            subset = [
                record for record in records
                if record["reference_proof_depth"] == depth
                and record["policy"] == policy_mode
            ]
            if subset:
                aggregate_depth[policy_mode][str(depth)] = {
                    "attempts": len(subset),
                    "certified": sum(
                        record["certified"] for record in subset
                    ),
                    "wilson_95": _wilson(
                        sum(record["certified"] for record in subset),
                        len(subset),
                    ),
                }
    report = {
        "format": "peano-baseline-mcts-evaluation-v2",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_metadata": _checkpoint_report_metadata(
            payload.get("metadata", {})
        ),
        "device": str(device),
        "configuration": asdict(cfg),
        "aggregate": aggregate,
        "aggregate_origin": aggregate_origin,
        "aggregate_reference_proof_depth": aggregate_depth,
        "certified_total": sum(
            record["certified"] for record in records
        ),
        "total": len(records),
        "elapsed_seconds": sum(
            record["elapsed_seconds"] for record in records
        ),
        "cases": records,
        "soundness_gate": {
            "internal_required": True,
            "external_required": cfg.require_external_verification,
            "external_verifier": cfg.external_verifier,
        },
    }
    destination_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from metamath_generator.parser import parse

from .benchmark import BenchmarkCase, load_benchmarks
from .certificate import CertificateError, compile_certificate, verify_certificate
from .data import load_examples
from .environment import BackwardEnvironment, ProofState
from .hybrid import HybridActionGenerator
from .mcts import MCTSConfig, ProofMCTS
from .model import ProofTransformer
from .search import (
    HeuristicPolicy,
    HybridPolicy,
    NeuralBestFirstSearch,
    TransformerPolicy,
)
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    device: str = "auto"
    easy_simulations: int = 100
    medium_simulations: int = 300
    hard_simulations: int = 1_000
    frontier_simulations: int = 100
    max_search_depth: int = 16
    branching: int = 32
    evaluate_heuristic: bool = True
    evaluate_neural: bool = True


@dataclass(frozen=True, slots=True)
class MCTSEvaluationConfig:
    device: str = "auto"
    simulations: int = 60
    max_search_depth: int = 16
    branching: int = 32
    cases_per_difficulty: int = 5
    include_frontier: bool = False
    origins: tuple[str, ...] = ()
    seed: int = 17


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
        state = torch.tensor(
            [example.state_ids],
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
    }


def _budget(case: BenchmarkCase, config: EvaluationConfig) -> int:
    return {
        "easy": config.easy_simulations,
        "medium": config.medium_simulations,
        "hard": config.hard_simulations,
        "frontier": config.frontier_simulations,
    }[case.difficulty]


def _run_policy(
    name: str,
    policy,
    cases: list[BenchmarkCase],
    environment: BackwardEnvironment,
    database,
    config: EvaluationConfig,
) -> dict:
    records: list[dict] = []
    for case in cases:
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
                verify_certificate(certificate, database)
                certified = True
        except (ValueError, RuntimeError, CertificateError) as exc:
            result = None
            error = str(exc)
        elapsed = time.perf_counter() - started
        records.append({
            "case_id": case.case_id,
            "title": case.title,
            "origin": case.origin,
            "difficulty": case.difficulty,
            "expected_status": case.expected_status,
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
            }
    return {
        "policy": name,
        "cases": records,
        "aggregate": aggregate,
        "certified_total": sum(record["certified"] for record in records),
        "total": len(records),
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
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    model.to(device).eval()
    supervised = evaluate_supervised(
        model,
        tokenizer,
        Path(corpus_directory) / "test.jsonl",
        device,
    )
    database = parse(database_path)
    environment = BackwardEnvironment(database)
    cases = load_benchmarks(benchmark_path)
    policies: list[dict] = []
    if cfg.evaluate_heuristic:
        policies.append(_run_policy(
            "heuristic",
            HeuristicPolicy(environment),
            cases,
            environment,
            database,
            cfg,
        ))
    if cfg.evaluate_neural:
        policies.append(_run_policy(
            "transformer",
            TransformerPolicy(
                model,
                tokenizer,
                environment,
                device,
            ),
            cases,
            environment,
            database,
            cfg,
        ))
    report = {
        "format": "peano-proof-evaluation-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_metadata": payload.get("metadata", {}),
        "device": str(device),
        "configuration": asdict(cfg),
        "supervised_test": supervised,
        "search": policies,
        "soundness_gate": (
            "a case counts as solved only after Metamath certificate replay"
        ),
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
    model.to(device).eval()
    database = parse(database_path)
    environment = BackwardEnvironment(database)
    all_cases = load_benchmarks(benchmark_path)
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
        cases.extend(subset[:cfg.cases_per_difficulty])
    neural = TransformerPolicy(
        model, tokenizer, environment, device
    )
    policy = HybridPolicy(environment, neural)
    generator = HybridActionGenerator(environment)
    records: list[dict] = []
    for index, case in enumerate(cases):
        target = case.theorem(database)
        prover = ProofMCTS(
            environment,
            generator,
            policy,
            MCTSConfig(
                simulations=cfg.simulations,
                max_depth=cfg.max_search_depth,
                branching=cfg.branching,
                seed=cfg.seed + index,
            ),
        )
        started = time.perf_counter()
        certified = False
        error = None
        try:
            result = prover.prove(ProofState.from_theorem(target))
            if result.search.solved:
                certificate = compile_certificate(
                    target,
                    result.search,
                    database,
                    name=f"mcts_eval_{case.case_id}",
                )
                verify_certificate(certificate, database)
                certified = True
        except (ValueError, RuntimeError, CertificateError) as exc:
            result = None
            error = str(exc)
        records.append({
            "case_id": case.case_id,
            "origin": case.origin,
            "difficulty": case.difficulty,
            "certified": certified,
            "simulations": (
                result.search.simulations if result else 0
            ),
            "nodes": result.search.nodes if result else 0,
            "elapsed_seconds": time.perf_counter() - started,
            "error": error,
        })
    aggregate = {}
    for difficulty in difficulties:
        subset = [
            record for record in records
            if record["difficulty"] == difficulty
        ]
        if subset:
            aggregate[difficulty] = {
                "cases": len(subset),
                "certified": sum(
                    record["certified"] for record in subset
                ),
            }
    aggregate_origin = {}
    for origin in sorted({record["origin"] for record in records}):
        subset = [
            record for record in records
            if record["origin"] == origin
        ]
        aggregate_origin[origin] = {
            "cases": len(subset),
            "certified": sum(
                record["certified"] for record in subset
            ),
        }
    report = {
        "format": "peano-hybrid-mcts-evaluation-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_metadata": payload.get("metadata", {}),
        "device": str(device),
        "configuration": asdict(cfg),
        "aggregate": aggregate,
        "aggregate_origin": aggregate_origin,
        "certified_total": sum(
            record["certified"] for record in records
        ),
        "total": len(records),
        "elapsed_seconds": sum(
            record["elapsed_seconds"] for record in records
        ),
        "cases": records,
        "soundness_gate": (
            "success requires replay by the independent Metamath verifier"
        ),
    }
    Path(destination).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

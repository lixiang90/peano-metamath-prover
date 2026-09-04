from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.parser import parse
from neural_prover.certificate import compile_certificate
from neural_prover.environment import ProofState
from neural_prover.data_contract import validate_checkpoint_tokenizer
from neural_prover.latent_model import LatentProofTransformer
from neural_prover.lemma import (
    LemmaActionGenerator,
    LemmaBackwardEnvironment,
    LemmaGeneratorConfig,
)
from neural_prover.rl import ReplayBuffer
from neural_prover.search import HybridPolicy, TransformerPolicy
from neural_prover.tokenizer import MetamathTokenizer
from neural_prover.verification import verification_record

from .hypergraph import (
    HTPSConfig,
    HyperTreeProofSearch,
    materialize_search_config,
)
from .training import (
    ReplayCollectionConfig,
    ReplayTrainConfig,
    _device,
    _load_records,
    collect_htps_replay,
    train_latent_from_replay,
)


@dataclass(frozen=True, slots=True)
class ClosedLoopConfig:
    iterations: int = 2
    replay_capacity: int = 50_000
    collection: ReplayCollectionConfig = ReplayCollectionConfig()
    training: ReplayTrainConfig = ReplayTrainConfig()


def _merge_replay(
    destination: ReplayBuffer,
    source: ReplayBuffer,
) -> None:
    for example in source.examples:
        existing = destination._by_state.get(example.state_ids)  # noqa: SLF001
        if existing is None:
            destination._by_state[example.state_ids] = len(  # noqa: SLF001
                destination.examples
            )
            destination.examples.append(example)
        else:
            destination.examples[existing] = example
    if len(destination.examples) > destination.capacity:
        destination.examples = destination.examples[-destination.capacity :]
        destination._by_state = {  # noqa: SLF001
            item.state_ids: index
            for index, item in enumerate(destination.examples)
        }


def run_closed_loop(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    policy_corpus_path: str | Path,
    database_path: str | Path,
    output_directory: str | Path,
    config: ClosedLoopConfig | None = None,
) -> dict:
    """Run synchronous HTPS actor/trainer iterations with a persistent queue."""

    cfg = config or ClosedLoopConfig()
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    current = Path(checkpoint)
    cumulative = ReplayBuffer(cfg.replay_capacity)
    iterations: list[dict] = []
    for iteration in range(1, cfg.iterations + 1):
        replay_path = output / f"replay-{iteration:03d}.jsonl"
        collection = collect_htps_replay(
            current,
            tokenizer_path,
            policy_corpus_path,
            database_path,
            replay_path,
            cfg.collection,
        )
        fresh = ReplayBuffer.load(replay_path, cfg.replay_capacity)
        _merge_replay(cumulative, fresh)
        cumulative_path = output / "replay-cumulative.jsonl"
        cumulative.save(cumulative_path)
        next_checkpoint = output / f"checkpoint-{iteration:03d}.pt"
        training = train_latent_from_replay(
            current, cumulative, next_checkpoint, cfg.training
        )
        iterations.append({
            "iteration": iteration,
            "input_checkpoint": str(current.resolve()),
            "collection": collection,
            "cumulative_replay_examples": len(cumulative),
            "training": training,
        })
        current = next_checkpoint
    summary = {
        "format": "peano-htps-closed-loop-v1",
        "configuration": asdict(cfg),
        "initial_checkpoint": str(Path(checkpoint).resolve()),
        "final_checkpoint": str(current.resolve()),
        "iterations": iterations,
    }
    (output / "closed-loop-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_htps(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    policy_corpus_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 32,
    device_name: str = "auto",
    search_config: HTPSConfig | None = None,
    external_verifier: str | None = None,
    require_external_verification: bool = False,
    external_timeout_seconds: float = 60.0,
    model_target_hints: bool = True,
    inference_target_guidance: bool = True,
) -> dict:
    """Evaluate only by independently verified final certificates."""

    device = _device(device_name)
    database = parse(database_path)
    tokenizer = MetamathTokenizer.load(tokenizer_path)
    model, payload = LatentProofTransformer.load_checkpoint(
        checkpoint, map_location=device
    )
    validate_checkpoint_tokenizer(model, payload, tokenizer)
    records = _load_records(policy_corpus_path)[:limit]
    episodes: list[dict] = []
    certified = 0
    base_search = search_config or HTPSConfig()
    for episode_index, record in enumerate(records):
        environment = LemmaBackwardEnvironment(
            database, excluded_assertions=record.get("excluded_labels", ())
        )
        environment.configure_from_tokenizer(
            tokenizer, target_guidance=inference_target_guidance
        )
        generator = LemmaActionGenerator(environment, LemmaGeneratorConfig())
        neural = TransformerPolicy(
            model, tokenizer.with_target_hints(model_target_hints), environment,
            device, configure_environment=False,
        )
        policy = HybridPolicy(environment, neural)
        theorem = tokenizer.theorem_from_state_tokens(
            record["state_tokens"],
            database,
            name=f"eval_{record['theorem_name']}",
        )
        realized_search = materialize_search_config(
            base_search, episode_index
        )
        searcher = HyperTreeProofSearch(
            environment,
            generator,
            policy,
            realized_search,
        )
        policy_before = neural.metrics()
        started = time.perf_counter()
        result, metrics = searcher.prove(ProofState.from_theorem(theorem))
        wall_seconds = time.perf_counter() - started
        policy_after = neural.metrics()
        valid = False
        error: str | None = None
        certificate_steps = 0
        verification = {
            "internal_verified": False,
            "external_verification": {"status": "not_run"},
            "certification_level": "none",
        }
        if result.search.solved:
            try:
                certificate = compile_certificate(
                    theorem,
                    result.search,
                    database,
                    name=f"eval_cert_{record['theorem_name']}",
                )
                valid, verification = verification_record(
                    certificate, database, database_path,
                    external_verifier=external_verifier,
                    require_external=require_external_verification,
                    timeout_seconds=external_timeout_seconds,
                )
                certificate_steps = len(certificate.proof.source_labels)
                certified += int(valid)
                if not valid:
                    error = "required external verification did not pass"
            except Exception as exc:
                error = str(exc)
        episodes.append({
            "theorem_name": record["theorem_name"],
            "proof_depth": record.get("proof_depth"),
            "certified": valid,
            "verification": verification,
            "environment": environment.configuration_record(),
            "certificate_error": error,
            "abstract_actions": len(result.search.actions),
            "certificate_steps": certificate_steps,
            "outcome": result.outcome,
            "selected_lemma_actions": sum(
                tactic.rule == "<PROPOSE_LEMMA>"
                for tactic in result.search.actions
            ),
            "wall_seconds": wall_seconds,
            "neural_candidates_scored": int(
                policy_after["candidates_scored"]
                - policy_before["candidates_scored"]
            ),
            "neural_action_tokens_scored": int(
                policy_after["action_tokens_scored"]
                - policy_before["action_tokens_scored"]
            ),
            "realized_search_configuration": asdict(realized_search),
            "search": asdict(metrics),
        })
    summary = {
        "format": "peano-htps-evaluation-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "attempted": len(records),
        "certified": certified,
        "certified_rate": certified / len(records) if records else 0.0,
        "pass_at_1": certified / len(records) if records else 0.0,
        "certificate_only_scoring": True,
        "soundness_gate": {
            "internal_required": True,
            "external_required": require_external_verification,
            "external_verifier": external_verifier,
            "external_timeout_seconds": external_timeout_seconds,
        },
        "guidance": {
            "model_target_hints": model_target_hints,
            "inference_target_guidance": inference_target_guidance,
        },
        "total_wall_seconds": sum(
            item["wall_seconds"] for item in episodes
        ),
        "total_neural_candidates_scored": sum(
            item["neural_candidates_scored"] for item in episodes
        ),
        "episodes": episodes,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

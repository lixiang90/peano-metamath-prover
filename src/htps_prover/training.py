from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from metamath_generator.parser import parse
from neural_prover.certificate import compile_certificate
from neural_prover.environment import ProofState
from neural_prover.latent_model import (
    LatentProofTransformer,
    LatentReasoningConfig,
)
from neural_prover.latent_train import LatentLossConfig, latent_policy_value_loss
from neural_prover.lemma import (
    LemmaActionGenerator,
    LemmaBackwardEnvironment,
    LemmaGeneratorConfig,
)
from neural_prover.rl import ReplayBuffer
from neural_prover.model import ProofTransformerConfig
from neural_prover.search import HybridPolicy, TransformerPolicy
from neural_prover.tokenizer import MetamathTokenizer
from neural_prover.verification import verification_record

from .hypergraph import (
    HTPSConfig,
    HyperTreeProofSearch,
    materialize_search_config,
)


@dataclass(frozen=True, slots=True)
class SupervisedTrainConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    value_weight: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 7
    device: str = "auto"
    max_examples: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayTrainConfig:
    epochs: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    value_weight: float = 1.0
    gradient_clip: float = 1.0
    gradient_accumulation: int = 8
    seed: int = 7
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class ReplayCollectionConfig:
    examples: int = 32
    capacity: int = 50_000
    seed: int = 7
    device: str = "auto"
    htps: HTPSConfig = HTPSConfig()
    external_verifier: str | None = None
    require_external_verification: bool = False
    external_timeout_seconds: float = 60.0
    model_target_hints: bool = True
    inference_target_guidance: bool = True


@dataclass(frozen=True, slots=True)
class FreshModelConfig:
    """Default to the established roughly GPT-2-small parameter scale."""

    d_model: int = 768
    nhead: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 3072
    dropout: float = 0.1
    max_state_tokens: int = 2304
    max_action_tokens: int = 2304
    gradient_checkpointing: bool = True
    max_thought_steps: int = 6
    min_thought_steps: int = 1
    halt_threshold: float = 0.85
    ponder_cost: float = 1.0e-3
    seed: int = 7


def initialize_fresh_latent_model(
    tokenizer_path: str | Path,
    output_checkpoint: str | Path,
    config: FreshModelConfig | None = None,
) -> dict:
    """Create a latent+lemma checkpoint without requiring a legacy model."""

    cfg = config or FreshModelConfig()
    if cfg.d_model <= 0 or cfg.nhead <= 0:
        raise ValueError("model width and attention heads must be positive")
    if cfg.d_model % cfg.nhead:
        raise ValueError("d_model must be divisible by nhead")
    torch.manual_seed(cfg.seed)
    tokenizer = MetamathTokenizer.load(tokenizer_path)
    if not tokenizer.supports_lemma_actions:
        raise ValueError(
            "tokenizer lacks lemma actions; use the HTPS generated tokenizer"
        )
    model = LatentProofTransformer(
        ProofTransformerConfig(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_state_tokens=cfg.max_state_tokens,
            max_action_tokens=cfg.max_action_tokens,
            gradient_checkpointing=cfg.gradient_checkpointing,
        ),
        LatentReasoningConfig(
            max_thought_steps=cfg.max_thought_steps,
            min_thought_steps=cfg.min_thought_steps,
            halt_threshold=cfg.halt_threshold,
            ponder_cost=cfg.ponder_cost,
            lemma_token_id=tokenizer.token_to_id["<PROPOSE_LEMMA>"],
        ),
    )
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        destination,
        metadata={
            "initialization": "fresh-htps-latent-model",
            "configuration": asdict(cfg),
        },
    )
    return {
        "checkpoint": str(destination.resolve()),
        "tokenizer": str(Path(tokenizer_path).resolve()),
        "parameter_count": model.parameter_count(),
        "configuration": asdict(cfg),
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_records(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pad(rows: list[list[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    width = max(len(row) for row in rows)
    result = torch.full(
        (len(rows), width), pad_id, dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(
            row, dtype=torch.long, device=device
        )
    return result


def train_supervised_latent(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    policy_path: str | Path,
    lemma_path: str | Path | None,
    output_checkpoint: str | Path,
    config: SupervisedTrainConfig | None = None,
) -> dict:
    """Warm-start latent policy/value/lemma heads from forward DAGs."""

    cfg = config or SupervisedTrainConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    tokenizer = MetamathTokenizer.load(tokenizer_path)
    model, payload = LatentProofTransformer.load_checkpoint(
        checkpoint, map_location=device
    )
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    records = _load_records(policy_path)
    if lemma_path is not None:
        records.extend({
            "state_ids": item["state_ids"],
            "action_ids": item["lemma_action_ids"],
            "value_target": item.get("utility_target", 0.5),
            "kind": "lemma",
        } for item in _load_records(lemma_path))
    records = [
        item for item in records
        if (
            len(item["state_ids"]) <= model.config.max_state_tokens
            and len(item["action_ids"]) <= model.config.max_action_tokens
        )
    ]
    random.shuffle(records)
    if cfg.max_examples is not None:
        records = records[: cfg.max_examples]
    if not records:
        raise ValueError("no compatible supervised examples")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(records)
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "ponder": 0.0}
        batches = 0
        for start in range(0, len(records), cfg.batch_size):
            batch = records[start : start + cfg.batch_size]
            states = _pad(
                [list(item["state_ids"]) for item in batch],
                tokenizer.pad_id,
                device,
            )
            actions = _pad(
                [list(item["action_ids"]) for item in batch],
                tokenizer.pad_id,
                device,
            )
            targets = torch.tensor(
                [float(item.get("value_target", 0.5)) for item in batch],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, parts = latent_policy_value_loss(
                model,
                states,
                actions,
                targets,
                LatentLossConfig(value_weight=cfg.value_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["policy"] += float(parts["policy_loss"].detach())
            totals["value"] += float(parts["value_loss"].detach())
            totals["ponder"] += float(parts["ponder_loss"].detach())
            batches += 1
        history.append({
            "epoch": epoch,
            **{key: value / max(1, batches) for key, value in totals.items()},
        })
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        destination,
        optimizer_state=optimizer.state_dict(),
        metadata={
            **payload.get("metadata", {}),
            "htps_supervised": {
                "configuration": asdict(cfg),
                "examples": len(records),
                "history": history,
            },
        },
    )
    return {
        "checkpoint": str(destination.resolve()),
        "device": str(device),
        "examples": len(records),
        "history": history,
    }


def train_latent_from_replay(
    checkpoint: str | Path,
    replay: ReplayBuffer,
    output_checkpoint: str | Path,
    config: ReplayTrainConfig | None = None,
) -> dict:
    """Train candidate policy, soft critic, latent halt and lemma gate."""

    cfg = config or ReplayTrainConfig()
    if not replay.examples:
        raise ValueError("replay buffer is empty")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    model, payload = LatentProofTransformer.load_checkpoint(
        checkpoint, map_location=device
    )
    compatible = [
        item for item in replay.examples
        if (
            len(item.state_ids) <= model.config.max_state_tokens
            and all(
                len(action) <= model.config.max_action_tokens
                for action in item.action_ids
            )
            and (item.action_ids or item.value_weight > 0)
        )
    ]
    if not compatible:
        raise ValueError("replay contains no checkpoint-compatible examples")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(compatible)
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "ponder": 0.0}
        used = 0
        for index, item in enumerate(compatible, start=1):
            state = torch.tensor(
                [item.state_ids], dtype=torch.long, device=device
            )
            memory, padding, value = model.encode(state)
            policy_loss = value.sum() * 0.0
            if item.action_ids and item.policy_weight > 0:
                candidates = _pad(
                    [list(action) for action in item.action_ids],
                    model.config.pad_id,
                    device,
                )
                logits, value = model.score_candidates_from_memory(
                    candidates, memory, padding, value
                )
                target = torch.tensor(
                    item.policy_target, dtype=logits.dtype, device=device
                )
                policy_loss = -(target * F.log_softmax(logits, dim=0)).sum()
            value_loss = value.sum() * 0.0
            if item.value_weight > 0:
                target_value = torch.tensor(
                    [item.value_target], dtype=value.dtype, device=device
                )
                value_loss = F.mse_loss(value, target_value)
            ponder = model.ponder_loss()
            loss = (
                item.policy_weight * policy_loss
                + cfg.value_weight * item.value_weight * value_loss
                + ponder
            )
            (loss / cfg.gradient_accumulation).backward()
            if index % cfg.gradient_accumulation == 0 or index == len(compatible):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals["loss"] += float(loss.detach())
            totals["policy"] += float(policy_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["ponder"] += float(ponder.detach())
            used += 1
        history.append({
            "epoch": epoch,
            **{key: value / max(1, used) for key, value in totals.items()},
        })
    model.candidate_head_trained = True
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        destination,
        optimizer_state=optimizer.state_dict(),
        metadata={
            **payload.get("metadata", {}),
            "htps_online": {
                "configuration": asdict(cfg),
                "replay_examples": len(compatible),
                "history": history,
                "targets": "minimal-proof policy plus soft HTPS critic",
            },
        },
    )
    return {
        "checkpoint": str(destination.resolve()),
        "device": str(device),
        "examples": len(compatible),
        "history": history,
    }


def collect_htps_replay(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    policy_corpus_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    config: ReplayCollectionConfig | None = None,
) -> dict:
    """Search generated training goals and retain only verified experience."""

    cfg = config or ReplayCollectionConfig()
    random.seed(cfg.seed)
    device = _device(cfg.device)
    database = parse(database_path)
    tokenizer = MetamathTokenizer.load(tokenizer_path)
    model, _ = LatentProofTransformer.load_checkpoint(
        checkpoint, map_location=device
    )
    model.to(device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    records = _load_records(policy_corpus_path)
    random.shuffle(records)
    records = records[: cfg.examples]
    replay = ReplayBuffer(cfg.capacity)
    episodes: list[dict] = []
    certified = 0
    rejected = 0
    for episode_index, record in enumerate(records):
        environment = LemmaBackwardEnvironment(
            database, excluded_assertions=record.get("excluded_labels", ())
        )
        environment.configure_from_tokenizer(
            tokenizer, target_guidance=cfg.inference_target_guidance
        )
        generator = LemmaActionGenerator(environment, LemmaGeneratorConfig())
        neural = TransformerPolicy(
            model, tokenizer.with_target_hints(cfg.model_target_hints), environment,
            device, configure_environment=False,
        )
        policy = HybridPolicy(environment, neural)
        theorem = tokenizer.theorem_from_state_tokens(
            record["state_tokens"],
            database,
            name=f"htps_{record['theorem_name']}",
        )
        realized_search = materialize_search_config(cfg.htps, episode_index)
        searcher = HyperTreeProofSearch(
            environment, generator, policy, realized_search
        )
        policy_before = neural.metrics()
        started = time.perf_counter()
        result, metrics = searcher.prove(ProofState.from_theorem(theorem))
        wall_seconds = time.perf_counter() - started
        policy_after = neural.metrics()
        certificate_ok = False
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
                    name=f"htps_cert_{record['theorem_name']}",
                )
                certificate_ok, verification = verification_record(
                    certificate, database, database_path,
                    external_verifier=cfg.external_verifier,
                    require_external=cfg.require_external_verification,
                    timeout_seconds=cfg.external_timeout_seconds,
                )
                certificate_steps = len(certificate.proof.source_labels)
                certified += int(certificate_ok)
                if not certificate_ok:
                    rejected += 1
                    error = "required external verification did not pass"
            except Exception as exc:
                error = str(exc)
                rejected += 1
        if certificate_ok or not result.search.solved:
            replay.add_mcts(
                result,
                tokenizer,
                environment,
                max_state_tokens=model.config.max_state_tokens,
                max_action_tokens=model.config.max_action_tokens,
            )
        episodes.append({
            "theorem_name": record["theorem_name"],
            "proof_depth": record.get("proof_depth"),
            "certified": certificate_ok,
            "verification": verification,
            "environment": environment.configuration_record(),
            "certificate_error": error,
            "outcome": result.outcome,
            "abstract_actions": len(result.search.actions),
            "certificate_steps": certificate_steps,
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
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay.save(destination)
    summary = {
        "format": "peano-htps-replay-v1",
        "configuration": asdict(cfg),
        "device": str(device),
        "attempted": len(records),
        "certified": certified,
        "rejected_certificates": rejected,
        "replay_examples": len(replay),
        "episodes": episodes,
        "replay": str(destination.resolve()),
    }
    destination.with_suffix(destination.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

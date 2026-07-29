from __future__ import annotations

import gzip
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import ProofTransformer, ProofTransformerConfig
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class ScaleTrainingConfig:
    max_steps: int = 100
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    value_loss_weight: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 20260729
    device: str = "auto"
    d_model: int = 768
    nhead: int = 12
    encoder_layers: int = 6
    decoder_layers: int = 6
    dim_feedforward: int = 3072
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    initial_context_tokens: int = 512
    context_warmup_steps: int = 50
    shuffle_buffer: int = 4096
    checkpoint_every: int = 25
    validation_batches: int = 4
    require_long_context_step: bool = True
    long_context_threshold: int = 2048


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffled_records(
    corpus: Path,
    split: str,
    seed: int,
    buffer_size: int,
) -> Iterator[dict]:
    manifest = json.loads(
        (corpus / "manifest.json").read_text(encoding="utf-8")
    )
    epoch = 0
    while True:
        rng = random.Random(seed + epoch * 1_000_003)
        shards = list(manifest["shards"][split])
        rng.shuffle(shards)
        buffer: list[dict] = []
        for shard in shards:
            with gzip.open(
                corpus / shard["path"],
                "rt",
                encoding="utf-8",
            ) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    buffer.append(json.loads(line))
                    if len(buffer) >= buffer_size:
                        index = rng.randrange(len(buffer))
                        buffer[index], buffer[-1] = (
                            buffer[-1],
                            buffer[index],
                        )
                        yield buffer.pop()
        while buffer:
            index = rng.randrange(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()
        epoch += 1


def _pad(sequences: list[list[int]], pad_id: int) -> Tensor:
    width = max(len(sequence) for sequence in sequences)
    result = torch.full(
        (len(sequences), width),
        pad_id,
        dtype=torch.long,
    )
    for row, sequence in enumerate(sequences):
        result[row, :len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
        )
    return result


def _collate(records: list[dict], pad_id: int) -> dict[str, Tensor]:
    actions = _pad([record["action"] for record in records], pad_id)
    return {
        "state": _pad(
            [record["state"] for record in records],
            pad_id,
        ),
        "action_input": actions[:, :-1],
        "action_target": actions[:, 1:],
        "value_target": torch.tensor(
            [float(record["value"]) for record in records],
            dtype=torch.float32,
        ),
    }


def _next_batch(
    records: Iterator[dict],
    batch_size: int,
    pad_id: int,
    maximum_tokens: int,
    *,
    minimum_tokens: int = 0,
) -> dict[str, Tensor]:
    selected: list[dict] = []
    while len(selected) < batch_size:
        record = next(records)
        length = max(len(record["state"]), len(record["action"]))
        if length > maximum_tokens or length < minimum_tokens:
            continue
        selected.append(record)
    return _collate(selected, pad_id)


def _context_limit(
    step: int,
    maximum: int,
    config: ScaleTrainingConfig,
) -> int:
    initial = min(maximum, config.initial_context_tokens)
    if config.context_warmup_steps <= 1:
        return maximum
    progress = min(1.0, step / (config.context_warmup_steps - 1))
    return int(round(initial + progress * (maximum - initial)))


def _learning_rate_factor(
    step: int,
    config: ScaleTrainingConfig,
) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return max(1.0e-6, (step + 1) / config.warmup_steps)
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(
        1.0,
        max(0.0, (step - config.warmup_steps) / decay_steps),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.min_learning_rate_ratio
        + (1.0 - config.min_learning_rate_ratio) * cosine
    )


def _evaluate(
    model: ProofTransformer,
    records: Iterator[dict],
    tokenizer: MetamathTokenizer,
    device: torch.device,
    config: ScaleTrainingConfig,
    maximum_tokens: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "correct": 0.0,
        "tokens": 0.0,
        "exact": 0.0,
        "examples": 0.0,
    }
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(config.validation_batches):
            batch = _next_batch(
                records,
                config.micro_batch_size,
                tokenizer.pad_id,
                maximum_tokens,
            )
            state = batch["state"].to(device)
            action_input = batch["action_input"].to(device)
            action_target = batch["action_target"].to(device)
            value_target = batch["value_target"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                logits, value = model(state, action_input)
                policy_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    action_target.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
                value_loss = F.mse_loss(value, value_target)
                loss = (
                    policy_loss
                    + config.value_loss_weight * value_loss
                )
            mask = action_target.ne(tokenizer.pad_id)
            predictions = logits.argmax(dim=-1)
            correct = (predictions == action_target) | ~mask
            totals["loss"] += float(loss.item()) * len(state)
            totals["policy_loss"] += (
                float(policy_loss.item()) * len(state)
            )
            totals["value_loss"] += (
                float(value_loss.item()) * len(state)
            )
            totals["correct"] += float(
                ((predictions == action_target) & mask).sum().item()
            )
            totals["tokens"] += float(mask.sum().item())
            totals["exact"] += float(correct.all(dim=1).sum().item())
            totals["examples"] += float(len(state))
    model.train()
    average_policy = (
        totals["policy_loss"] / max(1.0, totals["examples"])
    )
    return {
        "loss": totals["loss"] / max(1.0, totals["examples"]),
        "policy_loss": average_policy,
        "value_loss":
            totals["value_loss"] / max(1.0, totals["examples"]),
        "perplexity": math.exp(min(20.0, average_policy)),
        "token_accuracy": totals["correct"] / max(1.0, totals["tokens"]),
        "exact_action_accuracy":
            totals["exact"] / max(1.0, totals["examples"]),
    }


def _write_metrics(output: Path, payload: dict) -> None:
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_scale_model(
    corpus_directory: str | Path,
    output_directory: str | Path,
    config: ScaleTrainingConfig | None = None,
    *,
    resume_from: str | Path | None = None,
) -> dict:
    """Train/resume the first ~100M formal proof Transformer."""

    cfg = config or ScaleTrainingConfig()
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if cfg.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if cfg.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    _seed_everything(cfg.seed)
    corpus = Path(corpus_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    manifest = json.loads(
        (corpus / "manifest.json").read_text(encoding="utf-8")
    )
    maximum_state = int(
        manifest["configuration"]["max_state_tokens"]
    )
    maximum_action = int(
        manifest["configuration"]["max_action_tokens"]
    )
    maximum_context = max(maximum_state, maximum_action)
    if maximum_context <= cfg.long_context_threshold:
        raise ValueError(
            "scale corpus does not provide a context above "
            f"{cfg.long_context_threshold} tokens"
        )
    device = _device(cfg.device)
    model_config = ProofTransformerConfig(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.encoder_layers,
        num_decoder_layers=cfg.decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_state_tokens=maximum_state,
        max_action_tokens=maximum_action,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    start_step = 0
    resume_payload: dict | None = None
    if resume_from is not None:
        model, resume_payload = ProofTransformer.load_checkpoint(
            resume_from,
            map_location="cpu",
        )
        if model.config != model_config:
            raise ValueError(
                "resume checkpoint model configuration does not match"
            )
        start_step = int(
            resume_payload.get("metadata", {})
            .get("training_state", {})
            .get("step", 0)
        )
    else:
        model = ProofTransformer(model_config)
    parameter_count = model.parameter_count()
    if not 80_000_000 <= parameter_count <= 130_000_000:
        raise ValueError(
            "scale architecture is outside the intended 100M class: "
            f"{parameter_count:,} parameters"
        )
    model.to(device)
    model.train()
    optimizer_kwargs = {
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(step, cfg),
    )
    if resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = (
            resume_payload.get("metadata", {})
            .get("training_state", {})
            .get("scheduler_state")
        )
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    train_records = _shuffled_records(
        corpus,
        "train",
        cfg.seed,
        cfg.shuffle_buffer,
    )
    validation_records = _shuffled_records(
        corpus,
        "validation",
        cfg.seed + 17,
        max(1, min(cfg.shuffle_buffer, 512)),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history: list[dict] = []
    total_examples = 0
    total_target_tokens = 0
    latest_path = output / "latest.pt"
    for step in range(start_step, cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        limit = _context_limit(step, maximum_context, cfg)
        step_loss = 0.0
        step_policy_loss = 0.0
        step_value_loss = 0.0
        step_tokens = 0
        step_examples = 0
        long_context_used = False
        for micro_step in range(cfg.gradient_accumulation_steps):
            require_long = (
                cfg.require_long_context_step
                and step == cfg.max_steps - 1
                and micro_step == 0
            )
            batch = _next_batch(
                train_records,
                cfg.micro_batch_size,
                tokenizer.pad_id,
                maximum_context if require_long else limit,
                minimum_tokens=(
                    cfg.long_context_threshold + 1
                    if require_long else 0
                ),
            )
            state = batch["state"].to(device)
            action_input = batch["action_input"].to(device)
            action_target = batch["action_target"].to(device)
            value_target = batch["value_target"].to(device)
            long_context_used = long_context_used or max(
                state.shape[1], action_input.shape[1] + 1
            ) > cfg.long_context_threshold
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, value = model(state, action_input)
                policy_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    action_target.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
                value_loss = F.mse_loss(value, value_target)
                loss = (
                    policy_loss
                    + cfg.value_loss_weight * value_loss
                )
                scaled_loss = (
                    loss / cfg.gradient_accumulation_steps
                )
            scaled_loss.backward()
            tokens = int(
                action_target.ne(tokenizer.pad_id).sum().item()
            )
            step_loss += float(loss.item())
            step_policy_loss += float(policy_loss.item())
            step_value_loss += float(value_loss.item())
            step_tokens += tokens
            step_examples += len(state)
            del logits, value, loss, scaled_loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.gradient_clip,
        )
        optimizer.step()
        scheduler.step()
        total_examples += step_examples
        total_target_tokens += step_tokens
        elapsed = time.perf_counter() - started
        record = {
            "step": step + 1,
            "context_limit": limit,
            "long_context_used": long_context_used,
            "loss":
                step_loss / cfg.gradient_accumulation_steps,
            "policy_loss":
                step_policy_loss / cfg.gradient_accumulation_steps,
            "value_loss":
                step_value_loss / cfg.gradient_accumulation_steps,
            "gradient_norm": float(gradient_norm.item()),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "examples_per_second": total_examples / elapsed,
            "target_tokens_per_second": total_target_tokens / elapsed,
        }
        history.append(record)
        should_checkpoint = (
            (step + 1) % cfg.checkpoint_every == 0
            or step + 1 == cfg.max_steps
        )
        if should_checkpoint:
            training_state = {
                "step": step + 1,
                "scheduler_state": scheduler.state_dict(),
            }
            model.save_checkpoint(
                latest_path,
                optimizer_state=optimizer.state_dict(),
                metadata={
                    "training_state": training_state,
                    "latest_metrics": record,
                    "parameter_count": parameter_count,
                    "tokenizer": str(
                        (corpus / "tokenizer.json").resolve()
                    ),
                },
            )
            _write_metrics(output, {
                "format": "peano-scale-training-v1",
                "status": "running",
                "configuration": asdict(cfg),
                "model_configuration": asdict(model_config),
                "parameter_count": parameter_count,
                "history": history,
            })
    validation = _evaluate(
        model,
        validation_records,
        tokenizer,
        device,
        cfg,
        maximum_context,
    )
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0
    )
    final_path = output / "final.pt"
    model.save_checkpoint(
        final_path,
        metadata={
            "training_state": {"step": cfg.max_steps},
            "validation": validation,
            "parameter_count": parameter_count,
            "tokenizer": str(
                (corpus / "tokenizer.json").resolve()
            ),
        },
    )
    summary = {
        "format": "peano-scale-training-v1",
        "status": "complete",
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "configuration": asdict(cfg),
        "model_configuration": asdict(model_config),
        "parameter_count": parameter_count,
        "start_step": start_step,
        "completed_steps": cfg.max_steps,
        "train_examples_seen": total_examples,
        "target_tokens_seen": total_target_tokens,
        "elapsed_seconds": elapsed,
        "examples_per_second": total_examples / max(elapsed, 1.0e-9),
        "target_tokens_per_second":
            total_target_tokens / max(elapsed, 1.0e-9),
        "peak_cuda_memory_bytes": peak_memory,
        "validation": validation,
        "long_context_training_exercised": any(
            record["long_context_used"] for record in history
        ),
        "history": history,
        "resume_checkpoint": str(latest_path.resolve()),
        "final_checkpoint": str(final_path.resolve()),
    }
    _write_metrics(output, summary)
    return summary


def evaluate_scale_checkpoint(
    checkpoint: str | Path,
    corpus_directory: str | Path,
    destination: str | Path,
    *,
    split: str = "test",
    examples: int = 256,
    device_name: str = "auto",
    seed: int = 20260801,
) -> dict:
    """Teacher-force a deterministic held-out scale sample."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    corpus = Path(corpus_directory)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    device = _device(device_name)
    model, payload = ProofTransformer.load_checkpoint(
        checkpoint,
        map_location=device,
    )
    model.to(device)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint and corpus vocabularies differ")
    records = _shuffled_records(corpus, split, seed, 512)
    evaluation_config = ScaleTrainingConfig(
        micro_batch_size=1,
        validation_batches=examples,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = _evaluate(
        model,
        records,
        tokenizer,
        device,
        evaluation_config,
        max(
            model.config.max_state_tokens,
            model.config.max_action_tokens,
        ),
    )
    elapsed = time.perf_counter() - started
    report = {
        "format": "peano-scale-evaluation-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "corpus": str(corpus.resolve()),
        "split": split,
        "examples": examples,
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "parameter_count": model.parameter_count(),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1.0e-9),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "checkpoint_metadata": payload.get("metadata", {}),
    }
    Path(destination).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import ProverExample, load_examples
from .model import ProofTransformer, ProofTransformerConfig
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    value_loss_weight: float = 0.25
    candidate_loss_weight: float = 0.25
    pa_plus_balanced_sampling: bool = True
    gradient_clip: float = 1.0
    seed: int = 7
    device: str = "auto"
    num_workers: int = 0
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1


class ProofDataset(Dataset):
    def __init__(self, examples: list[ProverExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ProverExample:
        return self.examples[index]


def _pad(sequences: list[tuple[int, ...]], pad_id: int) -> Tensor:
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


def _collate(
    examples: list[ProverExample],
    pad_id: int,
) -> dict[str, Tensor]:
    actions = _pad([example.action_ids for example in examples], pad_id)
    return {
        "state": _pad(
            [example.state_ids for example in examples],
            pad_id,
        ),
        "action_input": actions[:, :-1],
        "action_target": actions[:, 1:],
        "action_full": actions,
        "value_target": torch.tensor(
            [example.value_target for example in examples],
            dtype=torch.float32,
        ),
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(
    model: ProofTransformer,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    value_weight: float,
    candidate_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "candidate_loss": 0.0,
        "candidate_correct": 0.0,
        "correct_tokens": 0.0,
        "tokens": 0.0,
        "examples": 0.0,
    }
    for batch in loader:
        state = batch["state"].to(device)
        action_input = batch["action_input"].to(device)
        action_target = batch["action_target"].to(device)
        action_full = batch["action_full"].to(device)
        value_target = batch["value_target"].to(device)
        with torch.set_grad_enabled(training):
            logits, value = model(state, action_input)
            policy_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                action_target.reshape(-1),
                ignore_index=pad_id,
            )
            value_loss = F.mse_loss(value, value_target)
            candidate_scores, _ = model.score_candidate_matrix(
                state, action_full
            )
            duplicate_actions = action_full[:, None, :].eq(
                action_full[None, :, :]
            ).all(dim=-1)
            positive_scores = candidate_scores.masked_fill(
                ~duplicate_actions,
                float("-inf"),
            )
            candidate_loss = -(
                torch.logsumexp(positive_scores, dim=1)
                - torch.logsumexp(candidate_scores, dim=1)
            ).mean()
            loss = (
                policy_loss
                + value_weight * value_loss
                + candidate_weight * candidate_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                )
                optimizer.step()
        mask = action_target.ne(pad_id)
        predictions = logits.argmax(dim=-1)
        token_count = int(mask.sum().item())
        totals["loss"] += float(loss.item()) * len(state)
        totals["policy_loss"] += (
            float(policy_loss.item()) * len(state)
        )
        totals["value_loss"] += float(value_loss.item()) * len(state)
        totals["candidate_loss"] += (
            float(candidate_loss.item()) * len(state)
        )
        best_candidates = candidate_scores.argmax(dim=1)
        totals["candidate_correct"] += int(
            duplicate_actions[
                torch.arange(len(state), device=device),
                best_candidates,
            ].sum().item()
        )
        totals["correct_tokens"] += int(
            ((predictions == action_target) & mask).sum().item()
        )
        totals["tokens"] += token_count
        totals["examples"] += len(state)
    denominator = max(1.0, totals["examples"])
    return {
        "loss": totals["loss"] / denominator,
        "policy_loss": totals["policy_loss"] / denominator,
        "value_loss": totals["value_loss"] / denominator,
        "candidate_loss": totals["candidate_loss"] / denominator,
        "candidate_accuracy": totals["candidate_correct"] / denominator,
        "token_accuracy": totals["correct_tokens"]
        / max(1.0, totals["tokens"]),
    }


def train_model(
    corpus_directory: str | Path,
    output_directory: str | Path,
    config: TrainingConfig | None = None,
) -> dict:
    cfg = config or TrainingConfig()
    _seed_everything(cfg.seed)
    corpus = Path(corpus_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    manifest = json.loads(
        (corpus / "manifest.json").read_text(encoding="utf-8")
    )
    train_examples = load_examples(corpus / "train.jsonl")
    validation_examples = load_examples(corpus / "validation.jsonl")
    if not train_examples:
        raise ValueError("training corpus is empty")
    if not validation_examples:
        raise ValueError("validation corpus is empty")
    device = _device(cfg.device)
    model_cfg = ProofTransformerConfig(
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
        max_state_tokens=int(
            manifest["configuration"]["max_state_tokens"]
        ),
        max_action_tokens=int(
            manifest["configuration"]["max_action_tokens"]
        ),
    )
    model = ProofTransformer(model_cfg).to(device)
    model.candidate_head_trained = cfg.candidate_loss_weight > 0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    collate = lambda examples: _collate(examples, tokenizer.pad_id)
    sampler = None
    if cfg.pa_plus_balanced_sampling and tokenizer.pa_plus_context.enabled:
        weights = {
            "source": 0.5,
            "definition_bridge": 0.75,
            "bounded_instance": 3.0,
            "search": 1.0,
        }
        sampler = WeightedRandomSampler(
            [
                weights.get(example.generation_kind, 1.0)
                for example in train_examples
            ],
            num_samples=len(train_examples),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        ProofDataset(train_examples),
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        ProofDataset(validation_examples),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_path = output / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            device,
            tokenizer.pad_id,
            cfg.value_loss_weight,
            cfg.candidate_loss_weight,
            optimizer,
            cfg.gradient_clip,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            device,
            tokenizer.pad_id,
            cfg.value_loss_weight,
            cfg.candidate_loss_weight,
            None,
            cfg.gradient_clip,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            model.save_checkpoint(
                best_path,
                optimizer_state=optimizer.state_dict(),
                metadata={
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "tokenizer": str(
                        (corpus / "tokenizer.json").resolve()
                    ),
                },
            )
    final_path = output / "final.pt"
    model.save_checkpoint(
        final_path,
        optimizer_state=optimizer.state_dict(),
        metadata={"epoch": cfg.epochs, "tokenizer": "tokenizer.json"},
    )
    summary = {
        "format": "peano-proof-training-v1",
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "configuration": asdict(cfg),
        "model_configuration": asdict(model_cfg),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "best_validation_loss": best_loss,
        "history": history,
        "best_checkpoint": str(best_path.resolve()),
        "final_checkpoint": str(final_path.resolve()),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

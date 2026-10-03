"""Standard next-token pretraining and action-masked agent SFT.

Both routes use the same shifted causal LM loss.  There is no bidirectional
attention or latent thought supervision.  Memory encodings are recomputed with
current weights so gradients reach the lemma decoder.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .agent import AgentAction, AgentBudget, AgentSession
from .kernel import theorem_from_data
from .model import save_checkpoint
from .tokenizer import ByteTokenizer


@dataclass
class EncodedSample:
    ids: list[int]
    labels: list[int]
    memories: dict[int, list[int]]


def action_text(action):
    data = action.to_dict() if isinstance(action, AgentAction) else action
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def encode_segments(segments, tokenizer, slots, action=None, *, action_only=False):
    ids = [tokenizer.bos_id]
    memories = {}
    for segment in segments:
        if isinstance(segment, str):
            ids.extend(tokenizer.encode(segment))
        else:
            memories[len(ids)] = tokenizer.encode(segment["text"], bos=True, eos=True)
            ids.extend([tokenizer.mem_id] * slots)
    labels = [-100] * len(ids) if action_only else list(ids)
    if action is not None:
        prefix = tokenizer.encode("\nNEXT_ACTION ")
        ids.extend(prefix)
        labels.extend([-100] * len(prefix) if action_only else prefix)
        suffix = tokenizer.encode(action_text(action) + "\n", eos=True)
        ids.extend(suffix)
        labels.extend(suffix)
    else:
        ids.append(tokenizer.eos_id)
        labels.append(tokenizer.eos_id)
    for start in memories:
        labels[start:start + slots] = [-100] * slots
    labels[0] = -100
    return EncodedSample(ids, labels, memories)


def collate(samples, device="cpu"):
    if not samples:
        raise ValueError("empty batch")
    width = max(len(s.ids) for s in samples)
    ids = torch.zeros((len(samples), width), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    memories = {}
    for batch, sample in enumerate(samples):
        length = len(sample.ids)
        ids[batch, :length] = torch.tensor(sample.ids, device=device)
        labels[batch, :length] = torch.tensor(sample.labels, device=device)
        mask[batch, :length] = True
        for start, text in sample.memories.items():
            memories[(batch, start)] = text
    return {"input_ids": ids, "labels": labels, "attention_mask": mask, "memories": memories}


def ntp_blocks(sample, block_size, slots):
    """Predict each language target once while retaining complete memory blocks.

    Usually adjacent blocks overlap by one token. At a memory boundary they
    overlap by its complete slot block instead; these repeated slots carry no
    targets. If consecutive memories cannot fit together, the next block may
    start directly at the next MEM (also not a language target).
    """
    if block_size < slots + 2:
        raise ValueError("block size cannot hold a memory block")
    start = 0
    while start < len(sample.ids) - 1:
        end = min(len(sample.ids), start + block_size)
        for pos in sample.memories:
            if pos < end < pos + slots:
                end = pos
        if end <= start + 1:
            raise ValueError("memory block crosses an unusable NTP boundary")
        memories = {pos - start: value for pos, value in sample.memories.items() if start <= pos and pos + slots <= end}
        labels = sample.labels[start:end]
        labels[0] = -100
        yield EncodedSample(sample.ids[start:end], labels, memories)
        if end == len(sample.ids):
            break
        # Keep the last memory whole; retreating across *all* adjacent memory
        # ranges could return to the same start forever on a long MEM run.
        overlap = end - 1
        for pos in sample.memories:
            if pos <= overlap < pos + slots:
                overlap = pos
                break
        if overlap <= start:
            if end in sample.memories:
                overlap = end
            else:
                raise ValueError("NTP boundary cannot advance without losing a language target")
        start = overlap


def build_training_samples(episodes, kernel, library, config, mode="ntp"):
    if mode not in {"ntp", "sft"}:
        raise ValueError("mode must be ntp or sft")
    tokenizer = ByteTokenizer()
    samples, stats = [], {"episodes": 0, "samples": 0, "filtered_long_context": 0, "filtered_long_lemma": 0,
                          "invalid_actions_in_context": 0, "memory_samples": 0,
                          "candidate_samples": 0, "filtered_no_targets": 0,
                          "source_language_targets": 0, "trained_language_targets": 0,
                          "filtered_language_targets": 0}

    def targets(sample):
        return sum(label == tokenizer.eos_id or label >= tokenizer.byte_offset for label in sample.labels[1:])

    def accept(sample, *, context_limit):
        stats["candidate_samples"] += 1
        count = targets(sample)
        if context_limit and len(sample.ids) > config.max_seq_len:
            stats["filtered_long_context"] += 1
        elif any(len(text) + config.lemma_slots > config.max_seq_len for text in sample.memories.values()):
            stats["filtered_long_lemma"] += 1
        elif not count:
            stats["filtered_no_targets"] += 1
        else:
            samples.append(sample)
            stats["trained_language_targets"] += count
            return
        stats["filtered_language_targets"] += count

    for episode in episodes:
        if episode.get("split") != "train":
            raise ValueError("training accepts only train episodes")
        target = theorem_from_data(episode["certificate"])
        kernel.verify(target)
        actions = [AgentAction.from_dict(a) for a in episode["actions"]]
        session = AgentSession(kernel, target, library, AgentBudget(max_steps=max(128, len(actions) + 8)))
        for action in actions:
            if session.status != "running":
                raise ValueError("teacher episode contains actions after a terminal state")
            # Capture only the prefix that the actor could see at this step.
            sample = encode_segments(session.context_segments(), tokenizer, config.lemma_slots, action, action_only=True)
            observation = session.step(action)
            if observation.status == "invalid":
                stats["invalid_actions_in_context"] += 1
            elif mode == "sft" and observation.status in {"running", "success"}:
                stats["source_language_targets"] += targets(sample)
                accept(sample, context_limit=True)
        if session.status != "success":
            raise ValueError("unverified teacher episode in training corpus")
        if mode == "ntp":
            # context_segments uses an initial header and chronological events;
            # the current workspace summary appears only after those events.
            full = encode_segments(session.context_segments(), tokenizer, config.lemma_slots)
            stats["source_language_targets"] += targets(full)
            for sample in ntp_blocks(full, config.max_seq_len, config.lemma_slots):
                accept(sample, context_limit=False)
        stats["episodes"] += 1
    stats["samples"] = len(samples)
    stats["memory_samples"] = sum(bool(s.memories) for s in samples)
    if not samples:
        raise ValueError(f"no trainable samples: {stats}")
    return samples, stats


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.steps, int) or not isinstance(self.batch_size, int) or min(self.steps, self.batch_size) <= 0:
            raise ValueError("positive integer training steps and batch_size required")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.grad_clip)):
            raise ValueError("training hyperparameters must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("positive learning rate/gradient clip and nonnegative weight decay required")


def train_language_model(model, samples, output_directory, config=None, *, mode="ntp", metadata=None):
    cfg = config or TrainConfig()
    if mode not in {"ntp", "sft"}:
        raise ValueError("mode must be ntp or sft")
    if cfg.steps <= 0 or cfg.batch_size <= 0 or not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        raise ValueError("invalid training budget or learning rate")
    if not samples:
        raise ValueError("empty training samples")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model.to(cfg.device).train()
    torch.manual_seed(cfg.seed)
    randomizer = random.Random(cfg.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses, lemma_gradient_steps = [], 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(cfg.steps):
            batch_samples = [samples[randomizer.randrange(len(samples))] for _ in range(cfg.batch_size)]
            batch = collate(batch_samples, cfg.device)
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            loss = result.loss
            if loss is None or not torch.isfinite(loss):
                raise ValueError("nonfinite language modeling loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            has_lemma_grad = any("lemma" in name and p.grad is not None and bool(p.grad.abs().sum()) for name, p in model.named_parameters())
            lemma_gradient_steps += int(has_lemma_grad)
            optimizer.step()
            record = {"step": step + 1, "mode": mode, "loss": float(loss.detach()), "grad_norm": float(norm),
                      "tokens": int((batch["labels"][:, 1:] >= 0).sum()), "lemma_gradient": has_lemma_grad}
            losses.append(record["loss"])
            log.write(json.dumps(record) + "\n")
            log.flush()
    summary = {"mode": mode, "steps": cfg.steps, "first_loss": losses[0], "last_loss": losses[-1],
               "mean_loss": sum(losses) / len(losses), "lemma_gradient_steps": lemma_gradient_steps,
               "train_config": asdict(cfg), **(metadata or {})}
    save_checkpoint(output / "model.pt", model, metadata=summary)
    # Optimizer state is explicit; model checkpoints do not pretend to support exact resume.
    torch.save({"optimizer": optimizer.state_dict(), "step": cfg.steps, "torch_rng": torch.get_rng_state(),
                "sampler_rng": randomizer.getstate()}, output / "optimizer.pt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

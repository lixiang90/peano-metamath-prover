"""Neural autoregressive scoring of bounded, symbolically proposed agent actions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .agent import AgentAction, AgentBudget, AgentSession
from .tokenizer import ByteTokenizer
from .training import EncodedSample, collate, encode_segments


@dataclass(frozen=True)
class PolicyConfig:
    max_candidates: int = 16
    temperature: float = 1.0
    length_penalty: float = 1.0
    exploration: float = 0.1
    score_batch_size: int = 4

    def __post_init__(self):
        if self.max_candidates < 1 or self.score_batch_size < 1:
            raise ValueError("positive candidate/scoring budgets required")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not 0 <= self.exploration < 1 or not 0 <= self.length_penalty <= 1:
            raise ValueError("invalid exploration or length penalty")


@dataclass
class Decision:
    samples: list[EncodedSample]
    actions: list[dict]
    selected: int
    old_log_prob: float


@dataclass
class Rollout:
    decisions: list[Decision] = field(default_factory=list)
    reward: float = 0.0
    status: str = "unknown"
    reason: str = ""
    session: AgentSession | None = None


def action_distribution(model, samples, policy=None):
    """A categorical policy over AR sequence scores, including exploration.

    length_penalty=0 is the conditional LM probability over this finite action
    set.  The default 1 ranks by mean token log-probability to avoid a systematic
    preference for short control actions.  RL optimizes this exact categorical
    policy, not an incorrectly substituted token-policy likelihood.
    """
    cfg = policy or PolicyConfig()
    if not samples:
        raise ValueError("no actions to score")
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(samples), cfg.score_batch_size):
        batch = collate(samples[start:start + cfg.score_batch_size], device)
        log_probs = model.sequence_log_probs(**batch, reduction="sum")
        lengths = (batch["labels"][:, 1:] >= 0).sum(dim=1).clamp_min(1)
        scores.append(log_probs / lengths.pow(cfg.length_penalty))
    logits = torch.cat(scores) / cfg.temperature
    probs = torch.softmax(logits.float(), dim=0)
    probs = probs * (1 - cfg.exploration) + cfg.exploration / len(samples)
    return probs


def prepare_actions(session, model, policy):
    if session.status != "running":
        return [], [], 0
    candidates = session.candidate_actions(limit=policy.max_candidates)
    segments, tokenizer = session.context_segments(), ByteTokenizer()
    actions, samples = [], []
    rejected = 0
    for action in candidates:
        sample = encode_segments(segments, tokenizer, model.config.lemma_slots, action, action_only=True)
        if len(sample.ids) > model.config.max_seq_len or any(
            len(text) + model.config.lemma_slots > model.config.max_seq_len for text in sample.memories.values()
        ):
            rejected += 1
            continue
        actions.append(action)
        samples.append(sample)
    return actions, samples, rejected


def rollout(model, kernel, target, library=None, *, budget=None, policy=None, seed=7, greedy=False):
    cfg = policy or PolicyConfig()
    session = AgentSession(kernel, target, library, budget or AgentBudget())
    result = Rollout(session=session)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(session.budget.max_steps):
            actions, samples, rejected = prepare_actions(session, model, cfg)
            if not actions:
                result.reason = "context_budget" if rejected else "no_candidates"
                break
            with torch.no_grad():
                probabilities = action_distribution(model, samples, cfg)
                index = int(probabilities.argmax()) if greedy else int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
                old_log_prob = float(probabilities[index].log())
            result.decisions.append(Decision(samples, [a.to_dict() for a in actions], index, old_log_prob))
            observation = session.step(actions[index])
            if session.status == "success":
                # FINISH has replayed the full target certificate.  No model or
                # shaping score can bypass this verifiable reward boundary.
                kernel.verify(session.certificate)
                result.reward, result.status = 1.0, "success"
                result.reason = "verified_certificate"
                break
            if session.status == "unknown":
                result.reason = observation.message
                break
        else:
            result.reason = "step_budget"
    finally:
        model.train(was_training)
    return result


def rollout_record(result):
    return {"status": result.status, "reason": result.reason, "reward": result.reward,
            "steps": len(result.decisions), "certification": result.session.certification,
            "actions": [d.actions[d.selected] for d in result.decisions],
            "trace": result.session.trace}

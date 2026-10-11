"""Online group-relative policy gradients with official proof rewards.

No benchmark targets are consumed. Failed bounded attempts receive no reward,
not an unprovability label; verifier failures censor the entire update group.
The cost value head retains its supervised meaning and is not fit to failures.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from metamath_generator.parser import parse
from .certificate import compile_certificate, export_certificate
from .data_contract import validate_checkpoint_tokenizer
from .environment import BackwardEnvironment, ProofState
from .hybrid import HybridActionGenerator, HybridGeneratorConfig
from .model import ProofTransformer
from .rl import training_candidates, sample_training_targets
from .search import SearchResult
from .tokenizer import MetamathTokenizer
from .train import _device, _pad
from .trajectories import tactic_tokens
from .verification import verification_record


@dataclass(frozen=True)
class RLVRConfig:
    groups: int = 8
    group_size: int = 4
    max_steps: int = 24
    candidate_limit: int = 32
    seconds_per_rollout: float = 10.0
    learning_rate: float = 1e-5
    clip_epsilon: float = .2
    kl_coefficient: float = .01
    update_epochs: int = 1
    seed: int = 7
    device: str = "auto"
    external_verifier: str | None = None
    verification_timeout: float = 30.0

    def __post_init__(self):
        if min(self.groups, self.group_size - 1, self.max_steps, self.candidate_limit, self.update_epochs) <= 0:
            raise ValueError("positive budgets and group_size >= 2 required")
        numbers = (self.seconds_per_rollout, self.learning_rate, self.clip_epsilon, self.verification_timeout)
        if any(not math.isfinite(x) or x <= 0 for x in numbers) or self.clip_epsilon >= 1:
            raise ValueError("invalid RLVR budget or learning rate/clip")
        if not math.isfinite(self.kl_coefficient) or self.kl_coefficient < 0:
            raise ValueError("invalid KL coefficient")


def distribution(model, state_ids, actions):
    device = next(model.parameters()).device
    state = torch.tensor([state_ids], device=device, dtype=torch.long)
    candidates = _pad(actions, model.config.pad_id).to(device)
    scores, _ = model.score_candidate_matrix(state, candidates)
    return scores[0].float().log_softmax(dim=0)


def rollout(model, tokenizer, environment, theorem, cfg, database_path, certificate_path=None):
    state = ProofState.from_theorem(theorem)
    trace, decisions = [], []
    deadline = time.monotonic() + cfg.seconds_per_rollout
    generator = HybridActionGenerator(environment, HybridGeneratorConfig(
        max_deterministic_tactics=cfg.candidate_limit,
        max_construction_tactics=cfg.candidate_limit,
        deterministic_candidates_per_variable=4, construction_candidates_per_variable=4,
    ))
    outcome = "unknown_step_budget"
    for _ in range(cfg.max_steps):
        if time.monotonic() >= deadline:
            outcome = "unknown_time_budget"
            break
        try:
            state_ids = tokenizer.encode(tokenizer.proof_state_tokens(state))
        except ValueError:
            outcome = "unknown_encoding_limit"
            break
        if len(state_ids) > model.config.max_state_tokens:
            outcome = "unknown_encoding_limit"
            break
        options = []
        for tactic in generator.actions(state).all:
            try:
                ids = tokenizer.encode(tactic_tokens(tactic, state, tokenizer, environment))
                after = environment.apply(state, tactic)
                if len(ids) <= model.config.max_action_tokens:
                    options.append((tactic, ids, after))
            except ValueError:
                continue
            if len(options) >= cfg.candidate_limit:
                break
        if not options:
            outcome = "unknown_candidate_limit"
            break
        if time.monotonic() >= deadline:
            outcome = "unknown_time_budget"
            break
        with torch.no_grad():
            logp = distribution(model, state_ids, [a[1] for a in options])
            index = int(torch.multinomial(logp.exp(), 1).item())
        decisions.append(dict(state=state_ids, actions=[a[1] for a in options], selected=index,
                              old_log_probs=logp.cpu().tolist()))
        transition = options[index][2]
        trace.append(transition)
        state = transition.after
        if state.solved:
            break
    verification = None
    reward = 0.0
    if state.solved:
        result = SearchResult(True, len(trace), len(trace) + 1, tuple(t.tactic for t in trace), tuple(trace), state)
        try:
            cert = compile_certificate(theorem, result, environment.database)
            passed, verification = verification_record(
                cert, environment.database, database_path, external_verifier=cfg.external_verifier,
                require_external=True, timeout_seconds=cfg.verification_timeout,
            )
            if passed:
                reward, outcome = 1.0, "officially_verified"
                if certificate_path is not None:
                    export_certificate(cert, certificate_path, ambient_database=environment.database)
            else:
                reward, outcome = None, "verification_rejected_or_unavailable"
        except Exception as exc:
            reward, outcome = None, "verification_error"
            verification = {"error": str(exc)}
    return dict(reward=reward, outcome=outcome, decisions=decisions, verification=verification,
                certificate=str(certificate_path) if reward == 1 and certificate_path else None)


def update_group(model, episodes, optimizer, cfg):
    if any(e['reward'] is None for e in episodes):
        return {"updated": False, "reason": "censored_verification"}
    rewards = torch.tensor([e['reward'] for e in episodes], dtype=torch.float32)
    if rewards.std(unbiased=False).item() < 1e-8:
        return {"updated": False, "reason": "equal_rewards"}
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    count = sum(len(e['decisions']) for e in episodes)
    if not count:
        return {"updated": False, "reason": "no_decisions"}
    # The rollout policy is the frozen reference for this online group. Exact
    # KL uses all candidates from that decision, never a different candidate set.
    model.eval()  # Disable dropout while retaining autograd for policy ratios.
    total = 0.0
    for _ in range(cfg.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        for i, episode in enumerate(episodes):
            for decision in episode['decisions']:
                logp = distribution(model, decision['state'], decision['actions'])
                old = torch.tensor(decision['old_log_probs'], device=logp.device)
                selected = decision['selected']
                ratio = (logp[selected] - old[selected]).exp()
                advantage = advantages[i].to(logp.device)
                surrogate = torch.minimum(ratio * advantage,
                    ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage)
                kl = (logp.exp() * (logp - old)).sum()
                loss = (-surrogate + cfg.kl_coefficient * kl) / count
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite RLVR loss")
                loss.backward()
                total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    model.candidate_head_trained = True
    return {"updated": True, "loss": total / cfg.update_epochs}


def train_rlvr(checkpoint, corpus_directory, database_path, output_directory, config=None):
    cfg = config or RLVRConfig()
    if not cfg.external_verifier or not Path(cfg.external_verifier).is_file():
        raise ValueError("RLVR requires an existing official Metamath executable")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    database = parse(database_path)
    corpus = Path(corpus_directory)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    model, payload = ProofTransformer.load_checkpoint(checkpoint, map_location=_device(cfg.device))
    validate_checkpoint_tokenizer(model, payload, tokenizer)
    model.eval()
    candidates = sample_training_targets(corpus, training_candidates(corpus, tokenizer, database,
                                        max(256, cfg.groups * 16), cfg.seed), cfg.groups, cfg.seed)
    if not candidates:
        raise ValueError("no training-only single-root targets for RLVR")
    environment = BackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "rlvr.json").exists() or (output / "final.pt").exists():
        raise ValueError("RLVR output must be a new directory")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    reports = []
    with (output / "rollouts.jsonl").open('w', encoding='utf-8') as stream:
        for group_index, example in enumerate(candidates):
            theorem = tokenizer.theorem_from_state_tokens(example.state_tokens, database,
                                                         name=f"rlvr_{example.example_id}")
            episodes = [rollout(model, tokenizer, environment, theorem, cfg, database_path,
                        output / f"proof-{group_index}-{i}.mm") for i in range(cfg.group_size)]
            for episode in episodes:
                stream.write(json.dumps({"target": example.example_id, "split": "train", **episode}) + '\n')
            stream.flush()
            update = update_group(model, episodes, optimizer, cfg)
            reports.append({"target": example.example_id, "rewards": [e['reward'] for e in episodes], **update})
    summary = {"format": "peano-rlvr-v1", "configuration": asdict(cfg), "groups": reports,
               "database": str(Path(database_path).resolve()), "corpus": str(corpus.resolve()),
               "checkpoint": str(Path(checkpoint).resolve()),
               "policy": "candidate ranking; autoregressive token decoder is not optimized by this RLVR objective",
               "reward_semantics": "officially verified success within a bounded attempt; zero is not unprovability"}
    model.save_checkpoint(output / "final.pt", optimizer_state=optimizer.state_dict(),
                          metadata={**payload.get('metadata', {}), "rlvr": summary})
    tokenizer.save(output / "tokenizer.json")
    (output / "rlvr.json").write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary

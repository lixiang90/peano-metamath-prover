"""Group-relative, clipped policy-gradient RL with verified proof rewards."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .agent import AgentBudget
from .inference import PolicyConfig, action_distribution, rollout, rollout_record
from .kernel import theorem_from_data
from .model import save_checkpoint
from .memory import incorporate_verified_rollouts


@dataclass(frozen=True)
class RLVRConfig:
    groups: int = 8
    group_size: int = 4
    update_epochs: int = 1
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7
    max_steps: int = 24
    require_external: bool = False

    def __post_init__(self):
        if min(self.groups, self.group_size - 1, self.update_epochs, self.max_steps) <= 0:
            raise ValueError("positive groups/epochs/steps and group_size >=2 required")
        if not all(math.isfinite(v) for v in (self.learning_rate, self.clip_epsilon, self.kl_coefficient, self.grad_clip)):
            raise ValueError("nonfinite RLVR hyperparameter")
        if min(self.learning_rate, self.clip_epsilon, self.grad_clip) <= 0 or self.kl_coefficient < 0:
            raise ValueError("invalid RLVR hyperparameter")


def verified_reward(result, kernel):
    """Replay again at the training trust boundary; status text is not a reward."""
    if result.status != "success" or result.session is None or result.session.certificate is None:
        return 0.0
    certificate, target = result.session.certificate, result.session.target
    if (certificate.conclusion != target.conclusion
            or [h.expr for h in certificate.hypotheses] != [h.expr for h in target.hypotheses]
            or certificate.d_constraints != target.d_constraints
            or certificate.variable_types != target.variable_types):
        return 0.0
    mandatory = {node.op for expression in [target.conclusion, *(h.expr for h in target.hypotheses)]
                 for node in expression.walk() if node.op in target.variable_types}
    if any(set(pair) <= mandatory and pair not in target.d_constraints
           for pair in certificate.proof_d_constraints):
        return 0.0
    try:
        kernel.verify(result.session.certificate)
        if result.session.budget.required_external:
            certification = kernel.certify(result.session.certificate, required_external=True)
            if not certification.get("external_verified", False):
                return 0.0
    except ValueError:
        return 0.0
    return 1.0


def update_group(model, reference_model, rollouts, optimizer, *, kernel, policy=None, config=None):
    cfg, pcfg = config or RLVRConfig(), policy or PolicyConfig()
    rewards = torch.tensor([verified_reward(r, kernel) for r in rollouts], dtype=torch.float32)
    if not len(rollouts):
        raise ValueError("empty rollout group")
    std = rewards.std(unbiased=False)
    if float(std) < 1e-8:
        return {"updated": False, "reason": "zero_group_advantage", "rewards": rewards.tolist(), "loss": 0.0}
    advantages = (rewards - rewards.mean()) / std.clamp_min(1e-8)
    # Evaluation mode disables dropout but leaves autograd enabled.  Importance
    # ratios must compare the same deterministic policy used during rollout.
    was_training = model.training
    model.eval()
    reference_model.eval()
    losses = []
    try:
        for _ in range(cfg.update_epochs):
            optimizer.zero_grad(set_to_none=True)
            count = sum(len(r.decisions) for r in rollouts)
            if not count:
                return {"updated": False, "reason": "no_decisions", "rewards": rewards.tolist(), "loss": 0.0}
            total = 0.0
            for index, result in enumerate(rollouts):
                for decision in result.decisions:
                    probabilities = action_distribution(model, decision.samples, pcfg)
                    log_probability = probabilities[decision.selected].clamp_min(1e-30).log()
                    ratio = torch.exp(log_probability - decision.old_log_prob)
                    advantage = advantages[index].to(probabilities.device)
                    policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage)
                    with torch.no_grad():
                        reference = action_distribution(reference_model, decision.samples, pcfg)
                    # Exact categorical KL over the SAME available action set.
                    kl = (probabilities * (probabilities.clamp_min(1e-30).log() - reference.clamp_min(1e-30).log())).sum()
                    loss = (policy_loss + cfg.kl_coefficient * kl) / count
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite RLVR loss")
                    loss.backward()
                    total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            losses.append(total)
    finally:
        model.train(was_training)
    return {"updated": True, "reason": "verified_group_policy_gradient", "rewards": rewards.tolist(),
            "loss": sum(losses) / len(losses), "epochs": cfg.update_epochs}


def train_rlvr(model, kernel, library, episodes, output_directory, config=None, policy=None):
    cfg, pcfg = config or RLVRConfig(), policy or PolicyConfig()
    if not episodes or any(e.get("split") != "train" for e in episodes):
        raise ValueError("RLVR requires nonempty training-split targets")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    updates, success, reports = 0, 0, []
    library_step = max((record.step for record in library.records), default=-1) + 1 if library is not None else 0
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as trace, (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for group in range(cfg.groups):
            episode = episodes[group % len(episodes)]
            target = theorem_from_data(episode["certificate"])
            # Remove the teacher proof before handing a target to inference.
            target.proof = None
            results = [rollout(model, kernel, target, library,
                               budget=AgentBudget(max_steps=cfg.max_steps, required_external=cfg.require_external),
                               policy=pcfg, seed=cfg.seed + group * cfg.group_size + j)
                       for j in range(cfg.group_size)]
            report = update_group(model, reference, results, optimizer, kernel=kernel, policy=pcfg, config=cfg)
            if library is not None:
                report["library"] = incorporate_verified_rollouts(library, results, step=library_step + group)
            report.update({"group": group, "target_id": episode["id"]})
            updates += int(report["updated"])
            success += sum(report["rewards"])
            reports.append(report)
            for result in results:
                trace.write(json.dumps({"group": group, "target_id": episode["id"], **rollout_record(result)}, ensure_ascii=False) + "\n")
            metrics.write(json.dumps(report) + "\n")
            trace.flush()
            metrics.flush()
    summary = {"mode": "rlvr", "groups": cfg.groups, "updated_groups": updates,
               "zero_advantage_groups": sum(r["reason"] == "zero_group_advantage" for r in reports),
               "verified_successes": int(success), "rollouts": cfg.groups * cfg.group_size,
               "theory_sha256": kernel.theory_fingerprint, "rlvr_config": asdict(cfg), "policy_config": asdict(pcfg)}
    save_checkpoint(output / "model.pt", model, metadata=summary)
    if library is not None:
        library.save(output / "library.json")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

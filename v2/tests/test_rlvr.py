import copy
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from metamath_generator.model import Node, Theorem
from pa_prover_v2.agent import AgentAction, AgentBudget, AgentSession
from pa_prover_v2.inference import Decision, PolicyConfig, Rollout, action_distribution, rollout
from pa_prover_v2.kernel import ProofKernel
from pa_prover_v2.model import LemmaDecoderLM, ModelConfig
from pa_prover_v2.rlvr import RLVRConfig, update_group, verified_reward
from pa_prover_v2.tokenizer import ByteTokenizer
from pa_prover_v2.training import encode_segments

ROOT = Path(__file__).resolve().parents[2]


def setup_task():
    kernel = ProofKernel(ROOT / "formal/peano-pa-plus.mm")
    p, q = Node("zzp"), Node("zzq")
    goal = Node("|-", (Node("implies", (p, Node("implies", (q, p)))),))
    target = Theorem("rl_target", [], goal, variable_types={"zzp": "wff", "zzq": "wff"})
    actions = [AgentAction("APPLY", {"rule": "ax-1", "substitution": {"phi": p, "psi": q}, "premises": []}),
               AgentAction("APPLY", {"rule": "ax-1", "substitution": {"phi": q, "psi": p}, "premises": []})]
    torch.manual_seed(19)
    model = LemmaDecoderLM(ModelConfig(d_model=16, n_heads=2, n_layers=1, lemma_layers=1, max_seq_len=2048)).eval()
    return kernel, target, actions, model


def actual_rollout(kernel, target, actions, model, selected, policy):
    session = AgentSession(kernel, target, budget=AgentBudget(max_steps=2))
    samples = [encode_segments(session.context_segments(), ByteTokenizer(), model.config.lemma_slots, a, action_only=True) for a in actions]
    with torch.no_grad():
        old = float(action_distribution(model, samples, policy)[selected].log())
    session.step(actions[selected])
    session.step(AgentAction("FINISH", {"fact_id": "f0"}))
    # Both branches create genuine well-typed facts.  Only one proves target.
    return Rollout([Decision(samples, [a.to_dict() for a in actions], selected, old)],
                   reward=1.0 if session.status == "success" else 0.0,
                   status=session.status, reason="test_actual_kernel_episode", session=session)


def test_verified_policy_gradient_increases_probability_of_real_proof():
    torch.set_num_threads(1)
    kernel, target, actions, model = setup_task()
    policy = PolicyConfig(exploration=0.2)
    good, bad = [actual_rollout(kernel, target, actions, model, i, policy) for i in range(2)]
    assert verified_reward(good, kernel) == 1
    assert verified_reward(bad, kernel) == 0
    assert bad.session.trace[-1]["observation"]["status"] == "unknown"
    reference = copy.deepcopy(model)
    before = float(action_distribution(model, good.decisions[0].samples, policy)[0].detach())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
    report = update_group(model, reference, [good, bad], optimizer, kernel=kernel, policy=policy,
                          config=RLVRConfig(update_epochs=1, kl_coefficient=0))
    after = float(action_distribution(model, good.decisions[0].samples, policy)[0].detach())
    assert report["updated"] and report["rewards"] == [1, 0]
    assert after > before + 1e-5


def test_zero_reward_variance_does_not_invent_learning_signal():
    kernel, target, actions, model = setup_task()
    policy = PolicyConfig()
    bad = actual_rollout(kernel, target, actions, model, 1, policy)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    report = update_group(model, copy.deepcopy(model), [bad, bad], optimizer, kernel=kernel)
    assert not report["updated"] and report["reason"] == "zero_group_advantage"
    assert all(torch.equal(before[k], p) for k, p in model.named_parameters())


def test_forged_success_status_does_not_grant_reward():
    kernel, target, actions, model = setup_task()
    bad = actual_rollout(kernel, target, actions, model, 1, PolicyConfig())
    bad.status, bad.reward = "success", 1.0
    assert verified_reward(bad, kernel) == 0
    # Even an internally valid proof of ANOTHER formula is not a target proof.
    bad.session.certificate = bad.session.facts["f0"]
    assert verified_reward(bad, kernel) == 0
    good = actual_rollout(kernel, target, actions, model, 0, PolicyConfig())
    good.session.certificate.proof.source_labels = ("nonexistent-proof-label",)
    assert verified_reward(good, kernel) == 0


def test_policy_recompute_matches_rollout_probability_with_dropout_disabled():
    kernel, target, actions, model = setup_task()
    policy = PolicyConfig(temperature=0.7, exploration=0.3, length_penalty=0)
    result = actual_rollout(kernel, target, actions, model, 0, policy)
    decision = result.decisions[0]
    recomputed = action_distribution(model, decision.samples, policy)[decision.selected].log()
    assert float(recomputed.detach()) == pytest.approx(decision.old_log_prob, abs=1e-6)


def test_inference_has_no_teacher_and_context_exhaustion_is_unknown():
    kernel, target, _, model = setup_task()
    assert target.proof is None
    result = rollout(model, kernel, target, budget=AgentBudget(max_steps=3), policy=PolicyConfig(max_candidates=2), seed=3)
    assert result.status in {"success", "unknown"}
    if result.status == "success":
        kernel.verify(result.session.certificate)
    short = LemmaDecoderLM(ModelConfig(d_model=16, n_heads=2, n_layers=1, lemma_layers=1, max_seq_len=32))
    result = rollout(short, kernel, target, budget=AgentBudget(max_steps=3))
    assert result.status == "unknown" and result.reason == "context_budget" and result.reward == 0

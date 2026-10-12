"""Scope, causality, executable feedback and training tests for B/C."""
import copy
import itertools
import json
import os
from pathlib import Path
import time

import pytest

from metamath_generator.model import Theorem
from metamath_generator.parser import parse
from pa_prover_v21.kernel import N, eq
from pa_prover_v21.strategy import Action, Session
from pa_prover_v21.search import SearchConfig, Classical, search
from pa_prover_v21.interaction import Interaction, command, replay_interaction, render, segments

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"formal/peano-pa-plus.mm"
MM = Path(os.environ.get("METAMATH_EXECUTABLE", ROOT/"outputs/tools/metamath/metamath.exe"))


@pytest.fixture(scope="module")
def db():
    return parse(SOURCE)


def target():
    x = N("x0")
    types = {"x0": "var", "x1": "var", "x2": "var"}
    return Theorem("test", [], N("|-", eq(N("+", N("0"), x), x)),
                   variable_types=types, d_constraints=set(itertools.combinations(types, 2)))


class NoClosure:
    def __init__(self, session):
        self.session = session
        self.deadline = time.perf_counter()+60

    def solve(self, task):
        return None


def initialized(db, mode="scratchpad"):
    s = Session(db, target())
    flow = Interaction(s, s.initial, mode)
    backend = NoClosure(s)
    flow.initialize(backend, lambda: 50.)
    return flow, backend


def test_backtrack_restores_scope_and_keeps_only_attempt_memory(db):
    flow, backend = initialized(db)
    original = flow.session.observation(flow.state, 40.)
    cfg = SearchConfig(reasoning_mode="scratchpad")
    text = command("TRY", Action("INDUCT", 0, variable="x0"))
    assert text in flow.options(cfg)
    flow.execute(text, backend, cfg, lambda: 45.)
    assert flow.pending is not None
    assert any(g["facts"] for g in flow.session.observation(flow.state, 40.)["goals"])
    assert len(flow.session.b.types) > len(original["types"])
    flow.execute(command("BACKTRACK"), backend, cfg, lambda: 40.)
    assert flow.session.observation(flow.state, 40.) == original
    assert flow.depth == 0 and flow.trials == 1 and flow.pending is None
    assert flow.memory[0]["outcome"] == "UNKNOWN"
    assert text not in flow.options(cfg)  # Do not repeat the same speculative call.
    assert command("APPLY", Action("INDUCT", 0, variable="x0")) in flow.options(cfg)
    assert "receipts" not in json.dumps(flow.observation())
    result = {"interaction": flow.events, "receipts": flow.receipts, "certified": False}
    replayed = replay_interaction(db, target(), None, result, cfg)
    assert replayed.session.observation(replayed.state, 40.) == original


def test_trial_limits_and_no_unproved_have_leak(db):
    flow, backend = initialized(db)
    cfg = SearchConfig(reasoning_mode="scratchpad", max_trials=1)
    flow.execute(command("TRY", Action("HAVE", 0, statement="= 0 x0")), backend, cfg, lambda: 40.)
    assert flow.state.frontier == (1,)
    assert not flow.state.tasks[2].facts
    with pytest.raises(ValueError):
        flow.execute(command("APPLY", Action("INDUCT", 1, variable="x0")), backend, cfg, lambda: 30.)
    flow.execute(command("BACKTRACK"), backend, cfg, lambda: 30.)
    assert all(json.loads(a)["op"] == "APPLY" for a in flow.options(cfg))
    with pytest.raises(ValueError):
        flow.execute(command("TRY", Action("INDUCT", 0, variable="x0")), backend, cfg, lambda: 30.)
    flow, backend = initialized(db, "feedback")
    with pytest.raises(ValueError):
        flow.execute(command("TRY", Action("INDUCT", 0, variable="x0")), backend, SearchConfig(reasoning_mode="feedback"), lambda: 30.)


def test_causal_prefix_and_tampered_feedback_rejected(db):
    flow, backend = initialized(db)
    before = flow.observation()
    cfg = SearchConfig(reasoning_mode="scratchpad")
    flow.execute(command("TRY", Action("INDUCT", 0, variable="x0")), backend, cfg, lambda: 40.)
    assert len(before["events"]) == 1 and "INDUCT" not in render(before["events"])
    result = {"interaction": flow.events, "receipts": flow.receipts, "certified": False}
    corrupt = copy.deepcopy(result)
    corrupt["interaction"][-1]["content"]["state"]["goals"] = []
    with pytest.raises(ValueError, match="feedback"):
        replay_interaction(db, target(), None, corrupt, cfg)


def test_commit_resumes_closure_and_forged_restore_is_rejected(db, tmp_path):
    flow, backend = initialized(db)
    cfg = SearchConfig(reasoning_mode="scratchpad", closure_seconds=2)
    flow.execute(command("TRY", Action("INDUCT", 0, variable="x0")), backend, cfg, lambda: 40.)
    assert not flow.state.solved
    actual = Classical(flow.session, cfg, time.perf_counter()+15, tmp_path)
    flow.execute(command("COMMIT"), actual, cfg, lambda: 30.)
    assert flow.state.solved and flow.pending is None
    result = {"interaction": flow.events, "receipts": flow.receipts, "certified": False}
    assert replay_interaction(db, target(), None, result, cfg).state.solved
    flow, backend = initialized(db)
    flow.execute(command("TRY", Action("INDUCT", 0, variable="x0")), backend, cfg, lambda: 40.)
    with pytest.raises(ValueError, match="inject"):
        flow.execute(command("BACKTRACK"), backend, cfg, lambda: 30., {"operations": ["forged"]})
    corrupt = copy.deepcopy(result)
    corrupt["receipts"][-1]["context"]["distinct"] = []
    with pytest.raises(ValueError, match="distinct"):
        replay_interaction(db, target(), None, corrupt, cfg)


@pytest.mark.skipif(not MM.exists(), reason="official verifier not installed")
@pytest.mark.parametrize("mode", ["baseline", "feedback", "scratchpad"])
def test_search_certificates_and_replay(db, tmp_path, mode):
    cfg = SearchConfig(seconds=20, reasoning_mode=mode, trial_seconds=2, max_candidates=8)
    result = search(db, SOURCE, target(), tmp_path/mode, MM, config=cfg)
    assert result["certified"], result["status"]
    if mode == "baseline":
        assert "interaction" not in result
        assert "protocol" not in result["trace"][0]["observation"]
    else:
        result = json.loads((tmp_path/mode/"result.json").read_text())
        flow = replay_interaction(db, target(), None, result, cfg)
        assert flow.state.solved and flow.pending is None
        if mode == "scratchpad":
            assert any(e["content"].get("op") == "TRY" for e in result["interaction"])
            assert any(e["content"].get("op") == "COMMIT" for e in result["interaction"])
        for i, entry in enumerate(result["trace"]):
            assert entry["observation"]["events"] == result["interaction"][:2*i+1]


@pytest.mark.skipif(not MM.exists(), reason="official verifier not installed")
def test_successful_repair_trajectory_is_not_a_free_lemma(db, tmp_path):
    cfg = SearchConfig(seconds=25, reasoning_mode="scratchpad", trial_seconds=.5, max_candidates=16)
    forced = command("TRY", Action("HAVE", 0, statement="= 0 x0"))
    result = search(db, SOURCE, target(), tmp_path, MM, config=cfg, forced=forced)
    assert result["certified"], result["status"]
    ops = [e["content"]["op"] for e in result["interaction"] if e["role"] == "model"]
    assert ops[:2] == ["TRY", "BACKTRACK"]
    assert replay_interaction(db, target(), None, result, cfg).state.solved


def test_budget_is_shared_and_not_refunded(db, tmp_path):
    result = search(db, SOURCE, target(), tmp_path, "unused", config=SearchConfig(
        seconds=.000001, reasoning_mode="scratchpad"))
    assert not result["certified"] and result["status"] == "unknown_timeout"
    assert result["trials"] == 0


def test_continuous_model_mask_and_candidate_likelihood(monkeypatch):
    torch = pytest.importorskip("torch")
    from pa_prover_v21.policy import MacroPolicy, PolicyConfig, prompt
    model = MacroPolicy(PolicyConfig(d_model=16, n_heads=2, n_layers=1, max_seq_len=2048, exploration=0))
    events = [{"role": "engine", "content": {"goal": "α"}},
              {"role": "model", "content": {"op": "BACKTRACK"}},
              {"role": "engine", "content": {"status": "OPEN"}}]
    import pa_prover_v21.policy as module
    original = module.F.cross_entropy
    captured = []
    def capture(logits, labels, **kwargs):
        captured.append(labels.detach().clone())
        return original(logits, labels, **kwargs)
    monkeypatch.setattr(module.F, "cross_entropy", capture)
    model.trajectory_loss(events, model_only=True).backward()
    expected = []
    for text, supervise in segments(events):
        expected.extend([supervise]*len(text.encode("utf-8")))
    assert (captured[-1] != -100).tolist() == expected+[False]
    model.trajectory_loss(events, model_only=False)
    assert (captured[-1] != -100).all()
    obs = {"protocol": "v21-interaction-1", "mode": "scratchpad", "events": events}
    actions = [command("COMMIT"), command("BACKTRACK")]
    lp, _ = model.distribution(obs, actions)
    scores = []
    offset = len(model.tokenizer.encode(prompt(obs), bos=True))
    for a in actions:
        ids = model.ids(prompt(obs)+a+"\n", bos=True)
        logits = model(ids[:, :-1])[0]
        scores.append(logits[:, offset-1:].log_softmax(-1).gather(-1, ids[:, offset:, None]).sum())
    assert torch.allclose(lp, torch.stack(scores).log_softmax(0), atol=1e-4)
    with pytest.raises(ValueError, match="context"):
        model.trajectory_loss(events*100)


def test_comparison_arms():
    from pa_prover_v21.evaluation import arm, arm_checkpoint
    assert arm("learned-b") == ("learned", "feedback")
    assert arm("learned-c") == ("learned", "scratchpad")
    assert arm("heuristic-a", "scratchpad") == ("heuristic", "baseline")
    assert arm("atp", "scratchpad") == ("atp", "baseline")
    assert arm_checkpoint({"checkpoint": "a", "checkpoint_feedback": "b"}, "feedback") == "b"
    with pytest.raises(ValueError):
        arm("invalid-c")


@pytest.mark.skipif(not MM.exists(), reason="official verifier not installed")
@pytest.mark.parametrize("mode", ["feedback", "scratchpad"])
def test_generation_audit_and_micro_training(db, tmp_path, monkeypatch, mode):
    torch = pytest.importorskip("torch")
    from pa_prover_v21 import curriculum
    from pa_prover_v21.data import record
    from pa_prover_v21.training import train, rlvr
    from pa_prover_v21.policy import MacroPolicy
    # A fixed test-only proposal isolates the plumbing from random pool size.
    # Production generation still uses the unchanged axioms-only proposal pool.
    case = record(target(), "test-fixture")
    case.update(id="self_00000", full_key="test-full", conclusion_key="test-conclusion",
                skeleton="test-skeleton", group="test-group", split="train")
    monkeypatch.setattr(curriculum, "proposal_pool", lambda *a: ([case], {}))
    corpus = tmp_path/"corpus"
    manifest = curriculum.generate(SOURCE, corpus, MM, count=1, seconds=8, trials=2,
                                   counterfactuals=1, reasoning_mode=mode, trial_seconds=1)
    report = curriculum.audit(SOURCE, corpus, MM)
    assert report["interactions_replayed"] >= 3
    assert report["continuous_trajectories"] and report["official_certificates_replayed"]
    _, _, rows = curriculum.read_corpus(corpus, SOURCE)
    trajectory = next(r for r in rows if r["kind"] == "trajectory")
    assert any(e["role"] == "model" for e in trajectory["events"])
    ntp = train(SOURCE, corpus, tmp_path/"ntp", stage="ntp", steps=1, d_model=16,
                n_layers=1, n_heads=2, max_seq_len=16384)
    assert ntp["updates"] == 1 and ntp["reasoning_mode"] == mode
    sft = train(SOURCE, corpus, tmp_path/"sft", stage="sft", steps=1, checkpoint=tmp_path/"ntp/policy.pt")
    assert sft["updates"] == 1
    m = MacroPolicy.load(tmp_path/"sft/policy.pt")
    assert m.provenance["reasoning_mode"] == mode
    # One tiny on-policy group exercises protocol-aware probabilities; no full training.
    rl = rlvr(SOURCE, corpus, tmp_path/"rlvr", MM, tmp_path/"sft/policy.pt",
              groups=1, group_size=2, seconds=1.)
    assert rl["reasoning_mode"] == mode
    # Dataset checksum rebinding must not allow future feedback to become input.
    rows_path = corpus/"examples.jsonl"
    rows[rows.index(trajectory)]["events"][0]["content"]["status"] = "FORGED"
    rows_path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    from pa_prover_v21.atp import sha256
    manifest["files"]["examples.jsonl"] = sha256(rows_path)
    (corpus/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        curriculum.read_corpus(corpus, SOURCE)

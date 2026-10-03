from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from metamath_generator.model import Hypothesis, Node, Theorem
from pa_prover_v2.agent import (
    AgentAction, AgentBudget, AgentSession, augment_teacher_actions,
    certificate_to_actions,
)
from pa_prover_v2.kernel import ProofKernel
from pa_prover_v2.library import TheoremLibrary


ROOT = Path(__file__).resolve().parents[2]


def imp(a, b):
    return Node("implies", (a, b))


def claim(a):
    return Node("|-", (a,))


@pytest.fixture
def kernel():
    return ProofKernel(ROOT / "formal" / "peano.mm")


@pytest.fixture
def example(kernel):
    p, q = Node("p"), Node("q")
    target = Theorem("agent_target", [Hypothesis("target_h0", claim(p))],
                     claim(imp(q, p)), variable_types={"p": "wff", "q": "wff"})
    lemma = kernel.compose(kernel.database.logical_assertions["ax-1"],
                           {"phi": p, "psi": q}, [], target)
    certificate = kernel.compose(kernel.database.logical_assertions["ax-mp"],
                                 {"phi": p, "psi": imp(q, p)},
                                 [kernel.assumption(target, 0), lemma], target)
    return target, lemma, certificate


def apply_ax1():
    return AgentAction("APPLY", {"rule": "ax-1", "substitution": {
        "phi": Node("p"), "psi": Node("q"),
    }, "premises": []})


def replay(kernel, target, actions, library=None):
    session = AgentSession(kernel, target, library)
    for action in actions:
        observation = session.step(action)
        assert observation.status in {"running", "success"}, observation
    assert session.status == "success"
    kernel.verify(session.certificate)
    return session


def test_teacher_compiles_postfix_and_replays(kernel, example):
    target, _, certificate = example
    actions = certificate_to_actions(certificate, kernel)
    assert [a.op for a in actions] == ["APPLY", "APPLY", "FINISH"]
    session = replay(kernel, target, actions)
    assert session.facts["h0"].conclusion == target.hypotheses[0].expr
    assert session.certificate.conclusion == target.conclusion
    assert "source_labels" not in session.context()
    segments = session.context_segments()
    assert set(json.loads(segments[0])["facts"]) == {"h0"}
    assert "f1" in json.loads(segments[-1])["current_workspace"]["facts"]
    for action in actions:
        assert AgentAction.from_dict(json.loads(json.dumps(action.to_dict()))).to_dict() == action.to_dict()


def test_invalid_action_and_unproved_proposal_never_add_facts(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target)
    before = dict(session.facts)
    proposed = session.step(AgentAction("PROPOSE", {"formula": target.conclusion}))
    assert proposed.status == "running" and proposed.data["trusted"] is False
    wrong = session.step(AgentAction("APPLY", {"rule": "ax-mp", "substitution": {
        "phi": Node("p"), "psi": imp(Node("q"), Node("p")),
    }, "premises": ["h0", "h0"]}))
    assert wrong.status == "invalid"
    assert session.facts == before
    assert session.pending_focus == [target.conclusion]
    assert session.step(AgentAction("SAVE", {"fact_id": "pending", "name": "fake"})).status == "invalid"
    assert not session.scratch
    assert session.step(AgentAction("FINISH", {"fact_id": "h0"})).status == "invalid"
    assert session.status == "running"
    assert session.step(AgentAction(1, {})).status == "invalid"
    assert session.step(AgentAction("APPLY", [])).status == "invalid"
    assert session.facts == before


def test_local_save_read_use_is_verified_and_has_memory_position(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target)
    assert session.step(apply_ax1()).status == "running"
    assert session.step(AgentAction("SAVE", {"fact_id": "f0", "name": "local"})).status == "running"
    assert session.step(AgentAction("USE", {"name": "local", "substitution": {}, "premises": []})).status == "invalid"
    before = set(session.facts)
    assert session.step(AgentAction("READ", {"name": "local"})).status == "running"
    assert set(session.facts) == before
    observation = session.step(AgentAction("USE", {"name": "local", "substitution": {
        "p": Node("p"), "q": Node("q"),
    }, "premises": ["h0"]}))
    assert observation.status == "running", observation
    assert session.facts[observation.data["fact_id"]].conclusion == session.facts["f0"].conclusion
    assert len(session.memory_occurrences) == 1
    segments = session.context_segments()
    memory_index = next(index for index, item in enumerate(segments) if isinstance(item, dict))
    assert json.loads(segments[memory_index - 1])["action"]["op"] == "READ"
    text = json.loads(segments[memory_index]["text"])
    assert {"hypotheses", "conclusion", "variable_types", "d_constraints"} <= set(text)


def test_backtrack_revokes_branch_facts_and_memory_but_keeps_failure_trace(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target)
    session.step(apply_ax1())
    session.step(AgentAction("SAVE", {"fact_id": "f0", "name": "local"}))
    session.step(AgentAction("READ", {"name": "local"}))
    session.step(AgentAction("READ", {"name": "missing"}))
    reads, tokens = session.reads, session.memory_tokens
    assert session.step(AgentAction("BACKTRACK", {"snapshot_id": "s0"})).status == "running"
    assert list(session.facts) == ["h0"]
    assert not session.scratch and not session.memory_occurrences
    assert session.reads == reads and session.memory_tokens == tokens
    assert any(event["observation"]["status"] == "invalid" for event in session.trace)
    assert session.step(AgentAction("BACKTRACK", {"snapshot_id": "s3"})).status == "invalid"
    assert session.step(AgentAction("USE", {"name": "local", "substitution": {}, "premises": []})).status == "invalid"


def test_strict_step_read_memory_budgets_return_unknown(kernel, example):
    target, lemma, _ = example
    session = AgentSession(kernel, target, budget=AgentBudget(max_steps=1))
    assert session.step(apply_ax1()).status == "unknown"
    assert session.step(AgentAction("FINISH", {"fact_id": "f0"})).status == "unknown"
    library = TheoremLibrary(kernel)
    ref = library.add(lemma)
    for budget in (AgentBudget(max_reads=0), AgentBudget(max_memory_tokens=1)):
        session = AgentSession(kernel, target, library, budget)
        initial = dict(session.facts)
        assert session.step(AgentAction("READ", {"lemma_id": ref})).status == "unknown"
        assert session.facts == initial
        assert not session.memory_occurrences
        assert session.reads == 0 and session.memory_tokens == 0


def test_proof_label_budget_exhaustion_is_unknown_without_a_new_fact(kernel, example):
    target, lemma, _ = example
    session = AgentSession(kernel, target, budget=AgentBudget(max_proof_labels=1))
    before = dict(session.facts)
    observation = session.step(apply_ax1())
    assert observation.status == "unknown"
    assert "proof" in observation.message.lower()
    assert session.facts == before and session.certificate is None
    assert session.candidate_actions() == []
    library = TheoremLibrary(kernel)
    lemma_id = library.add(lemma)
    session = AgentSession(kernel, target, library, AgentBudget(max_proof_labels=1))
    assert session.step(AgentAction("READ", {"lemma_id": lemma_id})).status == "unknown"
    assert not session.memory_occurrences and session.reads == 0
    assert list(session.facts) == ["h0"]


def test_use_respects_expanded_proof_budget_not_just_library_entry_size(kernel, example):
    target, lemma, _ = example
    library = TheoremLibrary(kernel)
    ref = library.add(lemma)
    # Reading the compact schematic proof fits.  A larger substitution makes
    # its instantiated syntax proof exceed the same output-label allowance.
    limit = len(lemma.proof.source_labels)
    session = AgentSession(kernel, target, library, AgentBudget(max_proof_labels=limit))
    assert session.step(AgentAction("READ", {"lemma_id": ref})).status == "running"
    before = dict(session.facts)
    action = AgentAction("USE", {"lemma_id": ref,
                        "substitution": {"p": Node("p"), "q": imp(Node("q"), Node("q"))},
                        "premises": ["h0"]})
    observation = session.step(action)
    assert observation.status == "unknown", observation
    assert session.facts == before and session.certificate is None


def context_size(session):
    return sum(len(segment) if isinstance(segment, str) else
               len(json.dumps(segment, sort_keys=True, ensure_ascii=False))
               for segment in session.context_segments())


def test_initial_context_limit_is_terminal_unknown_not_a_partial_state(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target, budget=AgentBudget(max_context_chars=1))
    assert session.status == "unknown"
    initial = dict(session.facts)
    assert session.step(apply_ax1()).status == "unknown"
    assert session.step(AgentAction("FINISH", {"fact_id": "h0"})).status == "unknown"
    assert session.facts == initial and session.certificate is None
    assert session.candidate_actions() == []
    notice = json.loads(session.context())
    assert notice["reason"] == "context_budget_exceeded"
    assert notice["full_context_omitted"] and notice["required_chars"] > 1
    assert session.context_segments() == [json.dumps(notice, sort_keys=True)]


def test_context_growth_rolls_back_fact_or_read_instead_of_truncating_it(kernel, example):
    target, lemma, _ = example
    limit = context_size(AgentSession(kernel, target)) + 1
    library = TheoremLibrary(kernel)
    lemma_id = library.add(lemma)
    for action in (apply_ax1(), AgentAction("READ", {"lemma_id": lemma_id})):
        session = AgentSession(kernel, target, library, AgentBudget(max_context_chars=limit))
        before = dict(session.facts)
        assert session.status == "running"
        observation = session.step(action)
        assert observation.status == "unknown"
        assert observation.data["reason"] == "context_budget_exceeded"
        assert session.status == "unknown" and session.facts == before
        assert not session.memory_occurrences and session.certificate is None
        assert session.candidate_actions() == []
        assert len(session.context_segments()) == 1


def test_finish_cannot_commit_success_after_context_budget_is_exceeded(kernel, example):
    target, _, certificate = example
    actions = certificate_to_actions(certificate, kernel)
    reference = AgentSession(kernel, target)
    for action in actions[:-1]:
        assert reference.step(action).status == "running"
    limit = context_size(reference) + 1
    bounded = AgentSession(kernel, target, budget=AgentBudget(max_context_chars=limit))
    for action in actions[:-1]:
        assert bounded.step(action).status == "running"
    assert bounded.step(actions[-1]).status == "unknown"
    assert bounded.status == "unknown"
    assert bounded.certificate is None and bounded.certification is None


def test_library_use_requires_read_and_cannot_add_axioms(kernel, example):
    target, lemma, _ = example
    library = TheoremLibrary(kernel)
    ref = library.add(lemma)
    session = AgentSession(kernel, target, library)
    assert session.step(AgentAction("SEARCH", {"query": target.conclusion.to_prefix()})).status == "running"
    assert ref in session.search_results
    assert session.step(AgentAction("READ", {"lemma_id": ref})).status == "running"
    action = AgentAction("USE", {"lemma_id": ref, "substitution": {
        "p": Node("p"), "q": Node("q"),
    }, "premises": ["h0"]})
    assert session.step(action).status == "running"
    assert session.step(AgentAction("APPLY", {"rule": "invented-axiom"})).status == "invalid"


def test_library_target_is_excluded_from_search_and_direct_read(kernel, example):
    target, lemma, certificate = example
    library = TheoremLibrary(kernel)
    target_ref = library.add(certificate)
    lemma_ref = library.add(lemma)
    session = AgentSession(kernel, target, library)
    assert session.step(AgentAction("SEARCH", {})).status == "running"
    assert target_ref not in session.search_results
    assert lemma_ref in session.search_results
    assert session.step(AgentAction("READ", {"lemma_id": target_ref})).status == "invalid"
    assert session.step(AgentAction("USE", {"lemma_id": target_ref})).status == "invalid"
    assert session.reads == 0 and not session.memory_occurrences


def test_vector_memory_has_no_search_or_read_text_bypass(kernel, example):
    target, _, _ = example
    left, right = Node("memory_left"), Node("memory_right")
    context = Theorem("distinct_library_context", [], claim(imp(left, imp(right, left))),
                      variable_types={"memory_left": "wff", "memory_right": "wff"})
    theorem = kernel.compose(kernel.database.logical_assertions["ax-1"],
                             {"phi": left, "psi": right}, [], context)
    library = TheoremLibrary(kernel)
    lemma_id = library.add(theorem)
    session = AgentSession(kernel, target, library)
    assert session.step(AgentAction("SEARCH", {})).status == "running"
    assert session.step(AgentAction("READ", {"lemma_id": lemma_id})).status == "running"
    audit_before = json.dumps(session.trace, sort_keys=True)
    vectors = session.context_segments()
    main_text = "".join(item for item in vectors if isinstance(item, str))
    encoder_inputs = [item for item in vectors if isinstance(item, dict)]
    assert "memory_left" not in main_text and "memory_right" not in main_text
    assert len(encoder_inputs) == 1
    assert "memory_left" in encoder_inputs[0]["text"]
    assert "memory_right" in encoder_inputs[0]["text"]
    for item in vectors[1:-1]:
        if not isinstance(item, str):
            continue
        event = json.loads(item)
        data = event["observation"]["data"]
        if event["action"]["op"] == "READ":
            assert "statement" not in data and "text" not in data["memory"]
            assert data["verified"] is True
        elif event["action"]["op"] == "SEARCH":
            assert all("statement" not in result for result in data["results"])
    text = session.context_segments(memory_mode="text")
    assert all(isinstance(item, str) for item in text)
    assert "memory_left" in "".join(text)
    both = session.context_segments(memory_mode="both")
    assert any(isinstance(item, dict) for item in both)
    assert "memory_left" in "".join(item for item in both if isinstance(item, str))
    assert json.dumps(session.trace, sort_keys=True) == audit_before
    assert "memory_left" in session.context()
    assert json.loads(vectors[0])["facts"] == {"h0": target.hypotheses[0].expr.to_prefix()}
    assert session.step(AgentAction("BACKTRACK", {"snapshot_id": "s0"})).status == "running"
    assert not json.loads(session.context_segments()[-1])["current_workspace"]["active_memories"]
    assert session.step(AgentAction("USE", {"lemma_id": lemma_id})).status == "invalid"
    with pytest.raises(ValueError, match="memory_mode"):
        session.context_segments(memory_mode="silent_truncation")


def test_pa_plus_teacher_finishes_with_official_certification():
    from neural_prover.external import discover_metamath_executable

    executable = discover_metamath_executable()
    if executable is None:
        executable = discover_metamath_executable(ROOT / "outputs/tools/metamath/metamath.exe")
    if executable is None:
        pytest.skip("official verifier not configured")
    kernel = ProofKernel(ROOT / "formal/peano-pa-plus.mm", external_verifier=executable)
    rule = kernel.database.logical_assertions["pa_ax1"]
    target = Theorem("agent_pa_target", list(rule.hypotheses), rule.conclusion,
                     variable_types=dict(rule.variable_types), d_constraints=set(rule.d_constraints))
    cert = kernel.compose(rule, {name: Node(name) for name in rule.variable_types}, [], target)
    actions = certificate_to_actions(cert, kernel)
    session = AgentSession(kernel, target, budget=AgentBudget(required_external=True))
    for action in actions:
        assert session.step(action).status in {"running", "success"}
    assert session.status == "success"
    assert session.certification["external_verified"]


def test_candidates_are_constructed_without_teacher_and_finish_is_checked(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target)
    session.step(AgentAction("PROPOSE", {"formula": claim(imp(Node("p"), imp(Node("q"), Node("p"))))}))
    candidates = session.candidate_actions()
    assert candidates
    ax1 = next(action for action in candidates if action.op == "APPLY" and action.args["rule"] == "ax-1")
    assert session.step(ax1).status == "running"
    candidates = session.candidate_actions()
    mp = next(action for action in candidates if action.op == "APPLY" and action.args["rule"] == "ax-mp")
    assert session.step(mp).status == "running"
    finish = session.candidate_actions()[0]
    assert finish.op == "FINISH"
    assert session.step(finish).status == "success"


def test_augmented_teacher_contains_recovery_and_real_local_macro_use(kernel, example):
    target, _, certificate = example
    actions = augment_teacher_actions(certificate, kernel)
    assert {"PROPOSE", "BACKTRACK", "SAVE", "READ", "USE", "FINISH"} <= {a.op for a in actions}
    session = AgentSession(kernel, target)
    for action in actions:
        observation = session.step(action)
        assert observation.status in {"running", "invalid", "success"}
    assert session.status == "success"
    assert sum(item["observation"]["status"] == "invalid" for item in session.trace) == 1
    kernel.verify(session.certificate)


@pytest.mark.parametrize("seed", [7, 19, 31])
def test_random_pa_plus_compositions_teach_real_library_reuse(seed):
    kernel = ProofKernel(ROOT / "formal/peano-pa-plus.mm")
    rng = random.Random(seed)
    number = Node("0")
    for _ in range(rng.randint(1, 5)):
        number = Node("S", (number,))
    p = Node("positive", (number,))
    q = Node("ge", (number, Node("0")))
    target = Theorem(f"random_agent_{seed}", [Hypothesis("hp", claim(p))], claim(imp(q, p)))
    ax1 = kernel.database.logical_assertions["ax-1"]
    axmp = kernel.database.logical_assertions["ax-mp"]
    source_lemma = kernel.compose(ax1, {"phi": p, "psi": q}, [], target)
    certificate = kernel.compose(axmp, {"phi": p, "psi": imp(q, p)},
                                 [kernel.assumption(target, 0), source_lemma], target)
    a, b = Node("memory_phi"), Node("memory_psi")
    library_context = Theorem("generic_memory", [], claim(imp(a, imp(b, a))),
                              variable_types={"memory_phi": "wff", "memory_psi": "wff"})
    library = TheoremLibrary(kernel)
    ref = library.add(kernel.compose(ax1, {"phi": a, "psi": b}, [], library_context))
    stats = {}
    actions = augment_teacher_actions(certificate, kernel, library, stats=stats)
    assert stats["searches"] >= 2
    assert stats["library_reads"] == stats["library_uses"] == 1
    assert any(a.op == "USE" and a.args.get("lemma_id") == ref for a in actions)
    session = AgentSession(kernel, target, library)
    for action in actions:
        observation = session.step(action)
        if action.op == "USE" and "lemma_id" in action.args:
            assert observation.status == "running", observation
            assert session.facts[observation.data["fact_id"]].conclusion == source_lemma.conclusion
    assert session.status == "success"
    kernel.verify(session.certificate)
    # The reused library fact becomes the premise of the final MP; this is
    # an actual proof dependency, not an unused READ of a similar statement.
    final_apply = next(a for a in reversed(actions) if a.op == "APPLY")
    assert final_apply.args["rule"] == "ax-mp"
    assert any(ref.startswith("f") for ref in final_apply.args["premises"])


def test_constrained_candidates_offer_bounded_nontrivial_backtracking(kernel, example):
    target, _, _ = example
    session = AgentSession(kernel, target, budget=AgentBudget(max_backtracks=1))
    assert not any(a.op == "BACKTRACK" for a in session.candidate_actions())
    session.step(AgentAction("PROPOSE", {"formula": target.conclusion}))
    choices = session.candidate_actions(limit=4)
    backtrack = next(a for a in choices if a.op == "BACKTRACK")
    assert backtrack.args["snapshot_id"] == "s0"
    assert session.step(backtrack).status == "running"
    assert not session.pending_focus
    session.step(AgentAction("PROPOSE", {"formula": target.conclusion}))
    assert not any(a.op == "BACKTRACK" for a in session.candidate_actions())
    assert session.step(AgentAction("BACKTRACK", {"snapshot_id": "s0"})).status == "unknown"


def test_teacher_obeys_interleaved_floating_essential_hypotheses(tmp_path):
    source = tmp_path / "interleaved.mm"
    source.write_text("""$c wff |- implies $.
$v p q $.
fp $f wff p $.
${
  hp $e |- p $.
  fq $f wff q $.
  wi $a wff implies p q $.
  rule $a |- implies p q $.
  teacher $p |- implies p q $= fp hp fq rule $.
$}
""", encoding="utf-8")
    kernel = ProofKernel(source)
    theorem = kernel.database.statements["teacher"]
    actions = certificate_to_actions(theorem, kernel)
    assert actions[0].args["premises"] == ["h0"]
    assert actions[0].args["substitution"] == {"p": Node("p"), "q": Node("q")}
    replay(kernel, theorem, actions)

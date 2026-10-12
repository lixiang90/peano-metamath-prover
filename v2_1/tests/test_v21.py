from pathlib import Path
import itertools
import time

import pytest

from metamath_generator.parser import parse
from metamath_generator.model import Theorem, Hypothesis
from metamath_generator.verifier import verify
from pa_prover_v21.kernel import Builder, N, eq, imp
from pa_prover_v21.solver import Reconstructor, BudgetExpired
from pa_prover_v21.atp import Translator, UnsupportedFragment, problem, verify_bundle

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"formal/peano-pa-plus.mm"


@pytest.fixture(scope="module")
def db():
    return parse(SOURCE)


def context(db):
    types={"x0":"var","x1":"var","x2":"var"}
    distinct=set(itertools.combinations(sorted(types),2))
    return Builder(db,types,distinct)


def certify(db,b,p,target,tmp_path,official=False):
    fragment=tmp_path/"fragment.mm"
    b.export(p,target,fragment)
    combined=tmp_path/"check.mm"
    combined.write_text(f"$[ {SOURCE.as_posix()} $]\n"+fragment.read_text(),encoding="utf-8")
    expanded=parse(combined)
    for theorem in expanded.proved_theorems.values():verify(theorem,expanded)
    assert expanded.proved_theorems["v21_result"].conclusion==target.conclusion
    if official:
        import os
        executable=Path(os.environ.get("METAMATH_EXECUTABLE",ROOT/"outputs/tools/metamath/metamath.exe"))
        if executable.is_file():
            assert verify_bundle(fragment,SOURCE,executable,30)["status"]=="passed"


def test_boolean_plumbing_and_equality_reconstruct(db,tmp_path):
    b=context(db);x,y,z=map(N,("x0","x1","x2"))
    h1,h2=b.hyp(eq(x,y)),b.hyp(eq(y,z))
    p=b.congr("+",(b.trans(h1,h2),b.refl(x)))
    target=Theorem("t",[Hypothesis("h1",N("|-",h1.formula)),Hypothesis("h2",N("|-",h2.formula))],N("|-",p.formula),
                   variable_types=b.types.copy(),d_constraints=b.distinct.copy())
    certify(db,b,p,target,tmp_path,True)


def test_quantified_horn_reconstruct(db,tmp_path):
    b=context(db);x,y,z=map(N,("x0","x1","x2"))
    h=N("forall",z,imp(eq(z,x),eq(N("S",z),y)))
    target=Theorem("t",[Hypothesis("h",N("|-",h))],N("|-",eq(N("S",x),y)),
                   variable_types=b.types.copy(),d_constraints=b.distinct.copy())
    p=Reconstructor(b,time.perf_counter()+10).solve(target.conclusion.args[0],(b.hyp(h),))
    assert p is not None
    certify(db,b,p,target,tmp_path,True)


def test_induction_is_a_proof_not_an_extra_axiom(db,tmp_path):
    b=context(db);x=N("x0")
    goal=N("forall",x,eq(N("+",N("0"),x),x))
    p=Reconstructor(b,time.perf_counter()+15).induction(goal)
    assert p is not None and not p.assumptions
    target=Theorem("t",[],N("|-",goal),variable_types={"x0":"var","x1":"var","x2":"var"},
                   d_constraints=set(itertools.combinations(("x0","x1","x2"),2)))
    certify(db,b,p,target,tmp_path,True)
    assert "$a " not in (tmp_path/"fragment.mm").read_text()


def test_unsupported_quantifier_and_meta_syntax():
    tr=Translator({"x":"var","phi":"wff"})
    with pytest.raises(UnsupportedFragment):tr.formula(N("exists",N("x"),eq(N("x"),N("0"))))
    with pytest.raises(UnsupportedFragment):tr.formula(N("phi"))
    with pytest.raises(UnsupportedFragment):tr.term(N("S",N("0"),N("0")),{})


def test_free_parameters_are_rigid_and_binders_shadow():
    tr=Translator({"x":"var"})
    assert tr.formula(eq(N("x"),N("x")))=="(c0 = c0)"
    assert "X0 = X0" in tr.formula(N("forall",N("x"),eq(N("x"),N("x"))))


def test_capture_and_eigenvariable_rejected(db):
    b=context(db);x=N("x0")
    with pytest.raises(ValueError):b.forall_intro(x,b.hyp(eq(x,N("0"))))
    with pytest.raises(ValueError):b.forall_elim(b.hyp(N("forall",x,eq(x,x))),N("S",x))


def test_cannot_export_wrong_target_or_open_lemma(db,tmp_path):
    b=context(db);x=N("x0")
    target=Theorem("t",[],N("|-",eq(x,N("0"))),variable_types=b.types)
    with pytest.raises(ValueError):b.export(b.refl(x),target,tmp_path/"bad.mm")
    with pytest.raises(ValueError):b.export(b.hyp(target.conclusion.args[0]),target,tmp_path/"bad.mm")


def test_false_equalities_and_budget_are_unknown(db):
    b=context(db)
    assert Reconstructor(b,time.perf_counter()+1).solve(eq(N("0"),N("S",N("0")))) is None
    with pytest.raises(BudgetExpired):Reconstructor(b,time.perf_counter()-1).solve(eq(N("0"),N("0")))


def test_definitional_extension_certifies(db,tmp_path):
    b=context(db);x=N("x0")
    goal=N("natdiff",x,x,N("0"))
    p=Reconstructor(b,time.perf_counter()+10).solve(goal)
    assert p is not None
    target=Theorem("t",[],N("|-",goal),variable_types=b.types.copy(),d_constraints=b.distinct.copy())
    certify(db,b,p,target,tmp_path)


def test_source_target_statements_never_exported(db):
    target=Theorem("t",[],N("|-",eq(N("0"),N("0"))),variable_types={})
    text=problem(db,target)
    assert "statement" not in text and "induction" not in text
    assert "a_pa_ax3" in text


def test_conditional_generation_avoids_vacuous_premises():
    from pa_prover_v21.data import premise_witness
    x,z=N("x0"),N("x2")
    assert premise_witness([eq(x,N("S",x))]) is None
    assert premise_witness([eq(x,N("0"))]) is not None
    assert premise_witness([N("forall",z,imp(eq(z,x),eq(N("S",z),x)))]) is None


def test_atp_claim_without_reconstruction_is_not_success(db,tmp_path,monkeypatch):
    from pa_prover_v21 import engine
    monkeypatch.setattr(engine,"run_vampire",lambda *a,**k:{"claimed":True,"status":"Theorem","used_inputs":[]})
    monkeypatch.setattr(engine,"verify_bundle",lambda *a,**k:pytest.fail("must not certify missing reconstruction"))
    target=Theorem("false",[],N("|-",eq(N("0"),N("S",N("0")))),variable_types={})
    result=engine.attempt(db,SOURCE,target,"atp",2,tmp_path/"attempt","unused","unused")
    assert not result["certified"] and result["reconstruction"]=="gap"


def test_old_artifact_cannot_be_reused(db,tmp_path):
    from pa_prover_v21.engine import attempt
    target=Theorem("t",[],N("|-",eq(N("0"),N("0"))),variable_types={})
    with pytest.raises(FileExistsError):attempt(db,SOURCE,target,"atp",1,tmp_path,"unused","unused")


def test_official_verifier_rejects_tampered_goal(db,tmp_path):
    import os
    executable=Path(os.environ.get("METAMATH_EXECUTABLE",ROOT/"outputs/tools/metamath/metamath.exe"))
    if not executable.is_file():pytest.skip("official Metamath unavailable")
    b=context(db);x=N("x0");p=b.refl(x)
    target=Theorem("t",[],N("|-",p.formula),variable_types=b.types,d_constraints=b.distinct)
    fragment=tmp_path/"fragment.mm";b.export(p,target,fragment)
    fragment.write_text(fragment.read_text().replace("v21_result $p |- = x0 x0", "v21_result $p |- = x0 0"))
    assert verify_bundle(fragment,SOURCE,executable,10)["status"]=="failed"

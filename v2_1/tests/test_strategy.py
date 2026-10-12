import itertools
import json
from pathlib import Path
import time
import os

import pytest

from metamath_generator.parser import parse
from metamath_generator.model import Theorem
from pa_prover_v21.kernel import N, eq, Builder
from pa_prover_v21.strategy import Action, Session
from pa_prover_v21.solver import Reconstructor
from pa_prover_v21.search import search, SearchConfig, Classical, legal_candidates
from pa_prover_v21.library import Library, archive, replay

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"formal/peano-pa-plus.mm"
MM=Path(os.environ.get("METAMATH_EXECUTABLE",ROOT/"outputs/tools/metamath/metamath.exe"))

@pytest.fixture(scope="module")
def db():return parse(SOURCE)

def target(goal):
    types={"x0":"var","x1":"var","x2":"var"}
    return Theorem("target",[],N("|-",goal),variable_types=types,d_constraints=set(itertools.combinations(types,2)))

def test_multigoal_induction_and_archive(db,tmp_path):
    x=N("x0");t=target(N("forall",x,eq(N("+",N("0"),x),x)))
    s=Session(db,t);st=s.apply(s.initial,Action("INDUCT",0,variable="x0"))
    assert len(st.frontier)==2
    obs=s.observation(st,1)
    assert {g["role"] for g in obs["goals"]}=={"base","step"}
    for i in tuple(st.frontier):
        p=Reconstructor(s.b,time.perf_counter()+10).solve(st.tasks[i].goal,st.tasks[i].facts)
        assert p is not None
        st=s.close(st,i,p)
    assert st.solved
    r=archive(s.b,st.proofs[0]);b=Builder(db,r["types"],map(tuple,r["distinct"]))
    p,_=replay(b,r);assert p.formula==t.conclusion.args[0] and not p.assumptions
    if MM.exists():
        from pa_prover_v21.atp import verify_bundle
        s.b.export(p,t,tmp_path/"fragment.mm")
        assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

def test_have_blocks_unproved_lemma_and_rejects_scope_escape(db):
    x=N("x0");s=Session(db,target(eq(N("+",N("0"),x),x)))
    st=s.apply(s.initial,Action("HAVE",0,statement=str(eq(x,x))))
    assert st.frontier==(1,) and len(s.observation(st,2)["goals"])==2
    with pytest.raises(ValueError):s.apply(st,Action("INDUCT",2,variable="x0"))
    with pytest.raises(ValueError):s.close(st,1,s.b.hyp(eq(x,x)))
    st=s.close(st,1,s.b.refl(x))
    assert st.frontier==(2,) and st.tasks[2].facts[-1].formula==eq(x,x)

def test_branch_isolation_and_legal_candidates(db):
    x=N("x0");s=Session(db,target(eq(N("+",N("0"),x),x)))
    before=s.observation(s.initial,1)
    a=legal_candidates(s,s.initial,8);b=legal_candidates(s,s.initial,8)
    assert [x[0].text() for x in a]==[x[0].text() for x in b]
    assert s.observation(s.initial,1)==before
    assert [x[1].observation(x[2],1) for x in a]==[x[1].observation(x[2],1) for x in b]

def test_type_and_distinct_constraints_checked_before_export(db):
    b=Builder(db,{"x0":"var","x1":"var"},())
    with pytest.raises(ValueError):b.ax("pa_ax3",{"x":N("+",N("x0"),N("0"))})
    with pytest.raises(ValueError):b.forall_elim(b.hyp(N("forall",N("x0"),eq(N("x0"),N("x1")))),N("x1"))

@pytest.mark.parametrize("kind",["HAVE","STRENGTHEN","UNFOLD"])
def test_macro_certificates(db,tmp_path,kind):
    x=N("x0")
    t=target(N("natdiff",x,x,N("0")) if kind=="UNFOLD" else eq(x,x))
    s=Session(db,t)
    st=s.apply(s.initial,Action(kind,0,statement="= 0 0" if kind!="UNFOLD" else ""))
    backend=Classical(s,SearchConfig(closure_seconds=2),time.perf_counter()+20,tmp_path)
    st=backend.close(st);assert st.solved
    if MM.exists():
        from pa_prover_v21.atp import verify_bundle
        s.b.export(st.proofs[0],t,tmp_path/"fragment.mm")
        assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

@pytest.mark.skipif(not MM.exists(),reason="official verifier not installed")
def test_conditional_library_matches_premise_parameters(db,tmp_path):
    from metamath_generator.model import Hypothesis
    x,y,z=N("x0"),N("x1"),N("x2");t=target(eq(x,z))
    t.hypotheses=[Hypothesis("h0",N("|-",eq(x,y))),Hypothesis("h1",N("|-",eq(y,z)))]
    s=Session(db,t);p=s.b.trans(*s.initial.tasks[0].facts)
    library=Library().admit(s.b,p,t,SOURCE,MM,tmp_path/"library",origin="unit-test")
    t2=target(eq(x,y));t2.hypotheses=[Hypothesis("h0",N("|-",eq(x,N("0")))),Hypothesis("h1",N("|-",eq(N("0"),y)))]
    u=Session(db,t2,library);st=u.apply(u.initial,Action("USE",0,lemma=next(iter(library.entries))))
    assert len(st.frontier)==2
    st=Classical(u,SearchConfig(closure_seconds=2),time.perf_counter()+20,tmp_path).close(st)
    assert st.solved
    from pa_prover_v21.atp import verify_bundle
    u.b.export(st.proofs[0],t2,tmp_path/"fragment.mm")
    assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

def test_source_fingerprint_binds_entry_point():
    from pa_prover_v21.library import source_digest
    assert source_digest(SOURCE)!=source_digest(ROOT/"formal/peano.mm")

@pytest.mark.skipif(not MM.exists(),reason="official verifier not installed")
def test_universally_quantified_library_lemma(db,tmp_path):
    x=N("x0");t=target(N("forall",x,eq(x,x)));s=Session(db,t)
    p=s.b.forall_intro(x,s.b.refl(x))
    library=Library().admit(s.b,p,t,SOURCE,MM,tmp_path/"memory",origin="unit-test")
    term=N("S",N("x1"));u=Session(db,target(eq(term,term)),library)
    st=u.apply(u.initial,Action("USE",0,lemma=next(iter(library.entries))))
    assert st.solved
    from pa_prover_v21.atp import verify_bundle
    u.b.export(st.proofs[0],u.target,tmp_path/"fragment.mm")
    assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

@pytest.mark.skipif(not MM.exists(),reason="official verifier not installed")
def test_search_certified_and_library_compound_instantiation(db,tmp_path):
    x=N("x0");t=target(eq(N("+",N("0"),x),x))
    result,s,st=search(db,SOURCE,t,tmp_path/"search",MM,config=SearchConfig(seconds=15,max_candidates=4),return_session=True)
    assert result["certified"] and len(result["trace"])>=1
    memory=Library().admit(s.b,st.proofs[0],t,SOURCE,MM,tmp_path/"memory",origin="train-round-0")
    path=tmp_path/"library.json";memory.save(path,SOURCE)
    memory=Library.load(path,db,SOURCE)
    baseline=search(db,SOURCE,t,tmp_path/"classical",MM,config=SearchConfig(seconds=10),policy="atp",library=memory)
    assert baseline["certified"]
    term=N("S",N("x1"));u=Session(db,target(eq(N("+",N("0"),term),term)),memory)
    solved=u.apply(u.initial,Action("USE",0,lemma=next(iter(memory.entries))))
    assert solved.solved
    from pa_prover_v21.atp import verify_bundle
    u.b.export(solved.proofs[0],u.target,tmp_path/"fragment.mm")
    assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

@pytest.mark.skipif(not MM.exists(),reason="official verifier not installed")
def test_conventional_induction_and_last_expansion_can_finish(db,tmp_path):
    x=N("x0");t=target(eq(N("+",N("0"),x),x))
    for method in ("induction","heuristic"):
        result=search(db,SOURCE,t,tmp_path/method,MM,policy=method,config=SearchConfig(seconds=15,max_expansions=1,max_candidates=1))
        assert result["certified"]

def test_archive_tampering_and_cycles_rejected(db):
    s=Session(db,target(eq(N("x0"),N("x0"))));r=archive(s.b,s.b.refl(N("x0")))
    r["nodes"][0]["parents"]=[0]
    with pytest.raises(ValueError):replay(s.b,r)

def test_induction_with_generalized_parameter(db,tmp_path):
    x,y=N("x0"),N("x1")
    t=target(N("and",eq(N("+",N("0"),x),x),eq(y,y)))
    s=Session(db,t);st=s.apply(s.initial,Action("INDUCT",0,variable="x0",generalize=("x1",)))
    for i in tuple(st.frontier):
        p=Reconstructor(s.b,time.perf_counter()+20).solve(st.tasks[i].goal,st.tasks[i].facts)
        assert p is not None
        st=s.close(st,i,p)
    assert st.solved and st.proofs[0].formula==t.conclusion.args[0]
    if MM.exists():
        from pa_prover_v21.atp import verify_bundle
        s.b.export(st.proofs[0],t,tmp_path/"fragment.mm")
        assert verify_bundle(tmp_path/"fragment.mm",SOURCE,MM,30)["status"]=="passed"

def test_group_splits_and_timeout_semantics():
    from pa_prover_v21.curriculum import split_groups,usable_outcome
    rows=[{"full_key":"a","conclusion_key":"a","skeleton":"s","lineage":"g"},
          {"full_key":"b","conclusion_key":"b","skeleton":"s","lineage":""},
          {"full_key":"c","conclusion_key":"c","skeleton":"t","lineage":"g"}]
    split_groups(rows)
    assert len({r["group"] for r in rows})==1 and len({r["split"] for r in rows})==1
    assert usable_outcome({"status":"unknown_timeout"})
    assert not usable_outcome({"status":"unknown_verification"})

def test_cross_round_leakage_and_rlvr_censoring():
    pytest.importorskip("torch")
    from pa_prover_v21.training import register_splits,rewards_and_advantages
    class Model:provenance={}
    model=Model();case={"full_key":"a","conclusion_key":"b","skeleton":"s","split":"test"}
    register_splits(model,[case])
    with pytest.raises(ValueError):register_splits(model,[{**case,"split":"train"}])
    assert rewards_and_advantages([{"status":"unknown_verification","certified":False}]) is None
    rewards,advantages=rewards_and_advantages([{"status":"proved","certified":True},{"status":"unknown_timeout","certified":False}])
    assert rewards.tolist()==[1.,0.] and advantages.tolist()==[1.,-1.]

def test_checkpoint_roundtrip_and_legacy_rejection(tmp_path):
    torch=pytest.importorskip("torch")
    from pa_prover_v21.policy import MacroPolicy,PolicyConfig
    m=MacroPolicy(PolicyConfig(d_model=16,n_heads=2,n_layers=1,max_seq_len=2048))
    path=tmp_path/"policy.pt";m.save(path);n=MacroPolicy.load(path)
    assert all(torch.equal(p,q) for p,q in zip(m.parameters(),n.parameters()))
    torch.save({"format":"old-v1"},path)
    with pytest.raises(ValueError):MacroPolicy.load(path)

def test_policy_kv_cache_and_learning_objectives():
    torch=pytest.importorskip("torch")
    from pa_prover_v21.policy import MacroPolicy,PolicyConfig,multi_positive_loss,ppo_loss
    torch.manual_seed(21);m=MacroPolicy(PolicyConfig(d_model=16,n_heads=2,n_layers=1,max_seq_len=2048))
    ids=m.ids("abcdef",bos=True);full=m(ids)[0];first,_,cache=m(ids[:,:3]);rest=m(ids[:,3:],cache)[0]
    assert torch.allclose(full[:,3:],rest,atol=1e-6)
    lp=torch.tensor([.2,.3,.5]).log().requires_grad_()
    assert multi_positive_loss(lp,[0,1]).item()==pytest.approx(-__import__("math").log(.5))
    loss=ppo_loss(lp,lp.detach(),2,torch.tensor(1.),lp.detach());loss.backward()
    assert torch.isfinite(lp.grad).all()
    with pytest.raises(ValueError):m(torch.ones((1,2049),dtype=torch.long))

def test_policy_cache_scores_equal_full_token_likelihood():
    torch=pytest.importorskip("torch")
    from pa_prover_v21.policy import MacroPolicy,PolicyConfig,prompt
    m=MacroPolicy(PolicyConfig(d_model=16,n_heads=2,n_layers=1,max_seq_len=2048,exploration=0));obs={"goals":[],"budget":1}
    actions=['{"a":1}','{"a":2}'];lp,_=m.distribution(obs,actions)
    scores=[];offset=len(m.tokenizer.encode(prompt(obs),bos=True))
    for a in actions:
        ids=m.ids(prompt(obs)+a,bos=True,eos=True)
        logits=m(ids[:,:-1])[0]
        scores.append(logits[:,offset-1:].log_softmax(-1).gather(-1,ids[:,offset:,None]).sum())
    assert torch.allclose(lp,torch.stack(scores).log_softmax(0),atol=1e-4)

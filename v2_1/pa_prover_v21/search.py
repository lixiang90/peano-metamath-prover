"""Alternating classical closure and bounded high-level policy search."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import json
from pathlib import Path
import random
import time

from .strategy import Session, Action
from .kernel import N, eq
from .solver import Reconstructor, BudgetExpired
from .atp import DEFINITIONS, problem, run_vampire, verify_bundle


@dataclass
class SearchConfig:
    seconds: float = 10.
    closure_seconds: float = .75
    max_expansions: int = 64
    max_depth: int = 5
    max_candidates: int = 32
    max_nodes: int = 600
    seed: int = 21
    stochastic: bool = False
    temperature: float = 1.
    verification_reserve: float = .2

    def validate(self):
        if min(self.seconds,self.closure_seconds,self.temperature) <= 0:
            raise ValueError("budgets and temperature must be positive")
        if min(self.max_expansions,self.max_depth,self.max_candidates,self.max_nodes) < 1:
            raise ValueError("search limits must be positive")


def proposals(task, builder, limit=8):
    """A small typed, target-conditioned grammar, without named theorems.

    All policies see this same grammar. Formula invention outside the bounded
    grammar is deliberately deferred; no answer-proof nodes enter the prompt.
    """
    g=task.goal; variables=sorted({n.op for n in g.walk() if not n.args and builder.types.get(n.op)=="var"})
    out=[]
    for v in variables:
        candidate=N("forall",N(v),g)
        if not any(N(v) in set(a.walk()) for p in task.facts for a in p.assumptions):out.append(candidate)
    terms=sorted({n for n in g.walk() if n.op in {"0","S","+","*"} or builder.types.get(n.op)=="var"},key=lambda n:(len(list(n.walk())),str(n)))[:8]
    for a,b in itertools.combinations(terms,2):
        if a.op==b.op and a.args and b.args and len(a.args)==len(b.args):
            out.extend(eq(x,y) for x,y in zip(a.args,b.args) if x!=y)
        out.append(eq(a,b))
    seen={g,*(p.formula for p in task.facts)}
    for n in out:
        if n in seen:continue
        seen.add(n)
        yield str(n)
        limit-=1
        if limit<=0:return


def candidates(session,state,limit=32):
    groups=[]
    for i in state.frontier:
        task=state.tasks[i];g=task.goal;b=session.b;actions=[]
        if "df-"+g.op in DEFINITIONS:actions.append(Action("UNFOLD",i))
        vs=sorted({n.op for n in g.walk() if not n.args and b.types.get(n.op)=="var"
                   and not any(n in set(a.walk()) for p in task.facts for a in p.assumptions)})
        for v in vs:
            actions.append(Action("INDUCT",i,variable=v))
        for v in vs:
            for w in vs:
                if v!=w:actions.append(Action("INDUCT",i,variable=v,generalize=(w,)))
        if session.library:
            actions.extend(Action("USE",i,lemma=e["id"]) for e in session.library.retrieve([g],8))
        for f in proposals(task,b):
            actions.extend((Action("HAVE",i,statement=f),Action("STRENGTHEN",i,statement=f)))
        groups.append(actions)
    # Round robin prevents one large goal from consuming the entire action set.
    result=[]
    for row in itertools.zip_longest(*groups):
        for a in row:
            if a is not None:result.append(a)
            if len(result)>=limit:return result
    return result


def legal_candidates(session,state,limit=32):
    result=[]
    for action in candidates(session,state,limit):
        child_session=session.fork()
        try:child=child_session.apply(state,action)
        except (ValueError,KeyError):continue
        result.append((action,child_session,child))
    return result


class Classical:
    def __init__(self,session,config,deadline,directory,vampire=None):
        self.session,self.config,self.deadline=session,config,deadline
        self.directory,self.vampire=Path(directory),vampire
        self.cache={};self.calls=0;self.events=[]

    def solve(self,task):
        b=self.session.b
        key=(task.goal,tuple((p.formula,p.assumptions) for p in task.facts),frozenset(b.distinct))
        # Positive proofs only: a failed bounded attempt is not a refutation.
        if key in self.cache:return self.cache[key]
        left=self.deadline-time.perf_counter()
        if left<=0:raise BudgetExpired()
        self.calls+=1
        deadline=min(self.deadline,time.perf_counter()+self.config.closure_seconds)
        try:p=Reconstructor(b,deadline,max_nodes=self.config.max_nodes).solve(task.goal,task.facts)
        except (BudgetExpired,ValueError):p=None
        if p is None and self.session.library:
            for entry in self.session.library.retrieve([task.goal],8):
                if time.perf_counter()>=self.deadline:raise BudgetExpired()
                try:
                    candidate,premises=self.session.library.instantiate(entry["id"],task,b)
                    solver=Reconstructor(b,min(self.deadline,time.perf_counter()+self.config.closure_seconds),max_nodes=self.config.max_nodes)
                    for premise in premises:
                        minor=solver.solve(premise,task.facts)
                        if minor is None:break
                        candidate=b.mp(minor,candidate)
                    else:p=candidate;break
                except (BudgetExpired,ValueError,KeyError):continue
        if p is None and self.vampire and self.deadline-time.perf_counter()>.05:
            text=problem(b.db,self.session.task_theorem(task))
            event=run_vampire(self.vampire,text,self.directory/f"atp-{self.calls}",
                              min(self.config.closure_seconds,self.deadline-time.perf_counter()))
            self.events.append(event)
            if event.get("claimed"):
                # Reconstruction is proof-producing; an external ATP claim
                # that cannot be reconstructed remains unknown.
                try:p=Reconstructor(b,min(self.deadline,time.perf_counter()+self.config.closure_seconds),
                                     max_nodes=self.config.max_nodes*2).solve(task.goal,task.facts)
                except (BudgetExpired,ValueError):p=None
        if p is not None:self.cache[key]=p
        return p

    def close(self,state):
        changed=True
        while changed and not state.solved:
            changed=False
            for i in tuple(state.frontier):
                if time.perf_counter()>=self.deadline:raise BudgetExpired()
                p=self.solve(state.tasks[i])
                if p is not None:
                    state=self.session.close(state,i,p);changed=True;break
                child=self.session.structural(state,i)
                if child is not None:state=child;changed=True;break
        return state


def search(database,source,target,output,metamath,*,config=None,policy="heuristic",model=None,
           library=None,vampire=None,forced=None,return_session=False):
    """A single shared wall budget includes closure, proposals and verification.

    `stochastic` follows one sampled trajectory, with exact categorical action
    log probabilities. Best-first search is used for evaluation, not PPO data.
    """
    cfg=config or SearchConfig();cfg.validate()
    if policy not in {"atp","induction","heuristic","random","learned"}:raise ValueError("unknown policy")
    if policy=="learned" and model is None:raise ValueError("learned policy requires model")
    if model is not None and model.provenance.get("source"):
        from .library import source_digest
        if model.provenance["source"]!=source_digest(source):raise ValueError("checkpoint formal source changed")
    started=time.perf_counter();end=started+cfg.seconds
    deadline=end-min(cfg.verification_reserve,cfg.seconds*.2)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    session=Session(database,target,library)
    backend=Classical(session,cfg,deadline,out,vampire)
    rng=random.Random(cfg.seed);queue=[(0.,0,session.initial,[],session)];serial=0;expanded=0;seen=set();attempts=[]
    result={"id":target.name,"method":policy,"certified":False,"status":"unknown_exhausted",
            "budget_seconds":cfg.seconds,"trace":[],"attempts":attempts}
    winner=None
    while queue:
        if time.perf_counter()>=deadline:result["status"]="unknown_timeout";break
        cost,_,state,trace,session=heapq.heappop(queue)
        backend.session=session
        try:state=backend.close(state)
        except BudgetExpired:result["status"]="unknown_timeout";break
        if policy=="induction" and not state.solved:
            # Conventional induction uses the original predicate and no
            # conjectured strengthening. All arms share classical lemma reuse.
            for i in tuple(state.frontier):
                try:
                    proof=Reconstructor(session.b,deadline,max_nodes=cfg.max_nodes).induction(state.tasks[i].goal,state.tasks[i].facts)
                    if proof is not None:state=session.close(state,i,proof)
                except (ValueError,BudgetExpired):continue
        if state.solved:
            fragment=out/"fragment.mm"
            session.b.export(state.proofs[0],target,fragment)
            left=end-time.perf_counter()
            if left<=0:result["status"]="unknown_timeout";break
            verification=verify_bundle(fragment,source,metamath,left)
            result["verification"]=verification
            if verification["status"]=="passed" and time.perf_counter()<=end:
                result.update(certified=True,status="proved",trace=trace,certificate=str(fragment.resolve()));winner=state
            else:result["status"]="unknown_verification" if verification["status"]!="passed" else "unknown_timeout"
            break
        if policy in {"atp","induction"} or len(trace)>=cfg.max_depth or expanded>=cfg.max_expansions:continue
        key=session.key(state)
        if key in seen:continue
        seen.add(key);expanded+=1
        prepared=legal_candidates(session,state,cfg.max_candidates)
        actions=[a for a,_,_ in prepared]
        if not actions:continue
        obs=session.observation(state,deadline-time.perf_counter())
        texts=[a.text() for a in actions]
        if policy=="learned":
            try:logps,value=model.distribution(obs,texts,cfg.temperature)
            except ValueError as exc:
                attempts.append({"status":"unknown_context", "error":str(exc)});continue
            weights=[float(p) for p in logps.exp().detach().cpu()]
        else:
            weights=[1. if policy=="random" else 1./(i+1) for i in range(len(actions))]
            total=sum(weights);weights=[w/total for w in weights];value=0.
        import math
        if forced is not None and not trace:
            # Counterfactual action must belong to the same frozen legal grammar.
            indices=[texts.index(forced)] if forced in texts else []
        elif cfg.stochastic:indices=[rng.choices(range(len(actions)),weights=weights,k=1)[0]]
        elif policy=="random":indices=rng.sample(range(len(actions)),len(actions))
        else:indices=sorted(range(len(actions)),key=lambda i:-weights[i])
        for rank,i in enumerate(indices):
            if time.perf_counter()>=deadline:break
            entry={"observation":obs,"candidates":texts,"action":texts[i],"index":i,
                   "log_prob":math.log(max(weights[i],1e-38)),"distribution":weights,
                   "log_distribution":[math.log(max(w,1e-38)) for w in weights],
                   "temperature":cfg.temperature,"stochastic":cfg.stochastic,"state_key":key}
            _,child_session,child=prepared[i]
            attempts.append({**entry,"status":"expanded"})
            serial+=1
            uncertainty=-.1*math.log(max(float(value),.01)) if policy=="learned" else 0.
            heapq.heappush(queue,(cost+1.-math.log(max(weights[i],1e-38))*.1+uncertainty,serial,child,trace+[entry],child_session))
            if cfg.stochastic:result["trace"]=trace+[entry]
    result.update(seconds=time.perf_counter()-started,expansions=expanded,classical_calls=backend.calls,
                  external_calls=len(backend.events),library_snapshot=library.fingerprint if library else "empty")
    (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return (result,session,winner) if return_session else result

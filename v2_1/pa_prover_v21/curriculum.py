"""Axioms-only proposal pools, certified macro traces and grouped curricula."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import random
import time

from metamath_generator.model import Theorem
from metamath_generator.parser import parse
from metamath_generator.quality import semantic_profile
from .kernel import N, eq, Builder
from .data import record, induction_candidates
from .engine import from_record
from .library import digest, source_digest, archive, replay, Library
from .search import search, SearchConfig, Classical, legal_candidates
from .strategy import Session, Action
from .atp import problem, verify_bundle


def split_groups(rows):
    """Union full/conclusion duplicates, structural families and graph ancestry.

    Splits are assigned *before* solving. A failed teacher cannot move a case
    into another split or silently remove it from the evaluation population.
    """
    parent=list(range(len(rows)));keys={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for key in ["f:"+row["full_key"],"c:"+row["conclusion_key"],"s:"+row["skeleton"],
                    *( ["l:"+row["lineage"]] if row.get("lineage") else [] )]:
            if key in keys:parent[root(i)]=root(keys[key])
            else:keys[key]=i
    groups=defaultdict(list)
    for i in range(len(rows)):groups[root(i)].append(i)
    for indices in groups.values():
        identity=digest(sorted(rows[i]["full_key"] for i in indices))
        bucket=int(identity[:8],16)%10;split="test" if bucket==0 else "validation" if bucket==1 else "train"
        for i in indices:rows[i].update(group=identity,split=split)
    return rows


def skeleton(target):
    def visit(n):
        return [target.variable_types.get(n.op,n.op),*[visit(a) for a in n.args]]
    return digest([visit(target.conclusion),*[visit(h.expr) for h in target.hypotheses]])


def proposal_pool(database,count,seed,graph_steps=100):
    from metamath_generator.generator import TheoremGenerator, GenerationConfig
    if database.proved_theorems:raise ValueError("strict self-generation forbids source $p theorem libraries")
    rng=random.Random(seed);rows=[];seen=set();rejected=Counter()
    def add(t,channel,lineage=""):
        try:
            problem(database,t);Session(database,t)
        except (ValueError,KeyError):rejected["unsupported_fragment"]+=1;return
        profile=semantic_profile(t)
        if profile.full_key in seen:rejected["duplicate"]+=1;return
        seen.add(profile.full_key)
        r=record(t,channel);r.update(id=f"self_{len(rows):05}",full_key=digest(profile.full_key),
            conclusion_key=digest(profile.conclusion_key),skeleton=skeleton(t),lineage=lineage)
        rows.append(r)
    # Forward proofs provide one channel; do not seed named target statements.
    generator=TheoremGenerator(database,GenerationConfig(seed=seed,max_proof_depth=12,max_hypotheses=3))
    generated=generator.generate("graph",steps=graph_steps)
    for t in generated:
        if len(rows)>=max(1,count//3):break
        add(t,"graph",f"graph-{seed}")
    graph_admitted=len(rows)
    # The second channel proposes identities from a bounded arithmetic grammar.
    # Finite evaluations filter proposals only; they are never proof labels.
    types={"x0":"var","x1":"var","x2":"var"};distinct=set(itertools.combinations(types,2))
    pairs=list(induction_candidates(rng))
    bands=defaultdict(list)
    for pair in pairs:
        bands[sum(len(list(t.walk())) for t in pair)].append(pair)
    # Mix small and larger constructions from the outset, rather than allowing
    # the combinatorial number of large terms to erase the bootstrap examples.
    buckets=[iter(bands[k]) for k in sorted(bands)]
    balanced=(p for row in itertools.zip_longest(*buckets) for p in row if p is not None)
    for a,b in balanced:
        if len(rows)>=count:break
        g=eq(a,b)
        if rng.random()<.5:g=N("forall",N("x0"),g)
        if rng.random()<.2:g=N("and",g,N("forall",N("x1"),eq(N("+",N("x1"),N("0")),N("x1"))))
        add(Theorem("proposal",[],N("|-",g),variable_types=types,d_constraints=distinct),"conjecture")
    if len(rows)<count:raise ValueError("proposal grammar exhausted before requested count")
    return split_groups(rows),{"graph_generated":len(generated),"graph_admitted":graph_admitted,"rejected":dict(rejected)}


def usable_outcome(result):
    # A measured budget failure is a zero for this operational success event,
    # never a mathematical refutation. Verifier/configuration failures censor.
    return result["status"] not in {"unknown_verification","error"} and not any(
        a.get("status")=="unknown_context" for a in result.get("attempts",()))


def corpus_weights(rows):
    counts=Counter((r["difficulty"],r["structure"],r["premises"]) for r in rows)
    for r in rows:r["sampling_weight"]=1/counts[r["difficulty"],r["structure"],r["premises"]]**.5


def preserve_splits(cases,model):
    if model is None:return
    registry=model.provenance.get("split_registry",{})
    groups=defaultdict(list)
    for c in cases:groups[c["group"]].append(c)
    for group in groups.values():
        existing={registry[k+":"+c[k]] for c in group for k in ("full_key","conclusion_key","skeleton") if k+":"+c[k] in registry}
        if len(existing)>1:raise ValueError("new ancestry joins prior disjoint splits; choose a different graph seed")
        if existing:
            label=existing.pop()
            for c in group:c["split"]=label


def generate(source,output,metamath,*,count=24,seed=21,seconds=5.,graph_steps=100,
             trials=2,counterfactuals=4,library=None,vampire=None,model=None):
    if count<1 or trials<2 or counterfactuals<0:raise ValueError("positive count and at least two value trials required")
    out=Path(output)
    if out.exists():raise ValueError("use a fresh corpus directory")
    out.mkdir(parents=True);db=parse(source)
    memory=Library.load(library,db,source) if library else Library()
    cases,pool_info=proposal_pool(db,count,seed,graph_steps)
    preserve_splits(cases,model)
    # Libraries must originate in older training rounds and carry split metadata.
    for e in memory.entries.values():
        origin=e.get("origin",{})
        if not isinstance(origin,dict) or origin.get("split")!="train":raise ValueError("curriculum library needs training provenance")
        if any(c["split"]!="train" and (c["conclusion_key"]==origin.get("conclusion_key") or c["skeleton"]==origin.get("skeleton")) for c in cases):
            raise ValueError("frozen library overlaps held-out proposal families")
    (out/"cases.jsonl").write_text("".join(json.dumps(c)+"\n" for c in cases),encoding="utf-8")
    memory.save(out/"library.json",source)
    examples=[];outcomes=[];cfg=SearchConfig(seconds=seconds,seed=seed)
    for ci,c in enumerate(cases):
        t=from_record(c,db);directory=out/"proofs"/c["id"]
        baseline=search(db,source,t,directory/"baseline",metamath,config=cfg,policy="atp",library=memory,vampire=vampire)
        successes=[]
        teacher,session,winner=search(db,source,t,directory/"teacher",metamath,config=cfg,
            policy="heuristic",library=memory,vampire=vampire,return_session=True)
        if teacher["certified"]:
            successes.append(teacher)
            (directory/"proof.json").write_text(json.dumps(archive(session.b,winner.proofs[0])),encoding="utf-8")
        if model:
            learned,ls,lw=search(db,source,t,directory/"learned-teacher",metamath,config=cfg,
                policy="learned",model=model,library=memory,vampire=vampire,return_session=True)
            if learned["certified"]:
                successes.append(learned)
                if not teacher["certified"]:
                    (directory/"proof.json").write_text(json.dumps(archive(ls.b,lw.proofs[0])),encoding="utf-8")
        # Counterfactuals use the same root and equal total budget. Costs include
        # proving an introduced lemma, so HAVE cannot receive free oracle facts.
        root_entry=next((a for a in teacher["attempts"] if "candidates" in a),None)
        counter=[]
        if root_entry:
            for ai,a in enumerate(root_entry["candidates"][:counterfactuals]):
                r=search(db,source,t,directory/f"counterfactual-{ai}",metamath,config=cfg,
                         policy="heuristic",library=memory,vampire=vampire,forced=a)
                counter.append({"action":a,"certified":r["certified"],"status":r["status"],"seconds":r["seconds"],
                                "gain_vs_atp":int(r["certified"])-int(baseline["certified"])})
                if r["certified"]:successes.append(r)
        # Repeated randomized rollouts calibrate the initial state's budgeted
        # value independently of selecting a successful teacher trajectory.
        sampled=[]
        for j in range(trials):
            r=search(db,source,t,directory/f"rollout-{j}",metamath,
                     config=replace(cfg,seed=seed+ci*trials+j,stochastic=True),policy="learned" if model else "random",
                     model=model,library=memory,vampire=vampire)
            sampled.append(r)
            if r["certified"]:successes.append(r)
        depth=min((len(r["trace"]) for r in successes),default=None)
        difficulty="atp_easy" if baseline["certified"] else "macro_solved" if successes else "budget_unsolved"
        structure=",".join(sorted({n.op for n in t.conclusion.walk() if n.args}))
        meta={"case":c["id"],"group":c["group"],"split":c["split"],"difficulty":difficulty,
              "structure":structure,"premises":len(t.hypotheses)}
        grouped={}
        for r in successes:
            for step in r["trace"]:
                key=digest([step["state_key"],step["candidates"]])
                if key not in grouped:grouped[key]={**meta,"kind":"policy","observation":step["observation"],
                    "candidates":step["candidates"],"positives":[],"evidence":[]}
                e=grouped[key]
                if step["action"] not in e["positives"]:e["positives"].append(step["action"])
                # Keep the complete trajectory and certificate for auditing.
                e["evidence"].append({"trace":r["trace"],"certificate_status":r["verification"]["status"],
                    "certificate":Path(r["certificate"]).relative_to(out.resolve()).as_posix()})
        examples.extend(grouped.values())
        valid=[r for r in sampled if usable_outcome(r)]
        if valid:
            initial=Session(db,t,memory)
            examples.append({**meta,"kind":"value","observation":initial.observation(initial.initial,seconds),
                "value":sum(r["certified"] for r in valid)/len(valid),"trials":len(valid),
                "policy":"learned-frozen-round" if model else "random-macro-v1","censored":len(sampled)-len(valid)})
        outcomes.append({**meta,"baseline":baseline["certified"],"teacher":teacher["certified"],
            "teacher_status":teacher["status"],"macro_depth":depth,"counterfactuals":counter,
            "rollout_successes":sum(r["certified"] for r in valid),"rollout_trials":len(valid)})
        print(f"{ci+1}/{count}: {difficulty}; macro depth={depth}; split={c['split']}",flush=True)
    corpus_weights(examples)
    for name,rows in (("examples",examples),("outcomes",outcomes)):
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    from .atp import sha256
    manifest={"format":"v21-strategy-corpus-1","source":source_digest(source),"seed":seed,
        "cases":len(cases),"examples":len(examples),"pool":pool_info,"search":asdict(cfg),
        "library_snapshot":memory.fingerprint,"human_theorem_library":False,
        "splits":dict(Counter(c["split"] for c in cases)),"groups":len({c["group"] for c in cases}),
        "files":{name:sha256(out/name) for name in ("cases.jsonl","examples.jsonl","outcomes.jsonl","library.json")},
        "limits":"bounded PA grammar and reconstructable fragment; unsolved proposals retained; not representative of all PA+"}
    manifest["files"].update({p.relative_to(out).as_posix():sha256(p) for p in (out/"proofs").rglob("*")
                              if p.name in {"proof.json","fragment.mm","result.json"}})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def read_corpus(path,source):
    from .atp import sha256
    root=Path(path);manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format")!="v21-strategy-corpus-1" or manifest["source"]!=source_digest(source):raise ValueError("corpus/source mismatch")
    for name,checksum in manifest["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError("corpus path escapes directory")
        if sha256(path)!=checksum:raise ValueError("corpus checksum mismatch: "+name)
    cases=[json.loads(l) for l in (root/"cases.jsonl").read_text(encoding="utf-8").splitlines()]
    examples=[json.loads(l) for l in (root/"examples.jsonl").read_text(encoding="utf-8").splitlines()]
    labels={c["id"]:c for c in cases};seen={}
    for c in cases:
        for k in (c["group"],c["full_key"],c["conclusion_key"],c["skeleton"]):
            if k in seen and seen[k]!=c["split"]:raise ValueError("train/evaluation leakage")
            seen[k]=c["split"]
    for e in examples:
        if e["split"]!=labels[e["case"]]["split"] or e["group"]!=labels[e["case"]]["group"]:raise ValueError("example split differs")
        if e["kind"]=="policy" and not set(e["positives"])<=set(e["candidates"]):raise ValueError("positive outside candidate set")
        if e["kind"]=="policy" and (not e.get("evidence") or any(x["certificate"] not in manifest["files"] for x in e["evidence"])):
            raise ValueError("policy target lacks a bound certificate artifact")
        if e["kind"]=="value" and (not 0<=e["value"]<=1 or e["trials"]<1):raise ValueError("invalid value evidence")
    return manifest,cases,examples


def audit(source,corpus,metamath):
    manifest,cases,examples=read_corpus(corpus,source);db=parse(source);root=Path(corpus)
    checked=0;certificates=0
    for c in cases:
        path=root/"proofs"/c["id"]/"proof.json"
        if not path.exists():continue
        r=json.loads(path.read_text());b=Builder(db,r["types"],map(tuple,r["distinct"]))
        p,_=replay(b,r);t=from_record(c,db)
        directory=path.parent/"audit";directory.mkdir(exist_ok=True)
        b.export(p,t,directory/"fragment.mm")
        if verify_bundle(directory/"fragment.mm",source,metamath,30)["status"]!="passed":raise ValueError("certificate audit failed")
        checked+=1
    for path in (root/"proofs").glob("*/*/result.json"):
        result=json.loads(path.read_text(encoding="utf-8"))
        if not result["certified"]:continue
        fragment=path.parent/"fragment.mm"
        if verify_bundle(fragment,source,metamath,30)["status"]!="passed":raise ValueError("trajectory certificate failed")
        certificates+=1
    report={"proof_archives_replayed":checked,"official_certificates_replayed":certificates,"examples":len(examples),"split_counts":manifest["splits"],
            "policy_examples":sum(e["kind"]=="policy" for e in examples),
            "value_histogram":dict(Counter(str(e["value"]) for e in examples if e["kind"]=="value"))}
    (root/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

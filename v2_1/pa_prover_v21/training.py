"""Explicit NTP/SFT, expert iteration and verifiable-reward training entrypoints.

Nothing in data generation or inference starts training implicitly.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from metamath_generator.parser import parse
from .curriculum import read_corpus, usable_outcome
from .engine import from_record
from .library import Library, digest
from .policy import MacroPolicy, PolicyConfig, prompt, multi_positive_loss, ppo_loss
from .search import search, SearchConfig


def train(source,corpus,output,*,stage="sft",steps=100,seed=21,device="cpu",checkpoint=None,
          learning_rate=3e-4,d_model=256,n_layers=4,n_heads=8,max_seq_len=8192):
    if stage not in {"ntp","sft"} or steps<1 or learning_rate<=0:raise ValueError("invalid training configuration")
    manifest,cases,rows=read_corpus(corpus,source)
    rows=[r for r in rows if r["split"]=="train"]
    if not rows:raise ValueError("no training examples; increase proposal pool without moving held-out cases")
    out=Path(output)
    if out.exists():raise ValueError("use a fresh training directory")
    out.mkdir(parents=True);torch.manual_seed(seed);rng=random.Random(seed)
    model=MacroPolicy.load(checkpoint,device) if checkpoint else MacroPolicy(PolicyConfig(
        d_model=d_model,n_heads=n_heads,n_layers=n_layers,max_seq_len=max_seq_len)).to(device)
    if model.provenance.get("source",manifest["source"])!=manifest["source"]:raise ValueError("checkpoint formal source changed")
    model.provenance["source"]=manifest["source"]
    register_splits(model,cases)
    model.provenance.setdefault("corpora",[]).append(digest(manifest))
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.01)
    weights=[r.get("sampling_weight",1.) for r in rows];model.train();history=[];skipped=0;started=time.perf_counter()
    for step in range(steps):
        row=rng.choices(rows,weights=weights,k=1)[0];optimizer.zero_grad(set_to_none=True)
        try:
            if stage=="ntp":
                suffix=rng.choice(row["positives"]) if row["kind"]=="policy" else ""
                # Conventional next-token prediction over the entire synthetic
                # sequence. No auxiliary token mask changes pretraining NTP.
                loss=model.sequence_loss(prompt(row["observation"])+suffix)
            elif row["kind"]=="policy":
                lp,_=model.distribution(row["observation"],row["candidates"],grad=True)
                pos=[row["candidates"].index(p) for p in row["positives"]]
                loss=multi_positive_loss(lp,pos)+.2*model.sequence_loss(rng.choice(row["positives"]),prompt(row["observation"]))
            else:
                _,v,_=model(model.ids(prompt(row["observation"]),bos=True))
                loss=F.binary_cross_entropy_with_logits(v,torch.tensor([row["value"]],device=model.device))
        except ValueError as exc:
            if "context" not in str(exc):raise
            skipped+=1;history.append({"step":step,"status":"context_overflow"});continue
        if not torch.isfinite(loss):raise ValueError("non-finite training loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        history.append({"step":step,"kind":row["kind"],"loss":float(loss.detach()),"case":row["case"]})
        (out/"live.json").write_text(json.dumps({"stage":stage,**history[-1],"elapsed":time.perf_counter()-started}),encoding="utf-8")
    if skipped==steps:raise ValueError("all complete states exceed context; no model update performed")
    summary={"stage":stage,"requested_steps":steps,"updates":steps-skipped,"context_overflows":skipped,
             "seconds":time.perf_counter()-started,"examples":len(rows),"seed":seed,"history":history,
             "full_training":False if steps<=2 else None,"library_snapshot":manifest["library_snapshot"]}
    model.save(out/"policy.pt",training=summary,optimizer=optimizer.state_dict())
    (out/"training.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def case_keys(case):
    return {name+":"+case[name] for name in ("full_key","conclusion_key","skeleton")}


def register_splits(model,cases):
    train_keys=set(model.provenance.get("training_keys",()))
    heldout=set(model.provenance.get("heldout_keys",()))
    registry=dict(model.provenance.get("split_registry",{}))
    for c in cases:
        keys=case_keys(c)
        if any(k in registry and registry[k]!=c["split"] for k in keys):raise ValueError("split changed across training rounds")
        registry.update({k:c["split"] for k in keys})
        if c["split"]=="train":
            if keys & heldout:raise ValueError("new round trains on a previous held-out family")
            train_keys.update(keys)
        else:
            if keys & train_keys:raise ValueError("new held-out family was previously trained")
            heldout.update(keys)
    if train_keys & heldout:raise ValueError("cross-round train/evaluation leakage")
    model.provenance.update(training_keys=sorted(train_keys),heldout_keys=sorted(heldout),split_registry=registry)


def rewards_and_advantages(results):
    if not results or any(not usable_outcome(r) for r in results):return None
    rewards=torch.tensor([float(r["certified"]) for r in results])
    std=rewards.std(unbiased=False)
    if std==0:return rewards,torch.zeros_like(rewards)
    return rewards,(rewards-rewards.mean())/std.clamp_min(1e-6)


def rlvr(source,corpus,output,metamath,checkpoint,*,groups=10,group_size=4,seconds=5.,seed=21,
         device="cpu",learning_rate=1e-5,vampire=None):
    if groups<1 or group_size<2:raise ValueError("positive groups and at least two rollouts per group required")
    manifest,all_cases,_=read_corpus(corpus,source);cases=[c for c in all_cases if c["split"]=="train"]
    if not cases:raise ValueError("no train cases")
    db=parse(source);library=Library.load(Path(corpus)/"library.json",db,source)
    model=MacroPolicy.load(checkpoint,device);register_splits(model,all_cases);reference=copy.deepcopy(model).eval()
    if model.provenance.get("source",manifest["source"])!=manifest["source"]:raise ValueError("checkpoint formal source changed")
    model.provenance["source"]=manifest["source"]
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate);rng=random.Random(seed)
    out=Path(output)
    if out.exists():raise ValueError("use a fresh RLVR directory")
    out.mkdir(parents=True);logs=[];updates=0
    for group in range(groups):
        case=rng.choice(cases);target=from_record(case,db);results=[]
        # Freeze the actor and library for the entire rollout group.
        model.eval()
        for j in range(group_size):
            r=search(db,source,target,out/f"group-{group}"/f"rollout-{j}",metamath,
                config=SearchConfig(seconds=seconds,seed=seed+group*group_size+j,stochastic=True),
                policy="learned",model=model,library=library,vampire=vampire)
            results.append(r)
        scored=rewards_and_advantages(results)
        if scored is None:logs.append({"group":group,"status":"censored_infrastructure_or_context"});continue
        rewards,advantages=scored
        if not advantages.any():logs.append({"group":group,"status":"zero_advantage","rewards":rewards.tolist()});continue
        optimizer.zero_grad(set_to_none=True);count=sum(len(r["trace"]) for r in results)
        if not count:logs.append({"group":group,"status":"no_policy_actions"});continue
        total=0.
        for ri,result in enumerate(results):
            for step in result["trace"]:
                if not step["stochastic"]:raise ValueError("best-first teacher trace is not on-policy PPO data")
                lp,v=model.distribution(step["observation"],step["candidates"],step["temperature"],grad=True)
                ref,_=reference.distribution(step["observation"],step["candidates"],step["temperature"])
                old=torch.tensor(step["log_distribution"],device=model.device)
                # Guard against accidentally changing scoring between rollout
                # and update (length normalization, masks, temperature, etc.).
                if not torch.allclose(lp.detach(),old,atol=1e-4,rtol=1e-4):raise ValueError("rollout/update action probabilities differ")
                adv=advantages[ri].to(model.device)
                loss=ppo_loss(lp,old,step["index"],adv,ref)+.1*F.binary_cross_entropy(v,rewards[ri].to(model.device))
                (loss/count).backward();total+=float(loss.detach())/count
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
        logs.append({"group":group,"status":"updated","loss":total,"rewards":rewards.tolist(),"actions":count})
        (out/"live.json").write_text(json.dumps(logs[-1]),encoding="utf-8")
    summary={"groups":groups,"updates":updates,"group_size":group_size,"logs":logs,
             "library_snapshot":library.fingerprint,"reward":"officially verified complete root certificate within budget"}
    model.save(out/"policy.pt",rlvr=summary,optimizer=optimizer.state_dict())
    (out/"rlvr.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def promote(source,corpus,output,metamath):
    """Create the next immutable snapshot from prior training proofs only."""
    from .library import replay
    from .kernel import Builder
    _,cases,_=read_corpus(corpus,source);db=parse(source)
    memory=Library.load(Path(corpus)/"library.json",db,source);out=Path(output)
    if out.exists():raise ValueError("use a fresh library directory")
    out.mkdir(parents=True);count=0
    for c in cases:
        proof=Path(corpus)/"proofs"/c["id"]/"proof.json"
        if c["split"]!="train" or not proof.exists():continue
        r=json.loads(proof.read_text());b=Builder(db,r["types"],map(tuple,r["distinct"]))
        p,_=replay(b,r);t=from_record(c,db)
        if p.formula!=t.conclusion.args[0]:raise ValueError("promotion target mismatch")
        memory=memory.admit(b,p,t,source,metamath,out/c["id"],origin={k:c[k] for k in ("id","split","group","conclusion_key","skeleton")})
        count+=1
    memory.save(out/"library.json",source)
    return {"promoted":count,"snapshot":memory.fingerprint,"entries":len(memory.entries)}


def expert_iteration(source,output,metamath,checkpoint,*,count=24,steps=100,seconds=5.,seed=22,device="cpu",library=None,vampire=None):
    """One explicit frozen round; no implicit unbounded self-training loop."""
    from .curriculum import generate,audit
    out=Path(output)
    if out.exists():raise ValueError("use a fresh expert iteration directory")
    out.mkdir(parents=True)
    actor=MacroPolicy.load(checkpoint,device)
    data=generate(source,out/"corpus",metamath,count=count,seconds=seconds,seed=seed,model=actor,library=library,vampire=vampire)
    audit_result=audit(source,out/"corpus",metamath)
    del actor
    trained=train(source,out/"corpus",out/"training",steps=steps,seed=seed,device=device,checkpoint=checkpoint,stage="sft")
    promoted=promote(source,out/"corpus",out/"next-library",metamath)
    result={"data":data,"audit":audit_result,"training":trained,"next_library":promoted}
    (out/"iteration.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

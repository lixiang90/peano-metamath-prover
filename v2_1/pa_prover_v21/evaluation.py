"""Process-isolated equal-wall-budget comparison of macro policies."""
import json
import multiprocessing as mp
from pathlib import Path
import time

from .benchmark import summarize
from .curriculum import read_corpus
from .library import Library
from .search import SearchConfig


def arm(method, default_mode="baseline"):
    parts=method.rsplit("-",1)
    if len(parts)==2 and parts[1] in {"a","b","c"}:
        policy=parts[0];mode={"a":"baseline","b":"feedback","c":"scratchpad"}[parts[1]]
    else:policy,mode=method,default_mode
    if policy not in {"atp","induction","heuristic","random","learned"}:
        raise ValueError("unknown comparison arm")
    if policy in {"atp","induction"}:mode="baseline"
    return policy,mode


def arm_checkpoint(cfg,mode):
    return cfg.get("checkpoint_"+mode) or cfg.get("checkpoint")


def worker(connection,cfg,method):
    try:
        from metamath_generator.parser import parse
        from .engine import from_record
        from .search import search
        db=parse(cfg["source"]);memory=Library.load(Path(cfg["corpus"])/"library.json",db,cfg["source"])
        policy,mode=arm(method,cfg["reasoning_mode"])
        model=None
        if policy=="learned":
            from .policy import MacroPolicy
            model=MacroPolicy.load(arm_checkpoint(cfg,mode),cfg["device"])
        connection.send({"ready":True})
        while True:
            case=connection.recv()
            if case is None:break
            target=from_record(case,db);out=Path(cfg["output"])/method/case["id"]
            result=search(db,cfg["source"],target,out,cfg["metamath"],config=SearchConfig(seconds=cfg["seconds"],seed=cfg["seed"],
                reasoning_mode=mode,trial_seconds=cfg["trial_seconds"],max_trials=cfg["max_trials"],max_interactions=cfg["max_interactions"]),
                policy=policy,model=model,library=memory,vampire=cfg["vampire"])
            for k in ("attempts","trace","interaction","receipts","initial_observation"):result.pop(k,None)
            result["reasoning_mode"]=mode
            result.update(id=case["id"],method=method);connection.send(result)
    except Exception as exc:
        connection.send({"error":f"{type(exc).__name__}: {exc}"})
    finally:connection.close()


def evaluate_strategy(source,corpus,output,metamath,vampire,*,checkpoint=None,device="cpu",seconds=10.,seed=21,
                      methods="atp,induction,heuristic,random",split="test",reasoning_mode="baseline",
                      trial_seconds=.25,max_trials=4,max_interactions=24,checkpoint_feedback=None,checkpoint_scratchpad=None):
    if seconds<=0 or split not in {"train","validation","test"}:raise ValueError("invalid evaluation budget/split")
    methods=methods.split(",")
    SearchConfig(reasoning_mode=reasoning_mode,trial_seconds=trial_seconds,max_trials=max_trials,max_interactions=max_interactions).validate()
    if len(set(methods))!=len(methods):raise ValueError("duplicate comparison arm")
    checkpoints={"baseline":checkpoint,"feedback":checkpoint_feedback or checkpoint,"scratchpad":checkpoint_scratchpad or checkpoint}
    for method in methods:
        policy,mode=arm(method,reasoning_mode)
        if policy=="learned" and not checkpoints[mode]:raise ValueError("learned arm needs checkpoint")
    manifest,cases,_=read_corpus(corpus,source);cases=[c for c in cases if c["split"]==split]
    if not cases:raise ValueError("chosen grouped split is empty; generate a larger pool")
    for method in methods:
        policy,mode=arm(method,reasoning_mode)
        if policy!="learned" or split=="train":continue
        from .policy import MacroPolicy
        from .training import case_keys
        prior=set(MacroPolicy.load(checkpoints[mode]).provenance.get("training_keys",()))
        if any(case_keys(c)&prior for c in cases):raise ValueError("evaluation overlaps checkpoint training families")
    out=Path(output)
    if out.exists():raise ValueError("use a fresh evaluation directory")
    out.mkdir(parents=True)
    cfg=dict(source=source,corpus=corpus,output=output,metamath=metamath,vampire=vampire,checkpoint=checkpoint,
             device=device,seconds=seconds,seed=seed,reasoning_mode=reasoning_mode,trial_seconds=trial_seconds,
             max_trials=max_trials,max_interactions=max_interactions,checkpoint_feedback=checkpoint_feedback,checkpoint_scratchpad=checkpoint_scratchpad)
    ctx=mp.get_context("spawn");results=[]
    def start(method):
        parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,cfg,method));p.start();child.close()
        if not parent.poll(120):p.terminate();p.join();raise RuntimeError("evaluation initialization timed out")
        r=parent.recv()
        if not r.get("ready"):p.join();raise RuntimeError(str(r))
        return p,parent
    for method in methods:
        process,connection=start(method)
        try:
            for c in cases:
                started=time.perf_counter();connection.send(c)
                if connection.poll(seconds+.5):
                    r=connection.recv()
                    if "error" in r:raise RuntimeError(str(r))
                else:
                    process.terminate();process.join();connection.close()
                    r={"id":c["id"],"method":method,"certified":False,"status":"unknown_watchdog","seconds":time.perf_counter()-started}
                    process,connection=start(method)
                results.append(r)
                (out/"results.jsonl").write_text("".join(json.dumps(x)+"\n" for x in results),encoding="utf-8")
        finally:
            if process.is_alive():connection.send(None);process.join(5)
            if process.is_alive():process.terminate();process.join()
            connection.close()
    from .atp import sha256
    report={**summarize(results,cases),"split":split,"cases":len(cases),"budget_seconds":seconds,
            "library_snapshot":manifest["library_snapshot"],"manifest":manifest,
            "reasoning_modes":{m:arm(m,reasoning_mode)[1] for m in methods},
            "interaction_limits":{"trial_seconds":trial_seconds,"max_trials":max_trials,"max_interactions":max_interactions},
            "provenance":{k:sha256(v) for k,v in (("metamath",metamath),("vampire",vampire),
                ("checkpoint",checkpoint),("checkpoint_feedback",checkpoint_feedback),("checkpoint_scratchpad",checkpoint_scratchpad)) if v},
            "note":"only certified root proofs count; frozen library; test is not used for updates"}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

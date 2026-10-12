"""Frozen equal-wall-time comparison, with a process watchdog and provenance."""
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

from .atp import sha256


def worker(connection, configuration, method):
    try:
        from metamath_generator.parser import parse
        from .engine import attempt, from_record
        db = parse(configuration["source"])
        model = None
        if method in {"model","hybrid"}:
            from .neural import FrozenModel
            model = FrozenModel(configuration["checkpoint"],configuration["tokenizer"],db,configuration["device"])
        connection.send({"ready":True})
        while True:
            row = connection.recv()
            if row is None: break
            target = from_record(row,db)
            result = attempt(db,configuration["source"],target,method,configuration["seconds"],
                Path(configuration["output"])/method/row["id"],configuration["vampire"],configuration["metamath"],model)
            connection.send(result)
    except Exception as exc:
        connection.send({"worker_error":f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def wilson(successes,total):
    if not total: return [0.,1.]
    z=1.959963984540054; p=successes/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.,center-half),min(1.,center+half)]


def summarize(results,cases):
    methods=sorted({r["method"] for r in results})
    families={c["id"]:c["family"] for c in cases}
    aggregate={}
    for m in methods:
        rs=[r for r in results if r["method"]==m]
        n=sum(r["certified"] for r in rs)
        aggregate[m]={"attempts":len(rs),"certified":n,"rate":n/len(rs),"wilson95":wilson(n,len(rs)),
            "seconds":sum(r["seconds"] for r in rs),
            "statuses":{s:sum(r["status"]==s for r in rs) for s in sorted({r["status"] for r in rs})},
            "by_family":{f:{"attempts":sum(families[r["id"]]==f for r in rs),
                                "certified":sum(r["certified"] and families[r["id"]]==f for r in rs)}
                         for f in sorted(set(families.values()))}}
    paired={}
    for a,b in itertools.combinations(methods,2):
        aa={r["id"]:r["certified"] for r in results if r["method"]==a}
        bb={r["id"]:r["certified"] for r in results if r["method"]==b}
        common=aa.keys()&bb.keys()
        aw=sum(aa[k] and not bb[k] for k in common); bw=sum(bb[k] and not aa[k] for k in common)
        n=aw+bw
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(aw,bw)+1))/2**n) if n else 1.
        paired[a+" vs "+b]={"a_only":aw,"b_only":bw,"both":sum(aa[k] and bb[k] for k in common),"mcnemar_exact_p":p}
    return {"aggregate":aggregate,"paired":paired}


def evaluate(configuration,methods=("atp","induction","model","hybrid")):
    cfg=dict(configuration)
    out=Path(cfg["output"])
    if out.exists(): raise ValueError("use a fresh evaluation directory")
    if cfg["seconds"]<=0: raise ValueError("positive per-case budget required")
    out.mkdir(parents=True)
    source=Path(cfg["source"]).resolve()
    manifest=json.loads((Path(cfg["benchmark"])/"manifest.json").read_text())
    for name,digest in manifest["source_files_sha256"].items():
        if sha256(source.parent/name)!=digest: raise ValueError("formal source changed since benchmark generation")
    casepath=Path(cfg["benchmark"])/"cases.jsonl"
    if sha256(casepath)!=manifest["cases_sha256"]: raise ValueError("benchmark cases changed")
    cases=[json.loads(line) for line in casepath.read_text().splitlines()]
    provenance={key:sha256(cfg[key]) for key in ("vampire","metamath")}
    provenance["v21_python_files"]={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if any(m in {"model","hybrid"} for m in methods):
        provenance.update({key:sha256(cfg[key]) for key in ("checkpoint","tokenizer")})
    results=[]; ctx=mp.get_context("spawn")
    def start(method):
        parent,child=ctx.Pipe()
        process=ctx.Process(target=worker,args=(child,cfg,method))
        process.start();child.close()
        if not parent.poll(120):
            process.terminate();process.join();raise RuntimeError("worker initialization timed out")
        ready=parent.recv()
        if not ready.get("ready"):
            process.join();raise RuntimeError(str(ready))
        return process,parent
    for method in methods:
        process,connection=start(method)
        try:
            for row in cases:
                began=time.perf_counter();connection.send(row)
                if connection.poll(cfg["seconds"]+.5):
                    result=connection.recv()
                    if "worker_error" in result: raise RuntimeError(str(result))
                else:
                    process.terminate();process.join()
                    connection.close()
                    result={"id":row["id"],"method":method,"certified":False,
                        "status":"unknown_watchdog","seconds":time.perf_counter()-began,"budget_seconds":cfg["seconds"]}
                    process,connection=start(method)
                results.append(result)
                with (out/"results.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(result)+"\n")
                print(f"{method} {row['id']} {result['status']} {result['seconds']:.2f}s",flush=True)
        finally:
            if process.is_alive(): connection.send(None)
            process.join(5)
            if process.is_alive():process.terminate();process.join()
            connection.close()
    report={"version":"2.1.0","configuration":cfg,"provenance":provenance,
        "benchmark_manifest":manifest,"results":results,**summarize(results,cases),
        "budget_policy":"same end-to-end wall-clock budget; includes ATP, reconstruction, export and official verification; excludes common source/model load; watchdog grace is never counted as success",
        "no_new_training":True,
        "model_prior_caveat":"existing checkpoint is a historical comparator; inference target hints are disabled, but past training priors cannot be removed without retraining"}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

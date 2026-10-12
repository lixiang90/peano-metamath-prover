import argparse
import json


def main(argv=None):
    p=argparse.ArgumentParser(description="PA+ v2.1 certified ATP and high-level neural policy")
    sub=p.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build-benchmark")
    build.add_argument("source");build.add_argument("output")
    build.add_argument("--metamath",required=True)
    build.add_argument("--count",type=int,default=24)
    build.add_argument("--seed",type=int,default=21012)
    build.add_argument("--training-corpus",action="append",default=[])
    ev=sub.add_parser("evaluate")
    ev.add_argument("source");ev.add_argument("benchmark");ev.add_argument("output")
    ev.add_argument("--vampire",required=True);ev.add_argument("--metamath",required=True)
    ev.add_argument("--checkpoint");ev.add_argument("--tokenizer")
    ev.add_argument("--device",default="cuda");ev.add_argument("--seconds",type=float,default=10.)
    ev.add_argument("--methods",default="atp,induction,model,hybrid")
    gen=sub.add_parser("generate-strategy",help="self-generate a grouped macro-policy corpus")
    gen.add_argument("source");gen.add_argument("output");gen.add_argument("--metamath",required=True)
    gen.add_argument("--count",type=int,default=24);gen.add_argument("--seed",type=int,default=21)
    gen.add_argument("--seconds",type=float,default=5.);gen.add_argument("--graph-steps",type=int,default=100)
    gen.add_argument("--trials",type=int,default=2);gen.add_argument("--counterfactuals",type=int,default=4)
    gen.add_argument("--library");gen.add_argument("--vampire");gen.add_argument("--checkpoint");gen.add_argument("--device",default="cpu")
    audit=sub.add_parser("audit-strategy")
    audit.add_argument("source");audit.add_argument("corpus");audit.add_argument("--metamath",required=True)
    tr=sub.add_parser("train-policy",help="explicit NTP or multi-positive SFT; never started by generation")
    tr.add_argument("source");tr.add_argument("corpus");tr.add_argument("output")
    tr.add_argument("--stage",choices=("ntp","sft"),default="sft");tr.add_argument("--steps",type=int,default=100)
    tr.add_argument("--checkpoint");tr.add_argument("--device",default="cpu");tr.add_argument("--seed",type=int,default=21)
    tr.add_argument("--learning-rate",type=float,default=3e-4);tr.add_argument("--d-model",type=int,default=256)
    tr.add_argument("--n-heads",type=int,default=8);tr.add_argument("--n-layers",type=int,default=4);tr.add_argument("--max-seq-len",type=int,default=8192)
    rl=sub.add_parser("rlvr")
    for name in ("source","corpus","output"):rl.add_argument(name)
    rl.add_argument("--metamath",required=True);rl.add_argument("--checkpoint",required=True);rl.add_argument("--vampire")
    rl.add_argument("--groups",type=int,default=10);rl.add_argument("--group-size",type=int,default=4)
    rl.add_argument("--seconds",type=float,default=5.);rl.add_argument("--seed",type=int,default=21)
    rl.add_argument("--device",default="cpu");rl.add_argument("--learning-rate",type=float,default=1e-5)
    promote=sub.add_parser("promote-library")
    for name in ("source","corpus","output"):promote.add_argument(name)
    promote.add_argument("--metamath",required=True)
    expert=sub.add_parser("expert-iterate",help="one frozen generation/audit/SFT/promotion round")
    expert.add_argument("source");expert.add_argument("output");expert.add_argument("--metamath",required=True)
    expert.add_argument("--checkpoint",required=True);expert.add_argument("--library");expert.add_argument("--vampire")
    expert.add_argument("--count",type=int,default=24);expert.add_argument("--steps",type=int,default=100)
    expert.add_argument("--seconds",type=float,default=5.);expert.add_argument("--seed",type=int,default=22);expert.add_argument("--device",default="cpu")
    strategy=sub.add_parser("evaluate-strategy")
    for name in ("source","corpus","output"):strategy.add_argument(name)
    strategy.add_argument("--metamath",required=True);strategy.add_argument("--vampire",required=True)
    strategy.add_argument("--checkpoint");strategy.add_argument("--device",default="cpu")
    strategy.add_argument("--seconds",type=float,default=10.);strategy.add_argument("--seed",type=int,default=21)
    strategy.add_argument("--methods",default="atp,induction,heuristic,random")
    strategy.add_argument("--split",choices=("train","validation","test"),default="test")
    prove=sub.add_parser("prove-strategy")
    for name in ("source","case","output"):prove.add_argument(name)
    prove.add_argument("--metamath",required=True);prove.add_argument("--vampire");prove.add_argument("--library")
    prove.add_argument("--policy",choices=("atp","induction","heuristic","random","learned"),default="heuristic")
    prove.add_argument("--checkpoint");prove.add_argument("--device",default="cpu");prove.add_argument("--seconds",type=float,default=10.)
    for parser in (gen,expert,strategy,prove):
        parser.add_argument("--reasoning-mode",choices=("baseline","feedback","scratchpad"),default="baseline")
        parser.add_argument("--trial-seconds",type=float,default=.25)
        parser.add_argument("--max-trials",type=int,default=4)
        parser.add_argument("--max-interactions",type=int,default=24)
    strategy.add_argument("--checkpoint-feedback")
    strategy.add_argument("--checkpoint-scratchpad")
    args=vars(p.parse_args(argv));command=args.pop("command")
    if command=="build-benchmark":
        from .data import build
        args["training_corpora"]=args.pop("training_corpus")
        result=build(**args)
    elif command=="evaluate":
        from .benchmark import evaluate
        methods=args.pop("methods").split(",")
        if any(m not in {"atp","induction","model","hybrid"} for m in methods):p.error("unknown method")
        if any(m in {"model","hybrid"} for m in methods) and not (args["checkpoint"] and args["tokenizer"]):p.error("model arms require checkpoint and tokenizer")
        result=evaluate(args,methods)
        result={k:result[k] for k in ("aggregate","paired")}
    elif command=="generate-strategy":
        from .curriculum import generate
        checkpoint=args.pop("checkpoint");device=args.pop("device")
        if checkpoint:
            from .policy import MacroPolicy
            args["model"]=MacroPolicy.load(checkpoint,device)
        result=generate(**args)
    elif command=="audit-strategy":
        from .curriculum import audit
        result=audit(**args)
    elif command=="expert-iterate":
        from .training import expert_iteration
        result=expert_iteration(**args)
    elif command in {"train-policy","rlvr","promote-library"}:
        from .training import train,rlvr,promote
        result={"train-policy":train,"rlvr":rlvr,"promote-library":promote}[command](**args)
    elif command=="evaluate-strategy":
        from .evaluation import evaluate_strategy
        result=evaluate_strategy(**args)
        result={k:result[k] for k in ("aggregate","paired","split","cases")}
    else:
        from pathlib import Path
        from metamath_generator.parser import parse
        from .engine import from_record
        from .library import Library
        from .search import search,SearchConfig
        db=parse(args["source"]);target=from_record(json.loads(Path(args.pop("case")).read_text(encoding="utf-8")),db)
        library=args.pop("library");args["library"]=Library.load(library,db,args["source"]) if library else None
        checkpoint=args.pop("checkpoint");device=args.pop("device")
        if checkpoint:
            from .policy import MacroPolicy
            args["model"]=MacroPolicy.load(checkpoint,device)
        args["config"]=SearchConfig(**{k:args.pop(k) for k in ("seconds","reasoning_mode","trial_seconds","max_trials","max_interactions")})
        result=search(db,target=target,**args)
    print(json.dumps(result,indent=2))

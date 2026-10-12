import argparse
import json


def main(argv=None):
    p=argparse.ArgumentParser(description="PA+ v2.1 certified ATP integration (no training)")
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
    args=vars(p.parse_args(argv));command=args.pop("command")
    if command=="build-benchmark":
        from .data import build
        args["training_corpora"]=args.pop("training_corpus")
        result=build(**args)
    else:
        from .benchmark import evaluate
        methods=args.pop("methods").split(",")
        if any(m not in {"atp","induction","model","hybrid"} for m in methods):p.error("unknown method")
        if any(m in {"model","hybrid"} for m in methods) and not (args["checkpoint"] and args["tokenizer"]):p.error("model arms require checkpoint and tokenizer")
        result=evaluate(args,methods)
        result={k:result[k] for k in ("aggregate","paired")}
    print(json.dumps(result,indent=2))

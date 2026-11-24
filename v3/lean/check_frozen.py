"""Type-check the natural-number interpretation of v2.1 frozen arithmetic goals."""

import argparse
import json
from pathlib import Path
import subprocess

from metamath_generator.model import Node, Theorem
from metamath_generator.parser import parse
from pa_prover_v21.engine import from_record


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "formal" / "peano-pa-plus.mm"
OPERATORS = {
    "S": ("Nat.succ", 1),
    "+": ("Nat.add", 2),
    "*": ("Nat.mul", 2),
    "=": ("Eq", 2),
    "and": ("And", 2),
}


def nat_proposition(target: Theorem) -> str:
    if target.hypotheses or target.conclusion.op != "|-" or len(target.conclusion.args) != 1:
        raise ValueError("expected an arithmetic assertion without open hypotheses")
    if any(kind != "var" for kind in target.variable_types.values()):
        raise ValueError("this probe interprets object variables only")
    names = {name: f"n{i}" for i, name in enumerate(sorted(target.variable_types))}
    free: set[str] = set()

    def emit(node: Node, bound: frozenset[str]) -> str:
        op, args = node.op, node.args
        if not args and op in names:
            if op not in bound:
                free.add(op)
            return names[op]
        if not args and op == "0":
            return "Nat.zero"
        if op == "forall" and len(args) == 2:
            binder, body = args
            if binder.args or binder.op not in names:
                raise ValueError(f"unsupported binder {binder}")
            return f"(∀ {names[binder.op]} : Nat, {emit(body, bound | {binder.op})})"
        operator = OPERATORS.get(op)
        if operator is None or len(args) != operator[1]:
            raise ValueError(f"unsupported arithmetic expression {node}")
        head, _ = operator
        return f"({head} {' '.join(emit(arg, bound) for arg in args)})"

    proposition = emit(target.conclusion.args[0], frozenset())
    for name in reversed(sorted(free)):
        proposition = f"(∀ {names[name]} : Nat, {proposition})"
    return proposition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="v2.1 cases.jsonl with a test split")
    parser.add_argument("--witness-id", help="also check the supplied n = 1*n proof against this case")
    args = parser.parse_args()
    database = parse(SOURCE)
    lines = [(HERE / "Probe.lean").read_text(encoding="utf-8"), "namespace Frozen"]
    labels = {}
    with args.cases.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "test":
                continue
            case_id = row["id"]
            if case_id in labels:
                parser.error(f"duplicate test case {case_id}")
            try:
                proposition = nat_proposition(from_record(row, database))
            except ValueError as error:
                parser.error(f"{case_id}: {error}")
            label = f"goal_{len(labels)}"
            labels[case_id] = label
            lines.append(f"def {label} : Prop := {proposition}")
    if not labels:
        parser.error("the file has no test cases")
    if args.witness_id is not None:
        if args.witness_id not in labels:
            parser.error(f"unknown test case {args.witness_id}")
        lines.append(f"theorem witness : {labels[args.witness_id]} := V3Probe.natOneMul")
        lines.append("#print axioms witness")
    lines.append("end Frozen")
    result = subprocess.run(
        ["lean", "-DwarningAsError=true", "--stdin"], cwd=HERE,
        input=("\n".join(lines) + "\n").encode("utf-8"),
    )
    if result.returncode == 0:
        print(f"Checked {len(labels)} goal declarations.")
        if args.witness_id is not None:
            print(f"Checked an explicit proof of {args.witness_id}.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

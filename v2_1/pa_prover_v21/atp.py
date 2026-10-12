"""Restricted, fail-closed PA-to-TPTP translation and isolated Vampire calls."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import subprocess
import time

from .kernel import N

ARITHMETIC = ("pa_ax3", "pa_ax4", "pa_ax5", "pa_ax6")
DEFINITIONS = ("df-natdiff", "df-inteq", "df-intadd", "df-intmul", "df-intneg",
               "df-pythagorean", "df-pellsolution", "df-arithterm")
PREDICATES = {"natdiff": 3, "inteq": 4, "intadd": 6, "intmul": 6,
              "intneg": 4, "pythagorean": 3, "pellsolution": 3, "arithterm": 4}


class UnsupportedFragment(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Translator:
    def __init__(self, types):
        self.types = types
        self.constants = {v: f"c{i}" for i, v in enumerate(sorted(types))}
        self.counter = 0

    def term(self, node, bound):
        if not node.args and node.op in bound:
            return bound[node.op]
        if not node.args and self.types.get(node.op) in {"var", "term"}:
            return self.constants[node.op]
        if node == N("0"):
            return "zero"
        symbols = {"S": ("succ", 1), "+": ("add", 2), "*": ("mul", 2)}
        if node.op not in symbols or len(node.args) != symbols[node.op][1]:
            raise UnsupportedFragment(f"unsupported term: {node}")
        return symbols[node.op][0] + "(" + ",".join(self.term(c, bound) for c in node.args) + ")"

    def formula(self, node, bound=None):
        bound = {} if bound is None else bound
        op, args = node.op, node.args
        if op == "forall" and len(args) == 2:
            x, body = args
            if x.args or self.types.get(x.op) != "var":
                raise UnsupportedFragment("binder is not a PA object variable")
            variable = f"X{self.counter}"
            self.counter += 1
            return f"(! [{variable}] : ({self.formula(body, {**bound, x.op: variable})}))"
        if op in {"and", "implies", "iff"} and len(args) == 2:
            connective = {"and": "&", "implies": "=>", "iff": "<=>"}[op]
            return f"({self.formula(args[0], bound)} {connective} {self.formula(args[1], bound)})"
        if op in {"=", "<"} and len(args) == 2:
            a, b = (self.term(c, bound) for c in args)
            return f"({a} = {b})" if op == "=" else f"less({a},{b})"
        if op in PREDICATES and len(args) == PREDICATES[op]:
            return f"{op}({','.join(self.term(c, bound) for c in args)})"
        raise UnsupportedFragment(f"unsupported formula: {node}")


def problem(database, target):
    tr = Translator(target.variable_types)
    lines = []
    # Each source axiom is universally closed at the object-logic boundary.
    # Logical Metamath schemas are NOT encoded as natural-number variables.
    for label in (*ARITHMETIC, *DEFINITIONS):
        ax = database.axioms.get(label)
        if ax is None:
            continue
        if ax.hypotheses or ax.d_constraints or any(t not in {"var", "term"} for t in ax.variable_types.values()):
            raise UnsupportedFragment(f"unexpected axiom contract: {label}")
        atr = Translator(ax.variable_types)
        bound = {v: f"V{i}" for i, v in enumerate(sorted(ax.variable_types))}
        formula = atr.formula(ax.conclusion.args[0], bound)
        if bound:
            formula = f"(! [{','.join(bound.values())}] : ({formula}))"
        lines.append(f"fof(a_{label.replace('-', '_')},axiom,{formula}).")
    for i, h in enumerate(target.hypotheses):
        lines.append(f"fof(h{i},axiom,{tr.formula(h.expr.args[0])}).")
    lines.append(f"fof(goal,conjecture,{tr.formula(target.conclusion.args[0])}).")
    return "\n".join(lines) + "\n"


def run_vampire(executable, text, directory, seconds):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "problem.p"
    path.write_text(text, encoding="utf-8")
    command = [str(Path(executable).resolve()), "--mode", "vampire", "--input_syntax", "tptp",
               "--proof", "tptp", "--output_axiom_names", "on", "--time_limit", str(max(1, math.ceil(seconds))),
               "--memory_limit", "1024", str(path.resolve())]
    start = time.perf_counter()
    status, output, code = "error", "", None
    try:
        r = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=max(.01, seconds))
        output = r.stdout + r.stderr
        code = r.returncode
        match = re.search(r"SZS status (\w+)", output)
        status = match.group(1) if match else "error"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        out = exc.stdout or b""
        output = out.decode(errors="replace") if isinstance(out, bytes) else out
    except OSError as exc:
        output = str(exc)
    (root / "vampire.tstp").write_text(output, encoding="utf-8")
    # Proof input names are hints only. The local reconstruction proves the
    # exact original target and has no access to reference certificates.
    used = sorted(set(re.findall(r"file\([^\n]*?,\s*(a_\w+|h\d+)\)", output)))
    return {"status": status, "claimed": status in {"Theorem", "Unsatisfiable"},
            "returncode": code, "seconds": time.perf_counter()-start,
            "used_inputs": used, "command": command}


def verify_bundle(fragment, source, executable, seconds):
    fragment, source = Path(fragment).resolve(), Path(source).resolve()
    bundle = fragment.with_name("certificate.mm")
    # Metamath resolves nested includes from cwd, unlike the Python parser.
    bundle.write_text(f"$[ {source.name} $]\n" + fragment.read_text(encoding="utf-8"), encoding="utf-8")
    start = time.perf_counter()
    command = [str(Path(executable).resolve())]
    try:
        r = subprocess.run(command, input=f'set scroll continuous\nread "{bundle.as_posix()}"\nverify proof *\nexit\n',
                           cwd=source.parent, capture_output=True, text=True, errors="replace", timeout=max(.01, seconds))
        output = r.stdout + r.stderr
        passed = r.returncode == 0 and bool(re.search(r"All proofs .* verified", output, re.I)) and not re.search(r"\?Error|\?Fatal|proof verification error", output, re.I)
        status = "passed" if passed else "failed"
    except subprocess.TimeoutExpired:
        status, output = "timeout", "official verification exceeded remaining budget"
    except OSError as exc:
        status, output = "error", str(exc)
    fragment.with_name("metamath.log").write_text(output, encoding="utf-8")
    return {"status": status, "seconds": time.perf_counter()-start,
            "executable_sha256": sha256(executable)}

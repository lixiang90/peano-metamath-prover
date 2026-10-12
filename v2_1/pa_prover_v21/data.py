"""Self-generated, frozen diagnostic tasks. Reference proofs stay separate."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from metamath_generator.model import Theorem, Hypothesis
from metamath_generator.parser import parse
from metamath_generator.quality import semantic_profile
from .kernel import Builder, N, eq, imp
from .solver import Reconstructor, BudgetExpired
from .atp import problem, verify_bundle, sha256


def record(target, family):
    return {"id": target.name, "family": family,
            "conclusion": str(target.conclusion),
            "hypotheses": [str(h.expr) for h in target.hypotheses],
            "variable_types": target.variable_types,
            "d_constraints": sorted(target.d_constraints)}


def premise_witness(hypotheses):
    """Exhibit one standard-natural assignment for this restricted Horn form."""
    def term(t, values):
        if t.op in values and not t.args:return values[t.op]
        if t == N("0"):return 0
        a=[term(c,values) for c in t.args]
        return a[0]+1 if t.op=="S" else a[0]+a[1] if t.op=="+" else a[0]*a[1]
    def holds(h,values):
        if h.op=="=":return term(h.args[0],values)==term(h.args[1],values)
        if h.op=="forall":
            z,body=h.args
            guard,conclusion=body.args
            if body.op!="implies" or guard.op!="=" or guard.args[0]!=z:
                raise ValueError("non-vacuity check requires a singleton equality guard")
            return holds(conclusion,{**values,z.op:term(guard.args[1],values)})
        raise ValueError("unsupported non-vacuity check")
    for a,b in itertools.product(range(6),repeat=2):
        values={"x0":a,"x1":b}
        if all(holds(h,values) for h in hypotheses):return values
    return None


def induction_candidates(rng):
    """Enumerate a term grammar; concrete testing only proposes conjectures."""
    x = N("x0")
    sizes = {1: [N("0"), x]}
    for size in range(2, 6):
        terms = {N("S", t) for t in sizes[size-1]}
        for i in range(1, size-1):
            for a in sizes[i]:
                for b in sizes[size-1-i]:
                    terms.update((N("+", a, b), N("*", a, b)))
        sizes[size] = sorted(terms, key=str)
    def value(t, n):
        if t == x: return n
        if t == N("0"): return 0
        vals = [value(c, n) for c in t.args]
        return vals[0]+1 if t.op == "S" else vals[0]+vals[1] if t.op == "+" else vals[0]*vals[1]
    buckets = {}
    for terms in sizes.values():
        for t in terms:
            buckets.setdefault(tuple(value(t,n) for n in range(6)), []).append(t)
    pairs = [(a,b) for group in buckets.values() for a,b in itertools.combinations(group,2)
             if x in set(a.walk()) | set(b.walk())]
    rng.shuffle(pairs)
    return iter(pairs)


def build(source, output, metamath, count=24, seed=21012, training_corpora=()):
    if count < 3:
        raise ValueError("at least three cases required")
    out = Path(output)
    if out.exists():
        raise ValueError("use a fresh benchmark directory")
    out.mkdir(parents=True)
    db = parse(source)
    if db.proved_theorems:
        raise ValueError("this axioms-only experiment rejects source $p libraries")
    if training_corpora:
        print("indexing training goals for overlap exclusion",flush=True)
        from neural_prover.benchmark import _training_leakage_index
        leakage = _training_leakage_index(tuple(training_corpora), db)
    else:
        leakage = None
    rng = random.Random(seed)
    pairs = induction_candidates(rng)
    seen, rows = set(), []
    attempts, rejected = 0, {}
    x,y,z = (N(f"x{i}") for i in range(3))
    def term(depth=2):
        if depth <= 0 or rng.random()<.35:
            return rng.choice((N("0"),x,y,N("S",N("0"))))
        op = rng.choice(("S","+","*"))
        return N(op,term(depth-1)) if op == "S" else N(op,term(depth-1),term(depth-1))
    while len(rows) < count and attempts < count*120:
        attempts += 1
        family = ("equational", "first_order", "induction")[len(rows)%3]
        types = {"x0":"var","x1":"var","x2":"var"}
        distinct = set(itertools.combinations(sorted(types),2))
        b = Builder(db,types,distinct)
        solver = Reconstructor(b,time.perf_counter()+8,max_nodes=600)
        hypotheses = []
        try:
            if family == "equational":
                a = term()
                p = solver.normalize(a,())
                left,right = p.formula.args
                if left == right: continue
                goal = p.formula
                # Fold definitions only where they exactly match a proved body.
                if rng.random()<.4 and left.op == "+":
                    goal = N("natdiff",right,*left.args)
                    p = solver.solve(goal)
            elif family == "first_order":
                a,c = term(1),term(1)
                if a == c: continue
                if rng.random()<.5:
                    hypotheses = [eq(x,a),eq(a,c)]
                    goal = eq(N("S",x),N("S",c))
                else:
                    # All terms avoid z, making quantified instantiation legal.
                    hypotheses = [N("forall",z,imp(eq(z,a),eq(N("S",z),c))),eq(x,a)]
                    goal = eq(N("S",x),c)
                if premise_witness(hypotheses) is None:
                    rejected["no_premise_witness"] = rejected.get("no_premise_witness",0)+1
                    continue
                p = solver.solve(goal,tuple(b.hyp(h) for h in hypotheses))
            else:
                a,c = next(pairs)
                goal = N("forall",x,eq(a,c))
                # Reject easy normalisation; induction data must discharge a
                # real induction, not simply reproduce the induction schema.
                if solver.solve(goal) is not None: continue
                p = solver.induction(goal)
            if p is None: continue
            target = Theorem(f"v21_case_{len(rows):03}",
                [Hypothesis(f"h{i}",N("|-",h)) for i,h in enumerate(hypotheses)], N("|-",goal),
                variable_types=types,d_constraints=distinct)
            key = semantic_profile(target).full_key
            if key in seen: continue
            profile = semantic_profile(target)
            collision = leakage is not None and (profile.full_key in leakage.full_keys or profile.conclusion_key in leakage.conclusion_keys)
            if collision:
                rejected["training_collision"] = rejected.get("training_collision",0)+1
                continue
            problem(db,target)  # Apply the same admission rule as every solver.
            reference = out/"references"/target.name
            reference.mkdir(parents=True)
            nodes = b.export(p,target,reference/"fragment.mm")
            verification = verify_bundle(reference/"fragment.mm",source,metamath,30)
            if verification["status"] != "passed":
                raise RuntimeError(f"reference verification failed: {reference}")
            row = record(target,family)
            row.update(reference_nodes=nodes, training_overlap_checked=leakage is not None)
            rows.append(row)
            seen.add(key)
            (out/"cases.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
            print(f"generated {len(rows)}/{count} {family}, {nodes} proof DAG nodes",flush=True)
        except (BudgetExpired, ValueError) as exc:
            name = type(exc).__name__
            rejected[name] = rejected.get(name,0)+1
    if len(rows) != count:
        raise RuntimeError(f"only {len(rows)}/{count} cases generated; partial output is not a complete benchmark")
    manifest = {"version":"2.1.0", "seed":seed,"count":count,"attempts":attempts,
        "rejected":rejected,"source":str(Path(source).resolve()),
        "source_files_sha256":{p.name:sha256(p) for p in sorted(Path(source).parent.glob('*.mm'))},
        "cases_sha256":sha256(out/"cases.jsonl"),"human_theorem_library":False,
        "reference_official_verification":"all passed", "training_corpora":list(training_corpora),
        "training_examples_indexed":leakage.examples if leakage else 0,
        "selection_bias":"restricted grammar; reference-reconstructor-solvable; stratified diagnostic, not representative PA+ benchmark"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

"""v2.1 orchestration with one end-to-end deadline per problem."""
import json
from pathlib import Path
import time

from metamath_generator.model import Theorem, Hypothesis
from .kernel import Builder, N
from .solver import Reconstructor, BudgetExpired
from .atp import problem, run_vampire, verify_bundle, UnsupportedFragment


def from_record(record, database):
    from metamath_generator.parser import MetamathParser
    p = MetamathParser()
    p.database = database
    old = dict(database.variable_types)
    try:
        database.variable_types.update(record["variable_types"])
        conclusion = p.parse_expression(record["conclusion"].split())
        hypotheses = [Hypothesis(f"h{i}", p.parse_expression(h.split())) for i, h in enumerate(record["hypotheses"])]
    finally:
        database.variable_types.clear()
        database.variable_types.update(old)
    return Theorem(record["id"], hypotheses, conclusion,
                   variable_types=dict(record["variable_types"]),
                   d_constraints={tuple(pair) for pair in record["d_constraints"]})


def attempt(database, source, target, method, seconds, directory, vampire, metamath, model=None):
    start = time.perf_counter()
    deadline = start + seconds
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    record = {"method": method, "id": target.name, "budget_seconds": seconds,
              "certified": False, "status": "unknown", "new_training": False}
    try:
        # Same supported fragment for every arm, even the existing model.
        text = problem(database, target)
        proof, b = None, None
        if method != "model":
            fraction = .6 if method == "atp" else .2
            record["atp"] = run_vampire(vampire, text, root, max(.01, seconds*fraction))
            if record["atp"]["status"] == "error":
                raise RuntimeError("Vampire failed; see vampire.tstp")
            b = Builder(database, target.variable_types, target.d_constraints)
            reconstruction_start = time.perf_counter()
            reconstruction_end = deadline if method != "hybrid" else min(deadline, time.perf_counter()+seconds*.2)
            reconstruct = Reconstructor(b, reconstruction_end)
            # The ATP result gates pure ATP acceptance. Proof search during
            # reconstruction never sees the generator's reference proof.
            if record["atp"]["claimed"]:
                used = set(record["atp"]["used_inputs"])
                facts = tuple(b.hyp(h.expr.args[0]) for i,h in enumerate(target.hypotheses)
                              if not used or f"h{i}" in used)
                try:
                    proof = reconstruct.solve(target.conclusion.args[0], facts)
                except BudgetExpired:
                    if method != "hybrid":
                        raise
                record["reconstruction"] = "found" if proof else "gap"
                if proof is None:
                    record["status"] = "reconstruction_gap"
            else:
                record["status"] = "unknown_atp"
            if proof is None and method == "induction":
                facts = tuple(b.hyp(h.expr.args[0]) for h in target.hypotheses)
                proof = reconstruct.induction(target.conclusion.args[0], facts)
                record["induction"] = "found" if proof else "unknown"
                if proof is None:
                    record["status"] = "unknown_induction"
            record["reconstruction_calls"] = reconstruct.calls
            record["reconstruction_seconds"] = time.perf_counter()-reconstruction_start
        if proof is not None:
            if time.perf_counter() >= deadline:
                raise BudgetExpired()
            export_start = time.perf_counter()
            record["proof_nodes"] = b.export(proof, target, root/"fragment.mm")
            record["export_seconds"] = time.perf_counter()-export_start
        elif method in {"model", "hybrid"}:
            if model is None:
                raise ValueError("frozen model is required")
            certificate, metrics = model.prove(target, deadline)
            record["model"] = metrics
            if certificate is not None:
                from neural_prover.certificate import export_certificate
                export_certificate(certificate, root/"fragment.mm", ambient_database=database)
            else:
                record["status"] = "unknown_search"
        if (root/"fragment.mm").exists():
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                raise BudgetExpired()
            record["verification"] = verify_bundle(root/"fragment.mm", source, metamath, remaining)
            record["certified"] = record["verification"]["status"] == "passed" and time.perf_counter() <= deadline
            record["status"] = "certified" if record["certified"] else "verification_"+record["verification"]["status"]
    except UnsupportedFragment as exc:
        record.update(status="unsupported", error=str(exc))
    except BudgetExpired:
        record["status"] = "unknown_budget"
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record["seconds"] = time.perf_counter()-start
    if record["seconds"] > seconds:
        record["certified"] = False
        record["status"] = "unknown_budget"
    (root/"result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

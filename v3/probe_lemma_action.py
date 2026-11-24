"""Probe whether a hand-supplied HAVE outside v2.1's candidate list can yield a certificate."""

import argparse
import json
import time
from pathlib import Path

from metamath_generator.model import Theorem
from metamath_generator.parser import parse
from pa_prover_v21.atp import verify_bundle
from pa_prover_v21.kernel import N, eq
from pa_prover_v21.search import Classical, SearchConfig, legal_candidates
from pa_prover_v21.strategy import Action, Session


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "formal/peano-pa-plus.mm"


def probe(output, verifier):
    output.mkdir(parents=True, exist_ok=True)
    n, zero = N("x0"), N("0")
    left = N("+", zero, n)
    goal = eq(N("+", left, N("S", zero)), N("S", n))
    target = Theorem("v3_control_probe", [], N("|-", goal), variable_types={"x0": "var"})
    session = Session(parse(SOURCE), target)
    cfg = SearchConfig(seconds=30)
    backend = Classical(session, cfg, time.perf_counter() + cfg.seconds, output)
    root = backend.close(session.initial)
    if root.solved:
        raise RuntimeError("the classical solver now closes this probe before a policy decision")

    options = [action for action, _, _ in legal_candidates(session, root, cfg.max_candidates)]
    have = Action("HAVE", 0, statement=str(eq(left, n)))
    state = session.apply(root, have)
    lemma, continuation = state.frames[-1].children
    continuation_blocked = continuation not in state.frontier
    induct = Action("INDUCT", lemma, variable=n.op)
    state = backend.close(session.apply(state, induct))
    if not state.solved:
        raise RuntimeError("the manually supplied actions left open obligations")

    # Check whether the proof of the intermediate equality actually occurs in the final proof DAG.
    pending, used = [state.proofs[0]], set()
    while pending:
        evidence = pending.pop()
        if evidence not in used:
            used.add(evidence)
            pending.extend(evidence.parents)
    fragment = output / "fragment.mm"
    session.b.export(state.proofs[0], target, fragment)
    check = verify_bundle(fragment, SOURCE, verifier, cfg.seconds)
    report = {
        "target": str(target.conclusion), "origin": "manually_supplied_actions",
        "root_candidates": [a.record() for a in options], "candidate_count": len(options),
        "have_listed": have in options, "actions": [have.record(), induct.record()],
        "continuation_blocked_until_lemma": continuation_blocked,
        "lemma_in_final_proof": state.proofs[lemma] in used, "verification": check,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metamath", type=Path, default=ROOT / "outputs/tools/metamath/metamath.exe")
    args = parser.parse_args()
    result = probe(args.output, args.metamath)
    print(json.dumps({k: v for k, v in result.items() if k != "root_candidates"}, indent=2))
    raise SystemExit(result["verification"]["status"] != "passed")

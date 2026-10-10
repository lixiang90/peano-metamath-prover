"""Generate generic proofs and replay them with independent Metamath verifiers.

Run from an installed checkout (or with PYTHONPATH=src). Download mmverify.py
from the official URL separately; this script never fetches or trusts it by
itself. Both external processes must finish successfully for a passing report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from metamath_generator.generic import GenericConfig, GenericGenerator
from metamath_generator.token_mm import TokenDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+")
    parser.add_argument("--metamath", required=True)
    parser.add_argument("--mmverify", help="optional official mmverify.py (does not support interleaved $f/$e)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs/generic-verification")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, source in enumerate(args.databases):
        start = perf_counter()
        db = TokenDatabase.from_file(source)
        generator = GenericGenerator(db, GenericConfig(seed=args.seed))
        generator.generate(args.steps)
        paths = generator.export(root / f"{index}-{Path(source).stem}")
        # Round-trip tests the source isolation, hypothesis declarations, proof
        # ordering and dummy-variable $d declarations, not just in-memory facts.
        replay = TokenDatabase.from_file(paths["metamath"])
        assert replay.verified_proofs == db.verified_proofs + len(generator.generated)
        commands = f'read "{paths["metamath"].as_posix()}"\nverify proof *\nexit\n'
        result = subprocess.run([str(Path(args.metamath).resolve())], input=commands,
                                text=True, capture_output=True, timeout=args.timeout)
        log = result.stdout + result.stderr
        (paths["metamath"].parent / "metamath.log").write_text(log, encoding="utf-8")
        if (result.returncode or "?Error" in log
                or "All proofs in the database were verified" not in log):
            raise RuntimeError(f"official verification failed: {paths['metamath']}")
        python_verifier = None
        if args.mmverify:
            verifier = Path(args.mmverify).resolve()
            with paths["metamath"].open("r", encoding="utf-8") as source_file:
                check = subprocess.run([sys.executable, str(verifier)], stdin=source_file,
                                       text=True, capture_output=True, timeout=args.timeout)
            (paths["metamath"].parent / "mmverify.log").write_text(
                check.stdout + check.stderr, encoding="utf-8")
            if check.returncode:
                raise RuntimeError(f"mmverify.py failed: {paths['metamath']}")
            # mmverify prints one line for each $p at its default verbosity.
            verified = sum(line.startswith("verifying ") for line in check.stderr.splitlines())
            if verified != replay.verified_proofs:
                raise RuntimeError("mmverify did not report every source/generated proof")
            python_verifier = {"sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
                               "verified": verified}
        report = {"source": str(Path(source).resolve()), "source_proofs": db.verified_proofs,
                  "generated": len(generator.generated),
                  "generated_by_type": dict(Counter(
                      g.assertion.expression[0] for g in generator.generated)),
                  "generated_closed": sum(not g.assertion.essential for g in generator.generated),
                  "official_verified": replay.verified_proofs,
                  "mmverify": python_verifier, "statistics": dict(generator.stats),
                  "seconds": round(perf_counter() - start, 3)}
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        (root / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

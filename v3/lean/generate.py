"""Generate a small, reproducible corpus of proof-carrying Lean problems."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
CONTEXT = """(R : Type) (add mul : R → R → R) (f g : R → R) (d x y z : R)
    (hdist : ∀ a b c, Eq (mul a (add b c)) (add (mul a b) (mul a c)))
    (hxy : Eq x y) (hyz : Eq y z) (hx : V3Probe.Divides mul d x)"""
RULES = ("trans", "congr", "witness", "sum", "transport")
MAX_DEPTH = 4


@dataclass(frozen=True)
class Equal:
    left: str
    right: str

    def lean(self) -> str:
        return f"Eq {self.left} {self.right}"


@dataclass(frozen=True)
class Multiple:
    value: str

    def lean(self) -> str:
        return f"V3Probe.Divides mul d {self.value}"


@dataclass(frozen=True)
class Derivation:
    goal: Equal | Multiple
    rule: str
    head: str
    premises: tuple[Derivation, ...] = ()

    def proof_term(self) -> str:
        arguments = [f"({p.proof_term()})" for p in self.premises]
        return " ".join([self.head, *arguments])

    def action(self) -> str:
        if not self.premises:
            return f"exact {self.head}"
        holes = " ".join(f"(?_ : {p.goal.lean()})" for p in self.premises)
        return f"refine {self.head} {holes}"

    def depth(self) -> int:
        return 1 + max((p.depth() for p in self.premises), default=0)


def random_value(rng: random.Random, depth: int) -> str:
    """Sample an expression of type R in the fixed context."""
    if depth == 0 or rng.randrange(3) == 0:
        return rng.choice(("x", "y", "z"))
    operation = rng.choice(("f", "g", "add", "mul"))
    first = random_value(rng, depth - 1)
    if operation in ("f", "g"):
        return f"({operation} {first})"
    return f"({operation} {first} {random_value(rng, depth - 1)})"


def propose(rule: str, facts: list[Derivation], rng: random.Random) -> Derivation | None:
    equalities = [p for p in facts if isinstance(p.goal, Equal)]
    multiples = [p for p in facts if isinstance(p.goal, Multiple)]
    if rule == "trans":
        first = rng.choice(equalities)
        following = [p for p in equalities if p.goal.left == first.goal.right]
        if not following:
            return None
        second = rng.choice(following)
        return Derivation(Equal(first.goal.left, second.goal.right), rule, "Eq.trans", (first, second))
    if rule == "congr":
        premise = rng.choice(equalities)
        fn = rng.choice(("f", "g", "mul d", "add x"))
        goal = Equal(f"({fn} {premise.goal.left})", f"({fn} {premise.goal.right})")
        return Derivation(goal, rule, f"congrArg ({fn})", (premise,))
    if rule == "witness":
        witness = random_value(rng, 2)
        head = f"V3Probe.Divides.intro (mul := mul) (d := d) {witness}"
        return Derivation(Multiple(f"(mul d {witness})"), rule, head)
    if rule == "sum":
        first, second = rng.choices(multiples, k=2)
        goal = Multiple(f"(add {first.goal.value} {second.goal.value})")
        return Derivation(goal, rule, "V3Probe.dividesAdd add mul hdist", (first, second))
    if rule == "transport":
        by_value = {p.goal.value: p for p in multiples}
        matches = [p for p in equalities if p.goal.left in by_value]
        if not matches:
            return None
        equality = rng.choice(matches)
        premise = by_value[equality.goal.left]
        head = "Eq.ndrec (motive := fun t => V3Probe.Divides mul d t)"
        return Derivation(Multiple(equality.goal.right), rule, head, (premise, equality))
    raise ValueError(f"unknown rule {rule!r}")


def generate(seed: int, count: int) -> list[Derivation]:
    rng = random.Random(seed)
    facts = [
        Derivation(Equal("x", "y"), "given", "hxy"),
        Derivation(Equal("y", "z"), "given", "hyz"),
        Derivation(Multiple("x"), "given", "hx"),
    ]
    seen = {p.goal for p in facts}
    output = []
    for _ in range(100 * count):
        proof = propose(rng.choice(RULES), facts, rng)
        if proof is None or proof.goal in seen:
            continue
        seen.add(proof.goal)
        output.append(proof)
        if proof.depth() < MAX_DEPTH:
            facts.append(proof)
        if len(output) == count:
            return output
    raise ValueError(f"found {len(output)} of {count} distinct goals within the sampling budget")


def training_trace(root: Derivation) -> list[dict]:
    """Open the root goal, fill one hole per action, and retain all sibling goals."""
    pending = [root]
    steps = []
    while pending:
        current = pending[0]
        steps.append({
            "goals": [p.goal.lean() for p in pending],
            "rule": current.rule,
            "action": current.action(),
        })
        pending = [*current.premises, *pending[1:]]
    return steps


def write_batch(proofs: list[Derivation], destination: Path) -> None:
    # Keep the mathematical rules identical to Probe.lean and load the basic tactic syntax.
    probe = (HERE / "Probe.lean").read_text(encoding="utf-8")
    source = [
        "prelude\nimport Init.Tactics\nimport Init.Guard\n" + probe.removeprefix("prelude\n"),
        "namespace Generated",
        "set_option linter.unusedVariables false",  # Every problem shares the same context.
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        batch = Path(work) / "batch"
        batch.mkdir()
        with (
            (batch / "problems.jsonl").open("w", encoding="utf-8") as problems,
            (batch / "solutions.jsonl").open("w", encoding="utf-8") as solutions,
        ):
            for proof in proofs:
                goal = proof.goal.lean()
                label = "thm_" + sha256(f"{CONTEXT}\n{goal}".encode("utf-8")).hexdigest()
                steps = training_trace(proof)
                term = proof.proof_term()
                problem = {"id": label, "context": CONTEXT, "goal": goal}
                teacher = {"id": label, "proof_term": term, "steps": steps}
                problems.write(json.dumps(problem, ensure_ascii=False) + "\n")
                solutions.write(json.dumps(teacher, ensure_ascii=False) + "\n")
                source.append(f"theorem {label} {CONTEXT} : {goal} :=\n  {term}")
                source.append(f"example {CONTEXT} : {goal} := by")
                for step in steps:
                    source.append(f"  guard_target = {step['goals'][0]}")
                    source.append(f"  {step['action']}")
        source.append("end Generated\n")
        certificate = "\n".join(source)
        (batch / "Certified.lean").write_text(certificate, encoding="utf-8")
        result = subprocess.run(
            ["lean", "-DwarningAsError=true", "--stdin"], cwd=HERE,
            input=certificate, capture_output=True, encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        if destination.exists():
            raise FileExistsError(f"output path already exists: {destination}")
        batch.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the certified batch")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.out.exists():
        parser.error(f"output path already exists: {args.out}")
    proofs = generate(args.seed, args.count)
    write_batch(proofs, args.out)
    print(f"Lean verified {len(proofs)} proof terms and {len(proofs)} tactic replays.")
    print("Final rules:", dict(sorted(Counter(p.rule for p in proofs).items())))


if __name__ == "__main__":
    main()

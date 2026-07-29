from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .model import Database, Hypothesis, Node, Proof, Theorem, normalized_pair


def _canonicalize(theorem: Theorem) -> tuple[Theorem, tuple]:
    renaming: dict[str, str] = {}
    known = theorem.variable_types

    def visit(node: Node) -> Node:
        replacement = (
            renaming.setdefault(node.op, f"v{len(renaming)}")
            if node.op in known
            else node.op
        )
        return Node(replacement, tuple(visit(arg) for arg in node.args))

    hypotheses = [
        Hypothesis(f"h{index}", visit(h.expr))
        for index, h in enumerate(theorem.hypotheses)
    ]
    conclusion = visit(theorem.conclusion)
    # Include variables used only by a d constraint deterministically.
    for pair in sorted(theorem.d_constraints):
        for variable in pair:
            renaming.setdefault(variable, f"v{len(renaming)}")
    constraints = {
        normalized_pair(renaming[a], renaming[b])
        for a, b in theorem.d_constraints
    }
    variable_types = {
        replacement: theorem.variable_types[original]
        for original, replacement in renaming.items()
    }
    proof = theorem.proof
    canonical_proof = None
    if proof is not None:
        canonical_proof = Proof(
            rule=proof.rule,
            parents=list(proof.parents),
            substitution={
                key: visit(value) for key, value in proof.substitution.items()
            },
            premise_map=list(proof.premise_map),
            depth=proof.depth,
            source_labels=proof.source_labels,
        )
    canonical = Theorem(
        name=theorem.name,
        hypotheses=hypotheses,
        conclusion=conclusion,
        d_constraints=constraints,
        proof=canonical_proof,
        variable_types=variable_types,
        floating=theorem.floating,
        kind=theorem.kind,
        source_tokens=theorem.source_tokens,
        id=theorem.id,
    )
    key = (
        tuple(_node_key(h.expr) for h in hypotheses),
        _node_key(conclusion),
        tuple(sorted(constraints)),
        tuple(sorted(variable_types.items())),
    )
    return canonical, key


def _node_key(node: Node) -> tuple:
    return (node.op, tuple(_node_key(arg) for arg in node.args))


def theorem_hash(theorem: Theorem) -> str:
    _, key = _canonicalize(theorem)
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TheoremDatabase:
    """Deduplicated theorem store with alpha-normalized hashing."""

    def __init__(self, parsed: Database | None = None) -> None:
        self.parsed = parsed
        self._items: list[Theorem] = []
        self._hash_to_id: dict[str, int] = {}
        self._name_to_id: dict[str, int] = {}

    @classmethod
    def from_parsed(
        cls,
        parsed: Database,
        *,
        logical_only: bool = True,
    ) -> "TheoremDatabase":
        store = cls(parsed)
        source = (
            parsed.logical_assertions.values()
            if logical_only
            else parsed.statements.values()
        )
        for theorem in source:
            store.add(theorem, canonicalize=False)
        return store

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, theorem_id: int) -> Theorem:
        return self._items[theorem_id]

    def add(
        self,
        theorem: Theorem,
        *,
        canonicalize: bool = True,
    ) -> tuple[int, bool]:
        canonical, _ = _canonicalize(theorem)
        digest = theorem_hash(canonical)
        if digest in self._hash_to_id:
            return self._hash_to_id[digest], False
        stored = canonical if canonicalize else theorem
        theorem_id = len(self._items)
        stored.id = theorem_id
        self._items.append(stored)
        self._hash_to_id[digest] = theorem_id
        self._name_to_id[stored.name] = theorem_id
        return theorem_id, True

    def get_by_name(self, name: str) -> Theorem:
        return self._items[self._name_to_id[name]]

    def generated(self) -> Iterable[Theorem]:
        return (item for item in self._items if item.kind == "generated")

    def candidates(self, conclusion: Node | None = None) -> list[Theorem]:
        if conclusion is None:
            return list(self._items)
        result = []
        for theorem in self._items:
            if theorem.conclusion.op != conclusion.op:
                continue
            if (
                theorem.conclusion.args
                and conclusion.args
                and theorem.conclusion.args[0].op != conclusion.args[0].op
                and not theorem.conclusion.args[0].args
                and not conclusion.args[0].args
            ):
                continue
            result.append(theorem)
        return result

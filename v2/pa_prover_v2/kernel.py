"""Small certificate boundary over the existing, independent Metamath kernel.

Only assertions present in the pinned source database are trusted axioms.
Everything else must carry a flattened source proof which is replayed before
it can be composed, persisted, or returned as a solved goal.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from metamath_generator.database import TheoremDatabase, theorem_hash
from metamath_generator.export import _SyntaxCompiler, _ordered_application_proof
from metamath_generator.model import Database, Hypothesis, Node, Proof, Theorem, normalized_pair
from metamath_generator.parser import parse
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import VerificationError, verify as verify_source


class KernelError(ValueError):
    pass


class ProofBudgetError(KernelError):
    """A valid composition would exceed its explicit certificate size budget."""


def canonical_json(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(data) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def node_to_data(node: Node) -> dict:
    return {"op": node.op, "args": [node_to_data(child) for child in node.args]}


def node_from_data(data: Mapping) -> Node:
    if not isinstance(data, Mapping) or set(data) != {"op", "args"}:
        raise KernelError("invalid node record")
    if not isinstance(data["op"], str) or not data["op"] or not isinstance(data["args"], list):
        raise KernelError("invalid node fields")
    return Node(data["op"], tuple(node_from_data(child) for child in data["args"]))


def _hyp_to_data(hyp: Hypothesis) -> dict:
    return {"label": hyp.label, "expr": node_to_data(hyp.expr)}


def _hyp_from_data(data) -> Hypothesis:
    return Hypothesis(str(data["label"]), node_from_data(data["expr"]))


def theorem_to_data(theorem: Theorem) -> dict:
    proof = theorem.proof
    return {
        "name": theorem.name,
        "hypotheses": [_hyp_to_data(hyp) for hyp in theorem.hypotheses],
        "conclusion": node_to_data(theorem.conclusion),
        "variable_types": dict(theorem.variable_types),
        "proof_variable_types": dict(theorem.proof_variable_types),
        "d_constraints": [list(pair) for pair in sorted(theorem.d_constraints)],
        "proof_d_constraints": [list(pair) for pair in sorted(theorem.proof_d_constraints)],
        "floating": [_hyp_to_data(hyp) for hyp in theorem.floating],
        "kind": theorem.kind,
        "source_tokens": list(theorem.source_tokens),
        "declaration_index": theorem.declaration_index,
        "active_hypothesis_labels": sorted(theorem.active_hypothesis_labels),
        "hypothesis_order": [list(item) for item in theorem.hypothesis_order],
        "id": theorem.id,
        "proof": None if proof is None else {
            "rule": proof.rule,
            "parents": list(proof.parents),
            "substitution": {key: node_to_data(value) for key, value in proof.substitution.items()},
            "premise_map": list(proof.premise_map),
            "depth": proof.depth,
            "source_labels": list(proof.source_labels),
        },
    }


def theorem_from_data(data: Mapping) -> Theorem:
    try:
        proof = data["proof"]
        return Theorem(
            name=str(data["name"]),
            hypotheses=[_hyp_from_data(hyp) for hyp in data["hypotheses"]],
            conclusion=node_from_data(data["conclusion"]),
            variable_types=dict(data["variable_types"]),
            proof_variable_types=dict(data.get("proof_variable_types", {})),
            d_constraints={normalized_pair(*pair) for pair in data["d_constraints"]},
            proof_d_constraints={normalized_pair(*pair) for pair in data.get("proof_d_constraints", [])},
            floating=tuple(_hyp_from_data(hyp) for hyp in data["floating"]),
            kind=str(data["kind"]),
            source_tokens=tuple(data.get("source_tokens", [])),
            declaration_index=int(data.get("declaration_index", -1)),
            active_hypothesis_labels=frozenset(data.get("active_hypothesis_labels", [])),
            hypothesis_order=tuple((kind, int(index)) for kind, index in data.get("hypothesis_order", [])),
            id=data.get("id"),
            proof=None if proof is None else Proof(
                rule=str(proof["rule"]), parents=list(proof.get("parents", [])),
                substitution={key: node_from_data(value) for key, value in proof.get("substitution", {}).items()},
                premise_map=list(proof.get("premise_map", [])),
                depth=int(proof.get("depth", 0)), source_labels=tuple(proof["source_labels"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KernelError(f"invalid theorem record: {exc}") from exc


@lru_cache(maxsize=8192)
def _statement_fingerprint_cached(hypotheses, conclusion, variable_types, constraints):
    types = dict(variable_types)
    used = {
        node.op for expression in (*hypotheses, conclusion)
        for node in expression.walk() if node.op in types
    }
    types = {name: typ for name, typ in types.items() if name in used}
    distinct = {pair for pair in constraints if set(pair) <= used}
    # Assumptions are a logical set, although certificate applications retain
    # their exact ordered $e stack interface.  Canonicalize only this identity.
    hypotheses = tuple(dict.fromkeys(hypotheses))

    def encode(node, mapping):
        if not node.args and node.op in types:
            if node.op not in mapping:
                mapping[node.op] = len(mapping)
            return (1, types[node.op], mapping[node.op])
        return (0, node.op, tuple(encode(child, mapping) for child in node.args))

    initial = {}
    encode(conclusion, initial)
    distinct_variables = {name for pair in distinct for name in pair}
    visits = 0

    def ordered(remaining, mapping):
        nonlocal visits
        visits += 1
        if visits > 20000:
            raise KernelError("statement alpha-canonicalization budget exceeded")
        if not remaining:
            dv = tuple(sorted(tuple(sorted((mapping[a], mapping[b]))) for a, b in distinct))
            return ((), dv), ()
        candidates = []
        for index, hypothesis in enumerate(remaining):
            updated = dict(mapping)
            key = encode(hypothesis, updated)
            candidates.append((key, index, updated))
        minimum = min(item[0] for item in candidates)
        best, best_order = None, None
        private_seen = False
        for key, index, updated in candidates:
            if key != minimum:
                continue
            rest = remaining[:index] + remaining[index + 1:]
            new_variables = set(updated) - set(mapping)
            other_variables = {node.op for expr in rest for node in expr.walk()}
            private = not (new_variables & (other_variables | distinct_variables))
            # Identical local shapes with independent private variables are
            # interchangeable, so do not enumerate their n! permutations.
            if private and private_seen:
                continue
            private_seen |= private
            tail_key, tail_order = ordered(rest, updated)
            candidate_key = ((key, *tail_key[0]), tail_key[1])
            if best is None or candidate_key < best:
                best, best_order = candidate_key, (remaining[index], *tail_order)
        return best, best_order

    _, canonical_hypotheses = ordered(hypotheses, initial)
    projected = Theorem(
        "statement_identity", [Hypothesis(f"h{i}", expr) for i, expr in enumerate(canonical_hypotheses)],
        conclusion, variable_types=types, d_constraints=distinct,
    )
    return theorem_hash(projected)


def statement_fingerprint(theorem: Theorem) -> str:
    """Identity of the mandatory logical statement, never its proof metadata.

    Unused variables, proof-only distinctness and repeated/reordered premises
    must not turn a held-out target into a supposedly different library item.
    Only syntactic alpha-equivalence is considered, not logical equivalence.
    """
    return _statement_fingerprint_cached(
        tuple(hyp.expr for hyp in theorem.hypotheses), theorem.conclusion,
        tuple(sorted(theorem.variable_types.items())), tuple(sorted(theorem.d_constraints)),
    )


def statement_text(theorem: Theorem) -> str:
    """Full logical signature only: no proof dependency traversal or labels."""
    return canonical_json({
        "hypotheses": [hyp.expr.to_prefix() for hyp in theorem.hypotheses],
        "conclusion": theorem.conclusion.to_prefix(),
        "variable_types": dict(theorem.variable_types),
        "d_constraints": [list(pair) for pair in sorted(theorem.d_constraints)],
    })


class ProofKernel:
    def __init__(self, database_path: str | Path, *, external_verifier=None):
        self.path = Path(database_path).resolve()
        self.database = parse(self.path)
        self.external_verifier = external_verifier
        # Include both root-file bytes and expanded parsed contents, so a
        # changed included assertion/proof cannot retain the theory identity.
        self.theory_fingerprint = content_hash({
            "root_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "statements": {label: theorem_to_data(item) for label, item in self.database.statements.items()},
            "floating": {label: _hyp_to_data(item) for label, item in self.database.floating_hypotheses.items()},
            "essential": {label: _hyp_to_data(item) for label, item in self.database.essential_hypotheses.items()},
            "symbols": sorted(self.database.symbols), "variables": sorted(self.database.variables),
        })
        for theorem in self.database.proved_theorems.values():
            verify_source(theorem, self.database)
        self._source_hashes = {
            label: self._source_identity(theorem)
            for label, theorem in self.database.logical_assertions.items()
        }

    @staticmethod
    def _source_identity(theorem: Theorem) -> str:
        data = theorem_to_data(theorem)
        # The existing generator assigns transient store IDs to parsed rules.
        # IDs are not part of the trusted declaration's logical identity.
        data.pop("id", None)
        return content_hash(data)

    def _is_source(self, theorem: Theorem) -> bool:
        return self._source_hashes.get(theorem.name) == self._source_identity(theorem)

    def _compiler(self, theorem: Theorem) -> _SyntaxCompiler:
        floating = {hyp.expr.args[0].op: (hyp.label, hyp.expr.op) for hyp in theorem.floating}
        return _SyntaxCompiler(TheoremDatabase(self.database), floating)

    def verify(self, theorem: Theorem) -> None:
        if theorem.proof is None or not theorem.proof.source_labels:
            raise KernelError("a replayable source proof is required; new axioms are forbidden")
        if self._is_source(theorem):
            verify_source(theorem, self.database)
            return
        # Never trust serialized scope/declaration metadata to activate source
        # hypotheses that are absent from the actual certificate statement.
        certificate = copy.deepcopy(theorem)
        certificate.declaration_index = -1
        certificate.active_hypothesis_labels = frozenset()
        types = dict(certificate.variable_types)
        for variable, typ in certificate.proof_variable_types.items():
            if variable in types and types[variable] != typ:
                raise KernelError("conflicting proof variable type")
            types[variable] = typ
        if any(variable in self.database.symbols for variable in types):
            raise KernelError("a variable shadows a theory constant")
        ambient_types = {
            hyp.expr.args[0].op: hyp.expr.op
            for hyp in self.database.active_floating_hypotheses.values()
        }
        if any(variable in ambient_types and ambient_types[variable] != typ for variable, typ in types.items()):
            raise KernelError("episode variable conflicts with an active source floating type; use a fresh variable")
        local = [*certificate.floating, *certificate.hypotheses]
        labels = [hyp.label for hyp in local]
        if len(set(labels)) != len(labels) or any(label in self.database.statements for label in labels):
            raise KernelError("local proof labels collide")
        for hypotheses, same, opposite in (
            (certificate.floating, self.database.floating_hypotheses, self.database.essential_hypotheses),
            (certificate.hypotheses, self.database.essential_hypotheses, self.database.floating_hypotheses),
        ):
            for hyp in hypotheses:
                if hyp.label in opposite or (hyp.label in same and same[hyp.label].expr != hyp.expr):
                    raise KernelError("local hypothesis shadows an incompatible source declaration")
        floating_types = {}
        for hyp in certificate.floating:
            if len(hyp.expr.args) != 1 or hyp.expr.args[0].args:
                raise KernelError("invalid floating declaration")
            variable = hyp.expr.args[0].op
            if variable in floating_types:
                raise KernelError("duplicate floating variable")
            floating_types[variable] = hyp.expr.op
        if floating_types != types:
            raise KernelError("certificate variable types and floating declarations differ")
        for left, right in certificate.d_constraints | certificate.proof_d_constraints:
            if left == right or left not in types or right not in types:
                raise KernelError("invalid certificate distinct-variable scope")
        statement_variables = {
            node.op for expr in [certificate.conclusion, *(hyp.expr for hyp in certificate.hypotheses)]
            for node in expr.walk() if node.op in types
        }
        if not statement_variables <= set(certificate.variable_types):
            raise KernelError("a mandatory statement variable was hidden as proof-only")
        mandatory_proof_d = {
            pair for pair in certificate.proof_d_constraints if set(pair) <= statement_variables
        }
        if not mandatory_proof_d <= certificate.d_constraints:
            raise KernelError("mandatory distinct-variable constraints were hidden as proof-only")
        compiler = self._compiler(certificate)
        for expr in [certificate.conclusion, *(hyp.expr for hyp in certificate.hypotheses)]:
            if expr.op != "|-" or len(expr.args) != 1:
                raise KernelError("certificates must contain logical assertions")
            compiler.compile("wff", expr.args[0])
        augmented = copy.copy(self.database)
        augmented.variables = self.database.variables | set(types)
        augmented.floating_hypotheses = dict(self.database.floating_hypotheses)
        augmented.essential_hypotheses = dict(self.database.essential_hypotheses)
        augmented.floating_hypotheses.update({hyp.label: hyp for hyp in certificate.floating})
        augmented.essential_hypotheses.update({hyp.label: hyp for hyp in certificate.hypotheses})
        try:
            verify_source(certificate, augmented)
        except VerificationError as exc:
            raise KernelError(str(exc)) from exc

    def _context_certificate(self, context: Theorem, conclusion: Node, name=None) -> Theorem:
        stem = name or "v2_" + content_hash({
            "context": statement_fingerprint(context), "conclusion": node_to_data(conclusion),
        })[:24]
        if stem in self.database.statements:
            raise KernelError("certificate name collides with a source assertion")
        result = Theorem(
            stem,
            [Hypothesis(f"{stem}_h{i}", hyp.expr) for i, hyp in enumerate(context.hypotheses)],
            conclusion,
            d_constraints=set(context.d_constraints), variable_types=dict(context.variable_types),
            floating=tuple(
                Hypothesis(f"{stem}_f{i}", Node(typ, (Node(variable),)))
                for i, (variable, typ) in enumerate(context.variable_types.items())
            ), kind="theorem",
        )
        return result

    def assumption(self, target_context: Theorem, index: int) -> Theorem:
        if index < 0 or index >= len(target_context.hypotheses):
            raise KernelError("assumption index outside the episode context")
        result = self._context_certificate(target_context, target_context.hypotheses[index].expr)
        result.proof = Proof("assumption", source_labels=(result.hypotheses[index].label,))
        self.verify(result)
        return result

    def _check_scope(self, premise: Theorem, context: Theorem) -> None:
        source = self._is_source(premise)
        types = dict(premise.variable_types) if source else {**premise.variable_types, **premise.proof_variable_types}
        if any(context.variable_types.get(variable) != typ for variable, typ in types.items()):
            raise KernelError("premise escapes the episode variable scope")
        constraints = premise.d_constraints | (set() if source else premise.proof_d_constraints)
        if not constraints <= context.d_constraints:
            raise KernelError("premise needs distinct-variable assumptions outside the episode")
        allowed = {hyp.expr for hyp in context.hypotheses}
        if any(hyp.expr not in allowed for hyp in premise.hypotheses):
            raise KernelError("premise uses an unproved hypothesis outside the episode")

    def _proof_labels(self, theorem, substitution, essential_proofs, compiler, max_proof_labels):
        floating_proofs = [
            compiler.compile(hyp.expr.op, substitution[hyp.expr.args[0].op])
            for hyp in theorem.floating
        ]
        if any(len(labels) > max_proof_labels for labels in (*floating_proofs, *essential_proofs)):
            raise ProofBudgetError("proof label budget exceeded by a premise")
        if self._is_source(theorem):
            # Count before _ordered_application_proof allocates the concatenation.
            if 1 + sum(map(len, floating_proofs)) + sum(map(len, essential_proofs)) > max_proof_labels:
                raise ProofBudgetError("proof label budget exceeded by source application")
            return _ordered_application_proof(theorem, floating_proofs, essential_proofs)
        replacements = {
            hyp.label: labels for hyp, labels in zip(theorem.floating, floating_proofs)
        }
        replacements.update({hyp.label: labels for hyp, labels in zip(theorem.hypotheses, essential_proofs)})
        # One essential premise can occur many times in a library certificate.
        # Preflight repeated expansion lengths instead of materializing an
        # exponentially growing list and checking its size afterwards.
        length = 0
        for label in theorem.proof.source_labels:
            length += len(replacements[label]) if label in replacements else 1
            if length > max_proof_labels:
                raise ProofBudgetError("proof label budget exceeded by library expansion")
        return [
            expanded for label in theorem.proof.source_labels
            for expanded in replacements.get(label, [label])
        ]

    def compose(self, rule: Theorem, substitution: dict[str, Node], premises: list[Theorem], target_context: Theorem, *, name=None, max_proof_labels=65536) -> Theorem:
        if not isinstance(max_proof_labels, int) or isinstance(max_proof_labels, bool) or max_proof_labels <= 0:
            raise KernelError("max_proof_labels must be a positive integer")
        source = self._is_source(rule)
        if not source:
            if rule.name in self.database.logical_assertions:
                raise KernelError("rule does not match the pinned source assertion")
            self.verify(rule)
        if len(premises) != len(rule.hypotheses):
            raise KernelError("every rule hypothesis requires exactly one matching premise")
        rule_types = dict(rule.variable_types)
        if not source:
            rule_types.update(rule.proof_variable_types)
        mapping = dict(substitution)
        if set(mapping) - set(rule_types):
            raise KernelError("substitution contains undeclared rule variables")
        for variable, typ in rule_types.items():
            if variable not in mapping:
                if target_context.variable_types.get(variable) != typ:
                    raise KernelError(f"missing typed substitution for {variable}")
                mapping[variable] = Node(variable)
        conclusion = substitute_simultaneous(rule.conclusion, mapping)
        result = self._context_certificate(target_context, conclusion, name)
        compiler = self._compiler(result)
        for variable, typ in rule_types.items():
            compiler.compile(typ, mapping[variable])
        distinct = rule.d_constraints | (set() if source else rule.proof_d_constraints)
        for left, right in distinct:
            lhs = {node.op for node in mapping[left].walk() if node.op in target_context.variable_types}
            rhs = {node.op for node in mapping[right].walk() if node.op in target_context.variable_types}
            if lhs & rhs or any(normalized_pair(a, b) not in target_context.d_constraints for a in lhs for b in rhs):
                raise KernelError("rule substitution violates the episode distinct-variable scope")
        context_labels = {hyp.expr: [hyp.label] for hyp in result.hypotheses}
        premise_proofs = []
        for expected, premise in zip(rule.hypotheses, premises):
            self.verify(premise)
            self._check_scope(premise, target_context)
            if premise.conclusion != substitute_simultaneous(expected.expr, mapping):
                raise KernelError("premise does not match the instantiated rule hypothesis")
            identity = {variable: Node(variable) for variable in {**premise.variable_types, **premise.proof_variable_types}}
            premise_proofs.append(self._proof_labels(
                premise, identity, [context_labels[hyp.expr] for hyp in premise.hypotheses], compiler,
                max_proof_labels,
            ))
        labels = self._proof_labels(rule, mapping, premise_proofs, compiler, max_proof_labels)
        result.proof = Proof(
            rule.name, substitution=mapping,
            depth=1 + max((premise.proof_depth for premise in premises), default=0),
            source_labels=tuple(labels),
        )
        self.verify(result)
        return result

    def certify(self, theorem: Theorem, *, required_external=False, external_verifier=None, timeout_seconds=60.0) -> dict:
        self.verify(theorem)
        record = {"internal_verified": True, "theory_fingerprint": self.theory_fingerprint,
                  "statement_fingerprint": statement_fingerprint(theorem),
                  "external_verified": False, "external_verification": {"status": "not_requested"}}
        executable = external_verifier or self.external_verifier
        if required_external or executable:
            from neural_prover.external import verify_certificate_external
            result = verify_certificate_external(theorem, self.path, executable=executable, timeout_seconds=timeout_seconds)
            record["external_verification"] = result.to_record()
            record["external_verified"] = result.passed
            if required_external and not result.passed:
                raise KernelError(f"required external proof verification failed: {result.status}")
        return record


Kernel = ProofKernel

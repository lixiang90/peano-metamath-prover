from __future__ import annotations

from .model import Database, Node, Theorem, normalized_pair
from .unification import substitute_simultaneous


class VerificationError(ValueError):
    pass


def _variables(expr: Node, database: Database) -> set[str]:
    return {
        node.op for node in expr.walk() if node.op in database.variables
    }


def verify(theorem: Theorem, database: Database) -> None:
    """Verify an uncompressed Metamath proof against the parsed database."""

    if theorem.proof is None:
        raise VerificationError(f"{theorem.name} has no proof")
    labels = theorem.proof.source_labels
    if not labels:
        raise VerificationError(f"{theorem.name} has an empty proof")
    if labels[0] == "(":
        raise VerificationError("compressed proofs are not supported by this verifier")

    stack: list[Node] = []
    active_d = theorem.d_constraints
    for label in labels:
        floating = database.floating_hypotheses.get(label)
        if floating is not None:
            stack.append(floating.expr)
            continue
        essential = database.essential_hypotheses.get(label)
        if essential is not None:
            stack.append(essential.expr)
            continue
        assertion = database.statements.get(label)
        if assertion is None:
            raise VerificationError(f"unknown proof label {label!r}")
        hypotheses = [*assertion.floating, *assertion.hypotheses]
        if len(stack) < len(hypotheses):
            raise VerificationError(f"stack underflow while applying {label}")
        actuals = stack[len(stack) - len(hypotheses):] if hypotheses else []
        if hypotheses:
            del stack[-len(hypotheses):]

        subst: dict[str, Node] = {}
        for expected, actual in zip(assertion.floating, actuals):
            variable = expected.expr.args[0].op
            if actual.op != expected.expr.op or len(actual.args) != 1:
                raise VerificationError(
                    f"type mismatch for {variable} while applying {label}"
                )
            subst[variable] = actual.args[0]
        essential_actuals = actuals[len(assertion.floating):]
        for expected, actual in zip(assertion.hypotheses, essential_actuals):
            wanted = substitute_simultaneous(expected.expr, subst)
            if wanted != actual:
                raise VerificationError(
                    f"essential hypothesis mismatch for {label}: "
                    f"expected {wanted}, got {actual}"
                )
        for left, right in assertion.d_constraints:
            left_vars = _variables(
                substitute_simultaneous(Node(left), subst), database
            )
            right_vars = _variables(
                substitute_simultaneous(Node(right), subst), database
            )
            if left_vars & right_vars:
                raise VerificationError(
                    f"distinct variables collapse while applying {label}"
                )
            for a in left_vars:
                for b in right_vars:
                    if normalized_pair(a, b) not in active_d:
                        raise VerificationError(
                            f"missing $d {a} {b} while applying {label}"
                        )
        stack.append(substitute_simultaneous(assertion.conclusion, subst))

    if stack != [theorem.conclusion]:
        raise VerificationError(
            f"{theorem.name}: proof ended with stack {stack}, "
            f"expected only {theorem.conclusion}"
        )

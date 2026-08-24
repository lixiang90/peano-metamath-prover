from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from metamath_generator.database import TheoremDatabase
from metamath_generator.export import _SyntaxCompiler
from metamath_generator.model import (
    Database,
    Hypothesis,
    Node,
    Proof,
    Theorem,
)
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import verify

from .environment import (
    LEMMA_BINDING,
    PROPOSE_LEMMA_RULE,
    Transition,
)

if TYPE_CHECKING:
    from .search import SearchResult


class CertificateError(ValueError):
    pass


@dataclass(slots=True)
class _ProofTree:
    transition: Transition
    children: list["_ProofTree"] = field(default_factory=list)


def _tree(transitions: tuple[Transition, ...]) -> _ProofTree:
    if not transitions:
        raise CertificateError("proof has no transitions")
    root = _ProofTree(transitions[0])
    # Each frame stores a node and the number of pending generated subgoals.
    stack: list[list] = [[root, len(transitions[0].generated_goals)]]
    for transition in transitions[1:]:
        while stack and stack[-1][1] == 0:
            stack.pop()
        if not stack:
            raise CertificateError("transition trace has extra actions")
        parent = stack[-1][0]
        child = _ProofTree(transition)
        parent.children.append(child)
        stack[-1][1] -= 1
        stack.append([child, len(transition.generated_goals)])
    while stack and stack[-1][1] == 0:
        stack.pop()
    if stack:
        raise CertificateError("transition trace leaves subgoals unproved")
    return root


def _hypothesis_label(theorem: Theorem, expression: Node) -> str:
    for hypothesis in theorem.hypotheses:
        if hypothesis.expr == expression:
            return hypothesis.label
    raise CertificateError(
        f"proof requested unavailable hypothesis {expression}"
    )


def _compile_tree(
    node: _ProofTree,
    theorem: Theorem,
    database: Database,
    compiler: _SyntaxCompiler,
    local_proofs: dict[Node, tuple[str, ...]] | None = None,
) -> list[str]:
    available = local_proofs or {}
    transition = node.transition
    if transition.tactic.rule == "<ASSUMPTION>":
        local = available.get(transition.before.current_goal)
        if local is not None:
            return list(local)
        return [_hypothesis_label(
            theorem,
            transition.before.current_goal,
        )]
    if transition.tactic.rule == PROPOSE_LEMMA_RULE:
        payload = transition.tactic.substitution_dict()
        lemma = payload.get(LEMMA_BINDING)
        if lemma is None or len(payload) != 1:
            raise CertificateError("malformed intermediate-lemma action")
        if len(node.children) != 2:
            raise CertificateError(
                "intermediate lemma must have lemma and continuation proofs"
            )
        lemma_child, continuation_child = node.children
        if lemma_child.transition.before.current_goal != lemma:
            raise CertificateError("lemma branch proves the wrong statement")
        if (
            continuation_child.transition.before.current_goal
            != transition.before.current_goal
        ):
            raise CertificateError(
                "lemma continuation proves the wrong original goal"
            )
        lemma_labels = tuple(_compile_tree(
            lemma_child,
            theorem,
            database,
            compiler,
            available,
        ))
        extended = dict(available)
        extended[lemma] = lemma_labels
        # The local lemma is a proof macro, not a new axiom or theorem label.
        # Inline it wherever the continuation consumes the activated lemma.
        return _compile_tree(
            continuation_child,
            theorem,
            database,
            compiler,
            extended,
        )
    assertion = database.logical_assertions.get(
        transition.tactic.rule
    )
    if assertion is None:
        raise CertificateError(
            f"certificate cannot use non-source assertion "
            f"{transition.tactic.rule!r}"
        )
    substitution = dict(transition.resolved_substitution)
    labels: list[str] = []
    floating = [
        (item.expr.args[0].op, item.expr.op)
        for item in assertion.floating
    ]
    if not floating:
        floating = list(assertion.variable_types.items())
    for variable, typecode in floating:
        value = substitution.get(variable)
        if value is None:
            raise CertificateError(
                f"missing substitution for {variable} in "
                f"{assertion.name}"
            )
        labels.extend(compiler.compile(typecode, value))

    children = iter(node.children)
    for hypothesis in assertion.hypotheses:
        instance = substitute_simultaneous(
            hypothesis.expr,
            substitution,
        )
        if any(h.expr == instance for h in theorem.hypotheses):
            labels.append(_hypothesis_label(theorem, instance))
            continue
        local = available.get(instance)
        if local is not None:
            labels.extend(local)
            continue
        try:
            child = next(children)
        except StopIteration as exc:
            raise CertificateError(
                f"no proof for premise {instance}"
            ) from exc
        if child.transition.before.current_goal != instance:
            raise CertificateError(
                f"trace proves {child.transition.before.current_goal}, "
                f"expected {instance}"
            )
        labels.extend(_compile_tree(
            child,
            theorem,
            database,
            compiler,
            available,
        ))
    try:
        next(children)
    except StopIteration:
        pass
    else:
        raise CertificateError("proof node has too many children")
    labels.append(assertion.name)
    return labels


def compile_certificate(
    target: Theorem,
    result: "SearchResult",
    database: Database,
    *,
    name: str = "neural_proof",
) -> Theorem:
    if not result.solved:
        raise CertificateError("cannot certify an unsolved search")
    floating = tuple(
        Hypothesis(
            f"{name}_f{index}",
            Node(typecode, (Node(variable),)),
        )
        for index, (variable, typecode) in enumerate(
            target.variable_types.items()
        )
    )
    certificate = Theorem(
        name=name,
        hypotheses=[
            Hypothesis(f"{name}_h{index}", hypothesis.expr)
            for index, hypothesis in enumerate(target.hypotheses)
        ],
        conclusion=target.conclusion,
        d_constraints=set(target.d_constraints),
        variable_types=dict(target.variable_types),
        floating=floating,
        kind="theorem",
    )
    if not result.transitions:
        if target.conclusion in [
            h.expr for h in certificate.hypotheses
        ]:
            labels = [
                _hypothesis_label(certificate, target.conclusion)
            ]
        else:
            raise CertificateError("empty proof does not prove target")
    else:
        floating_map = {
            variable: (f"{name}_f{index}", typecode)
            for index, (variable, typecode) in enumerate(
                target.variable_types.items()
            )
        }
        store = TheoremDatabase(database)
        compiler = _SyntaxCompiler(store, floating_map)
        labels = _compile_tree(
            _tree(result.transitions),
            certificate,
            database,
            compiler,
        )
    certificate.proof = Proof(name, source_labels=tuple(labels))
    return certificate


def verify_certificate(
    certificate: Theorem,
    database: Database,
) -> None:
    augmented = copy.deepcopy(database)
    augmented.variables.update(certificate.variable_types)
    for floating in certificate.floating:
        augmented.floating_hypotheses[floating.label] = floating
    for hypothesis in certificate.hypotheses:
        augmented.essential_hypotheses[hypothesis.label] = hypothesis
    verify(certificate, augmented)


def export_certificate(
    certificate: Theorem,
    destination: str | Path,
    *,
    ambient_database: Database | None = None,
) -> None:
    lines = [
        "$( Neural search proof; every label is kernel replayable. $)",
    ]
    ambient_variables = (
        ambient_database.variables if ambient_database is not None else set()
    )
    variables = [
        variable for variable in certificate.variable_types
        if variable not in ambient_variables
    ]
    if variables:
        lines.append(f"$v {' '.join(variables)} $.")
    lines.append("${")
    ambient_by_expression = {
        hypothesis.expr: label
        for label, hypothesis in (
            ambient_database.active_floating_hypotheses.items()
            if ambient_database is not None else ()
        )
    }
    floating_replacements: dict[str, str] = {}
    for floating in certificate.floating:
        ambient_label = ambient_by_expression.get(floating.expr)
        if ambient_label is not None:
            floating_replacements[floating.label] = ambient_label
            continue
        lines.append(
            f"  {floating.label} $f {floating.expr.to_prefix()} $."
        )
    for left, right in sorted(certificate.d_constraints):
        lines.append(f"  $d {left} {right} $.")
    for hypothesis in certificate.hypotheses:
        lines.append(
            f"  {hypothesis.label} $e {hypothesis.expr.to_prefix()} $."
        )
    if certificate.proof is None:
        raise CertificateError("certificate has no proof")
    lines.append(
        f"  {certificate.name} $p "
        f"{certificate.conclusion.to_prefix()} $= "
        f"{' '.join(floating_replacements.get(label, label) for label in certificate.proof.source_labels)} $."
    )
    lines.append("$}")
    Path(destination).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .model import (
    Database,
    Hypothesis,
    Node,
    Proof,
    Theorem,
    normalized_pair,
)
from .parser import MetamathParser, ParseError
from .unification import UnificationError, substitute, unify


class CompositionError(ValueError):
    pass


def _rename_node(expr: Node, renaming: Mapping[str, str]) -> Node:
    return Node(
        renaming.get(expr.op, expr.op),
        tuple(_rename_node(arg, renaming) for arg in expr.args),
    )


def _standardize(theorem: Theorem, namespace: str) -> Theorem:
    renaming = {name: f"__{namespace}_{name}" for name in theorem.variable_types}
    return Theorem(
        name=theorem.name,
        hypotheses=[
            Hypothesis(h.label, _rename_node(h.expr, renaming))
            for h in theorem.hypotheses
        ],
        conclusion=_rename_node(theorem.conclusion, renaming),
        d_constraints={
            normalized_pair(renaming[a], renaming[b])
            for a, b in theorem.d_constraints
        },
        proof=theorem.proof,
        variable_types={
            renaming[name]: typ for name, typ in theorem.variable_types.items()
        },
        floating=tuple(
            Hypothesis(h.label, _rename_node(h.expr, renaming))
            for h in theorem.floating
        ),
        kind=theorem.kind,
        source_tokens=theorem.source_tokens,
        id=theorem.id,
    )


def _variable_leaves(expr: Node, variables: set[str]) -> set[str]:
    return {
        node.op
        for node in expr.walk()
        if node.op in variables
    }


def _instantiate_constraints(
    constraints: set[tuple[str, str]],
    subst: Mapping[str, Node],
    variables: set[str],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for left, right in constraints:
        left_vars = _variable_leaves(substitute(Node(left), subst), variables)
        right_vars = _variable_leaves(substitute(Node(right), subst), variables)
        overlap = left_vars & right_vars
        if overlap:
            names = ", ".join(sorted(overlap))
            raise CompositionError(
                f"distinct-variable constraint ({left}, {right}) collapses at {names}"
            )
        for a in left_vars:
            for b in right_vars:
                if a != b:
                    result.add(normalized_pair(a, b))
    return result


def compose(
    rule: Theorem,
    matches: Sequence[Theorem | None] | Mapping[int, Theorem],
    *,
    name: str | None = None,
    database: Database | None = None,
) -> Theorem:
    """Partially discharge a rule's essential hypotheses with known theorems.

    Parent variables and rule variables are standardized apart before the
    one-pass unification.  Unmatched rule hypotheses and all hypotheses of
    matched parents remain open in the result.
    """

    if isinstance(matches, Mapping):
        selected = [matches.get(i) for i in range(len(rule.hypotheses))]
    else:
        selected = list(matches)
        if len(selected) != len(rule.hypotheses):
            raise CompositionError(
                f"{rule.name} has {len(rule.hypotheses)} premises, "
                f"but {len(selected)} match slots were supplied"
            )
    if not any(parent is not None for parent in selected):
        raise CompositionError("composition must discharge at least one premise")

    standardized_rule = _standardize(rule, "rule")
    standardized_parents: list[Theorem | None] = []
    for index, parent in enumerate(selected):
        standardized_parents.append(
            _standardize(parent, f"p{index}") if parent is not None else None
        )

    variable_types = dict(standardized_rule.variable_types)
    for parent in standardized_parents:
        if parent is not None:
            variable_types.update(parent.variable_types)
    variables = set(variable_types)
    subst: dict[str, Node] = {}
    try:
        for premise, parent in zip(
            standardized_rule.hypotheses, standardized_parents
        ):
            if parent is not None:
                subst = unify(
                    substitute(premise.expr, subst),
                    substitute(parent.conclusion, subst),
                    variables,
                    subst,
                )
    except UnificationError as exc:
        raise CompositionError(str(exc)) from exc

    if database is not None:
        parser = MetamathParser()
        parser.database = database
        old_types = dict(database.variable_types)
        database.variable_types.update(variable_types)
        try:
            for variable, typecode in variable_types.items():
                value = substitute(Node(variable), subst)
                parsed = parser.parse_expression([
                    typecode,
                    *value.to_prefix().split(),
                ])
                if parsed.args != (value,):
                    raise CompositionError(
                        f"{variable} is not a {typecode}: {value}"
                    )
        except ParseError as exc:
            raise CompositionError(
                f"ill-typed unification substitution: {exc}"
            ) from exc
        finally:
            database.variable_types.clear()
            database.variable_types.update(old_types)

    open_hypotheses: list[Hypothesis] = []
    for premise, parent in zip(
        standardized_rule.hypotheses, standardized_parents
    ):
        if parent is None:
            open_hypotheses.append(
                Hypothesis(premise.label, substitute(premise.expr, subst))
            )
        else:
            open_hypotheses.extend(
                Hypothesis(h.label, substitute(h.expr, subst))
                for h in parent.hypotheses
            )

    # The same formula need only be assumed once.
    unique_hypotheses: list[Hypothesis] = []
    seen_hypotheses: set[Node] = set()
    for index, hypothesis in enumerate(open_hypotheses):
        if hypothesis.expr not in seen_hypotheses:
            unique_hypotheses.append(
                Hypothesis(f"h{index}", hypothesis.expr)
            )
            seen_hypotheses.add(hypothesis.expr)

    constraints = _instantiate_constraints(
        standardized_rule.d_constraints, subst, variables
    )
    for parent in standardized_parents:
        if parent is not None:
            constraints.update(
                _instantiate_constraints(parent.d_constraints, subst, variables)
            )

    conclusion = substitute(standardized_rule.conclusion, subst)
    used_variables = {
        node.op
        for expr in [*(h.expr for h in unique_hypotheses), conclusion]
        for node in expr.walk()
        if node.op in variables
    }
    constraints = {
        pair for pair in constraints
        if pair[0] in used_variables and pair[1] in used_variables
    }
    parent_ids = [
        parent.id for parent in selected
        if parent is not None and parent.id is not None
    ]
    parent_depth = max(
        (parent.proof_depth for parent in selected if parent is not None),
        default=0,
    )
    premise_map = [
        parent.id if parent is not None else None for parent in selected
    ]
    # Store a total instantiation for every local rule/parent variable.  This
    # makes the DAG independently replayable and is also what the Metamath
    # exporter needs to reconstruct assertion applications.
    total_substitution = {
        variable: substitute(Node(variable), subst)
        for variable in variable_types
    }
    return Theorem(
        name=name or f"gen_{rule.name}",
        hypotheses=unique_hypotheses,
        conclusion=conclusion,
        d_constraints=constraints,
        proof=Proof(
            rule=rule.name,
            parents=parent_ids,
            substitution=total_substitution,
            premise_map=premise_map,
            depth=parent_depth + 1,
        ),
        variable_types={
            variable: variable_types[variable] for variable in used_variables
        },
        kind="generated",
    )

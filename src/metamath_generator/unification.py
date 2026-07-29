from __future__ import annotations

from collections.abc import Iterable, Mapping

from .model import Node

Substitution = dict[str, Node]


class UnificationError(ValueError):
    pass


class OccursCheckError(UnificationError):
    pass


def substitute(expr: Node, subst: Mapping[str, Node]) -> Node:
    """Recursively apply a substitution until it reaches a fixed point."""

    def visit(node: Node, trail: frozenset[str]) -> Node:
        if node.op in subst:
            if subst[node.op] == Node(node.op):
                if not node.args:
                    return node
                return Node(
                    node.op,
                    tuple(visit(arg, trail) for arg in node.args),
                )
            if node.op in trail:
                raise OccursCheckError(f"cyclic substitution involving {node.op!r}")
            replacement = visit(subst[node.op], trail | {node.op})
            if not node.args:
                return replacement
            if replacement.args:
                raise UnificationError(
                    f"operator variable {node.op!r} was bound to non-operator "
                    f"{replacement}"
                )
            return Node(
                replacement.op,
                tuple(visit(arg, trail) for arg in node.args),
            )
        if not node.args:
            return node
        args = tuple(visit(arg, trail) for arg in node.args)
        return node if args == node.args else Node(node.op, args)

    return visit(expr, frozenset())


def substitute_simultaneous(expr: Node, subst: Mapping[str, Node]) -> Node:
    """Apply a Metamath-style simultaneous substitution exactly once."""

    replacement = subst.get(expr.op)
    if replacement is not None:
        if not expr.args:
            return replacement
        if replacement.args:
            raise UnificationError(
                f"operator variable {expr.op!r} was bound to non-operator "
                f"{replacement}"
            )
        return Node(
            replacement.op,
            tuple(substitute_simultaneous(arg, subst) for arg in expr.args),
        )
    return Node(
        expr.op,
        tuple(substitute_simultaneous(arg, subst) for arg in expr.args),
    )


def _occurs(variable: str, expr: Node, subst: Mapping[str, Node]) -> bool:
    resolved = substitute(expr, subst)
    return any(
        node.op == variable
        for node in resolved.walk()
    )


def _bind(
    variable: str,
    value: Node,
    subst: Substitution,
) -> None:
    value = substitute(value, subst)
    if value == Node(variable):
        return
    if _occurs(variable, value, subst):
        raise OccursCheckError(f"{variable!r} occurs in {value}")

    # Keep the substitution idempotent.  This also merges bindings made by
    # unifying several premises one after another.
    one = {variable: value}
    for key, old_value in list(subst.items()):
        subst[key] = substitute(old_value, one)
    subst[variable] = value


def unify(
    expr1: Node,
    expr2: Node,
    variables: Iterable[str] | None = None,
    subst: Mapping[str, Node] | None = None,
) -> Substitution:
    """First-order unification with occurs check.

    ``variables`` identifies which leaf names are metavariables.  When omitted,
    leaf names beginning with ``?`` are treated as metavariables.
    """

    variable_set = set(variables) if variables is not None else {
        node.op
        for expr in (expr1, expr2)
        for node in expr.walk()
        if node.op.startswith("?")
    }
    result: Substitution = dict(subst or {})
    agenda: list[tuple[Node, Node]] = [(expr1, expr2)]

    while agenda:
        left, right = agenda.pop()
        left = substitute(left, result)
        right = substitute(right, result)
        if left == right:
            continue
        if left.op in variable_set:
            if left.args:
                if len(left.args) != len(right.args):
                    raise UnificationError(f"cannot unify {left} with {right}")
                _bind(left.op, Node(right.op), result)
                agenda.extend(zip(left.args, right.args))
                continue
            _bind(left.op, right, result)
            continue
        if right.op in variable_set:
            if right.args:
                if len(left.args) != len(right.args):
                    raise UnificationError(f"cannot unify {left} with {right}")
                _bind(right.op, Node(left.op), result)
                agenda.extend(zip(left.args, right.args))
                continue
            _bind(right.op, left, result)
            continue
        if left.op != right.op or len(left.args) != len(right.args):
            raise UnificationError(f"cannot unify {left} with {right}")
        agenda.extend(zip(left.args, right.args))

    return {key: substitute(value, result) for key, value in result.items()}


def compose_substitutions(
    first: Mapping[str, Node],
    second: Mapping[str, Node],
) -> Substitution:
    merged = {key: substitute(value, second) for key, value in first.items()}
    for key, value in second.items():
        if key in merged and merged[key] != value:
            merged = unify(merged[key], value, set(merged) | set(second), merged)
        else:
            merged[key] = substitute(value, merged)
    return {key: substitute(value, merged) for key, value in merged.items()}

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from .database import TheoremDatabase
from .model import Node, Theorem
from .unification import substitute_simultaneous


def expression_text(expr: Node) -> str:
    return expr.to_prefix()


def proof_records(theorem: Theorem, store: TheoremDatabase) -> list[dict]:
    records: list[dict] = []
    visited: set[int] = set()

    def visit(theorem_id: int) -> None:
        if theorem_id in visited:
            return
        visited.add(theorem_id)
        item = store[theorem_id]
        if item.proof:
            for parent in item.proof.parents:
                visit(parent)
            records.append(
                {
                    "id": theorem_id,
                    "rule": item.proof.rule,
                    "parents": item.proof.parents,
                    "premise_map": item.proof.premise_map,
                    "substitution": {
                        key: expression_text(value)
                        for key, value in item.proof.substitution.items()
                    },
                }
            )

    if theorem.id is not None:
        visit(theorem.id)
    return records


def theorem_record(theorem: Theorem, store: TheoremDatabase) -> dict:
    proof = theorem.proof
    return {
        "id": theorem.id,
        "name": theorem.name,
        "premises": [expression_text(h.expr) for h in theorem.hypotheses],
        "conclusion": expression_text(theorem.conclusion),
        "variable_types": dict(sorted(theorem.variable_types.items())),
        "proof_variable_types": dict(
            sorted(theorem.proof_variable_types.items())
        ),
        "d_constraints": [list(pair) for pair in sorted(theorem.d_constraints)],
        "proof_d_constraints": [
            list(pair) for pair in sorted(theorem.proof_d_constraints)
        ],
        "proof": proof_records(theorem, store),
        "rule": proof.rule if proof else theorem.name,
        "substitution": {
            key: expression_text(value)
            for key, value in (proof.substitution.items() if proof else ())
        },
    }


def export_jsonl(
    store: TheoremDatabase,
    destination: str | Path | TextIO,
    *,
    generated_only: bool = True,
) -> None:
    close = False
    if hasattr(destination, "write"):
        stream = destination  # type: ignore[assignment]
    else:
        stream = Path(destination).open("w", encoding="utf-8")
        close = True
    try:
        items = store.generated() if generated_only else iter(store)
        for theorem in items:
            stream.write(json.dumps(
                theorem_record(theorem, store), ensure_ascii=False
            ) + "\n")
    finally:
        if close:
            stream.close()


def export_graphviz(
    theorem: Theorem,
    store: TheoremDatabase,
    destination: str | Path,
) -> None:
    lines = ["digraph proof {", "  rankdir=BT;"]
    records = proof_records(theorem, store)
    ids = {record["id"] for record in records}
    if theorem.id is not None:
        ids.add(theorem.id)
    for theorem_id in sorted(ids):
        item = store[theorem_id]
        label = f"{item.name}\\n{expression_text(item.conclusion)}"
        lines.append(f'  n{theorem_id} [shape=box,label={json.dumps(label)}];')
    for record in records:
        for parent in record["parents"]:
            lines.append(f'  n{parent} -> n{record["id"]};')
    lines.append("}")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _SyntaxCompiler:
    def __init__(self, store: TheoremDatabase, floating: dict[str, tuple[str, str]]):
        if store.parsed is None:
            raise ValueError("Metamath export requires the parsed source database")
        self.parsed = store.parsed
        self.floating = floating

    def compile(self, expected_type: str, node: Node) -> list[str]:
        direct = self.floating.get(node.op)
        if not node.args and direct is not None and direct[1] == expected_type:
            return [direct[0]]

        for rule in self.parsed.syntax_rules.get(expected_type, ()):
            variables = [
                part for part in rule.pattern if part in rule.variable_types
            ]
            children: list[Node] | None = None
            if len(rule.pattern) == 1:
                part = rule.pattern[0]
                if part in rule.variable_types:
                    children = [node]
                elif not node.args and node.op == part:
                    children = []
            elif rule.pattern and rule.pattern[0] in rule.variable_types:
                if len(variables) == len(node.args) + 1:
                    children = [Node(node.op), *node.args]
            elif rule.pattern and node.op == rule.pattern[0]:
                if len(variables) == len(node.args):
                    children = list(node.args)
            if children is None:
                continue
            child_by_variable = dict(zip(variables, children))
            proof: list[str] = []
            try:
                assertion = self.parsed.statements[rule.label]
                for floating in assertion.floating:
                    variable = floating.expr.args[0].op
                    proof.extend(
                        self.compile(
                            rule.variable_types[variable],
                            child_by_variable[variable],
                        )
                    )
            except ValueError:
                continue
            proof.append(rule.label)
            return proof
        raise ValueError(f"cannot compile {node} as {expected_type}")


def _instantiate(expr: Node, mapping: dict[str, Node]) -> Node:
    return substitute_simultaneous(
        expr,
        {
            variable: value
            for variable, value in mapping.items()
            if value != Node(variable)
        },
    )


def _hypothesis_label(theorem: Theorem, expr: Node) -> str:
    for index, hypothesis in enumerate(theorem.hypotheses):
        if hypothesis.expr == expr:
            return f"g{theorem.id}_h{index}"
    raise ValueError(f"proof requires an unavailable hypothesis: {expr}")


def _application_proof(
    assertion: Theorem,
    mapping: dict[str, Node],
    current: Theorem,
    store: TheoremDatabase,
    compiler: _SyntaxCompiler,
) -> list[str]:
    labels: list[str] = []
    floating_order: list[tuple[str, str]] = []
    if assertion.floating:
        for floating in assertion.floating:
            variable = floating.expr.args[0].op
            floating_order.append((variable, floating.expr.op))
    else:
        # Generated assertions have no source floating-hypothesis tuple.
        # Their export order is canonical rather than dict-insertion based.
        floating_order.extend(sorted(assertion.variable_types.items()))
    for variable, typecode in floating_order:
        labels.extend(compiler.compile(typecode, mapping.get(variable, Node(variable))))
    for hypothesis in assertion.hypotheses:
        labels.append(_hypothesis_label(current, _instantiate(hypothesis.expr, mapping)))
    labels.append(assertion.name)
    return labels


def _generated_proof(
    theorem: Theorem,
    store: TheoremDatabase,
    compiler: _SyntaxCompiler,
) -> list[str]:
    if theorem.proof is None or store.parsed is None:
        raise ValueError(f"{theorem.name} has no generated proof")
    proof = theorem.proof
    rule = store.parsed.statements.get(proof.rule)
    if rule is None:
        try:
            rule = store.get_by_name(proof.rule)
        except KeyError as exc:
            raise ValueError(
                f"{theorem.name} uses unknown proof rule {proof.rule!r}"
            ) from exc
    rule_mapping = {
        variable: proof.substitution.get(f"__rule_{variable}", Node(variable))
        for variable in rule.variable_types
    }
    labels: list[str] = []
    floating_order = [
        (floating.expr.args[0].op, floating.expr.op)
        for floating in rule.floating
    ]
    if not floating_order:
        floating_order = sorted(rule.variable_types.items())
    for variable, typecode in floating_order:
        labels.extend(compiler.compile(typecode, rule_mapping[variable]))

    for index, premise in enumerate(rule.hypotheses):
        parent_id = proof.premise_map[index]
        if parent_id is None:
            labels.append(
                _hypothesis_label(theorem, _instantiate(premise.expr, rule_mapping))
            )
            continue
        parent = store[parent_id]
        parent_mapping = {
            variable: proof.substitution.get(
                f"__p{index}_{variable}", Node(variable)
            )
            for variable in parent.variable_types
        }
        labels.extend(
            _application_proof(parent, parent_mapping, theorem, store, compiler)
        )
    labels.append(rule.name)
    return labels


def export_metamath(
    theorem: Theorem,
    store: TheoremDatabase,
    destination: str | Path,
) -> None:
    """Write valid uncompressed Metamath declarations for a generated DAG.

    The fragment is intended to be appended to the source database.  Generated
    ancestors are emitted in dependency order, so every `$p` only references
    original assertions or earlier generated theorems.
    """

    if theorem.id is None:
        raise ValueError("the theorem must belong to a TheoremDatabase")
    order: list[Theorem] = []
    visited: set[int] = set()

    def visit(theorem_id: int) -> None:
        if theorem_id in visited:
            return
        visited.add(theorem_id)
        item = store[theorem_id]
        if item.proof:
            for parent_id in item.proof.parents:
                visit(parent_id)
        if item.kind == "generated":
            order.append(item)

    visit(theorem.id)
    variables = sorted({
        variable
        for item in order
        for variable in (
            set(item.variable_types)
            | set(item.proof_variable_types)
        )
    })
    lines = ["$( Append this generated proof fragment to the source database. $)"]
    if variables:
        lines.append(f"$v {' '.join(variables)} $.")
    for item in order:
        lines.append("${")
        all_variable_types = {
            **item.proof_variable_types,
            **item.variable_types,
        }
        floating = {
            variable: (f"g{item.id}_f{index}", typecode)
            for index, (variable, typecode) in enumerate(
                sorted(all_variable_types.items())
            )
        }
        for variable, (label, typecode) in floating.items():
            lines.append(f"  {label} $f {typecode} {variable} $.")
        for left, right in sorted(
            item.d_constraints | item.proof_d_constraints
        ):
            lines.append(f"  $d {left} {right} $.")
        for index, hypothesis in enumerate(item.hypotheses):
            lines.append(
                f"  g{item.id}_h{index} $e {expression_text(hypothesis.expr)} $."
            )
        compiler = _SyntaxCompiler(store, floating)
        labels = _generated_proof(item, store, compiler)
        lines.append(
            f"  {item.name} $p {expression_text(item.conclusion)} "
            f"$= {' '.join(labels)} $."
        )
        lines.append("$}")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


# Backwards-compatible descriptive alias.
export_metamath_trace = export_metamath

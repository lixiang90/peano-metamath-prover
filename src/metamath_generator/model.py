from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Iterator, Mapping


@dataclass(frozen=True, slots=True)
class Node:
    """A small immutable AST node.

    Complete Metamath expressions use the typecode as their root, for example
    ``Node("|-", (Node("implies", (...)),))``.  Schematic variables and
    constants are leaf nodes.
    """

    op: str
    args: tuple["Node", ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.op
        return " ".join((self.op, *(str(arg) for arg in self.args)))

    @property
    def depth(self) -> int:
        return 1 + max((arg.depth for arg in self.args), default=0)

    def walk(self) -> Iterator["Node"]:
        yield self
        for arg in self.args:
            yield from arg.walk()

    def to_prefix(self, include_typecode: bool = True) -> str:
        if not include_typecode and len(self.args) == 1:
            return self.args[0].to_prefix()
        if not self.args:
            return self.op
        return " ".join([self.op, *(arg.to_prefix() for arg in self.args)])


@dataclass(frozen=True, slots=True)
class Hypothesis:
    label: str
    expr: Node


@dataclass(slots=True)
class Proof:
    """Origin of a theorem.

    ``parents`` contains theorem database ids.  ``premise_map`` maps each rule
    premise to either a parent id or ``None`` (an open premise).
    """

    rule: str
    parents: list[int] = field(default_factory=list)
    substitution: dict[str, Node] = field(default_factory=dict)
    premise_map: list[int | None] = field(default_factory=list)
    depth: int = 0
    source_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class Theorem:
    name: str
    hypotheses: list[Hypothesis]
    conclusion: Node
    d_constraints: set[tuple[str, str]] = field(default_factory=set)
    proof: Proof | None = None
    variable_types: dict[str, str] = field(default_factory=dict)
    # Variables and distinct-variable conditions needed only while replaying
    # the proof.  They are deliberately separate from the mandatory
    # variables/conditions of the theorem statement.
    proof_variable_types: dict[str, str] = field(default_factory=dict)
    proof_d_constraints: set[tuple[str, str]] = field(default_factory=set)
    floating: tuple[Hypothesis, ...] = ()
    kind: str = "generated"
    source_tokens: tuple[str, ...] = ()
    # Source-order and scope metadata used by the independent verifier.
    # Generated in-memory certificates use -1 and derive a virtual position
    # after the parsed database.
    declaration_index: int = -1
    active_hypothesis_labels: frozenset[str] = frozenset()
    id: int | None = None

    @property
    def proof_depth(self) -> int:
        return self.proof.depth if self.proof is not None else 0

    def variables(self) -> set[str]:
        known = set(self.variable_types)
        return {
            node.op
            for expr in [*(h.expr for h in self.hypotheses), self.conclusion]
            for node in expr.walk()
            if node.op in known
        }


@dataclass(frozen=True, slots=True)
class SyntaxRule:
    label: str
    output_type: str
    pattern: tuple[str, ...]
    variable_types: Mapping[str, str]


@dataclass(slots=True)
class Database:
    symbols: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    variable_types: dict[str, str] = field(default_factory=dict)
    # The indexes below intentionally describe different logical roles:
    # axioms are logical $a declarations, rules are logical assertions with
    # open $e hypotheses, and proved_theorems are logical $p declarations.
    # A logical $a with hypotheses therefore belongs to both axioms and rules.
    axioms: dict[str, Theorem] = field(default_factory=dict)
    rules: dict[str, Theorem] = field(default_factory=dict)
    proved_theorems: dict[str, Theorem] = field(default_factory=dict)
    # Syntax assertions never appear in any of the three logical indexes.
    syntax_rules: dict[str, list[SyntaxRule]] = field(default_factory=dict)
    syntax_statements: dict[str, Theorem] = field(default_factory=dict)
    logical_assertions: dict[str, Theorem] = field(default_factory=dict)
    statements: dict[str, Theorem] = field(default_factory=dict)
    floating_hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    essential_hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    # Hypotheses still active at end-of-file.  External certificates appended
    # to a database must reuse these instead of redeclaring the same $v/$f.
    active_floating_hypotheses: dict[str, Hypothesis] = field(
        default_factory=dict
    )
    # Global lookup remains useful for diagnostics, but verification must
    # additionally consult source order and the theorem-local active scope.
    label_order: dict[str, int] = field(default_factory=dict)
    d_declarations: list[set[tuple[str, str]]] = field(default_factory=list)
    source_path: str | None = None

    def add_syntax_rule(self, rule: SyntaxRule) -> None:
        self.syntax_rules.setdefault(rule.output_type, []).append(rule)
        self.types.add(rule.output_type)


def normalized_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def distinct_pairs(names: Iterable[str]) -> set[tuple[str, str]]:
    return {normalized_pair(a, b) for a, b in combinations(dict.fromkeys(names), 2)}

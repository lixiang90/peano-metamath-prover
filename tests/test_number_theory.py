from __future__ import annotations

import unittest
from itertools import combinations
from pathlib import Path

from metamath_generator.database import TheoremDatabase
from metamath_generator.model import Node, normalized_pair
from metamath_generator.parser import parse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "formal" / "peano.mm"
NUMBER_THEORY = ROOT / "formal" / "peano-number-theory.mm"

TARGETS = {
    "flt-statement",
    "goldbach-statement",
    "pnt-statement",
    "riemann-von-koch-statement",
}


def quantified_variables(node: Node) -> set[str]:
    result: set[str] = set()
    for item in node.walk():
        if item.op in {"forall", "exists"} and len(item.args) == 2:
            result.add(item.args[0].op)
    return result


def free_object_variables(node: Node, bound: frozenset[str] = frozenset()) -> set[str]:
    if node.op in {"forall", "exists"} and len(node.args) == 2:
        variable = node.args[0].op
        return free_object_variables(node.args[1], bound | {variable})
    result = set()
    if not node.args and node.op.startswith("x") and node.op[1:].isdigit():
        if node.op not in bound:
            result.add(node.op)
    for arg in node.args:
        result.update(free_object_variables(arg, bound))
    return result


class NumberTheoryExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = parse(BASE)
        cls.database = parse(NUMBER_THEORY)

    def test_extension_is_partitioned(self) -> None:
        self.assertEqual(len(self.database.logical_assertions), 59)
        self.assertEqual(len(self.database.syntax_statements), 52)
        self.assertEqual(len(self.database.rules), 5)
        self.assertFalse(TARGETS & set(self.database.logical_assertions))
        self.assertTrue(TARGETS <= set(self.database.syntax_statements))

        store = TheoremDatabase.from_parsed(self.database)
        self.assertFalse(TARGETS & {theorem.name for theorem in store})

    def test_new_logical_assertions_are_acyclic_explicit_definitions(self) -> None:
        base_labels = set(self.base.logical_assertions)
        defined_predicates: set[str] = set()
        new_predicates = {
            theorem.conclusion.args[0].args[0].op
            for label, theorem in self.database.logical_assertions.items()
            if label not in base_labels
        }

        for label, theorem in self.database.logical_assertions.items():
            if label in base_labels:
                continue
            self.assertTrue(label.startswith("df-"), label)
            body = theorem.conclusion.args[0]
            self.assertEqual(body.op, "iff", label)
            lhs, rhs = body.args
            self.assertNotIn(lhs.op, {node.op for node in rhs.walk()}, label)
            dependencies = {
                node.op for node in rhs.walk() if node.op in new_predicates
            }
            self.assertTrue(dependencies <= defined_predicates, (label, dependencies))
            defined_predicates.add(lhs.op)

        self.assertEqual(defined_predicates, new_predicates)
        self.assertEqual(len(defined_predicates), 29)

    def test_definition_binders_have_freshness_constraints(self) -> None:
        base_labels = set(self.base.logical_assertions)
        for label, theorem in self.database.logical_assertions.items():
            if label in base_labels:
                continue
            body = theorem.conclusion.args[0]
            lhs, rhs = body.args
            binders = quantified_variables(rhs)
            lhs_variables = {
                node.op
                for node in lhs.walk()
                if node.op in theorem.variable_types
            }
            for binder in binders:
                for parameter in lhs_variables:
                    self.assertIn(
                        normalized_pair(binder, parameter),
                        theorem.d_constraints,
                        (label, binder, parameter),
                    )
            for left, right in combinations(sorted(binders), 2):
                self.assertIn(
                    normalized_pair(left, right),
                    theorem.d_constraints,
                    (label, left, right),
                )

    def test_named_formulas_are_closed_and_have_expected_vocabulary(self) -> None:
        expected_symbols = {
            "flt-statement": {"pow"},
            "goldbach-statement": {"prime", "even"},
            "pnt-statement": {"pntat"},
            "riemann-von-koch-statement": {"rhat"},
        }
        for label, required in expected_symbols.items():
            theorem = self.database.syntax_statements[label]
            self.assertEqual(theorem.conclusion.op, "statement")
            formula = theorem.conclusion.args[0]
            self.assertFalse(free_object_variables(formula), label)
            operations = {node.op for node in formula.walk()}
            self.assertTrue(required <= operations, label)


if __name__ == "__main__":
    unittest.main()

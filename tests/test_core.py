from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metamath_generator.compose import (
    CompositionError,
    compose,
    instantiate_assertion,
)
from metamath_generator.database import TheoremDatabase
from metamath_generator.export import export_metamath
from metamath_generator.generator import (
    GenerationConfig,
    TheoremGenerator,
)
from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import MetamathParser, ParseError, parse
from metamath_generator.unification import (
    OccursCheckError,
    substitute,
    unify,
)
from metamath_generator.verifier import VerificationError, verify

ROOT = Path(__file__).resolve().parents[1]
PEANO = ROOT / "formal" / "peano.mm"


def typed(typecode: str, body: Node) -> Node:
    return Node(typecode, (body,))


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO)

    def test_parses_peano_and_scope_hypotheses(self) -> None:
        self.assertEqual(len(self.database.statements), 48)
        mp = self.database.statements["ax-mp"]
        self.assertEqual([h.label for h in mp.hypotheses], ["min", "maj"])
        self.assertEqual(
            str(mp.conclusion),
            "|- psi",
        )
        self.assertNotIn("min", [
            h.label for h in self.database.statements["eq-refl"].hypotheses
        ])

    def test_separates_syntax_and_logical_assertions(self) -> None:
        syntax_labels = set(self.database.syntax_statements)
        logical_labels = set(self.database.logical_assertions)
        self.assertEqual(len(syntax_labels), 18)
        self.assertEqual(len(logical_labels), 30)
        self.assertFalse(syntax_labels & logical_labels)
        self.assertFalse(syntax_labels & set(self.database.axioms))
        self.assertFalse(syntax_labels & set(self.database.rules))
        self.assertFalse(syntax_labels & set(self.database.proved_theorems))
        self.assertIn("wff_logbinop", syntax_labels)
        self.assertIn("ax-1", self.database.axioms)
        self.assertIn("ax-mp", self.database.rules)
        self.assertTrue(all(
            theorem.conclusion.op == "|-"
            for theorem in self.database.rules.values()
        ))
        self.assertTrue(all(
            theorem.hypotheses for theorem in self.database.rules.values()
        ))

    def test_turnstile_requires_a_wff(self) -> None:
        malformed = PEANO.read_text(encoding="utf-8") + "\nbad $a |- 0 $.\n"
        with self.assertRaises(ParseError):
            MetamathParser().parse_text(malformed)

    def test_builds_prefix_ast(self) -> None:
        expression = self.database.statements["pa_ax4"].conclusion
        self.assertEqual(expression.op, "|-")
        self.assertEqual(expression.args[0].op, "=")
        self.assertEqual(expression.args[0].args[0].op, "S")
        self.assertEqual(expression.args[0].args[1].op, "+")

    def test_preserves_distinct_constraints(self) -> None:
        theorem = self.database.statements["pa_ax7"]
        self.assertEqual(theorem.d_constraints, {("x", "z"), ("y", "z")})
        self.assertEqual(len(self.database.d_declarations), 14)


class UnificationTests(unittest.TestCase):
    def test_recursive_unification_and_substitution(self) -> None:
        left = Node("implies", (Node("?a"), Node("?b")))
        right = Node("implies", (Node("P"), Node("not", (Node("P"),))))
        result = unify(left, right)
        self.assertEqual(result["?a"], Node("P"))
        self.assertEqual(result["?b"], Node("not", (Node("P"),)))
        self.assertEqual(substitute(left, result), right)

    def test_operator_metavariable(self) -> None:
        left = Node("?op", (Node("?x"), Node("?y")))
        right = Node("=", (Node("s"), Node("t")))
        result = unify(left, right)
        self.assertEqual(result["?op"], Node("="))
        self.assertEqual(substitute(left, result), right)

    def test_occurs_check(self) -> None:
        with self.assertRaises(OccursCheckError):
            unify(Node("?x"), Node("S", (Node("?x"),)))


class VerifierVisibilityTests(unittest.TestCase):
    def test_rejects_future_assertion_reference(self) -> None:
        database = MetamathParser().parse_text(
            "$c |- wff $. $v ph $. wph $f wff ph $. "
            "bad $p |- ph $= wph later $. "
            "later $a |- ph $."
        )
        with self.assertRaisesRegex(
            VerificationError, "not declared before"
        ):
            verify(database.statements["bad"], database)

    def test_rejects_hypothesis_after_scope_exit(self) -> None:
        database = MetamathParser().parse_text(
            "$c |- wff $. $v ph $. wph $f wff ph $. "
            "${ h $e |- ph $. inside $a |- ph $. $} "
            "bad $p |- ph $= h $."
        )
        with self.assertRaisesRegex(VerificationError, "not active"):
            verify(database.statements["bad"], database)

    def test_rejects_floating_hypothesis_after_scope_exit(self) -> None:
        database = MetamathParser().parse_text(
            "$c wff $. $v ph ps $. "
            "${ wph $f wff ph $. inside $a wff ph $. $} "
            "wps $f wff ps $. bad $p wff ps $= wph $."
        )
        with self.assertRaisesRegex(VerificationError, "not active"):
            verify(database.statements["bad"], database)

    def test_rejects_self_reference(self) -> None:
        database = MetamathParser().parse_text(
            "$c |- wff $. $v ph $. wph $f wff ph $. "
            "bad $p |- ph $= wph bad $."
        )
        with self.assertRaisesRegex(
            VerificationError, "not declared before"
        ):
            verify(database.statements["bad"], database)

    def test_accepts_currently_active_hypothesis(self) -> None:
        database = MetamathParser().parse_text(
            "$c |- wff $. $v ph $. wph $f wff ph $. "
            "${ h $e |- ph $. good $p |- ph $= h $. $}"
        )
        verify(database.statements["good"], database)


class CompositionTests(unittest.TestCase):
    def test_assertion_instantiation_is_simultaneous(self) -> None:
        database = parse(PEANO)
        assertion = Theorem(
            "synthetic-distinct",
            [],
            typed("|-", Node("=", (Node("s"), Node("t")))),
            d_constraints={("s", "t")},
            variable_types={"s": "term", "t": "term"},
        )
        result = instantiate_assertion(
            assertion,
            {"s": Node("t"), "t": Node("u")},
            database=database,
        )
        self.assertEqual(
            result.conclusion,
            typed("|-", Node("=", (Node("t"), Node("u")))),
        )
        self.assertEqual(result.d_constraints, {("t", "u")})
        self.assertEqual(result.variable_types, {"t": "term", "u": "term"})

    def test_cross_type_unification_is_rejected(self) -> None:
        database = parse(PEANO)
        with self.assertRaises(CompositionError):
            compose(
                database.logical_assertions["ax-mp"],
                [
                    database.logical_assertions["pa_ax3"],
                    database.logical_assertions["eq-sym"],
                ],
                database=database,
            )

    def test_partial_modus_ponens_keeps_open_premises(self) -> None:
        mp = Theorem(
            "mp",
            [
                Hypothesis("minor", typed("|-", Node("A"))),
                Hypothesis(
                    "major",
                    typed("|-", Node("implies", (Node("A"), Node("B")))),
                ),
            ],
            typed("|-", Node("B")),
            variable_types={"A": "wff", "B": "wff"},
        )
        parent = Theorem(
            "conditional",
            [Hypothesis("context", typed("|-", Node("G")))],
            typed("|-", Node("implies", (Node("P"), Node("Q")))),
            variable_types={"G": "wff", "P": "wff", "Q": "wff"},
            id=3,
        )
        result = compose(mp, [None, parent])
        self.assertEqual(
            [h.expr for h in result.hypotheses],
            [typed("|-", Node("__p1_P")), typed("|-", Node("__p1_G"))],
        )
        self.assertEqual(result.conclusion, typed("|-", Node("__p1_Q")))
        self.assertEqual(result.proof.premise_map, [None, 3])

    def test_distinct_variable_collapse_is_rejected(self) -> None:
        rule = Theorem(
            "distinct",
            [Hypothesis("h", typed("|-", Node("=", (Node("x"), Node("y")))))],
            typed("|-", Node("x")),
            d_constraints={("x", "y")},
            variable_types={"x": "term", "y": "term"},
        )
        parent = Theorem(
            "refl",
            [],
            typed("|-", Node("=", (Node("z"), Node("z")))),
            variable_types={"z": "term"},
            id=1,
        )
        with self.assertRaises(CompositionError):
            compose(rule, [parent])

    def test_alpha_equivalent_theorems_are_deduplicated(self) -> None:
        one = Theorem(
            "one", [], typed("|-", Node("P")), variable_types={"P": "wff"}
        )
        two = Theorem(
            "two", [], typed("|-", Node("Q")), variable_types={"Q": "wff"}
        )
        store = TheoremDatabase()
        self.assertTrue(store.add(one)[1])
        self.assertFalse(store.add(two)[1])


class IntegrationTests(unittest.TestCase):
    def test_generation_json_shape_and_metamath_proof(self) -> None:
        parsed = parse(PEANO)
        generator = TheoremGenerator(
            parsed,
            GenerationConfig(seed=7, max_proof_depth=5),
        )
        generated = generator.generate("random", 100)
        self.assertGreater(len(generated), 0)

        with tempfile.TemporaryDirectory() as directory:
            fragment = Path(directory) / "generated.mm"
            export_metamath(generated[-1], generator.store, fragment)
            combined = (
                PEANO.read_text(encoding="utf-8")
                + "\n"
                + fragment.read_text(encoding="utf-8")
            )
            exported = MetamathParser().parse_text(combined)
            generated_statements = [
                theorem
                for theorem in exported.statements.values()
                if theorem.name.startswith("gen")
            ]
            self.assertGreater(len(generated_statements), 0)
            for theorem in generated_statements:
                verify(theorem, exported)

    def test_export_declares_variables_used_only_by_proof(self) -> None:
        source = (
            PEANO.read_text(encoding="utf-8")
            + "\n"
            + "$v a b $. ta $f term a $. tb $f term b $. "
            + "proof-minor $a |- = a a $. "
            + "proof-major $a |- implies = a a and = b b = b b $.\n"
        )
        parsed = MetamathParser().parse_text(source)
        store = TheoremDatabase.from_parsed(parsed)
        generated = compose(
            parsed.statements["ax-mp"],
            [
                store.get_by_name("eq-refl"),
                store.get_by_name("proof-major"),
            ],
            name="proof-variable-closure",
            database=parsed,
        )
        theorem_id, added = store.add(generated)
        self.assertTrue(added)
        generated = store[theorem_id]
        self.assertEqual(len(generated.variable_types), 1)
        self.assertEqual(len(generated.proof_variable_types), 2)

        with tempfile.TemporaryDirectory() as directory:
            fragment = Path(directory) / "proof-closure.mm"
            export_metamath(generated, store, fragment)
            exported = MetamathParser().parse_text(
                source + "\n" + fragment.read_text(encoding="utf-8")
            )
            verify(
                exported.statements["proof-variable-closure"],
                exported,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from metamath_generator.export import export_metamath
from metamath_generator.datasets import DATASET_FILENAMES, export_datasets
from metamath_generator.generator import (
    GenerationConfig,
    TheoremGenerator,
)
from metamath_generator.model import Hypothesis, Node, Proof, Theorem
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.quality import (
    QualityAnalyzer,
    remove_vacuous_quantifiers,
    semantic_profile,
)
from metamath_generator.verifier import verify

ROOT = Path(__file__).resolve().parents[1]
PEANO = ROOT / "formal" / "peano.mm"


def typed(typecode: str, body: Node) -> Node:
    return Node(typecode, (body,))


class SemanticQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO)

    def test_vacuous_quantifiers_are_removed_only_in_quality_view(self) -> None:
        original = typed(
            "|-",
            Node("forall", (Node("x"), Node("P"))),
        )
        normalized, removed = remove_vacuous_quantifiers(
            original,
            {"x": "var", "P": "wff"},
        )
        self.assertEqual(removed, 1)
        self.assertEqual(normalized, typed("|-", Node("P")))
        # The immutable proof-bearing expression was not rewritten.
        self.assertEqual(original.args[0].op, "forall")

        used = typed(
            "|-",
            Node(
                "forall",
                (Node("x"), Node("=", (Node("x"), Node("0")))),
            ),
        )
        self.assertEqual(
            remove_vacuous_quantifiers(used, {"x": "var"})[0],
            used,
        )

    def test_semantic_key_ignores_premise_order_and_variable_names(self) -> None:
        one = Theorem(
            "one",
            [
                Hypothesis(
                    "h1",
                    typed("|-", Node("implies", (Node("P"), Node("Q")))),
                ),
                Hypothesis("h2", typed("|-", Node("P"))),
            ],
            typed("|-", Node("Q")),
            variable_types={"P": "wff", "Q": "wff"},
        )
        two = Theorem(
            "two",
            [
                Hypothesis("g2", typed("|-", Node("R"))),
                Hypothesis(
                    "g1",
                    typed("|-", Node("implies", (Node("R"), Node("S")))),
                ),
            ],
            typed("|-", Node("S")),
            variable_types={"R": "wff", "S": "wff"},
        )
        self.assertEqual(
            semantic_profile(one).full_key,
            semantic_profile(two).full_key,
        )

    def test_quality_demotes_bare_and_rejects_vacuous(self) -> None:
        analyzer = QualityAnalyzer(self.database)
        bare = Theorem(
            "bare",
            [Hypothesis("h", typed("|-", Node("P")))],
            typed("|-", Node("Q")),
            variable_types={"P": "wff", "Q": "wff"},
        )
        bare_assessment, _ = analyzer.assess(bare)
        self.assertEqual(bare_assessment.category, "proof_states")
        self.assertFalse(bare_assessment.active)

        vacuous = Theorem(
            "vacuous",
            [],
            typed("|-", Node("forall", (Node("x"), Node("P")))),
            variable_types={"x": "var", "P": "wff"},
        )
        vacuous_assessment, _ = analyzer.assess(vacuous)
        self.assertTrue(vacuous_assessment.hard_reject)
        self.assertEqual(vacuous_assessment.vacuous_quantifiers, 1)

        quantified = Theorem(
            "quantified",
            [],
            typed(
                "|-",
                Node(
                    "forall",
                    (Node("x"), Node("=", (Node("x"), Node("0")))),
                ),
            ),
            variable_types={"x": "var"},
        )
        quantified_assessment, _ = analyzer.assess(quantified)
        self.assertEqual(quantified_assessment.nonvacuous_quantifiers, 1)


class TheoremGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = parse(PEANO)

    def test_subsumption_replaces_more_demanding_generated_rule(self) -> None:
        generator = TheoremGenerator(
            self.database,
            GenerationConfig(seed=1),
        )
        conclusion = typed(
            "|-",
            Node("implies", (Node("P"), Node("P"))),
        )
        stronger = Theorem(
            "stronger",
            [
                Hypothesis("a", typed("|-", Node("A"))),
                Hypothesis("b", typed("|-", Node("B"))),
            ],
            conclusion,
            proof=Proof("test", depth=1),
            variable_types={"P": "wff", "A": "wff", "B": "wff"},
        )
        weaker = Theorem(
            "weaker",
            [Hypothesis("a", typed("|-", Node("A")))],
            conclusion,
            proof=Proof("test", depth=1),
            variable_types={"P": "wff", "A": "wff"},
        )
        self.assertTrue(generator._admit(stronger))
        stronger_id = generator._new_ids[-1]
        self.assertTrue(generator._admit(weaker))
        weaker_id = generator._new_ids[-1]
        self.assertIn(stronger_id, generator.dominated_ids)
        self.assertNotIn(stronger_id, generator.active_ids)
        self.assertIn(weaker_id, generator.active_ids)

    def test_rejects_weakening_of_an_already_closed_theorem(self) -> None:
        generator = TheoremGenerator(
            self.database,
            GenerationConfig(seed=1),
        )
        weakened = Theorem(
            "weak-pa-ax1",
            [],
            typed(
                "|-",
                Node(
                    "implies",
                    (
                        Node("P"),
                        Node(
                            "not",
                            (
                                Node(
                                    "=",
                                    (Node("0"), Node("S", (Node("x"),))),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            proof=Proof("test", depth=1),
            variable_types={"P": "wff", "x": "var"},
        )
        self.assertFalse(generator._admit(weakened))
        self.assertEqual(
            generator.rejected["known_consequence_weakening"],
            1,
        )

    def test_generation_filters_bad_active_candidates(self) -> None:
        generator = TheoremGenerator(
            self.database,
            GenerationConfig(seed=7, max_proof_depth=5),
        )
        generated = generator.generate("random", 600)
        self.assertGreater(len(generated), 0)
        active_generated = [
            theorem_id
            for theorem_id in generator._new_ids
            if theorem_id in generator.active_ids
        ]
        self.assertGreater(len(active_generated), 0)
        for theorem_id in active_generated:
            assessment = generator.assessments[theorem_id]
            self.assertEqual(assessment.vacuous_quantifiers, 0)
            self.assertFalse(assessment.bare_conclusion)
            self.assertLessEqual(generator._alpha_chain(theorem_id), 1)
        self.assertGreater(
            generator.stats["full_discharge_plans"],
            generator.stats["partial_discharge_plans"],
        )

    def test_split_export_and_metamath_replay(self) -> None:
        generator = TheoremGenerator(
            self.database,
            GenerationConfig(seed=11, max_proof_depth=5),
        )
        generator.generate("random", 500)
        categorized = generator.categorized()
        preferred = [
            *categorized["closed_theorems"],
            *categorized["inference_rules"],
        ]
        self.assertTrue(preferred)

        with tempfile.TemporaryDirectory() as directory:
            paths = export_datasets(generator, directory)
            for category, filename in DATASET_FILENAMES.items():
                self.assertEqual(paths[category].name, filename)
                self.assertTrue(paths[category].exists())
            records = []
            for category in DATASET_FILENAMES:
                records.extend(
                    json.loads(line)
                    for line in paths[category].read_text(
                        encoding="utf-8"
                    ).splitlines()
                )
            self.assertTrue(records)
            self.assertTrue(all("quality" in record for record in records))
            for category, path in paths.items():
                if category == "summary":
                    continue
                scores = [
                    json.loads(line)["quality"]["score"]
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(scores, sorted(scores, reverse=True))

            fragment = Path(directory) / "generated.mm"
            export_metamath(preferred[-1], generator.store, fragment)
            combined = (
                PEANO.read_text(encoding="utf-8")
                + "\n"
                + fragment.read_text(encoding="utf-8")
            )
            exported = MetamathParser().parse_text(combined)
            generated = [
                theorem
                for theorem in exported.statements.values()
                if theorem.name.startswith("gen")
            ]
            self.assertTrue(generated)
            for theorem in generated:
                verify(theorem, exported)


if __name__ == "__main__":
    unittest.main()

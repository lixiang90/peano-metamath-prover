from __future__ import annotations

import unittest
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from metamath_generator.database import TheoremDatabase
from metamath_generator.definitions import (
    DefinitionCatalogError,
    audit_catalog_requirements,
    audit_conservative_extension,
    load_definition_catalog,
    render_definition_catalog,
)
from metamath_generator.parser import MetamathParser, ParseError, parse


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "formal" / "peano-number-theory.mm"
CATALOG = ROOT / "formal" / "pa-plus-definitions.json"
EXTENSION = ROOT / "formal" / "peano-pa-plus.mm"


class PaPlusDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_definition_catalog(CATALOG)
        cls.base = parse(BASE)
        cls.extension = parse(EXTENSION)

    def test_generated_file_is_deterministic(self) -> None:
        self.assertEqual(
            EXTENSION.read_text(encoding="utf-8"),
            render_definition_catalog(self.catalog),
        )

    def test_all_fresh_assertions_are_conservative_definitions(self) -> None:
        report = audit_conservative_extension(
            self.base,
            self.extension,
            expected_definitions=self.catalog.definition_names,
            expected_statements=self.catalog.statement_names,
            allowed_unused_parameters=self.catalog.allowed_unused_parameters,
        )
        self.assertEqual(len(report.definitions), 68)
        self.assertEqual(len(report.statements), 35)
        self.assertEqual(set(report.definition_names), set(self.catalog.definition_names))
        self.assertLessEqual(max(item.body_nodes for item in report.definitions), 64)

    def test_statement_dependency_metadata_matches_formal_asts(self) -> None:
        audit_catalog_requirements(self.catalog, self.extension)

    def test_target_formulas_are_not_available_as_theorems(self) -> None:
        targets = set(self.catalog.statement_names)
        self.assertFalse(targets & set(self.extension.logical_assertions))
        self.assertTrue(targets <= set(self.extension.syntax_statements))
        theorem_names = {
            theorem.name for theorem in TheoremDatabase.from_parsed(self.extension)
        }
        self.assertFalse(targets & theorem_names)

    def test_targets_do_not_register_as_syntax_rules(self) -> None:
        target_labels = set(self.catalog.statement_names)
        syntax_rule_labels = {
            rule.label
            for rules in self.extension.syntax_rules.values()
            for rule in rules
        }
        self.assertFalse(target_labels & syntax_rule_labels)

    def test_statement_payload_cannot_validate_by_self_registration(self) -> None:
        malformed = "$c statement 0 $. bad $a statement 0 $."
        with self.assertRaises(ParseError):
            MetamathParser().parse_text(malformed)

    def test_audit_rejects_changed_base_and_extra_vocabulary(self) -> None:
        changed_base = deepcopy(self.extension)
        inherited_label = next(iter(self.base.statements))
        changed_base.statements[inherited_label].kind = "tampered"
        with self.assertRaisesRegex(
            DefinitionCatalogError, "changes inherited base declaration"
        ):
            audit_conservative_extension(self.base, changed_base)

        extra_constant = deepcopy(self.extension)
        extra_constant.symbols.add("rogue")
        with self.assertRaisesRegex(DefinitionCatalogError, "fresh constant mismatch"):
            audit_conservative_extension(self.base, extra_constant)

    def test_audit_rejects_malformed_syntax_and_unapproved_unused_parameter(
        self,
    ) -> None:
        malformed_syntax = deepcopy(self.extension)
        rules = malformed_syntax.syntax_rules["wff"]
        index = next(
            index for index, rule in enumerate(rules) if rule.label == "wff_congruent"
        )
        rules[index] = replace(rules[index], pattern=("congruent", "a", "b"))
        with self.assertRaisesRegex(
            DefinitionCatalogError, "syntax declaration does not match left side"
        ):
            audit_conservative_extension(self.base, malformed_syntax)

        with self.assertRaisesRegex(
            DefinitionCatalogError, "unused parameters mismatch"
        ):
            audit_conservative_extension(self.base, self.extension)

    def test_beta_is_hidden_behind_one_sequence_interface(self) -> None:
        base_labels = set(self.base.logical_assertions)
        direct_beta_users: set[str] = set()
        for label, theorem in self.extension.logical_assertions.items():
            if label in base_labels:
                continue
            rhs = theorem.conclusion.args[0].args[1]
            if "beta" in {node.op for node in rhs.walk()}:
                direct_beta_users.add(label)
        self.assertEqual(direct_beta_users, {"df-seqat"})

    def test_mainstream_number_theory_domains_are_explicit(self) -> None:
        for name in ("congruent", "divisorcount", "divisorsum", "eulerphi"):
            rhs = self.extension.logical_assertions[f"df-{name}"].conclusion.args[0].args[1]
            self.assertEqual(rhs.op, "and", name)
            self.assertEqual(rhs.args[0].op, "positive", name)

        order_rhs = self.extension.logical_assertions[
            "df-multiplicativeorder"
        ].conclusion.args[0].args[1]
        order_ops = {node.op for node in order_rhs.walk()}
        self.assertIn("coprime", order_ops)
        self.assertIn("congruent", order_ops)
        self.assertIn("pow", order_ops)

        divisor_target = self.extension.syntax_statements[
            "divisorcount-square-statement"
        ].conclusion.args[0]
        self.assertIn("positive", {node.op for node in divisor_target.walk()})

    def test_binomial_is_total_and_root_criterion_is_nontrivial(self) -> None:
        binomial_rhs = self.extension.logical_assertions[
            "df-binomial"
        ].conclusion.args[0].args[1]
        self.assertEqual(binomial_rhs.op, "or")
        outside_domain = binomial_rhs.args[1]
        self.assertEqual(outside_domain.op, "and")
        self.assertEqual(outside_domain.args[0].op, "<")
        self.assertEqual(outside_domain.args[1].op, "=")

        root_target = self.extension.syntax_statements[
            "nthroot-rational-criterion-statement"
        ].conclusion.args[0]
        root_ops = {node.op for node in root_target.walk()}
        self.assertIn("nthrootexact", root_ops)
        self.assertIn("coprime", root_ops)
        self.assertIn("=", root_ops)

    def test_catalog_covers_all_first_wave_domains(self) -> None:
        definition_areas = Counter(item.area for item in self.catalog.definitions)
        statement_areas = Counter(item.area for item in self.catalog.statements)
        self.assertGreaterEqual(definition_areas["number-theory"], 20)
        self.assertGreaterEqual(definition_areas["finite-sequences"], 10)
        self.assertGreaterEqual(definition_areas["signed-arithmetic"], 7)
        self.assertGreaterEqual(definition_areas["signed-rationals"], 8)
        self.assertGreaterEqual(definition_areas["analysis-certificates"], 9)
        self.assertGreaterEqual(statement_areas["number-theory"], 15)
        self.assertGreaterEqual(statement_areas["analysis-certificates"], 5)


if __name__ == "__main__":
    unittest.main()

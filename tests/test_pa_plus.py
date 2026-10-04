from __future__ import annotations

import tempfile
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
from metamath_generator.export import export_metamath
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.parser import MetamathParser, ParseError, parse
from metamath_generator.verifier import verify
from neural_prover.data import (
    CorpusBuildConfig,
    build_corpus,
    load_examples,
)


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

    def test_definition_guidance_builds_bidirectional_certified_bridges(
        self,
    ) -> None:
        focus = ("positive", "multiplicativeorder")
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                seed=7,
                bootstrap_definitions=True,
                focus_predicates=focus,
                definition_coverage_weight=3.0,
            ),
        )
        generated = generator.generate("random", 0)
        self.assertEqual(len(generated), 2 * len(focus))
        for predicate in focus:
            bodies = [
                theorem.conclusion.args[0]
                for theorem in generated
                if predicate in {
                    node.op for node in theorem.conclusion.walk()
                }
            ]
            self.assertEqual(len(bodies), 2, predicate)
            self.assertTrue(all(body.op == "implies" for body in bodies))
            self.assertEqual(sum(body.args[0].op == predicate for body in bodies), 1)
            self.assertEqual(sum(body.args[1].op == predicate for body in bodies), 1)

        summary = generator.summary()
        self.assertEqual(summary.definition_predicates_seen, len(focus))
        self.assertEqual(summary.definition_coverage, 1.0)
        self.assertTrue(all(summary.definition_usage[name] == 2 for name in focus))

    def test_definition_bridge_exports_and_replays(self) -> None:
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                focus_predicates=("positive",),
                bounded_nat_max=2,
                ground_instances_per_predicate=1,
            ),
        )
        generated = generator.generate("random", 0)
        instance = next(
            theorem for theorem in generated if "bounded" in theorem.name
        )
        with tempfile.TemporaryDirectory() as directory:
            fragment = Path(directory) / "generated.mm"
            export_metamath(instance, generator.store, fragment)
            parser = MetamathParser()
            expanded = " ".join(parser._tokens_with_includes(EXTENSION, set()))
            combined = parser.parse_text(
                expanded + "\n" + fragment.read_text(encoding="utf-8")
            )
            replayed = [
                theorem
                for theorem in combined.proved_theorems.values()
                if theorem.name.startswith("gen")
            ]
            self.assertEqual(len(replayed), 2)
            for theorem in replayed:
                verify(theorem, combined)

    def test_bounded_instances_are_canonical_and_targets_stay_nonlogical(
        self,
    ) -> None:
        focus = ("positive", "multiplicativeorder", "binomial")
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                seed=7,
                focus_predicates=focus,
                bounded_nat_max=2,
                ground_instances_per_predicate=1,
                target_statements=self.catalog.statement_names,
                target_guidance_weight=4.0,
            ),
        )
        generated = generator.generate("random", 0)
        bounded = [
            theorem for theorem in generated if "bounded" in theorem.name
        ]
        self.assertEqual(len(bounded), 2 * len(focus))

        def is_numeral(node) -> bool:
            return node.op == "0" or (
                node.op == "S"
                and len(node.args) == 1
                and is_numeral(node.args[0])
            )

        for theorem in bounded:
            self.assertNotIn("term", theorem.variable_types.values())
            predicate = next(
                name for name in focus if name in theorem.name
            )
            occurrence = next(
                node for node in theorem.conclusion.walk()
                if node.op == predicate
            )
            self.assertTrue(all(is_numeral(arg) for arg in occurrence.args))
        summary = generator.summary()
        self.assertEqual(summary.bounded_ground_instances, len(bounded))
        self.assertEqual(summary.search_stored, 0)
        self.assertEqual(summary.target_statements_total, 35)
        self.assertFalse(
            set(self.catalog.statement_names)
            & set(generator.parsed.logical_assertions)
        )

    def test_bounded_instances_enumerate_unique_assignments(self) -> None:
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                focus_predicates=("positive",),
                bounded_nat_max=2,
                ground_instances_per_predicate=4,
            ),
        )
        generated = generator.generate("random", 0)
        bounded = [
            theorem for theorem in generated if "bounded" in theorem.name
        ]
        # positive has one term parameter, so 0..2 gives three assignments
        # per direction even though the requested cap is four.
        self.assertEqual(len(bounded), 6)
        self.assertEqual(len({theorem.conclusion for theorem in bounded}), 6)

    def test_pa_plus_neural_corpus_keeps_bootstrap_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            manifest = build_corpus(
                EXTENSION,
                output,
                CorpusBuildConfig(
                    generation_mode="random",
                    seeds=(7,),
                    steps_per_seed=0,
                    definition_catalog=str(CATALOG),
                    bootstrap_definitions=True,
                    definition_coverage_weight=3.0,
                    bounded_nat_max=2,
                    ground_instances_per_predicate=1,
                    target_guidance_weight=4.0,
                    max_definition_only_search_per_predicate=1,
                    max_state_tokens=1024,
                    max_action_tokens=512,
                    include_source_actions=False,
                ),
            )
            examples = sum((
                load_examples(output / f"{split}.jsonl")
                for split in ("train", "validation", "test")
            ), [])
        self.assertEqual(manifest["runs"][0]["definition_bridges"], 136)
        self.assertEqual(
            manifest["runs"][0]["bounded_ground_instances"], 136
        )
        kinds = {example.generation_kind for example in examples}
        self.assertEqual(kinds, {"definition_bridge", "bounded_instance"})
        bounded = next(
            example for example in examples
            if example.generation_kind == "bounded_instance"
        )
        self.assertTrue(bounded.rule.startswith("gen_df_"))
        self.assertIn("<PA_PLUS_CONTEXT>", bounded.state_tokens)
        self.assertIn("<DEFINITION_BRIDGE>", bounded.action_tokens)
        self.assertTrue(
            "<UNFOLD>" in bounded.action_tokens
            or "<FOLD>" in bounded.action_tokens
        )
        self.assertTrue(any(
            token.startswith("<V:term:")
            for token in bounded.action_tokens
        ))

    def test_backward_guidance_improves_target_similarity(self) -> None:
        common = dict(
            seed=7,
            max_proof_depth=8,
            focus_predicates=self.catalog.definition_names,
            bootstrap_definitions=True,
            bounded_nat_max=2,
            ground_instances_per_predicate=1,
            target_statements=self.catalog.statement_names,
            definition_coverage_weight=3.0,
        )
        unguided = TheoremGenerator(
            self.extension,
            GenerationConfig(**common),
        )
        unguided.generate("random", 150)
        guided = TheoremGenerator(
            self.extension,
            GenerationConfig(**common, target_guidance_weight=4.0),
        )
        guided.generate("random", 150)
        unguided_summary = unguided.summary()
        guided_summary = guided.summary()
        self.assertGreater(
            guided_summary.target_mean_best_similarity,
            unguided_summary.target_mean_best_similarity,
        )
        self.assertGreater(
            guided_summary.target_statements_touched,
            unguided_summary.target_statements_touched,
        )

    def test_definition_only_search_quota_does_not_remove_bootstraps(
        self,
    ) -> None:
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                seed=7,
                focus_predicates=("positive",),
                bounded_nat_max=2,
                ground_instances_per_predicate=1,
                max_definition_only_search_per_predicate=0,
            ),
        )
        generator.generate("random", 200)
        summary = generator.summary()
        self.assertEqual(summary.definition_bridges, 2)
        self.assertEqual(summary.bounded_ground_instances, 2)
        self.assertEqual(summary.definition_only_search_admitted, 0)
        self.assertGreater(
            summary.rejected.get("definition_only_search_quota", 0),
            0,
        )

    def test_guided_random_pa_plus_dag_exports_and_replays(self) -> None:
        focus = ("positive", "multiplicativeorder", "binomial")
        generator = TheoremGenerator(
            self.extension,
            GenerationConfig(
                seed=11,
                max_proof_depth=6,
                bootstrap_definitions=True,
                focus_predicates=focus,
                definition_coverage_weight=3.0,
            ),
        )
        generator.generate("random", 250)
        candidates = [
            theorem
            for theorem in generator.store.generated()
            if theorem.proof_depth >= 2
            and {
                node.op for node in theorem.conclusion.walk()
            } & set(focus)
        ]
        self.assertTrue(candidates)

        with tempfile.TemporaryDirectory() as directory:
            fragment = Path(directory) / "generated.mm"
            export_metamath(candidates[-1], generator.store, fragment)
            parser = MetamathParser()
            expanded = " ".join(parser._tokens_with_includes(EXTENSION, set()))
            combined = parser.parse_text(
                expanded + "\n" + fragment.read_text(encoding="utf-8")
            )
            replayed = [
                theorem
                for theorem in combined.proved_theorems.values()
                if theorem.name.startswith("gen")
            ]
            self.assertGreaterEqual(len(replayed), 2)
            for theorem in replayed:
                verify(theorem, combined)

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

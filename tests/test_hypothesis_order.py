from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from metamath_generator.compose import _standardize, compose, instantiate_assertion
from metamath_generator.database import TheoremDatabase, _canonicalize
from metamath_generator.export import export_metamath
from metamath_generator.model import Node, Theorem
from metamath_generator.parser import MetamathParser
from metamath_generator.verifier import VerificationError, verify
from neural_prover.certificate import (
    compile_certificate,
    export_certificate,
    verify_certificate,
)
from neural_prover.environment import BackwardEnvironment, ProofState, Tactic
from neural_prover.external import verify_certificate_external


SOURCE = """
$c |- wff not $.
$v p q unused $.
wp $f wff p $.
wn $a wff not p $.
${
  h $e |- p $.
  wq $f wff q $.
  wu $f wff unused $.
  r $a |- q $.
  good $p |- q $= wp h wq r $.
  bad $p |- q $= wp wq h r $.
$}
${
  oq $f wff q $.
  oh $e |- q $.
  outer $a |- not q $.
$}
"""


class HypothesisOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = MetamathParser().parse_text(SOURCE)

    def test_verifier_accepts_declaration_order_and_rejects_grouped_order(self):
        assertion = self.database.statements["r"]
        self.assertEqual(
            [hypothesis.label for hypothesis in assertion.mandatory_hypotheses],
            ["wp", "h", "wq"],
        )
        # The unused floating hypothesis is active but not mandatory.
        self.assertNotIn("wu", [h.label for h in assertion.mandatory_hypotheses])
        verify(self.database.statements["good"], self.database)
        with self.assertRaises(VerificationError):
            verify(self.database.statements["bad"], self.database)
        # The inner essential and floating declarations leave scope together.
        self.assertEqual(
            [h.label for h in self.database.statements["outer"].mandatory_hypotheses],
            ["oq", "oh"],
        )

    def test_normalization_preserves_interleaving(self):
        assertion = self.database.statements["r"]
        canonical, _ = _canonicalize(assertion)
        for renamed in (canonical, _standardize(assertion, "rule")):
            self.assertEqual(renamed.hypothesis_order, assertion.hypothesis_order)
            self.assertEqual(
                [h.expr.op for h in renamed.mandatory_hypotheses],
                ["wff", "|-", "wff"],
            )

    def verify_export(self, theorem, store):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.mm"
            export_metamath(theorem, store, path)
            database = MetamathParser().parse_text(SOURCE + path.read_text(encoding="utf-8"))
        for generated in store.generated():
            verify(database.statements[generated.name], database)
        return database.statements[theorem.name]

    def test_generated_rule_and_generated_ancestor_use_their_own_orders(self):
        store = TheoremDatabase(self.database)
        instance = instantiate_assertion(
            self.database.statements["r"], {}, name="instance", database=self.database,
        )
        first_id, _ = store.add(instance)
        first = store[first_id]
        parsed = self.verify_export(first, store)
        self.assertEqual(
            parsed.proof.source_labels,
            (f"g{first_id}_f0", f"g{first_id}_h0", f"g{first_id}_f1", "r"),
        )
        # Exported generated ancestors declare all $f before all $e.
        repeated = instantiate_assertion(
            first, {"v1": Node("v0")}, name="repeated", database=self.database,
        )
        repeated_id, _ = store.add(repeated)
        self.verify_export(store[repeated_id], store)

    def test_composed_parent_application_retains_source_order(self):
        store = TheoremDatabase.from_parsed(self.database)
        theorem = compose(
            self.database.statements["outer"],
            [self.database.statements["r"]],
            name="composed", database=self.database,
        )
        theorem_id, added = store.add(theorem)
        self.assertTrue(added)
        parsed = self.verify_export(store[theorem_id], store)
        r_position = parsed.proof.source_labels.index("r")
        preceding = parsed.proof.source_labels[r_position - 3:r_position]
        self.assertEqual(preceding, (
            f"g{theorem_id}_f0", f"g{theorem_id}_h0", f"g{theorem_id}_f1",
        ))

    def test_neural_certificate_uses_interleaved_assertion_order(self):
        assertion = self.database.statements["r"]
        target = Theorem(
            "target", list(assertion.hypotheses), assertion.conclusion,
            variable_types=dict(assertion.variable_types),
        )
        environment = BackwardEnvironment(self.database)
        tactic = Tactic.create("r", {"p": Node("p")})
        transition = environment.apply(ProofState.from_theorem(target), tactic)
        self.assertTrue(transition.after.solved)
        result = SimpleNamespace(solved=True, transitions=(transition,))
        certificate = compile_certificate(target, result, self.database)
        verify_certificate(certificate, self.database)
        labels = certificate.proof.source_labels
        self.assertEqual(labels[1], "neural_proof_h0")
        self.assertEqual(labels[-1], "r")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.mm"
            export_certificate(certificate, path, ambient_database=self.database)
            database = MetamathParser().parse_text(SOURCE + path.read_text(encoding="utf-8"))
        verify(database.statements[certificate.name], database)


    @unittest.skipUnless(os.environ.get("METAMATH_EXECUTABLE"), "set METAMATH_EXECUTABLE for official Metamath tests")
    def test_official_verifier_accepts_interleaved_certificate(self):
        # Keep the deliberate negative fixture out of VERIFY PROOF *.
        valid_source = SOURCE.replace("  bad $p |- q $= wp wq h r $.\n", "")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mm"
            source.write_text(valid_source, encoding="utf-8")
            database = MetamathParser().parse_file(source)
            target = database.statements["r"]
            environment = BackwardEnvironment(database)
            tactic = Tactic.create("r", {"p": Node("p")})
            transition = environment.apply(ProofState.from_theorem(target), tactic)
            self.assertTrue(transition.after.solved)
            certificate = compile_certificate(
                target, SimpleNamespace(solved=True, transitions=(transition,)), database,
            )
            verify_certificate(certificate, database)
            result = verify_certificate_external(certificate, source)
            self.assertTrue(result.passed, result.output_tail)

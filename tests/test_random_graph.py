"""End-to-end contracts for fresh, replayable proof graphs."""
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import pytest

from metamath_generator.certificates import replay_graph, flatten_certificate, certificate_steps
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.verifier import verify
from neural_prover.audit import audit_corpus_actions, audit_scale_corpus
from neural_prover.data import CorpusBuildConfig, build_corpus, load_examples
from neural_prover.scale_data import ScaleCorpusConfig, build_scale_corpus, iter_scale_records

ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "formal/peano-pa-plus.mm"


PARTIAL_SOURCE = """
$c wff |- P R S f $.
$v ph $.
wph $f wff ph $.
wp $a wff P $.
wr $a wff R $.
ws $a wff S $.
wf $a wff f ph $.
factp $a |- P $.
factfp $a |- f P $.
${
  first $e |- ph $.
  second $e |- f ph $.
  join $a |- R $.
$}
${
  inherited $e |- f P $.
  parent $a |- P $.
$}
${
  single $e |- P $.
  lift $a |- S $.
$}
"""


def partial_graph(probability, seed=7):
    from metamath_generator.random_graph import RandomProofGraph
    database = MetamathParser().parse_text(PARTIAL_SOURCE)
    g = TheoremGenerator(database, GenerationConfig(seed=seed,
        graph_instance_probability=0, graph_derived_rule_probability=0,
        graph_partial_premise_probability=probability))
    graph = RandomProofGraph(g)
    graph.rules = [database.rules["join"]]
    # Both premises always have compatible certified parents. This isolates
    # deliberate open-premise sampling from candidate-search failures.
    graph.candidates = lambda premise, rule: [g.store.get_by_name(
        "factfp" if premise.args[0].op == "f" else "factp")]
    return g, graph


@pytest.mark.parametrize("probability", [0, 1])
def test_partial_sampling_with_all_parents_available(probability):
    g, graph = partial_graph(probability)
    graph.run(30)
    nodes = list(g.store.generated())
    assert nodes
    for t in nodes:
        assert t.proof.premise_map.count(None) == probability
        assert len(t.hypotheses) == probability
        if probability:
            # The shared variable is solved by the chosen old node; the new
            # assumption must be P or f P, never unconstrained ph.
            assert t.hypotheses[0].expr.to_prefix() in {"|- P", "|- f P"}
            assert not t.variable_types
    assert g.stats["graph_partial_plan_admitted"] == probability * len(nodes)
    assert g.stats["graph_new_hypotheses"] == probability * len(nodes)
    assert g.stats["graph_nodes_with_reserved_new_hypothesis"] == probability * len(nodes)
    expanded = replay_graph(g.store, PARTIAL_SOURCE)
    for t in nodes:
        verify(expanded.proved_theorems[t.name], expanded)


def test_partial_sampling_frequency_and_single_premise_rules():
    g, graph = partial_graph(.3)
    for _ in range(1000):
        assert graph.combine() is not None
    assert g.stats["graph_multi_premise_plans"] == 1000
    assert 250 <= g.stats["graph_partial_plans"] <= 350
    graph.rules = [g.parsed.rules["lift"]]
    g.config.graph_partial_premise_probability = 1
    theorem = graph.combine()
    assert theorem.proof.premise_map == [g.store.get_by_name("factp").id]
    assert not theorem.hypotheses
    assert g.stats["graph_single_premise_plans"] == 1


def test_new_assumptions_are_distinguished_from_inherited_ones():
    from metamath_generator.compose import compose
    from metamath_generator.random_graph import premise_statistics
    g, _ = partial_graph(1)
    rule = g.parsed.rules["join"]
    t = compose(rule, [g.store.get_by_name("parent"), None], database=g.parsed)
    # Both the inherited and reserved assumption become f P after unification.
    assert [h.expr.to_prefix() for h in t.hypotheses] == ["|- f P"]
    stats = premise_statistics(t, rule, g.store, reserved_premise=1)
    assert stats["graph_inherited_hypotheses"] == 1
    assert stats["graph_new_hypotheses"] == 0
    assert stats["graph_nodes_with_reserved_new_hypothesis"] == 0
    node_id, _ = g.store.add(t)
    assert premise_statistics(g.store[node_id], rule, g.store, 1) == stats
    expanded = replay_graph(g.store, PARTIAL_SOURCE)
    verify(expanded.proved_theorems[t.name], expanded)


def test_failed_matching_still_preserves_open_premises():
    g, graph = partial_graph(0)
    graph.candidates = lambda premise, rule: (
        [] if premise.args[0].op == "f" else [g.store.get_by_name("factp")])
    graph.run(1)
    theorem = next(g.store.generated())
    assert [h.expr.to_prefix() for h in theorem.hypotheses] == ["|- f P"]
    assert g.stats["graph_partial_plan_admitted"] == 0
    assert g.stats["graph_new_hypotheses"] == 1
    assert g.stats["graph_unmatched_after_search"] == 1


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan")])
def test_partial_probability_rejects_invalid_values(probability):
    g, _ = partial_graph(probability)
    with pytest.raises(ValueError, match="graph_partial_premise_probability"):
        g.generate("graph", 1)


def test_graph_steps_with_action_only_variables_are_reported(tmp_path):
    source = tmp_path / "dummy.mm"
    source.write_text("""
$c wff |- P f $.
$v ph $.
wph $f wff ph $.
wp $a wff P $.
wf $a wff f ph $.
ax $a |- f ph $.
${
  premise $e |- f ph $.
  forget $a |- P $.
$}
""", encoding="utf-8")
    corpus = tmp_path / "corpus"
    result = build_corpus(source, corpus, CorpusBuildConfig(seeds=(7,), steps_per_seed=1,
        graph_instance_probability=0, include_source_actions=False))
    # The complete proof is valid. Its ax step has ph in the goal; forget
    # uses ph only in its action, beyond what the v1 state protocol supports.
    assert result["total"] == 1
    assert result["runs"][0]["rejected"]["graph_action_only_variables"] == 1
    assert audit_corpus_actions(source, corpus)["invalid_actions"] == 0


def test_flattening_repeated_lemmas_freshens_proof_local_variables(tmp_path):
    from metamath_generator.compose import compose
    from pa_prover_v2.kernel import ProofKernel
    source = """
$c wff |- P R f $.
$v ph ps $.
wph $f wff ph $.
wps $f wff ps $.
wp $a wff P $.
wr $a wff R $.
wf $a wff f ph ps $.
${
  $d ph ps $.
  ax $a |- f ph ps $.
$}
${
  premise $e |- f ph ps $.
  forget $a |- P $.
$}
${
  first $e |- P $.
  second $e |- P $.
  join $a |- R $.
$}
"""
    path = tmp_path / "source.mm"
    path.write_text(source, encoding="utf-8")
    database = parse(path)
    g = TheoremGenerator(database)
    lemma = compose(database.rules["forget"], [g.store.get_by_name("ax")],
                    name="local_lemma", database=database)
    lemma_id, _ = g.store.add(lemma)
    lemma = g.store[lemma_id]
    root = compose(database.rules["join"], [lemma, lemma], name="twice", database=database)
    root_id, _ = g.store.add(root)
    expanded = replay_graph(g.store, source)
    certificate = flatten_certificate(expanded.proved_theorems[g.store[root_id].name], expanded, database)
    assert len(certificate.floating) == len(certificate.variable_types) == 4
    assert len(certificate.d_constraints) == 2
    assert all(label in database.statements or label in {h.label for h in certificate.floating}
               for label in certificate.proof.source_labels)
    ProofKernel(path).verify(certificate)
    assert list(certificate_steps(certificate, database))[-1].conclusion == certificate.conclusion


def test_graph_without_random_substitution_reuses_derived_rules_and_proves_deeper():
    database = parse(PA)
    g = TheoremGenerator(database, GenerationConfig(seed=7, max_proof_depth=12,
                                                   graph_instance_probability=0))
    g.generate("graph", 220)
    generated = list(g.store.generated())
    assert g.stats["random_instance"] == 0
    assert max(t.proof_depth for t in generated) >= 8
    assert g.stats["graph_reused_parents"] > 0
    derived_uses = 0
    for theorem in generated:
        assert all(parent < theorem.id for parent in theorem.proof.parents)
        if theorem.proof.rule not in database.logical_assertions:
            derived_uses += 1
            rule = g.store.get_by_name(theorem.proof.rule)
            assert rule.id in theorem.proof.parents
            assert theorem.proof_depth > rule.proof_depth
    assert derived_uses > 0
    text = " ".join(MetamathParser()._tokens_with_includes(PA, set()))
    expanded = replay_graph(g.store, text)
    for theorem in generated:
        verify(expanded.proved_theorems[theorem.name], expanded)
    deepest = max(generated, key=lambda t: t.proof_depth)
    cert = flatten_certificate(expanded.proved_theorems[deepest.name], expanded, database)
    steps = list(certificate_steps(cert, database))
    assert len(steps) > 1
    assert steps[-1].conclusion == cert.conclusion
    assert all(s.proof.rule in database.logical_assertions for s in steps)


def test_graph_data_contains_all_source_steps_and_audits(tmp_path):
    result = build_corpus(PA, tmp_path, CorpusBuildConfig(seeds=(7,), steps_per_seed=60,
        graph_instance_probability=0, include_source_actions=False, max_state_tokens=1024,
        max_action_tokens=1024, max_proof_depth=10))
    examples = [e for s in ("train", "validation", "test") for e in load_examples(tmp_path / f"{s}.jsonl")]
    assert result["total"] > result["runs"][0]["stored"]
    assert all(e.value_target == 1 for e in examples)
    assert all(e.rule in parse(PA).logical_assertions for e in examples)
    assert audit_corpus_actions(PA, tmp_path)["invalid_actions"] == 0
    assert list((tmp_path / "proof-graphs").glob("*.mm"))


def test_graph_scale_resume_matches_uninterrupted_stream(tmp_path):
    base = tmp_path / "base"
    build_corpus(PA, base, CorpusBuildConfig(seeds=(7,), steps_per_seed=2))
    cfg = ScaleCorpusConfig(train_examples=60, validation_examples=12, test_examples=12,
                            graph_steps=40, shard_size=12, workers=1, seed=123)
    partial = tmp_path / "partial"
    first = build_scale_corpus(PA, base, partial, replace(cfg, max_new_records=17))
    assert first["total"] == 17 and not first["complete"]
    last = build_scale_corpus(PA, base, partial, cfg)
    whole = tmp_path / "whole"
    build_scale_corpus(PA, base, whole, cfg)
    assert last["complete"]
    for split in ("train", "validation", "test"):
        assert list(iter_scale_records(partial, split)) == list(iter_scale_records(whole, split))
    assert audit_scale_corpus(PA, partial, sample_size=1000)["invalid_actions"] == 0


def test_graph_generation_is_independent_of_python_hash_seed():
    script = """
import json
from metamath_generator.generator import TheoremGenerator, GenerationConfig
from metamath_generator.database import theorem_hash
from metamath_generator.parser import parse
g=TheoremGenerator(parse('formal/peano-pa-plus.mm'),GenerationConfig(seed=19))
g.generate('graph',80)
print(json.dumps([(theorem_hash(t), t.proof.rule, t.proof.depth) for t in g.store.generated()]))
"""
    outputs = []
    for seed in ("3", "71"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT / "src"))
        outputs.append(subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, env=env, timeout=30))
    assert outputs[0] == outputs[1]


def test_v2_uses_graph_and_operator_variable_identity(tmp_path):
    from pa_prover_v2.data import DataConfig, generate_corpus, load_episodes
    from pa_prover_v2.kernel import ProofKernel, statement_fingerprint
    from metamath_generator.model import Node, Theorem
    quantified = Theorem("quant", [], Node("|-", (Node("q", (Node("x"), Node("p"))),)),
                         variable_types={"q": "QUANT", "x": "var", "p": "wff"},
                         d_constraints={("q", "x")})
    assert statement_fingerprint(quantified)
    cfg = DataConfig(seeds=(7, 11), steps_per_seed=60, max_examples=16,
                     bootstrap_definitions=False, graph_instance_probability=0)
    result = generate_corpus(PA, tmp_path, cfg)
    assert result["generation_algorithm"] == "graph"
    assert result["generation_kind"].get("random_instance", 0) == 0
    assert result["generation_kind"]["random_graph"] > 0
    assert not any(k.startswith("KeyError") for k in result["filtered"])
    kernel = ProofKernel(PA)
    for split in ("train", "validation", "test"):
        load_episodes(tmp_path, split, kernel=kernel)

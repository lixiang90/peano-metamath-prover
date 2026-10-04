"""End-to-end contracts for fresh, replayable proof graphs."""
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from metamath_generator.certificates import replay_graph, flatten_certificate, certificate_steps
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.verifier import verify
from neural_prover.audit import audit_corpus_actions, audit_scale_corpus
from neural_prover.data import CorpusBuildConfig, build_corpus, load_examples
from neural_prover.scale_data import ScaleCorpusConfig, build_scale_corpus, iter_scale_records

ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "formal/peano-pa-plus.mm"


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

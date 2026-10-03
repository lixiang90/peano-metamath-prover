from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from metamath_generator.database import theorem_hash
from metamath_generator.model import Hypothesis, Node, Proof, Theorem
from pa_prover_v2.kernel import (
    KernelError, ProofBudgetError, ProofKernel, content_hash, node_from_data, node_to_data,
    statement_fingerprint, statement_text, theorem_from_data, theorem_to_data,
)
from pa_prover_v2.library import (
    LibraryBudgetError, LibraryError, LibraryLeakageError, StaleIndexError, TheoremLibrary,
)

ROOT = Path(__file__).resolve().parents[2]


def imp(a, b):
    return Node("implies", (a, b))


def claim(a):
    return Node("|-", (a,))


@pytest.fixture(scope="module")
def kernel():
    return ProofKernel(ROOT / "formal" / "peano.mm")


def ax1(kernel, *, variant=0):
    p, q = Node("p"), Node("q")
    right = q if variant == 0 else imp(q, q) if variant == 1 else imp(p, q)
    target = Theorem("target", [], claim(imp(p, imp(right, p))), variable_types={"p": "wff", "q": "wff"})
    return kernel.compose(kernel.database.logical_assertions["ax-1"], {"phi": p, "psi": right}, [], target)


def test_codecs_preserve_all_theorem_metadata_and_hash_contract(kernel):
    theorem = ax1(kernel)
    theorem.proof.parents = [2, 3]
    theorem.proof.premise_map = [2, None]
    theorem.proof_variable_types = {"r": "wff"}
    theorem.proof_d_constraints = {("p", "r")}
    theorem.hypothesis_order = (("f", 0), ("f", 1))
    theorem.declaration_index = 12
    theorem.active_hypothesis_labels = frozenset({"f0", "f1"})
    theorem.source_tokens = ("|-", "p")
    theorem.id = 17
    assert theorem_from_data(theorem_to_data(theorem)) == theorem
    assert node_from_data(node_to_data(theorem.conclusion)) == theorem.conclusion
    assert statement_fingerprint(theorem) == theorem_hash(theorem)


def test_no_new_axioms_and_no_spoofed_source_rule(kernel):
    theorem = ax1(kernel)
    forged = Theorem("invented", [], claim(Node("p")), variable_types={"p": "wff"})
    with pytest.raises(KernelError):
        kernel.verify(forged)
    with pytest.raises(KernelError):
        kernel.compose(forged, {"p": Node("p")}, [], theorem)
    spoofed = copy.deepcopy(kernel.database.logical_assertions["ax-1"])
    spoofed.conclusion = claim(Node("phi"))
    with pytest.raises(KernelError):
        kernel.compose(spoofed, {"phi": Node("p"), "psi": Node("q")}, [], theorem)


def scoped_kernel(tmp_path, *, distinct=False):
    source = tmp_path / "ordered.mm"
    source.write_text("""$c wff |- not $.
$v p q $.
fp $f wff p $.
wn $a wff not p $.
${
 hp $e |- p $.
 fq $f wff q $.
""" + (" $d p q $.\n" if distinct else "") + " r $a |- q $.\n$}\n", encoding="utf-8")
    return ProofKernel(source)


def context(x="x", y="y", *, distinct=False):
    return Theorem("context", [Hypothesis("h", claim(Node(x)))], claim(Node(y)),
                   variable_types={x: "wff", y: "wff"},
                   d_constraints={(x, y)} if distinct else set())


def test_interleaved_source_order_and_hypothesis_bearing_library_macro(tmp_path):
    kernel = scoped_kernel(tmp_path)
    initial = context("p", "q")
    assumption = kernel.assumption(initial, 0)
    lemma = kernel.compose(kernel.database.logical_assertions["r"], {"p": Node("p"), "q": Node("q")}, [assumption], initial)
    assert lemma.proof.source_labels == (lemma.floating[0].label, lemma.hypotheses[0].label, lemma.floating[1].label, "r")
    library = TheoremLibrary(kernel)
    ref = library.add(lemma)
    new_context = context()
    instantiated = kernel.compose(library.get(ref), {"p": Node("x"), "q": Node("y")}, [kernel.assumption(new_context, 0)], new_context)
    kernel.verify(instantiated)
    assert instantiated.conclusion == new_context.conclusion
    assert lemma.name not in instantiated.proof.source_labels
    assert instantiated.proof.source_labels[-1] == "r"
    with pytest.raises(KernelError):
        kernel.compose(lemma, {"p": Node("x"), "q": Node("y")}, [], new_context)
    foreign = context("z", "y")
    with pytest.raises(KernelError):
        kernel.compose(lemma, {"p": Node("x"), "q": Node("y")}, [kernel.assumption(foreign, 0)], new_context)


def test_composition_preflights_repeated_library_premises_and_exact_budget(tmp_path):
    source = tmp_path / "duplication.mm"
    source.write_text("""$c wff |- $.
$v p $.
fp $f wff p $.
${
 h1 $e |- p $.
 h2 $e |- p $.
 both $a |- p $.
$}
""", encoding="utf-8")
    kernel = ProofKernel(source)
    target = Theorem("target", [Hypothesis("h", claim(Node("x")))], claim(Node("x")),
                     variable_types={"x": "wff"})
    assumption = kernel.assumption(target, 0)
    rule = kernel.database.logical_assertions["both"]
    with pytest.raises(ProofBudgetError, match="source application"):
        kernel.compose(rule, {"p": Node("x")}, [assumption, assumption], target, max_proof_labels=3)
    macro = kernel.compose(rule, {"p": Node("x")}, [assumption, assumption], target, max_proof_labels=4)
    assert len(macro.proof.source_labels) == 4
    assert macro.proof.source_labels.count(macro.hypotheses[0].label) == 2
    doubled = kernel.compose(macro, {"x": Node("x")}, [macro], target, max_proof_labels=10)
    assert len(doubled.proof.source_labels) == 10
    with pytest.raises(ProofBudgetError, match="library expansion"):
        kernel.compose(macro, {"x": Node("x")}, [doubled], target, max_proof_labels=21)
    exact = kernel.compose(macro, {"x": Node("x")}, [doubled], target, max_proof_labels=22)
    assert len(exact.proof.source_labels) == 22
    kernel.verify(exact)
    # Even a single supplied premise is checked before assembling the parent.
    with pytest.raises(ProofBudgetError):
        kernel.compose(rule, {"p": Node("x")}, [exact, assumption], target, max_proof_labels=21)
    with pytest.raises(KernelError, match="positive integer"):
        kernel.compose(rule, {"p": Node("x")}, [assumption, assumption], target, max_proof_labels=0)


def test_distinct_variable_collapse_and_missing_scope_are_rejected(tmp_path):
    kernel = scoped_kernel(tmp_path, distinct=True)
    rule = kernel.database.logical_assertions["r"]
    valid = context(distinct=True)
    kernel.verify(kernel.compose(rule, {"p": Node("x"), "q": Node("y")}, [kernel.assumption(valid, 0)], valid))
    missing = context()
    with pytest.raises(KernelError):
        kernel.compose(rule, {"p": Node("x"), "q": Node("y")}, [kernel.assumption(missing, 0)], missing)
    with pytest.raises(KernelError):
        kernel.compose(rule, {"p": Node("x"), "q": Node("x")}, [kernel.assumption(valid, 0)], valid)
    with pytest.raises(ValueError):
        kernel.compose(rule, {"p": Node("x"), "q": Node("out_of_scope")}, [kernel.assumption(valid, 0)], valid)


def test_library_roundtrip_replays_every_certificate_and_rejects_tamper(kernel, tmp_path):
    library = TheoremLibrary(kernel)
    ref = library.add(ax1(kernel))
    path = tmp_path / "library.json"
    library.save(path)
    loaded = TheoremLibrary.load(path, kernel)
    assert loaded.record_ids == (ref,)
    kernel.verify(loaded.get(ref))
    payload = json.loads(path.read_text())
    record = payload["records"][0]
    record["certificate"]["proof"]["source_labels"] = ["invented"]
    # Even an attacker who recomputes every unauthenticated metadata hash
    # cannot replace a valid proof with a false one.
    record["content_hash"] = content_hash(record["certificate"])
    record["id"] = record["content_hash"]
    del record["record_hash"]
    record["record_hash"] = content_hash(record)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KernelError):
        TheoremLibrary.load(path, kernel)


def test_theory_identity_uses_actual_source_contents(kernel, tmp_path):
    source = tmp_path / "peano.mm"
    source.write_bytes((ROOT / "formal" / "peano.mm").read_bytes() + b"\n$( changed source $)\n")
    other = ProofKernel(source)
    assert other.theory_fingerprint != kernel.theory_fingerprint
    library = TheoremLibrary(kernel)
    library.add(ax1(kernel))
    path = tmp_path / "library.json"
    library.save(path)
    with pytest.raises(LibraryError, match="different theory"):
        TheoremLibrary.load(path, other)


def test_train_only_target_and_future_guards_survive_roundtrip(kernel, tmp_path):
    first, held_out = ax1(kernel), ax1(kernel, variant=1)
    excluded = statement_fingerprint(held_out)
    library = TheoremLibrary(kernel, exclude_fingerprints={excluded}, max_step=4)
    for provenance in ("validation", "test", {"split": "validation"}):
        with pytest.raises(LibraryLeakageError):
            library.add(first, provenance=provenance)
    with pytest.raises(LibraryLeakageError):
        library.add(held_out)
    with pytest.raises(LibraryLeakageError):
        library.add(first, step=5)
    library.add(first, step=3)
    assert not library.search("implies", max_step=2)
    assert not library.search("implies", exclude_fingerprints={statement_fingerprint(first)})
    path = tmp_path / "library.json"
    library.save(path)
    loaded = TheoremLibrary.load(path, kernel, max_step=100, exclude_fingerprints={"another_target"})
    assert loaded.max_step == 4
    assert loaded.exclude_fingerprints == {excluded, "another_target"}
    with pytest.raises(LibraryLeakageError):
        loaded.add(held_out)
    with pytest.raises(LibraryLeakageError):
        loaded.add(first, step=5)


def test_count_and_actual_serialized_byte_budgets_are_hard(kernel, tmp_path):
    first, second = ax1(kernel), ax1(kernel, variant=1)
    library = TheoremLibrary(kernel, max_items=1)
    first_id = library.add(first, importance=1)
    second_id = library.add(second, importance=10)
    assert library.record_ids == (second_id,)
    assert first_id not in library.record_ids
    single_size = library.bytes_used
    bounded = TheoremLibrary(kernel, max_items=10, max_bytes=single_size + 50)
    bounded.add(first)
    bounded.add(second, importance=20)
    assert len(bounded) == 1 and bounded.bytes_used <= bounded.max_bytes
    path = tmp_path / "bounded.json"
    bounded.save(path)
    assert path.stat().st_size <= bounded.max_bytes
    with pytest.raises(LibraryBudgetError):
        TheoremLibrary.load(path, kernel, max_bytes=100)
    with pytest.raises(LibraryBudgetError):
        TheoremLibrary(kernel, max_bytes=100)


def test_score_refresh_and_library_records_are_immutable(kernel):
    library = TheoremLibrary(kernel, max_items=2)
    first_id = library.add(ax1(kernel), importance=1)
    second_id = library.add(ax1(kernel, variant=1), importance=1)
    library.update_scores({first_id: {"importance": 100, "reuse_count": 20}})
    assert library.search("no lexical match", limit=1)[0].id == first_id
    theorem = library.get(first_id)
    theorem.proof.source_labels = ("corrupt",)
    kernel.verify(library.get(first_id))
    assert set(library.record_ids) == {first_id, second_id}


def test_statement_only_single_batch_encoder_and_staleness(kernel, tmp_path):
    library = TheoremLibrary(kernel)
    first = ax1(kernel)
    first.proof.parents = [1, 1]  # dependency metadata never enters encoding
    first_id = library.add(first)
    calls = []
    def encoder(texts):
        calls.append(list(texts))
        return [[float(text.count("implies")), float(len(text))] for text in texts]
    metadata = library.build_index(encoder, "encoder-v1")
    assert len(calls) == 1 and calls[0] == [statement_text(first)]
    assert first.name not in calls[0][0] and "source_labels" not in calls[0][0]
    assert metadata["library_version"] == library.version
    assert library.search(first, encoder=encoder, encoder_fingerprint="encoder-v1")[0].id == first_id
    with pytest.raises(StaleIndexError):
        library.search(first, encoder=encoder, encoder_fingerprint="encoder-v2")
    library.add(ax1(kernel, variant=1))
    with pytest.raises(StaleIndexError):
        library.search(first, encoder=encoder, encoder_fingerprint="encoder-v1")
    library.build_index(encoder, "encoder-v2")
    path = tmp_path / "index.json"
    library.save(path)
    loaded = TheoremLibrary.load(path, kernel)
    assert loaded.search(first, encoder=encoder, encoder_fingerprint="encoder-v2")


def test_optional_cpu_torch_encoder(kernel):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    library = TheoremLibrary(kernel)
    theorem = ax1(kernel)
    library.add(theorem)
    def encoder(texts):
        return torch.tensor([[len(text), 1.0] for text in texts], device="cpu")
    library.build_index(encoder, "cpu-torch-v1")
    assert library.search(theorem, encoder=encoder, encoder_fingerprint="cpu-torch-v1")


def test_required_external_verifier_failure_cannot_be_success(kernel):
    with pytest.raises(KernelError, match="external"):
        kernel.certify(ax1(kernel), required_external=True, external_verifier="Z:/does-not-exist/metamath.exe")


def test_episode_cannot_retype_an_active_source_variable(tmp_path):
    source = tmp_path / "active.mm"
    source.write_text("""$c wff term |- implies $.
$v p q x $.
fp $f wff p $.
fq $f wff q $.
wi $a wff implies p q $.
ax $a |- implies p implies q p $.
tx $f term x $.
""", encoding="utf-8")
    kernel = ProofKernel(source)
    target = Theorem("target", [], claim(imp(Node("x"), imp(Node("x"), Node("x")))), variable_types={"x": "wff"})
    with pytest.raises(KernelError, match="active source floating"):
        kernel.compose(kernel.database.logical_assertions["ax"], {"p": Node("x"), "q": Node("x")}, [], target)


def test_flatten_counts_all_proof_labels_and_split_components(kernel):
    from pa_prover_v2.data import DataConfig, _assign_splits, flatten_certificate
    theorem = ax1(kernel)
    expanded = copy.copy(kernel.database)
    expanded.floating_hypotheses = {**expanded.floating_hypotheses, **{hyp.label: hyp for hyp in theorem.floating}}
    with pytest.raises(ValueError, match="budget"):
        flatten_certificate(theorem, expanded, kernel.database, max_labels=2)
    kernel.verify(flatten_certificate(theorem, expanded, kernel.database, max_labels=3))
    records = [
        {"statement_sha256": "a", "skeleton_sha256": "x"},
        {"statement_sha256": "b", "skeleton_sha256": "x"},
        {"statement_sha256": "b", "skeleton_sha256": "y"},
    ]
    _assign_splits(records, DataConfig())
    assert len({record["split_group"] for record in records}) == 1
    assert len({record["split"] for record in records}) == 1


def test_dataset_loader_replays_proofs_even_after_file_hash_is_recomputed(kernel, tmp_path):
    import hashlib
    from pa_prover_v2.data import _digest, load_episodes
    theorem = ax1(kernel)
    library = TheoremLibrary(kernel)
    library.add(theorem)
    library.save(tmp_path / "library.json")
    record = {
        "split": "train", "theory_sha256": kernel.theory_fingerprint,
        "statement_sha256": statement_fingerprint(theorem),
        "skeleton_sha256": _digest(["ax-1"]), "certificate": theorem_to_data(theorem),
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 3,
        "theory_sha256": kernel.theory_fingerprint, "counts": {"train": 1},
        "files": {"train.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()},
        "library_sha256": hashlib.sha256((tmp_path / "library.json").read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert len(load_episodes(tmp_path, kernel=kernel)) == 1
    record["certificate"]["proof"]["source_labels"] = ["unproved-label"]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest["files"]["train.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(KernelError):
        load_episodes(tmp_path, kernel=kernel)


def test_small_generated_corpus_reloads_library_and_all_teacher_certificates(tmp_path):
    from pa_prover_v2.data import DataConfig, generate_corpus, load_episodes
    source = ROOT / "formal" / "peano-pa-plus.mm"
    manifest = generate_corpus(source, tmp_path, DataConfig(
        seeds=(7,), steps_per_seed=20, max_examples=12,
        max_proof_labels=256, max_depth=4, library_items=4,
        axiom_instances_per_seed=4,
    ))
    kernel = ProofKernel(source)
    assert set(manifest["counts"]) == {"train", "validation", "test"}
    assert manifest["augmentation"]["library_uses"] >= 0
    assert manifest["augmentation"]["scratch_uses"] >= 0
    library = TheoremLibrary.load(tmp_path / "library.json", kernel)
    assert len(library) == manifest["library_count"]
    records = [record for split in ("train", "validation", "test")
               for record in load_episodes(tmp_path, split, kernel=kernel)]
    assert len(records) == sum(manifest["counts"].values())
    assert manifest["generation_kind"]["random_axiom_instance"] == 4
    assert manifest["generation_kind"]["composed_proof_dag"] > 0
    for record in records:
        cert = theorem_from_data(record["certificate"])
        assert len(cert.proof.source_labels) <= 256
        if record["generation_kind"] == "random_axiom_instance":
            assert not cert.hypotheses and not cert.variable_types
            assert record["proof_depth"] == 1
            assert len([label for label in cert.proof.source_labels
                        if label in kernel.database.logical_assertions]) == 1
    assert sum(record["augmentation"]["library_uses"] for record in records) == manifest["augmentation"]["library_uses"]
    assert manifest["library_reuse_score"] == "logical_rule_frequency_proxy_not_measured_theorem_reuse"


def test_warmup_budget_seed_quota_and_disable(tmp_path):
    from pa_prover_v2.data import DataConfig, generate_corpus, _random_axiom_instances
    source = ROOT / "formal" / "peano-pa-plus.mm"
    kernel = ProofKernel(source)
    assert list(_random_axiom_instances(kernel, 7, DataConfig(axiom_instances_per_seed=0))) == []
    with pytest.raises(ValueError, match="positive generation budgets"):
        generate_corpus(source, tmp_path / "zero", DataConfig(max_examples=0))
    with pytest.raises(ValueError, match="nonnegative integer"):
        DataConfig(axiom_instances_per_seed=-1).validate()
    manifest = generate_corpus(source, tmp_path / "three-seeds", DataConfig(
        seeds=(7, 11, 19), max_examples=9, steps_per_seed=1,
        validation_fraction=0, test_fraction=0, library_items=3,
    ))
    assert manifest["used_seeds"] == [7, 11, 19]
    assert manifest["seed_certificate_counts"] == {"7": 3, "11": 3, "19": 3}
    assert manifest["generation_kind"] == {"random_axiom_instance": 9, "composed_proof_dag": 0}
    assert sum(manifest["counts"].values()) == 9
    # A small label budget filters even verified warmup candidates before storage.
    tight = generate_corpus(source, tmp_path / "tight", DataConfig(
        seeds=(7,), max_examples=2, steps_per_seed=1, max_proof_labels=3,
        validation_fraction=0, test_fraction=0, library_items=2,
    ))
    from pa_prover_v2.data import load_episodes
    assert tight["filtered"].get("ValueError:proof label budget exceeded", 0) > 0
    assert all(len(theorem_from_data(r["certificate"]).proof.source_labels) <= 3
               for r in load_episodes(tmp_path / "tight", kernel=kernel))


def test_target_exclusion_ignores_unused_variables_and_auxiliary_constraints(kernel):
    target = ax1(kernel)
    expanded = copy.deepcopy(target)
    expanded.variable_types["unused_wff"] = "wff"
    expanded.floating += (Hypothesis("unused_float", Node("wff", (Node("unused_wff"),))),)
    expanded.proof_variable_types = {"private_wff": "wff"}
    expanded.floating += (Hypothesis("private_float", Node("wff", (Node("private_wff"),))),)
    expanded.proof_d_constraints = {("p", "private_wff")}
    kernel.verify(expanded)
    assert statement_fingerprint(expanded) == statement_fingerprint(target)
    library = TheoremLibrary(kernel, exclude_fingerprints={statement_fingerprint(target)})
    with pytest.raises(LibraryLeakageError):
        library.add(expanded)
    constrained = copy.deepcopy(target)
    constrained.d_constraints = {("p", "q")}
    assert statement_fingerprint(constrained) != statement_fingerprint(target)


def test_statement_identity_is_alpha_invariant_and_premise_order_independent():
    from pa_prover_v2.data import DataConfig, _assign_splits
    p, q, r, z = map(Node, ("p", "q", "r", "z"))
    first = Theorem("one", [Hypothesis("a", claim(imp(p, q))), Hypothesis("b", claim(imp(q, r)))],
                    claim(z), variable_types={v: "wff" for v in ("p", "q", "r", "z")})
    a, b, c, out = map(Node, ("zeta", "alpha", "beta", "goal"))
    second = Theorem("two", [Hypothesis("b", claim(imp(b, c))), Hypothesis("a", claim(imp(a, b))), Hypothesis("duplicate", claim(imp(a, b)))],
                     claim(out), variable_types={v: "wff" for v in ("zeta", "alpha", "beta", "goal", "unused")})
    assert statement_fingerprint(first) == statement_fingerprint(second)
    records = [{"statement_sha256": statement_fingerprint(first), "skeleton_sha256": "proof-a"},
               {"statement_sha256": statement_fingerprint(second), "skeleton_sha256": "proof-b"}]
    _assign_splits(records, DataConfig())
    assert records[0]["split_group"] == records[1]["split_group"]
    assert records[0]["split"] == records[1]["split"]


def test_mandatory_variable_and_dv_cannot_be_hidden_as_proof_only(kernel, tmp_path):
    theorem = ax1(kernel)
    theorem.proof_variable_types["p"] = theorem.variable_types.pop("p")
    with pytest.raises(KernelError, match="hidden as proof-only"):
        kernel.verify(theorem)
    scoped = scoped_kernel(tmp_path, distinct=True)
    target = context(distinct=True)
    certificate = scoped.compose(scoped.database.logical_assertions["r"],
                                 {"p": Node("x"), "q": Node("y")},
                                 [scoped.assumption(target, 0)], target)
    certificate.proof_d_constraints = certificate.d_constraints
    certificate.d_constraints = set()
    with pytest.raises(KernelError, match="hidden as proof-only"):
        scoped.verify(certificate)


def test_old_statement_identity_formats_are_rejected(kernel, tmp_path):
    from pa_prover_v2.data import load_episodes
    library = TheoremLibrary(kernel)
    library.add(ax1(kernel))
    library.save(tmp_path / "library.json")
    data = json.loads((tmp_path / "library.json").read_text())
    data["format"] = "pa-prover-v2-library-v1"
    (tmp_path / "library.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LibraryError, match="format"):
        TheoremLibrary.load(tmp_path / "library.json", kernel)
    (tmp_path / "manifest.json").write_text(json.dumps({"format_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="regenerate"):
        load_episodes(tmp_path, kernel=kernel)

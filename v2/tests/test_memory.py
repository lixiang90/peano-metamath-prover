from __future__ import annotations

from pathlib import Path

import pytest
import torch

from metamath_generator.model import Hypothesis, Node, Theorem
from pa_prover_v2.agent import AgentAction, AgentSession
from pa_prover_v2.inference import PolicyConfig, Rollout, action_distribution
from pa_prover_v2.kernel import ProofKernel
from pa_prover_v2.library import StaleIndexError, TheoremLibrary
from pa_prover_v2.memory import FrozenLemmaRetriever, incorporate_verified_rollouts
from pa_prover_v2.model import LemmaDecoderLM, ModelConfig
from pa_prover_v2.tokenizer import ByteTokenizer
from pa_prover_v2.training import encode_segments


ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def imp(left, right):
    return Node("implies", (left, right))


def claim(expression):
    return Node("|-", (expression,))


@pytest.fixture(scope="module")
def kernel():
    return ProofKernel(ROOT / "formal/peano-pa-plus.mm")


def proof(kernel, variant=0):
    p, q = Node("memory_p"), Node("memory_q")
    right = q if variant == 0 else imp(q, q) if variant == 1 else imp(q, p)
    target = Theorem("memory-target", [], claim(imp(p, imp(right, p))),
                     variable_types={"memory_p": "wff", "memory_q": "wff"})
    return kernel.compose(kernel.database.logical_assertions["ax-1"], {"phi": p, "psi": right}, [], target)


def model():
    torch.manual_seed(12)
    return LemmaDecoderLM(ModelConfig(d_model=16, n_heads=2, n_layers=1, lemma_layers=1,
                                      lemma_slots=2, max_seq_len=4096)).eval()


def library_and_session(kernel):
    library = TheoremLibrary(kernel)
    lemma_id = library.add(proof(kernel))
    p, q = Node("memory_p"), Node("memory_q")
    target = Theorem("new-target", [Hypothesis("h", claim(p))], claim(imp(q, p)),
                     variable_types={"memory_p": "wff", "memory_q": "wff"})
    return library, lemma_id, AgentSession(kernel, target)


def test_frozen_retriever_encodes_real_model_without_copying_main_decoder_or_rng(kernel):
    live = model()
    library = TheoremLibrary(kernel)
    key = library.add(proof(kernel))
    library.add(proof(kernel, 1))
    rng = torch.get_rng_state().clone()
    frozen = FrozenLemmaRetriever(library, live)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(frozen) == len(library) == 2
    assert not hasattr(frozen.encoder, "decoder")
    assert not hasattr(frozen.encoder, "lm_head")
    assert all(not parameter.requires_grad and parameter.device.type == "cpu"
               for parameter in frozen.encoder.parameters())
    texts = [record.statement for record in library.records]
    tokens = [ByteTokenizer().encode(text, bos=True, eos=True) for text in texts]
    with torch.no_grad():
        expected = live.encode_lemmas(tokens).mean(1)
    torch.testing.assert_close(frozen.encode(texts), expected, rtol=0, atol=0)
    assert frozen.search(next(record for record in library.records if record.id == key).statement, limit=1)[0].id == key


def test_library_mutation_requires_explicit_reindex_with_same_frozen_encoder(kernel):
    library = TheoremLibrary(kernel)
    library.add(proof(kernel))
    frozen = FrozenLemmaRetriever(library, model())
    fingerprint = frozen.encoder_fingerprint
    added = library.add(proof(kernel, 1), step=1)
    with pytest.raises(StaleIndexError):
        frozen.search("implies")
    frozen.refresh()
    assert frozen.encoder_fingerprint == fingerprint
    assert library.indexed_version == library.version
    assert added in {record.id for record in frozen.search("implies", limit=10)}


def test_live_memory_candidate_gradients_and_post_training_index_refresh(kernel):
    live = model()
    library, key, initial = library_and_session(kernel)
    frozen = FrozenLemmaRetriever(library, live)
    session = AgentSession(kernel, initial.target, frozen)
    assert session.step(AgentAction("READ", {"lemma_id": key})).status == "running"
    p, q = Node("memory_p"), Node("memory_q")
    actions = [AgentAction("USE", {"lemma_id": key, "premises": [], "substitution": mapping})
               for mapping in ({"memory_p": p, "memory_q": q}, {"memory_p": q, "memory_q": p})]
    for action in actions:
        # Verify both branches on independent sessions before treating them as candidates.
        probe = AgentSession(kernel, initial.target, frozen)
        probe.step(AgentAction("READ", {"lemma_id": key}))
        assert probe.step(action).status == "running"
    samples = [encode_segments(session.context_segments(), ByteTokenizer(), 2, action, action_only=True)
               for action in actions]
    assert all(sample.memories for sample in samples)
    old_fingerprint = frozen.encoder_fingerprint
    statement = next(record for record in library.records if record.id == key).statement
    old_vectors = frozen.encode([statement]).clone()
    live_before = live.lemma_decoder.layers[0].attention.qkv.weight.detach().clone()
    optimizer = torch.optim.AdamW(live.parameters(), lr=0.01, weight_decay=0)
    probabilities = action_distribution(live, samples, PolicyConfig(exploration=0.1, score_batch_size=1))
    assert probabilities.isfinite().all() and probabilities.sum().item() == pytest.approx(1)
    (-probabilities[0].log()).backward()
    for parameter in (live.summary_slots, live.lemma_embedding.weight,
                      live.lemma_decoder.layers[0].attention.qkv.weight):
        assert parameter.grad is not None and parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(live_before, live.lemma_decoder.layers[0].attention.qkv.weight)
    torch.testing.assert_close(old_vectors, frozen.encode([statement]), rtol=0, atol=0)
    frozen.refresh(live)
    assert frozen.encoder_fingerprint != old_fingerprint
    assert library._index["encoder_fingerprint"] == frozen.encoder_fingerprint
    assert not torch.equal(old_vectors, frozen.encode([statement]))
    assert frozen.search(statement, 1)[0].id == key


def completed_rollout(kernel, library, key, target):
    session = AgentSession(kernel, target, library)
    p, q = Node("memory_p"), Node("memory_q")
    actions = [
        AgentAction("READ", {"lemma_id": key}),
        AgentAction("USE", {"lemma_id": key, "substitution": {"memory_p": p, "memory_q": q}, "premises": []}),
        AgentAction("APPLY", {"rule": "ax-mp", "substitution": {"phi": p, "psi": imp(q, p)}, "premises": ["h0", "f0"]}),
        AgentAction("FINISH", {"fact_id": "f1"}),
    ]
    for action in actions:
        assert session.step(action).status in {"running", "success"}
    assert session.status == "success"
    kernel.verify(session.certificate)
    return Rollout(status="success", reward=1, session=session)


def test_verified_online_accumulation_invalidates_then_refreshes_index(kernel):
    library, key, initial = library_and_session(kernel)
    live = model()
    frozen = FrozenLemmaRetriever(library, live)
    result = completed_rollout(kernel, frozen, key, initial.target)
    report = incorporate_verified_rollouts(frozen, [result], step=2)
    assert report["added"] == 1 and report["library_count"] == 2
    assert next(record for record in library.records if record.id == key).reuse_count == 1
    with pytest.raises(StaleIndexError):
        frozen.search("implies")
    frozen.refresh(live)
    assert len(frozen.search("implies", 8)) == 2
    assert all(record.provenance["split"] == "train" for record in library.records)


def test_forged_success_and_invalid_use_do_not_promote_or_increment_reuse(kernel):
    library, key, initial = library_and_session(kernel)
    session = AgentSession(kernel, initial.target, library)
    assert session.step({"malformed": "action"}).status == "invalid"
    invalid = AgentAction("USE", {"lemma_id": key, "substitution": {}, "premises": []})
    assert session.step(invalid).status == "invalid"
    session.certificate = proof(kernel, 2)  # Valid proof, wrong episode goal.
    forged = Rollout(status="success", reward=1, session=session)
    report = incorporate_verified_rollouts(library, [forged], step=2)
    assert report["added"] == 0 and len(library) == 1
    assert next(record for record in library.records if record.id == key).reuse_count == 0


def test_too_large_or_excluded_new_proof_does_not_abort_online_training(kernel):
    library, key, initial = library_and_session(kernel)
    result = completed_rollout(kernel, library, key, initial.target)
    library.max_bytes = library.bytes_used
    report = incorporate_verified_rollouts(library, [result], step=1)
    assert report["skipped_budget"] == 1
    library, key, initial = library_and_session(kernel)
    result = completed_rollout(kernel, library, key, initial.target)
    from pa_prover_v2.kernel import statement_fingerprint
    library.exclude_fingerprints = frozenset({statement_fingerprint(result.session.certificate)})
    report = incorporate_verified_rollouts(library, [result], step=1)
    assert report["skipped_visibility"] == 1 and len(library) == 1

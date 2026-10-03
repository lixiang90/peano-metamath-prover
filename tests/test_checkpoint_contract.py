"""Inference must reject equal-sized vocabularies with different meanings."""

from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from metamath_generator.model import Node
from metamath_generator.parser import parse
from neural_prover.benchmark import BenchmarkCase
from neural_prover.cli import main
from neural_prover.data_contract import (
    tokenizer_fingerprint,
    validate_checkpoint_tokenizer,
)
from neural_prover.evaluate import (
    EvaluationConfig,
    MCTSEvaluationConfig,
    evaluate_checkpoint,
    evaluate_mcts_checkpoint,
)
from neural_prover.latent_model import LatentProofTransformer, LatentReasoningConfig
from neural_prover.model import ProofTransformer, ProofTransformerConfig
from neural_prover.scale_train import (
    ScaleTrainingConfig,
    evaluate_scale_checkpoint,
    train_scale_model,
)
from neural_prover.tokenizer import MetamathTokenizer


DATABASE = Path(__file__).resolve().parents[1] / "formal" / "peano.mm"


@pytest.fixture
def corpus(tmp_path):
    database = parse(DATABASE)
    tokenizer = MetamathTokenizer.from_database(database)
    tokenizer.save(tmp_path / "tokenizer.json")
    # Real, well-typed ax-1 application, so the tiny training fixture is also
    # usable if corpus validation is tightened in the future.
    theorem = database.logical_assertions["ax-1"]
    variables = tokenizer.canonical_variables(theorem)
    state = tokenizer.encode(tokenizer.state_tokens(theorem, variables))
    action = tokenizer.encode(tokenizer.assertion_tactic_tokens(
        theorem, {name: Node(name) for name in theorem.variables()}, variables,
    ))
    shards = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "id": split, "state": state, "action": action, "value": 1.0,
            }) + "\n")
        shards[split] = [{"path": path.name, "records": 1}]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "configuration": {"max_state_tokens": 128, "max_action_tokens": 128},
        "shards": shards,
    }), encoding="utf-8")
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    case = BenchmarkCase(
        "reflexive", "reflexive", "synthetic", "easy", "provable", (),
        "|- = x x", (), (("x", "var"),),
    )
    (tmp_path / "benchmark.json").write_text(json.dumps({
        "format": "peano-proof-benchmark-v2", "cases": [case.to_record()],
    }), encoding="utf-8")
    return tmp_path, tokenizer


def _model_config(tokenizer):
    return ProofTransformerConfig(
        len(tokenizer), tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id,
        d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, max_state_tokens=128, max_action_tokens=128,
    )


def _swap_ordinary_tokens(tokenizer, path):
    tokens = list(tokenizer.tokens)
    a, b = tokens.index("="), tokens.index("implies")
    tokens[a], tokens[b] = tokens[b], tokens[a]
    changed = MetamathTokenizer(
        tokens, config=tokenizer.config, preserve_token_order=True,
        pa_plus_context=tokenizer.pa_plus_context,
    )
    assert len(changed) == len(tokenizer)
    assert changed.bos_id == tokenizer.bos_id
    changed.save(path)
    return changed


@pytest.mark.parametrize("entry", [
    "supervised", "mcts", "scale", "prove", "prove-decomposed", "replay",
])
def test_inference_rejects_reordered_token_ids(corpus, entry):
    root, tokenizer = corpus
    config = _model_config(tokenizer)
    model = (
        LatentProofTransformer(config, LatentReasoningConfig(max_thought_steps=1))
        if entry == "prove-decomposed" else ProofTransformer(config)
    )
    checkpoint = root / "model.pt"
    model.save_checkpoint(checkpoint, metadata={
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
    })
    _swap_ordinary_tokens(tokenizer, root / "tokenizer.json")
    benchmark, report = root / "benchmark.json", root / "report.json"
    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        if entry == "supervised":
            evaluate_checkpoint(
                checkpoint, root, benchmark, DATABASE, report,
                EvaluationConfig(device="cpu"),
            )
        elif entry == "mcts":
            evaluate_mcts_checkpoint(
                checkpoint, root, benchmark, DATABASE, report,
                MCTSEvaluationConfig(device="cpu"),
            )
        elif entry == "scale":
            evaluate_scale_checkpoint(checkpoint, root, report, device_name="cpu")
        elif entry == "replay":
            from neural_prover.rl import ReplayCollectionConfig, collect_replay_from_corpus
            collect_replay_from_corpus(
                checkpoint, root, DATABASE, report,
                ReplayCollectionConfig(device="cpu", examples=1),
            )
        else:
            tokenizer_argument = root / "tokenizer.json" if entry == "prove-decomposed" else root
            main([
                entry, str(checkpoint), str(tokenizer_argument), str(benchmark),
                "reflexive", str(DATABASE), "--device", "cpu",
            ])
    assert not report.exists()


def _training_config():
    return ScaleTrainingConfig(
        max_steps=2, run_steps=1, micro_batch_size=1,
        gradient_accumulation_steps=1, device="cpu", d_model=16, nhead=2,
        encoder_layers=1, decoder_layers=1, dim_feedforward=32, dropout=0.0,
        gradient_checkpointing=False, initial_context_tokens=128,
        context_warmup_steps=1, shuffle_buffer=1, checkpoint_every=1,
        validation_batches=0, require_long_context_step=False,
        enforce_scale_parameter_range=False,
    )


def test_scale_saves_fingerprint_and_rejects_wrong_resume(corpus):
    root, tokenizer = corpus
    output = root / "trained"
    config = _training_config()
    train_scale_model(root, output, config)
    latest = output / "latest.pt"
    model, payload = ProofTransformer.load_checkpoint(latest)
    assert payload["metadata"]["tokenizer_sha256"] == tokenizer_fingerprint(tokenizer)
    validate_checkpoint_tokenizer(model, payload, tokenizer)
    # A new checkpoint may retain historical migration metadata from an
    # earlier fine-tuning run; its current encoder and optimizer are complete.
    payload["metadata"]["candidate_policy_migration"] = {
        "optimizer_restart_required": True,
    }
    torch.save(payload, latest)
    _swap_ordinary_tokens(tokenizer, root / "tokenizer.json")
    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        train_scale_model(root, output, config, resume_from=latest)
    tokenizer.save(root / "tokenizer.json")
    train_scale_model(root, output, replace(config, run_steps=0), resume_from=latest)
    _, final = ProofTransformer.load_checkpoint(output / "final.pt")
    assert final["metadata"]["tokenizer_sha256"] == tokenizer_fingerprint(tokenizer)


def test_legacy_encoder_does_not_silently_reset_exact_resume(corpus):
    root, tokenizer = corpus
    checkpoint = root / "legacy.pt"
    model = ProofTransformer(_model_config(tokenizer))
    model.save_checkpoint(checkpoint, metadata={
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
    })
    payload = torch.load(checkpoint, weights_only=False)
    payload["metadata"]["candidate_policy"].pop("encoder")
    payload["model_state"] = {
        name: value for name, value in payload["model_state"].items()
        if not name.startswith(("candidate_encoder.", "candidate_projection."))
    }
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fresh optimizer.*exact scale resume"):
        train_scale_model(root, root / "resumed", _training_config(), resume_from=checkpoint)


@pytest.mark.parametrize("name", ["pad_id", "bos_id", "eos_id"])
def test_legacy_special_token_mismatch_is_rejected(corpus, name):
    _, tokenizer = corpus
    model = ProofTransformer(replace(_model_config(tokenizer), **{
        name: tokenizer.token_to_id["="],
    }))
    with pytest.raises(ValueError, match=f"{name} mismatch"):
        validate_checkpoint_tokenizer(model, {}, tokenizer)

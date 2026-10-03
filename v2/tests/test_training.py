from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import torch

from metamath_generator.model import Hypothesis, Node, Theorem
from pa_prover_v2.agent import AgentAction, AgentSession
from pa_prover_v2.kernel import ProofKernel, theorem_to_data
from pa_prover_v2.model import LemmaDecoderLM, ModelConfig, load_checkpoint
from pa_prover_v2.tokenizer import ByteTokenizer
from pa_prover_v2.training import (
    TrainConfig, action_text, build_training_samples,
    collate, encode_segments, ntp_blocks, train_language_model,
)


ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def real_episode():
    """A real PA+ theorem with a failed read, certified local memory and reuse."""
    kernel = ProofKernel(ROOT / "formal/peano-pa-plus.mm")
    p, q = Node("train_phi"), Node("train_psi")
    implication = Node("implies", (q, p))
    target = Theorem("training_target", [Hypothesis("h", Node("|-", (p,)))],
                     Node("|-", (implication,)), variable_types={"train_phi": "wff", "train_psi": "wff"})
    lemma = kernel.compose(kernel.database.logical_assertions["ax-1"],
                           {"phi": p, "psi": q}, [], target)
    certificate = kernel.compose(kernel.database.logical_assertions["ax-mp"],
                                 {"phi": p, "psi": implication},
                                 [kernel.assumption(target, 0), lemma], target)
    kernel.verify(certificate)
    actions = [
        AgentAction("READ", {"name": "missing_training_probe"}),
        AgentAction("APPLY", {"rule": "ax-1", "substitution": {"phi": p, "psi": q}, "premises": []}),
        AgentAction("SAVE", {"fact_id": "f0", "name": "cached"}),
        AgentAction("READ", {"name": "cached"}),
        AgentAction("USE", {"name": "cached", "substitution": {"train_phi": p, "train_psi": q}, "premises": ["h0"]}),
        AgentAction("APPLY", {"rule": "ax-mp", "substitution": {"phi": p, "psi": implication}, "premises": ["h0", "f1"]}),
        AgentAction("FINISH", {"fact_id": "f2"}),
    ]
    session = AgentSession(kernel, target)
    observations = [session.step(action) for action in actions]
    assert observations[0].status == "invalid"
    assert all(observation.status in {"running", "success"} for observation in observations[1:])
    assert session.status == "success"
    kernel.verify(session.certificate)
    # Independent reference executable when bundled locally; no mock verifier.
    executable = ROOT / "outputs/tools/metamath/metamath.exe"
    if executable.exists():
        result = kernel.certify(session.certificate, external_verifier=executable)
        assert result["external_verified"], result
    episode = {"id": "real-training-episode", "split": "train",
               "certificate": theorem_to_data(certificate),
               "actions": [action.to_dict() for action in actions]}
    return kernel, episode, session, actions


def model_config(max_seq_len=512, slots=2):
    return ModelConfig(d_model=16, n_heads=2, n_layers=1, lemma_layers=1,
                       lemma_slots=slots, max_seq_len=max_seq_len, ffn_multiplier=2)


@pytest.mark.parametrize("slots", [1, 2, 3, 5])
def test_ntp_blocks_keep_memory_whole_and_cover_each_language_target_once(slots):
    tokenizer = ByteTokenizer()
    for block_size in (slots + 2, slots * 2 + 2, slots * 3 + 1):
        for contiguous in (1, 3, 8):
            segments = ["abc", *[{"text": f"lemma {i}"} for i in range(contiguous)],
                        "defgh", {"text": "later"}, "ijklm"]
            sample = encode_segments(segments, tokenizer, slots)
            original_labels = list(sample.labels)
            cap = len(sample.ids) * 2
            blocks = list(itertools.islice(ntp_blocks(sample, block_size, slots), cap))
            assert len(blocks) < cap, "NTP chunker failed to advance over adjacent memory blocks"
            expected = [value for value in sample.labels[1:] if value >= 0]
            actual = [value for block in blocks for value in block.labels[1:] if value >= 0]
            assert actual == expected
            assert sample.labels == original_labels
            for block in blocks:
                assert len(block.ids) <= block_size
                assert block.labels[0] == -100
                covered = set()
                for start in block.memories:
                    assert block.ids[start:start + slots] == [tokenizer.mem_id] * slots
                    covered.update(range(start, start + slots))
                assert covered == {i for i, value in enumerate(block.ids) if value == tokenizer.mem_id}


def test_sft_targets_only_successful_actions_and_keeps_failure_in_context(real_episode):
    kernel, episode, _, actions = real_episode
    samples, stats = build_training_samples([episode], kernel, None, model_config(8192), "sft")
    tokenizer = ByteTokenizer()
    assert len(samples) == len(actions) - 1
    assert stats["invalid_actions_in_context"] == 1
    for sample, action in zip(samples, actions[1:]):
        supervised = tokenizer.decode([value for value in sample.labels if value >= 0])
        assert supervised == action_text(action) + "\n"
        context = tokenizer.decode(value for value, label in zip(sample.ids, sample.labels) if label == -100)
        assert "missing_training_probe" in context
        assert '"status": "invalid"' in context
        assert "missing_training_probe" not in supervised
    assert stats["memory_samples"] >= 1
    assert stats["trained_language_targets"] == stats["source_language_targets"]
    assert stats["filtered_language_targets"] == 0


def test_real_ntp_sequence_is_chronological_and_counts_all_targets(real_episode):
    kernel, episode, session, _ = real_episode
    tokenizer = ByteTokenizer()
    segments = session.context_segments()
    header = json.loads(segments[0])
    assert header["episode_start"] is True
    assert set(header["facts"]) == {"h0"}
    assert header["resources"]["steps"] == 0
    assert json.loads(segments[-1])["current_workspace"]["status"] == "success"
    full = encode_segments(segments, tokenizer, 2)
    samples, stats = build_training_samples([episode], kernel, None, model_config(), "ntp")
    expected = [value for value in full.labels[1:] if value >= 0]
    actual = [value for sample in samples for value in sample.labels[1:] if value >= 0]
    assert actual == expected
    assert stats["source_language_targets"] == stats["trained_language_targets"] == len(expected)
    assert stats["memory_samples"] >= 1


def test_ntp_updates_main_and_lemma_parameters_and_saves_checkpoint(real_episode, tmp_path):
    kernel, episode, _, _ = real_episode
    cfg = model_config()
    samples, stats = build_training_samples([episode], kernel, None, cfg, "ntp")
    memory_samples = [sample for sample in samples if sample.memories]
    assert memory_samples
    torch.manual_seed(5)
    model = LemmaDecoderLM(cfg)
    main_before = model.decoder.layers[0].attention.qkv.weight.detach().clone()
    lemma_before = model.lemma_decoder.layers[0].attention.qkv.weight.detach().clone()
    slots_before = model.summary_slots.detach().clone()
    summary = train_language_model(
        model, memory_samples, tmp_path,
        TrainConfig(steps=2, batch_size=1, learning_rate=1e-3, weight_decay=0),
        mode="ntp", metadata={"samples": stats},
    )
    assert summary["lemma_gradient_steps"] == 2
    assert not torch.equal(main_before, model.decoder.layers[0].attention.qkv.weight)
    assert not torch.equal(lemma_before, model.lemma_decoder.layers[0].attention.qkv.weight)
    assert not torch.equal(slots_before, model.summary_slots)
    restored, metadata = load_checkpoint(tmp_path / "model.pt")
    assert metadata["mode"] == "ntp" and metadata["steps"] == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(record["tokens"] > 0 for record in records)
    assert (tmp_path / "optimizer.pt").exists()


def test_collated_ntp_loss_matches_single_standard_shift_ce():
    tokenizer = ByteTokenizer()
    samples = [encode_segments(["abc"], tokenizer, 2), encode_segments(["中文"], tokenizer, 2)]
    batch = collate(samples)
    model = LemmaDecoderLM(model_config()).eval()
    output = model(**batch)
    manual = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, 260), batch["labels"][:, 1:].reshape(-1), ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, manual)
    assert (batch["labels"][~batch["attention_mask"]] == -100).all()


def test_long_sample_filtering_reports_lost_language_targets(real_episode):
    kernel, episode, _, _ = real_episode
    _, stats = build_training_samples([episode], kernel, None, model_config(128), "ntp")
    assert stats["filtered_long_lemma"] > 0
    assert stats["memory_samples"] == 0
    assert stats["filtered_language_targets"] > 0
    assert stats["source_language_targets"] == stats["trained_language_targets"] + stats["filtered_language_targets"]
    full, _ = build_training_samples([episode], kernel, None, model_config(8192), "sft")
    samples, stats = build_training_samples([episode], kernel, None, model_config(len(full[0].ids)), "sft")
    assert len(samples) < len(full)
    assert stats["filtered_long_context"] == len(full) - len(samples)
    assert stats["source_language_targets"] == stats["trained_language_targets"] + stats["filtered_language_targets"]


def test_unverified_traces_and_post_terminal_actions_are_rejected(real_episode):
    kernel, episode, _, _ = real_episode
    unfinished = {**episode, "actions": episode["actions"][:-1]}
    with pytest.raises(ValueError, match="unverified teacher"):
        build_training_samples([unfinished], kernel, None, model_config(8192), "sft")
    trailing = {**episode, "actions": [*episode["actions"], {"op": "READ", "args": {"name": "after-finish"}}]}
    with pytest.raises(ValueError, match="terminal"):
        build_training_samples([trailing], kernel, None, model_config(8192), "sft")
    with pytest.raises(ValueError, match="train episodes"):
        build_training_samples([{**episode, "split": "test"}], kernel, None, model_config(), "ntp")


@pytest.mark.parametrize("kwargs", [{"grad_clip": -1}, {"learning_rate": float("nan")},
                                    {"weight_decay": -1}, {"steps": 0}])
def test_training_hyperparameters_cannot_reverse_or_corrupt_updates(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)

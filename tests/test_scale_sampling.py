from __future__ import annotations

import gzip
import json
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

pytest.importorskip("torch")

from neural_prover import scale_train
from neural_prover.scale_train import (
    ScaleTrainingConfig,
    _ResumableRecordStream,
    _next_batch,
    _shuffled_records,
    train_scale_model,
)
from neural_prover.tokenizer import MetamathTokenizer


def _write_corpus(
    root: Path,
    *,
    train: list[int],
    validation: list[int] | None = None,
    maximum: int = 16,
) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    MetamathTokenizer([]).save(corpus / "tokenizer.json")
    shards = {}
    for split, lengths in (
        ("train", train),
        ("validation", [3] if validation is None else validation),
        ("test", [3]),
    ):
        path = corpus / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for index, length in enumerate(lengths):
                record = {
                    "id": f"{split}-{index}",
                    "state": [index + 1] * length,
                    "action": [2, 3],
                    "value": 1.0,
                }
                stream.write(json.dumps(record) + "\n")
        # Deliberately incorrect metadata must not conceal empty files or
        # missing length ranges. The sampler must inspect the actual records.
        shards[split] = [{"path": path.name, "records": 999}]
    (corpus / "manifest.json").write_text(
        json.dumps({
            "configuration": {
                "max_state_tokens": maximum,
                "max_action_tokens": maximum,
            },
            "shards": shards,
        }),
        encoding="utf-8",
    )
    return corpus


@pytest.mark.parametrize("empty_shard_list", [False, True])
def test_empty_split_fails_instead_of_repeating_empty_epochs(
    tmp_path: Path, empty_shard_list: bool,
) -> None:
    corpus = _write_corpus(tmp_path, train=[])
    if empty_shard_list:
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shards"]["train"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split 'train' is empty"):
        next(_shuffled_records(corpus, "train", 7, 3))
    with pytest.raises(ValueError, match="split 'train' is empty"):
        _next_batch(_ResumableRecordStream(corpus, "train", 7, 3), 1, 0, 16)


@pytest.mark.parametrize("minimum, maximum", [(0, 2), (9, 16), (5, 6)])
def test_missing_token_range_fails_before_consuming_the_stream(
    tmp_path: Path, minimum: int, maximum: int,
) -> None:
    corpus = _write_corpus(tmp_path, train=[3, 8])
    records = _ResumableRecordStream(corpus, "train", 7, 3)
    with pytest.raises(ValueError, match="split 'train' has no records"):
        _next_batch(records, 1, 0, maximum, minimum_tokens=minimum)
    assert records.consumed == 0


def test_preflight_uses_one_early_stopping_scan_and_caches_witnesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(tmp_path, train=[3, 12, 8, 5])
    records = _ResumableRecordStream(corpus, "train", 7, 3)
    actual_reader = scale_train._records_from_shards
    scans = []
    visited = []

    def counted_reader(corpus_path, shards):
        scans.append(tuple(shard["path"] for shard in shards))
        for record in actual_reader(corpus_path, shards):
            visited.append(record["id"])
            yield record

    monkeypatch.setattr(scale_train, "_records_from_shards", counted_reader)
    records.require_token_ranges([(0, 4), (9, 16)])
    for limit in range(4, 17):
        records.require_token_ranges([(0, limit)])
    records.require_token_ranges([(9, 16)])
    assert len(scans) == 1
    assert visited == ["train-0", "train-1"]
    assert records.consumed == 0


def test_eligibility_checks_preserve_shuffle_order_and_resume_offset(
    tmp_path: Path,
) -> None:
    corpus = _write_corpus(tmp_path, train=[3, 12, 8, 5])
    expected = list(islice(_shuffled_records(corpus, "train", 7, 3), 20))
    checked = _ResumableRecordStream(corpus, "train", 7, 3)
    checked.require_token_ranges([(0, 4), (9, 16)])
    assert list(islice(checked, 20)) == expected
    assert checked.consumed == 20
    resumed = _ResumableRecordStream(corpus, "train", 7, 3, consumed=9)
    resumed.require_token_ranges([(0, 4), (9, 16)])
    assert list(islice(resumed, 11)) == expected[9:]
    assert resumed.consumed == 20


def test_batch_can_repeat_an_eligible_record_across_epochs(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path, train=[3, 12])
    records = _ResumableRecordStream(corpus, "train", 7, 3)
    batch = _next_batch(records, 4, 0, 12, minimum_tokens=9)
    assert tuple(batch["state"].shape) == (4, 12)
    assert batch["state"].eq(2).all()


def _preflight_config() -> ScaleTrainingConfig:
    return ScaleTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        initial_context_tokens=4,
        context_warmup_steps=2,
        long_context_threshold=8,
        validation_batches=1,
        device="cpu",
        enforce_scale_parameter_range=False,
    )


@pytest.mark.parametrize(
    "train, validation, message",
    [
        ([], [3], "split 'train' is empty"),
        ([6, 12], [3], "split 'train' has no records.*\\[0, 4\\]"),
        ([3, 8], [3], "split 'train' has no records.*\\[9, 16\\]"),
        ([3, 12], [], "split 'validation' is empty"),
        ([3, 12], [17], "split 'validation' has no records"),
    ],
)
def test_training_rejects_impossible_ranges_before_building_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train: list[int],
    validation: list[int],
    message: str,
) -> None:
    corpus = _write_corpus(tmp_path, train=train, validation=validation)

    def unexpected_model(*args, **kwargs):
        pytest.fail("training allocated a model before validating its corpus")

    monkeypatch.setattr(scale_train, "ProofTransformer", unexpected_model)
    with pytest.raises(ValueError, match=message):
        train_scale_model(corpus, tmp_path / "output", _preflight_config())


@pytest.mark.parametrize("long_only", [False, True])
def test_preflight_checks_only_ranges_that_training_will_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, long_only: bool,
) -> None:
    corpus = _write_corpus(tmp_path, train=[12] if long_only else [3])
    config = replace(
        _preflight_config(),
        max_steps=1,
        require_long_context_step=long_only,
        # No long-context requirement means a large threshold is harmless.
        long_context_threshold=8 if long_only else 2048,
    )

    class PreflightPassed(Exception):
        pass

    def checked_model(*args, **kwargs):
        raise PreflightPassed

    monkeypatch.setattr(scale_train, "ProofTransformer", checked_model)
    with pytest.raises(PreflightPassed):
        train_scale_model(corpus, tmp_path / "output", config)


def test_finite_record_stream_reports_an_incomplete_batch() -> None:
    with pytest.raises(ValueError, match="record stream ended"):
        _next_batch(iter([]), 1, 0, 16)

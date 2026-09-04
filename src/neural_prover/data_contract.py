"""Serialized training data contracts shared by the MCTS and HTPS paths."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .tokenizer import MetamathTokenizer


def tokenizer_fingerprint(tokenizer: MetamathTokenizer) -> str:
    payload = {
        "tokens": tokenizer.tokens,
        "configuration": asdict(tokenizer.config),
        "pa_plus_context": asdict(tokenizer.pa_plus_context),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_record_encoding(record: dict, tokenizer: MetamathTokenizer, *, lemma=False) -> None:
    """Do not silently train on IDs produced by a different tokenizer."""
    if tokenizer.pa_plus_context.enabled and tokenizer.pa_plus_context.bridge_variable_order != "sorted-v1":
        raise ValueError("legacy PA+ bridge ordering: regenerate and audit the corpus")
    expected = record.get("tokenizer_sha256")
    if expected is not None and expected != tokenizer_fingerprint(tokenizer):
        raise ValueError("record tokenizer fingerprint mismatch")
    for prefix in ("state", "lemma_action" if lemma else "action"):
        tokens = record[f"{prefix}_tokens"]
        ids = record[f"{prefix}_ids"]
        if not tokens or tokenizer.encode(tokens) != list(ids):
            raise ValueError(f"{prefix} token/ID mismatch")


def validate_checkpoint_tokenizer(model, payload: dict, tokenizer: MetamathTokenizer) -> None:
    expected = payload.get("metadata", {}).get("tokenizer_sha256")
    if expected is not None and expected != tokenizer_fingerprint(tokenizer):
        raise ValueError("checkpoint tokenizer fingerprint mismatch")
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    for name in ("pad_id", "bos_id", "eos_id"):
        if getattr(model.config, name) != getattr(tokenizer, name):
            raise ValueError(f"checkpoint tokenizer {name} mismatch")

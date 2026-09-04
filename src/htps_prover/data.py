"""Replay audits for forward-DAG policy and guarded lemma supervision."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from metamath_generator.parser import parse
from neural_prover.data_contract import tokenizer_fingerprint, validate_record_encoding
from neural_prover.environment import ProofState, parse_tactic_tokens
from neural_prover.lemma import LemmaBackwardEnvironment
from neural_prover.tokenizer import MetamathTokenizer


def replay_record(record, tokenizer, database, environment, *, lemma=False):
    validate_record_encoding(record, tokenizer, lemma=lemma)
    theorem = tokenizer.theorem_from_state_tokens(record["state_tokens"], database)
    action = record["lemma_action_tokens" if lemma else "action_tokens"]
    tactic = parse_tactic_tokens(action, theorem, tokenizer, database, environment=environment)
    return environment.apply(ProofState.from_theorem(theorem), tactic)


def audit_forward_dataset(database_path, directory, destination=None):
    """Audit all emitted actions; this is not full-proof or external certification."""
    directory = Path(directory)
    database = parse(database_path)
    tokenizer = MetamathTokenizer.load(directory / "tokenizer.json")
    environment = LemmaBackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    counts = Counter()
    kinds = Counter()
    failures = []
    skeleton_splits = {}
    state_splits = {}
    for split in ("train", "validation", "test"):
        for kind in ("policy", "lemma"):
            for index, line in enumerate((directory / f"{kind}_{split}.jsonl").read_text(encoding="utf-8").splitlines()):
                record = json.loads(line)
                counts[f"{kind}_{split}"] += 1
                try:
                    replay_record(record, tokenizer, database, environment, lemma=kind == "lemma")
                    skeleton = record.get("proof_skeleton_sha256")
                    if skeleton:
                        previous = skeleton_splits.setdefault(skeleton, split)
                        if previous != split:
                            raise ValueError("proof skeleton crosses splits")
                    state = tuple(record["state_tokens"])
                    previous = state_splits.setdefault(state, split)
                    if previous != split:
                        raise ValueError("serialized proof state crosses splits")
                    kinds[record.get("generation_kind", "search")] += 1
                except Exception as exc:
                    failures.append({"split": split, "kind": kind, "index": index, "error": str(exc)})
    report = {
        "format": "peano-htps-action-audit-v1", "counts": dict(counts),
        "valid_actions": sum(counts.values()) - len(failures),
        "invalid_actions": len(failures), "failures": failures,
        "generation_kinds": dict(kinds), "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "environment": environment.configuration_record(),
        "scope": "serialized action replay, not complete proof or external verification",
    }
    if destination is not None:
        Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

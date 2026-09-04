from __future__ import annotations

import json
import random
from pathlib import Path

from metamath_generator.parser import parse

from .data import load_examples
from .environment import (
    BackwardEnvironment,
    ProofState,
    parse_tactic_tokens,
)
from .tokenizer import MetamathTokenizer


def audit_corpus_actions(
    database_path: str | Path,
    corpus_directory: str | Path,
    destination: str | Path | None = None,
) -> dict:
    """Replay every serialized supervised action through the typed kernel."""

    database = parse(database_path)
    corpus = Path(corpus_directory)
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    environment = BackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    counts: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    for split in ("train", "validation", "test"):
        valid = 0
        examples = load_examples(corpus / f"{split}.jsonl")
        for example in examples:
            try:
                theorem = tokenizer.theorem_from_state_tokens(
                    example.state_tokens,
                    database,
                    name=f"audit_{example.example_id}",
                )
                tactic = parse_tactic_tokens(
                    example.action_tokens,
                    theorem,
                    tokenizer,
                    database,
                    environment=environment,
                )
                environment.apply(
                    ProofState.from_theorem(theorem),
                    tactic,
                )
                valid += 1
            except Exception as exc:
                failures.append({
                    "split": split,
                    "example_id": example.example_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
        counts[split] = {
            "examples": len(examples),
            "valid_actions": valid,
            "invalid_actions": len(examples) - valid,
        }
    report = {
        "format": "peano-corpus-action-audit-v1",
        "database": str(Path(database_path).resolve()),
        "corpus": str(corpus.resolve()),
        "counts": counts,
        "valid_actions": sum(
            item["valid_actions"] for item in counts.values()
        ),
        "invalid_actions": len(failures),
        "failures": failures,
        "environment": environment.configuration_record(),
    }
    if destination is not None:
        Path(destination).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def audit_scale_corpus(
    database_path: str | Path,
    corpus_directory: str | Path,
    destination: str | Path | None = None,
    *,
    sample_size: int = 10_000,
    seed: int = 20260729,
) -> dict:
    """Replay an exact deterministic sample from a sharded scale corpus."""

    from .scale_data import iter_scale_records

    database = parse(database_path)
    corpus = Path(corpus_directory)
    manifest = json.loads(
        (corpus / "manifest.json").read_text(encoding="utf-8")
    )
    tokenizer = MetamathTokenizer.load(corpus / "tokenizer.json")
    environment = BackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    split_counts = manifest["counts"]
    total = sum(int(value) for value in split_counts.values())
    wanted = min(max(0, sample_size), total)
    selected = set(random.Random(seed).sample(range(total), wanted))
    failures: list[dict] = []
    audited = 0
    valid = 0
    duplicates = 0
    seen_ids: set[str] = set()
    maximum_state = 0
    maximum_action = 0
    global_index = 0
    audited_by_split: dict[str, int] = {}
    audited_by_depth: dict[str, int] = {}
    valid_by_depth: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_audited = 0
        for record in iter_scale_records(corpus, split):
            if global_index in selected:
                audited += 1
                split_audited += 1
                record_id = str(record["id"])
                proof_depth = str(int(record.get("proof_depth", 0)))
                audited_by_depth[proof_depth] = (
                    audited_by_depth.get(proof_depth, 0) + 1
                )
                if record_id in seen_ids:
                    duplicates += 1
                seen_ids.add(record_id)
                state_ids = record["state"]
                action_ids = record["action"]
                maximum_state = max(maximum_state, len(state_ids))
                maximum_action = max(maximum_action, len(action_ids))
                try:
                    state_tokens = tokenizer.decode(state_ids)
                    action_tokens = tokenizer.decode(action_ids)
                    theorem = tokenizer.theorem_from_state_tokens(
                        state_tokens,
                        database,
                        name=f"scale_audit_{record_id}",
                    )
                    tactic = parse_tactic_tokens(
                        action_tokens,
                        theorem,
                        tokenizer,
                        database,
                        environment=environment,
                    )
                    environment.apply(
                        ProofState.from_theorem(theorem),
                        tactic,
                    )
                    valid += 1
                    valid_by_depth[proof_depth] = (
                        valid_by_depth.get(proof_depth, 0) + 1
                    )
                except Exception as exc:
                    failures.append({
                        "split": split,
                        "record_id": record_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
            global_index += 1
        audited_by_split[split] = split_audited
    report = {
        "format": "peano-scale-corpus-audit-v1",
        "database": str(Path(database_path).resolve()),
        "corpus": str(corpus.resolve()),
        "total_records": total,
        "sample_requested": sample_size,
        "sample_audited": audited,
        "sample_by_split": audited_by_split,
        "sample_by_proof_depth": dict(sorted(
            audited_by_depth.items(), key=lambda item: int(item[0])
        )),
        "valid_by_proof_depth": dict(sorted(
            valid_by_depth.items(), key=lambda item: int(item[0])
        )),
        "valid_actions": valid,
        "invalid_actions": len(failures),
        "duplicate_ids_in_sample": duplicates,
        "maximum_sampled_state_tokens": maximum_state,
        "maximum_sampled_action_tokens": maximum_action,
        "environment": environment.configuration_record(),
        "failures": failures,
    }
    if destination is not None:
        Path(destination).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report

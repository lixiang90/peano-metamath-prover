"""HTPS policy/cut supervision from the shared random proof graph generator."""
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path

from metamath_generator.parser import parse
from neural_prover.data import CorpusBuildConfig, build_corpus, load_examples, _conclusion_bytes
from neural_prover.data_contract import tokenizer_fingerprint
from neural_prover.lemma import LemmaBackwardEnvironment
from neural_prover.tokenizer import MetamathTokenizer
from .data import replay_record, audit_forward_dataset


def build_graph_dataset(database_path, output_directory, cfg, base_tokenizer_path):
    output = Path(output_directory)
    options = {k: v for k, v in asdict(cfg).items() if k in {f.name for f in fields(CorpusBuildConfig)}}
    options.update(include_source_actions=False, base_tokenizer=base_tokenizer_path)
    manifest = build_corpus(database_path, output, CorpusBuildConfig(**options))
    database = parse(database_path)
    tokenizer = MetamathTokenizer.load(output / "tokenizer.json").upgraded_for_lemma_actions()
    tokenizer.save(output / "tokenizer.json")
    fingerprint = tokenizer_fingerprint(tokenizer)
    environment = LemmaBackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    counts, kinds = {}, {"policy": Counter(), "lemma": Counter()}
    filtered = Counter()
    for split in ("train", "validation", "test"):
        policies, lemmas = [], []
        seen_lemmas = set()
        for example in load_examples(output / f"{split}.jsonl"):
            theorem = tokenizer.theorem_from_state_tokens(example.state_tokens, database)
            group = hashlib.sha256(_conclusion_bytes(theorem)).hexdigest()
            shared = {"split": split, "split_group_sha256": group,
                      "tokenizer_sha256": fingerprint, "generation_kind": example.generation_kind,
                      "definition_support": list(example.definition_support),
                      "state_tokens": list(example.state_tokens),
                      "state_ids": tokenizer.encode(example.state_tokens)}
            record = {**example.to_record(), **shared, "value_target": 1.0,
                      "action_ids": tokenizer.encode(example.action_tokens)}
            transition = replay_record(record, tokenizer, database, environment)
            policies.append(record)
            kinds["policy"][example.generation_kind] += 1
            canonical = tokenizer.canonical_variables(theorem)
            for goal in transition.generated_goals:
                if goal == theorem.conclusion or goal in [h.expr for h in theorem.hypotheses]:
                    continue
                try:
                    tokens = tokenizer.lemma_tactic_tokens(goal, canonical)
                    if len(tokens) > cfg.max_action_tokens:
                        filtered["lemma_length"] += 1
                        continue
                    lemma = {**shared, "lemma_action_tokens": tokens,
                             "lemma_action_ids": tokenizer.encode(tokens),
                             "lemma": goal.to_prefix(), "root_proof_depth": example.proof_depth,
                             "utility_target": 1.0}
                    replay_record(lemma, tokenizer, database, environment, lemma=True)
                except ValueError:
                    filtered["lemma_kernel_or_encoding"] += 1
                    continue
                key = (tuple(example.state_tokens), tuple(tokens))
                if key not in seen_lemmas:
                    seen_lemmas.add(key)
                    lemmas.append(lemma)
                    kinds["lemma"][example.generation_kind] += 1
        for name, rows in (("policy", policies), ("lemma", lemmas)):
            (output / f"{name}_{split}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        counts[split] = {"policy": len(policies), "lemma": len(lemmas)}
    manifest.update(format="peano-htps-random-graph-v1", counts=counts,
        tokenizer_sha256=fingerprint, vocabulary_size=len(tokenizer),
        configuration=asdict(cfg), filtered=dict(filtered), environment=environment.configuration_record(),
        generation_kinds={k: dict(v) for k, v in kinds.items()},
        proof_dag_nodes=sum(r["stored"] for r in manifest["runs"]),
        split_policy="alpha-normalized conclusion family of each expanded source step",
        independent_verification={"replayed_generated_declarations": sum(r["stored"] for r in manifest["runs"]),
                                  "method": "every generated declaration exported, reparsed and replayed"})
    audit = audit_forward_dataset(database_path, output, output / "action-audit.json")
    if audit["invalid_actions"]:
        raise ValueError(f"HTPS graph action audit failed: {audit['failures'][:3]}")
    manifest["action_audit"] = {k: audit[k] for k in ("valid_actions", "invalid_actions", "scope")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

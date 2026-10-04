"""Resumable sharded data from fresh random proof graphs, not templates."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.parser import parse
from .data import CorpusBuildConfig, _graph_examples
from .data_contract import tokenizer_fingerprint
from .tokenizer import MetamathTokenizer


def build_graph_scale_corpus(database_path, base_directory, output_directory, cfg):
    from .scale_data import _length_bucket
    if min(cfg.shard_size, cfg.graph_steps, cfg.graph_max_proof_depth, cfg.max_graph_batches) <= 0:
        raise ValueError("positive shard and graph budgets required")
    targets = dict(train=cfg.train_examples, validation=cfg.validation_examples, test=cfg.test_examples)
    if min(targets.values()) < 0 or cfg.max_new_records < 0:
        raise ValueError("record budgets must be non-negative")
    tokenizer = MetamathTokenizer.load(Path(base_directory) / "tokenizer.json")
    database = parse(database_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    stable = {k: v for k, v in asdict(cfg).items() if k not in {"workers", "max_new_records", "max_graph_batches"}}
    # Includes are part of the theory fingerprint.
    from metamath_generator.parser import MetamathParser
    source = " ".join(MetamathParser()._tokens_with_includes(Path(database_path).resolve(), set()))
    identity = {"theory_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer)}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != stable or any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("existing graph corpus configuration/theory/tokenizer does not match")
    else:
        manifest = {"format": "peano-scale-corpus-v3", "generation_algorithm": "random_proof_graph",
                    "configuration": stable, **identity, "database": str(Path(database_path).resolve()),
                    "targets": targets, "counts": {s: 0 for s in targets},
                    "shards": {s: [] for s in targets}, "cursor": [0, 0],
                    "proof_depth_histogram": {s: {} for s in targets}, "length_histogram": {},
                    "complete": False, "total": 0, "maximum_proof_depth": 0, "graph_runs": {},
                    "leakage_control": "alpha-normalized conclusion families; all source steps audited",
                    "worker_policy": "bounded graphs processed serially; workers is reserved for template mode"}
    tokenizer.save(output / "tokenizer.json")
    seen = set()
    for split in targets:
        for shard in manifest["shards"][split]:
            shard_path = output / shard["path"]
            if hashlib.sha256(shard_path.read_bytes()).hexdigest() != shard["sha256"]:
                raise ValueError("graph corpus shard hash mismatch")
            with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
                seen.update(json.loads(line)["id"] for line in stream if line.strip())
    remaining = cfg.max_new_records or sum(targets[s] - manifest["counts"][s] for s in targets)
    pending = {s: [] for s in targets}

    def save():
        # Commit all splits and the exact graph/record cursor together. A
        # partial graph is regenerated deterministically when resuming.
        for split, rows in pending.items():
            for start in range(0, len(rows), cfg.shard_size):
                batch = rows[start:start + cfg.shard_size]
                shard_path = output / split / f"part-{len(manifest['shards'][split]):05d}.jsonl.gz"
                shard_path.parent.mkdir(exist_ok=True)
                temporary = shard_path.with_suffix(".tmp")
                with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=cfg.gzip_level) as stream:
                    for record in batch:
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                temporary.replace(shard_path)
                depths = Counter(str(r["proof_depth"]) for r in batch)
                manifest["shards"][split].append({"path": shard_path.relative_to(output).as_posix(),
                    "records": len(batch), "bytes": shard_path.stat().st_size,
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                    "proof_depth_histogram": dict(depths)})
                for depth, n in depths.items():
                    hist = manifest["proof_depth_histogram"][split]
                    hist[depth] = hist.get(depth, 0) + n
                for r in batch:
                    key = _length_bucket(max(len(r["state"]), len(r["action"])))
                    hist = manifest["length_histogram"]
                    hist[key] = hist.get(key, 0) + 1
            rows.clear()
        manifest["total"] = sum(manifest["counts"].values())
        manifest["maximum_proof_depth"] = max((int(d) for h in manifest["proof_depth_histogram"].values() for d in h), default=0)
        manifest["complete"] = all(manifest["counts"][s] == targets[s] for s in targets)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    data_cfg = CorpusBuildConfig(max_state_tokens=cfg.max_state_tokens, max_action_tokens=cfg.max_action_tokens,
                                include_source_actions=False, max_proof_depth=cfg.graph_max_proof_depth)
    while remaining and not all(manifest["counts"][s] == targets[s] for s in targets):
        graph_index, offset = manifest["cursor"]
        if graph_index >= cfg.max_graph_batches:
            save()
            raise ValueError("graph generation budget exhausted; corpus saved incomplete, increase max_graph_batches")
        generator = TheoremGenerator(database, GenerationConfig(
            seed=cfg.seed + graph_index, max_proof_depth=cfg.graph_max_proof_depth,
            graph_instance_probability=cfg.graph_instance_probability,
        ))
        generator.generate("graph", cfg.graph_steps)
        records = list(_graph_examples(generator, tokenizer, data_cfg))
        manifest["graph_runs"][str(graph_index)] = {
            "statistics": dict(generator.stats), "filtered": dict(generator.rejected),
            "nodes": generator.generated_count, "training_steps": len(records),
        }
        for index in range(offset, len(records)):
            example = records[index]
            split = example.split
            manifest["cursor"] = [graph_index, index + 1]
            if example.example_id in seen or manifest["counts"][split] >= targets[split]:
                continue
            seen.add(example.example_id)
            pending[split].append({"id": example.example_id, "state": list(example.state_ids),
                "action": list(example.action_ids), "value": example.value_target,
                "difficulty": example.difficulty, "proof_depth": example.proof_depth,
                "generation_kind": example.generation_kind, "definition_support": list(example.definition_support),
                "graph_index": graph_index})
            manifest["counts"][split] += 1
            remaining -= 1
            if sum(map(len, pending.values())) >= cfg.shard_size:
                save()
            if not remaining or all(manifest["counts"][s] == targets[s] for s in targets):
                break
        else:
            manifest["cursor"] = [graph_index + 1, 0]
        save()
    save()
    return manifest

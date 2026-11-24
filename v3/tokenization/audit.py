"""Compare reversible encodings of the PA+ grammar and saved v2.1 model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from metamath_generator.parser import parse
from pa_prover_v2.tokenizer import BYTE_OFFSET, ByteTokenizer
from pa_prover_v21.interaction import PROTOCOL, segments
from pa_prover_v21.policy import prompt

from pa_symbols import PASymbolTokenizer


ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prompt_parts(observation):
    if observation.get("protocol") == PROTOCOL:
        return [s for s, _ in segments(observation["events"], awaiting_model=True)]
    return [prompt(observation)]


def training_texts(run, frozen):
    train_keys = {r["full_key"] for r in frozen if r["split"] == "train"}
    test_keys = {r["full_key"] for r in frozen if r["split"] == "test"}
    if train_keys & test_keys:
        raise ValueError("overlapping theorem keys in frozen split")
    texts = set()
    for shard in sorted(run.glob("data-*")):
        cases = {r["id"]: r for r in rows(shard / "cases.jsonl") if r["split"] == "train"}
        for r in cases.values():
            if r["full_key"] not in train_keys:
                raise ValueError("BPE fit case outside frozen train split")
            texts.update([r["conclusion"], *r["hypotheses"]])
        for r in rows(shard / "examples.jsonl"):
            if r["split"] != "train":
                continue
            if r["case"] not in cases:
                raise ValueError("BPE fit example outside shard train split")
            if r["kind"] == "trajectory":
                texts.update(s for s, _ in segments(r["events"]))
            else:
                texts.update(prompt_parts(r["observation"]))
                suffix = "\n" if r["observation"].get("protocol") == PROTOCOL else ""
                texts.update(c + suffix for c in r.get("candidates", []))
    return texts


def fit_bpe(texts, vocab_size):
    model = Tokenizer(models.BPE())
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    model.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - BYTE_OFFSET,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    model.train_from_iterator(texts, trainer)
    return model


class Audit:
    def __init__(self, pa, bpe_models):
        self.byte = ByteTokenizer()
        self.pa = pa
        self.bpe_models = bpe_models
        self.groups = defaultdict(lambda: defaultdict(list))

    def measure(self, group, parts):
        text = "".join(parts)
        byte_ids = self.byte.encode(text)
        pa_parts = self.pa.encode_segments(parts)
        pa_ids = [i for span in pa_parts for i in span]
        if not (self.byte.decode(byte_ids) == self.pa.decode(pa_ids) == text):
            raise ValueError(f"round-trip failed for {group}")
        sizes = {"byte": len(byte_ids), "pa": len(pa_ids)}
        for name, model in self.bpe_models.items():
            ids = [i for part in parts for i in model.encode(part).ids]
            if model.decode(ids) != text:
                raise ValueError(f"{name} round-trip failed for {group}")
            sizes[name] = len(ids)
        for name, n in sizes.items():
            self.groups[group][name].append(n)
        return sizes

    def summary(self):
        result = {}
        for group, methods in self.groups.items():
            result[group] = {}
            for name, sizes in methods.items():
                ordered = sorted(sizes)
                result[group][name] = {
                    "n": len(sizes), "sum": sum(sizes), "median": statistics.median(ordered),
                    "p95": ordered[math.ceil(.95 * len(sizes)) - 1], "single": sizes.count(1),
                }
        return result


def run_audit(run):
    frozen = list(rows(run / "frozen" / "cases.jsonl"))
    test = {r["id"]: r for r in frozen if r["split"] == "test"}
    catalog = json.loads((ROOT / "formal" / "pa-plus-definitions.json").read_text(encoding="utf-8"))
    # Reproduce the original 701-piece comparison with the same BPE fit budget.
    pa = PASymbolTokenizer(parse(ROOT / "formal" / "peano-pa-plus.mm"), file_tokens=False)
    texts = training_texts(run, frozen)
    # Give every BPE size the identical training texts and public lexer dictionary.
    fit_texts = sorted(texts | set(pa.pieces))
    bpe_sizes = {"bpe": pa.vocab_size, "bpe_512": 512, "bpe_1024": 1024}
    bpe_models = {name: fit_bpe(fit_texts, size) for name, size in bpe_sizes.items()}
    audit = Audit(pa, bpe_models)
    examples = {}
    definition_names = {r["name"] for r in catalog["definitions"]}
    formula_coverage = {}
    for split in ("train", "test"):
        cases = [r for r in frozen if r["split"] == split]
        frequencies = Counter(atom for r in cases for f in [r["conclusion"], *r["hypotheses"]]
                              for atom in f.split())
        formula_coverage[split] = {
            "cases": len(cases), "distinct_atoms": len(frequencies),
            "catalog_definition_occurrences": {n: frequencies[n] for n in sorted(definition_names & frequencies.keys())},
        }
    for r in test.values():
        size = audit.measure("heldout_conclusions", [r["conclusion"]])
        if r["id"] == "p0004":
            examples["p0004_conclusion"] = [r["conclusion"], size]
    for r in catalog["definitions"]:
        audit.measure("definition_names", [r["name"]])
        audit.measure("definition_bodies", [r["body"]])
    for r in catalog["statements"]:
        size = audit.measure("catalog_statements", [r["body"]])
        if r["name"] == "euclid-primes-statement":
            examples["euclid_primes"] = [r["body"], size]
    snapshots = set()
    for mode in "abc":
        for path in sorted((run / "eval" / f"learned-{mode}").glob("*/result.json")):
            if path.parent.name not in test:
                continue
            record = json.loads(path.read_bytes())
            attempt = next((a for a in record["attempts"] if a.get("distribution") and a.get("candidates")), None)
            if attempt is None:
                continue
            observation = attempt["observation"]
            parts = prompt_parts(observation)
            if "".join(parts) != prompt(observation):
                raise ValueError("prompt serialization drift")
            size = audit.measure(f"{mode}_first_prompt", parts)
            if mode == "a":
                snapshots.add(observation["library_snapshot"])
                # Diagnostic only. Vary one identifier while keeping the rest of the prompt identical.
                fresh = hashlib.sha256(("tokenizer-audit/" + path.parent.name).encode()).hexdigest()
                audit.measure("a_changed_snapshot", prompt_parts({**observation, "library_snapshot": fresh}))
            if mode == "a" and path.parent.name == "p0004":
                examples["p0004_prompt"] = size
            suffix = "\n" if observation.get("protocol") == PROTOCOL else ""
            for candidate in attempt["candidates"]:
                size = audit.measure(f"{mode}_candidate", [candidate + suffix])
                if mode == "a" and path.parent.name == "p0004" and json.loads(candidate)["kind"] == "INDUCT":
                    examples["p0004_induct"] = [candidate, size]
    return {
        "vocabulary": {"byte": audit.byte.vocab_size, "pa": pa.vocab_size,
                       **{name: model.get_vocab_size() + BYTE_OFFSET for name, model in bpe_models.items()}},
        "pa_tokenizer_sha256": pa.fingerprint(), "tokenizers_version": tokenizers.__version__,
        "unique_fit_chunks_before_public_grammar": len(texts), "unique_a_snapshots": len(snapshots),
        "frozen_conclusions_and_hypotheses": formula_coverage,
        "groups": audit.summary(), "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = json.dumps(run_audit(args.run), ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)

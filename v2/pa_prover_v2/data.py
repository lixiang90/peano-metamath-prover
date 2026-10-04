"""Rule-generated PA+ certificates and fully replayed agent supervision."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.certificates import flatten_certificate, replay_graph, certificate_steps
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.model import Node, Theorem
from metamath_generator.parser import MetamathParser
from metamath_generator.unification import substitute_simultaneous

from .kernel import ProofKernel, statement_fingerprint, theorem_from_data, theorem_to_data
from .library import TheoremLibrary


@dataclass(frozen=True)
class DataConfig:
    """Generation limits; seeds share the global example budget."""

    seeds: tuple[int, ...] = (7, 11, 19)
    steps_per_seed: int = 200
    max_examples: int = 120
    max_proof_labels: int = 4096
    max_depth: int = 10
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    library_items: int = 32
    library_bytes: int = 2_000_000
    bootstrap_definitions: bool = False
    axiom_instances_per_seed: int = 0
    generation_mode: str = "graph"
    graph_instance_probability: float = 0.10
    graph_partial_premise_probability: float = 0.30

    def validate(self):
        if self.generation_mode not in {"graph", "random"}:
            raise ValueError("generation_mode must be graph or random")
        if not 0 <= self.graph_instance_probability <= 1:
            raise ValueError("graph_instance_probability must be in [0, 1]")
        if not 0 <= self.graph_partial_premise_probability <= 1:
            raise ValueError("graph_partial_premise_probability must be in [0, 1]")
        if not self.seeds or min(self.steps_per_seed, self.max_examples, self.max_proof_labels, self.max_depth) <= 0:
            raise ValueError("positive generation budgets and seeds required")
        if min(self.validation_fraction, self.test_fraction) < 0 or self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("invalid split fractions")
        if self.library_items <= 0 or self.library_bytes <= 0:
            raise ValueError("invalid library budget")
        if not isinstance(self.axiom_instances_per_seed, int) or self.axiom_instances_per_seed < 0:
            raise ValueError("axiom_instances_per_seed must be a nonnegative integer")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _random_ground_expression(rng, typ, depth):
    """Small typed PA expressions; no hand-selected mathematical conclusions."""
    if typ == "term":
        op = "0" if depth <= 1 else rng.choice(("0", "S", "+", "*"))
        arity = 0 if op == "0" else 1 if op == "S" else 2
        return Node(op, tuple(_random_ground_expression(rng, "term", depth - 1) for _ in range(arity)))
    op = "=" if depth <= 2 else rng.choice(("=", "not", "implies"))
    child_type = "term" if op == "=" else "wff"
    return Node(op, tuple(_random_ground_expression(rng, child_type, depth - 1)
                          for _ in range(1 if op == "not" else 2)))


def _random_axiom_instances(kernel, seed, cfg):
    """Yield bounded attempts at single source-rule applications, independently checked."""
    rules = [rule for rule in kernel.database.logical_assertions.values()
             if not rule.hypotheses and set(rule.variable_types.values()) <= {"term", "wff"}]
    rules = sorted(rules, key=lambda rule: (sum(1 for _ in rule.conclusion.walk()), rule.name))[:16]
    if not rules or not cfg.axiom_instances_per_seed:
        return
    rng = random.Random(f"pa-plus-axiom-instances:{seed}")
    for attempt in range(max(32, cfg.axiom_instances_per_seed * 16)):
        rule = rng.choice(rules)
        mapping = {name: _random_ground_expression(rng, typ, max(2, min(cfg.max_depth, 3)))
                   for name, typ in rule.variable_types.items()}
        target = Theorem(f"warmup_{seed}_{attempt}", [], substitute_simultaneous(rule.conclusion, mapping))
        # compose checks every syntax proof, source identity, $d and the result.
        yield kernel.compose(rule, mapping, [], target)



def _assign_splits(records, cfg):
    """Connected components of alpha-equivalent statements AND proof skeletons."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for r in records:
        a, b = find("s:" + r["statement_sha256"]), find("p:" + r["skeleton_sha256"])
        parents[max(a, b)] = min(a, b)
    for r in records:
        group = _digest(find("s:" + r["statement_sha256"]))
        unit = int(group[:16], 16) / (1 << 64)
        r["split_group"] = group
        r["split"] = ("test" if unit < cfg.test_fraction else
                      "validation" if unit < cfg.test_fraction + cfg.validation_fraction else "train")


def generate_corpus(database_path, output_directory, config=None):
    """Generate, certify, split, select train-only memory, and replay episodes.

    ``max_examples`` is a global cap shared across seeds; unused quota is
    offered to later seeds. Initial library reuse scores are a logical-rule frequency
    proxy, not measurements of the theorem itself being reused.  Selection
    uses independently certified training declarations before episode budget
    filtering; a certificate can remain valid train-only memory even if its
    teacher trajectory later exceeds an agent budget.
    """
    from .agent import AgentBudget, AgentSession, augmented_teacher_actions

    cfg = config or DataConfig()
    cfg.validate()
    kernel = ProofKernel(database_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    parser = MetamathParser()
    source = " ".join(parser._tokens_with_includes(Path(database_path).resolve(), set()))
    records, seen, filtered = [], set(), Counter()
    seed_counts = {str(seed): 0 for seed in cfg.seeds}
    used_seeds = []
    graph_runs = []

    def append_certificate(cert, seed, kind, proof_depth):
        if len(cert.proof.source_labels) > cfg.max_proof_labels:
            raise ValueError("proof label budget exceeded")
        kernel.verify(cert)
        statement = statement_fingerprint(cert)
        if statement in seen:
            filtered["duplicate_statement"] += 1
            return False
        # Logical labels retain proof structure; syntax instantiations are removed.
        skeleton = [label for label in cert.proof.source_labels if label in kernel.database.logical_assertions]
        records.append({"id": f"g{seed}_{len(records)}", "seed": seed,
                        "statement_sha256": statement, "skeleton_sha256": _digest(skeleton),
                        "certificate": theorem_to_data(cert), "proof_depth": proof_depth,
                        "generation_kind": kind,
                        "expanded_proof_depth": max((s.proof_depth for s in certificate_steps(cert, kernel.database)), default=0)})
        seen.add(statement)
        seed_counts[str(seed)] += 1
        return True

    for seed_index, seed in enumerate(cfg.seeds):
        if len(records) >= cfg.max_examples:
            break
        used_seeds.append(seed)
        remaining_seeds = len(cfg.seeds) - seed_index
        seed_quota = (cfg.max_examples - len(records) + remaining_seeds - 1) // remaining_seeds
        seed_limit = len(records) + seed_quota
        warmup_count = 0
        for cert in _random_axiom_instances(kernel, seed, cfg):
            if len(records) >= seed_limit or warmup_count >= cfg.axiom_instances_per_seed:
                break
            try:
                warmup_count += append_certificate(cert, seed, "random_axiom_instance", 1)
            except (ValueError, KeyError, RecursionError) as exc:
                filtered[type(exc).__name__ + ":" + str(exc)[:100]] += 1
        if len(records) >= seed_limit:
            continue
        generator = TheoremGenerator(kernel.database, GenerationConfig(
            seed=seed, max_proof_depth=cfg.max_depth, depth_parent_bias=0.8,
            graph_instance_probability=cfg.graph_instance_probability,
            graph_partial_premise_probability=cfg.graph_partial_premise_probability,
            max_ast_depth=24, max_hypotheses=6, max_variables=12,
            bootstrap_definitions=cfg.bootstrap_definitions,
            definition_coverage_weight=1.0 if cfg.bootstrap_definitions else 0.0,
        ))
        generator.generate(cfg.generation_mode, cfg.steps_per_seed)
        graph_runs.append({"seed": seed, "statistics": dict(generator.stats),
                           "rejected": dict(generator.rejected),
                           "depth_histogram": dict(Counter(t.proof_depth for t in generator.store.generated()))})
        # Interleave shallow and deeper proofs; do not fill the corpus only with bridges.
        generated = sorted(generator.store.generated(), key=lambda t: (t.proof_depth, t.name))
        random.Random(seed).shuffle(generated)
        expanded = replay_graph(generator.store, source)
        for theorem in generated:
            if len(records) >= seed_limit:
                break
            try:
                cert = flatten_certificate(expanded.proved_theorems[theorem.name], expanded, kernel.database, cfg.max_proof_labels)
                kind = generator.generation_context.get(theorem.id, "composed_proof_dag")
                append_certificate(cert, seed, kind, theorem.proof_depth)
            except (ValueError, KeyError, RecursionError) as exc:
                filtered[type(exc).__name__ + ":" + str(exc)[:100]] += 1
    if not records:
        raise ValueError("generation produced no replayable certificates")
    _assign_splits(records, cfg)
    train = [r for r in records if r["split"] == "train"]
    if not train:
        raise ValueError("no training group: increase seeds/max_examples")
    excluded = {r["statement_sha256"] for r in records if r["split"] != "train"}
    library = TheoremLibrary(kernel, max_items=cfg.library_items, max_bytes=cfg.library_bytes,
                             exclude_fingerprints=excluded)
    logical_use = Counter(label for r in train for label in theorem_from_data(r["certificate"]).proof.source_labels)
    for index, record in enumerate(train):
        cert = theorem_from_data(record["certificate"])
        reuse = sum(logical_use[label] for label in set(cert.proof.source_labels) if label in kernel.database.logical_assertions)
        library.add(cert, provenance="train", reuse_count=reuse, step=index)
    library.save(output / "library.json")
    action_coverage = Counter()
    augmentation = Counter({
        "searches": 0, "library_reads": 0, "library_uses": 0,
        "scratch_uses": 0, "library_candidates_checked": 0,
    })
    episodes = []
    for record in records:
        cert = theorem_from_data(record["certificate"])
        try:
            episode_augmentation = {}
            actions = augmented_teacher_actions(cert, kernel, library, stats=episode_augmentation)
            episode_coverage = Counter()
            session = AgentSession(kernel, cert, library, AgentBudget(max_steps=max(128, len(actions) + 8)))
            for action in actions:
                observation = session.step(action)
                episode_coverage[action.op] += 1
                if observation.status == "unknown":
                    raise ValueError("teacher episode exceeded resource budget")
            if session.status != "success":
                raise ValueError("teacher episode failed final proof verification")
            action_coverage.update(episode_coverage)
            augmentation.update(episode_augmentation)
            episodes.append({**record, "actions": [a.to_dict() for a in actions],
                             "augmentation": episode_augmentation,
                             "text": "\n".join(session.context_segments(memory_mode="text")),
                             "audit_context": session.context(), "theory_sha256": kernel.theory_fingerprint})
        except (ValueError, KeyError) as exc:
            filtered["episode:" + str(exc)[:100]] += 1
    if not any(r["split"] == "train" for r in episodes):
        raise ValueError("no verified train episodes; inspect teacher generation")
    for split in ("train", "validation", "test"):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for record in episodes:
                if record["split"] == split:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"format_version": 3, "source": "random_pa_plus_rule_composition",
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
                "theory_sha256": kernel.theory_fingerprint, "config": asdict(cfg),
                "counts": {split: sum(r["split"] == split for r in episodes) for split in ("train", "validation", "test")},
                "filtered": dict(filtered), "augmentation": dict(augmentation),
                "generation_kind": dict(Counter(r["generation_kind"] for r in episodes)),
                "generation_algorithm": cfg.generation_mode, "graph_runs": graph_runs,
                "proof_depth_kind": "stored dependency DAG; expanded_proof_depth counts source logical steps",
                "proof_depth_histogram": dict(Counter(r["proof_depth"] for r in episodes)),
                "seed_certificate_counts": seed_counts, "used_seeds": used_seeds,
                "action_coverage": dict(action_coverage), "library_count": len(library.record_ids),
                "library_reuse_score": "logical_rule_frequency_proxy_not_measured_theorem_reuse",
                "seed_budget_note": "remaining max_examples is shared across remaining seeds; unused quota passes forward",
                "split_rule": "connected components of alpha-normalized statement and logical proof skeleton",
                "library_sha256": hashlib.sha256((output / "library.json").read_bytes()).hexdigest(),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.jsonl")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_episodes(directory, split="train", *, kernel=None):
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid split")
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 3:
        raise ValueError("dataset statement-identity format changed; regenerate the corpus")
    path = root / f"{split}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
        raise ValueError("dataset hash mismatch")
    if kernel is not None and manifest["theory_sha256"] != kernel.theory_fingerprint:
        raise ValueError("dataset theory mismatch")
    if "library_sha256" in manifest:
        library_path = root / "library.json"
        if hashlib.sha256(library_path.read_bytes()).hexdigest() != manifest["library_sha256"]:
            raise ValueError("dataset library hash mismatch")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != manifest["counts"].get(split, 0):
        raise ValueError("dataset split count mismatch")
    for record in records:
        if record.get("split") != split or record.get("theory_sha256") != manifest["theory_sha256"]:
            raise ValueError("episode split or theory metadata mismatch")
        certificate = theorem_from_data(record["certificate"])
        if statement_fingerprint(certificate) != record["statement_sha256"]:
            raise ValueError("episode statement fingerprint mismatch")
        if kernel is not None:
            kernel.verify(certificate)
            skeleton = [label for label in certificate.proof.source_labels if label in kernel.database.logical_assertions]
            if _digest(skeleton) != record["skeleton_sha256"]:
                raise ValueError("episode proof skeleton mismatch")
    return records

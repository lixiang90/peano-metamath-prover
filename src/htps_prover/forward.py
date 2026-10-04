from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.export import export_metamath, theorem_record
from metamath_generator.definitions import load_definition_catalog
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.model import Node, Theorem
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import verify
from neural_prover.tokenizer import MetamathTokenizer
from neural_prover.data_contract import tokenizer_fingerprint
from neural_prover.lemma import LemmaBackwardEnvironment

from .data import replay_record, audit_forward_dataset


@dataclass(frozen=True, slots=True)
class ForwardDAGConfig:
    """Configuration for certified forward proof-DAG generation."""

    seeds: tuple[int, ...] = (7, 11, 19, 23)
    generation_mode: str = "graph"
    graph_instance_probability: float = 0.10
    graph_partial_premise_probability: float = 0.30
    steps_per_seed: int = 5_000
    max_proof_depth: int = 12
    max_ast_depth: int = 40
    max_hypotheses: int = 8
    max_variables: int = 16
    depth_parent_bias: float = 0.8
    full_discharge_probability: float = 0.78
    closed_parent_probability: float = 0.82
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    verify_deepest_per_seed: int = 4
    max_state_tokens: int = 384
    max_action_tokens: int = 256
    definition_catalog: str | None = None
    bootstrap_definitions: bool = False
    definition_coverage_weight: float = 0.0
    bounded_nat_max: int = -1
    ground_instances_per_predicate: int = 0
    target_guidance_weight: float = 0.0
    max_definition_only_search_per_predicate: int = -1
    max_target_hints: int = 3

    def validate(self) -> None:
        if not self.seeds or self.steps_per_seed <= 0:
            raise ValueError("at least one seed and positive steps are required")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions must sum to < 1")
        if min(self.validation_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions must be non-negative")
        if min(self.max_state_tokens, self.max_action_tokens) <= 0:
            raise ValueError("token limits must be positive")
        if self.verify_deepest_per_seed < 0:
            raise ValueError("verification count must be non-negative")
        if self.definition_catalog is None and (
            self.bootstrap_definitions or self.ground_instances_per_predicate
            or self.target_guidance_weight or self.definition_coverage_weight
        ):
            raise ValueError("PA+ generation options require a definition catalog")


def _proof_skeleton(theorem_id: int, generator: TheoremGenerator) -> tuple:
    memo: dict[int, tuple] = {}

    def visit(item_id: int) -> tuple:
        cached = memo.get(item_id)
        if cached is not None:
            return cached
        item = generator.store[item_id]
        if item.proof is None or item.kind != "generated":
            result = ("source", item.name)
        else:
            result = (
                item.proof.rule,
                tuple(
                    visit(parent)
                    for parent in item.proof.premise_map
                    if parent is not None
                ),
                tuple(parent is None for parent in item.proof.premise_map),
            )
        memo[item_id] = result
        return result

    return visit(theorem_id)


def _split(skeleton: tuple, cfg: ForwardDAGConfig) -> str:
    digest = hashlib.sha256(repr(skeleton).encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit < cfg.test_fraction:
        return "test"
    if unit < cfg.test_fraction + cfg.validation_fraction:
        return "validation"
    return "train"


def _instantiated_parent(
    theorem: Theorem,
    premise_index: int,
    parent: Theorem,
) -> Node:
    if theorem.proof is None:
        raise ValueError("generated theorem has no proof")
    mapping = {
        variable: theorem.proof.substitution.get(
            f"__p{premise_index}_{variable}", Node(variable)
        )
        for variable in parent.variable_types
    }
    return substitute_simultaneous(parent.conclusion, mapping)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_splits(nodes, policies, lemmas, cfg):
    """Keep both recursive skeletons and identical root states together."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for node in nodes:
        a = find("proof:" + node["proof_skeleton_sha256"])
        b = find("state:" + node["state_sha256"])
        parents[max(a, b)] = min(a, b)
    for collection in (nodes, policies, lemmas):
        for record in collection:
            group = find("proof:" + record["proof_skeleton_sha256"])
            record["split_group_sha256"] = hashlib.sha256(group.encode()).hexdigest()
            record["split"] = _split((group,), cfg)


def _deduplicate(records, action_key):
    seen = set()
    result = []
    for record in records:
        key = (tuple(record["state_tokens"]), tuple(record[action_key]))
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def _independently_verify(
    source_path: Path,
    generator: TheoremGenerator,
    theorem: Theorem,
) -> int:
    """Export, reparse and replay one complete generated dependency DAG."""

    with tempfile.TemporaryDirectory() as directory:
        fragment = Path(directory) / "generated.mm"
        export_metamath(theorem, generator.store, fragment)
        parser = MetamathParser()
        expanded_source = " ".join(parser._tokens_with_includes(  # noqa: SLF001
            source_path.resolve(), set()
        ))
        combined = parser.parse_text(
            expanded_source + "\n" + fragment.read_text(encoding="utf-8"),
            source_name=str(source_path.resolve()),
        )
        generated = [
            item
            for item in combined.proved_theorems.values()
            if item.name.startswith("gen")
        ]
        for item in generated:
            verify(item, combined)
        return len(generated)


def build_forward_dag_dataset(
    database_path: str | Path,
    output_directory: str | Path,
    config: ForwardDAGConfig | None = None,
    *,
    base_tokenizer_path: str | Path | None = None,
) -> dict:
    """Build HTPS supervision from verified forward proof DAGs.

    Splits are assigned by recursive proof skeleton rather than only by the
    final formula, preventing type-preserving variants of one derivation from
    leaking across train and evaluation splits.
    """

    cfg = config or ForwardDAGConfig()
    cfg.validate()
    if cfg.generation_mode == "graph":
        from .graph_data import build_graph_dataset
        return build_graph_dataset(database_path, output_directory, cfg, base_tokenizer_path)
    if cfg.generation_mode != "random":
        raise ValueError("generation_mode must be graph or random")
    source = Path(database_path)
    database = parse(source)
    catalog = load_definition_catalog(cfg.definition_catalog) if cfg.definition_catalog else None
    tokenizer = (
        MetamathTokenizer.load(base_tokenizer_path)
        if base_tokenizer_path is not None
        else MetamathTokenizer.from_database(database)
    )
    if catalog is not None:
        tokenizer = tokenizer.upgraded_for_pa_plus(
            database, catalog.definition_names, catalog.statement_names,
            bounded_nat_max=cfg.bounded_nat_max, max_target_hints=cfg.max_target_hints,
        )
    elif tokenizer.pa_plus_context.enabled:
        raise ValueError("PA+ base tokenizer requires an explicit matching definition catalog")
    tokenizer = tokenizer.upgraded_for_lemma_actions()
    fingerprint = tokenizer_fingerprint(tokenizer)
    environment = LemmaBackwardEnvironment(database)
    environment.configure_from_tokenizer(tokenizer)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")

    nodes: list[dict] = []
    policies: dict[str, list[dict]] = {
        "train": [], "validation": [], "test": []
    }
    lemmas: dict[str, list[dict]] = {
        "train": [], "validation": [], "test": []
    }
    run_summaries: list[dict] = []
    verified_certificates = 0
    verified_declarations = 0
    filtered = Counter()
    encoding_failures = []

    for seed in cfg.seeds:
        generator = TheoremGenerator(
            database,
            GenerationConfig(
                seed=seed,
                max_proof_depth=cfg.max_proof_depth,
                max_ast_depth=cfg.max_ast_depth,
                max_hypotheses=cfg.max_hypotheses,
                max_variables=cfg.max_variables,
                depth_parent_bias=cfg.depth_parent_bias,
                full_discharge_probability=cfg.full_discharge_probability,
                closed_parent_probability=cfg.closed_parent_probability,
                bootstrap_definitions=cfg.bootstrap_definitions,
                focus_predicates=catalog.definition_names if catalog else (),
                definition_coverage_weight=cfg.definition_coverage_weight,
                bounded_nat_max=cfg.bounded_nat_max,
                ground_instances_per_predicate=cfg.ground_instances_per_predicate,
                target_statements=catalog.statement_names if catalog else (),
                target_guidance_weight=cfg.target_guidance_weight,
                max_definition_only_search_per_predicate=cfg.max_definition_only_search_per_predicate,
            ),
        )
        generator.generate("random", cfg.steps_per_seed)
        rule_lookup = {item.name: item for item in generator.store}
        generated = [
            theorem for theorem in generator.store.generated()
            if theorem.proof is not None and theorem.id is not None
        ]
        selected = sorted(
            generated,
            key=lambda item: (item.proof_depth, item.id or -1),
            reverse=True,
        )[: cfg.verify_deepest_per_seed]
        for theorem in selected:
            verified_declarations += _independently_verify(
                source, generator, theorem
            )
            verified_certificates += 1

        for theorem in generated:
            assert theorem.id is not None and theorem.proof is not None
            skeleton = _proof_skeleton(theorem.id, generator)
            split = _split(skeleton, cfg)
            record = theorem_record(theorem, generator.store)
            record.update({
                "run_seed": seed,
                "split": split,
                "proof_depth": theorem.proof_depth,
                "proof_skeleton_sha256": hashlib.sha256(
                    repr(skeleton).encode("utf-8")
                ).hexdigest(),
                "independent_certificate_family_verified": theorem in selected,
                "generation_kind": generator.generation_context.get(theorem.id, "search"),
                "guidance_target": generator.guidance_targets.get(theorem.id),
                "definition_support": sorted(generator.definition_support.get(theorem.id, ())),
            })
            nodes.append(record)

            canonical = tokenizer.canonical_variables(theorem)
            state_tokens = tokenizer.state_tokens(theorem, canonical)
            record["state_sha256"] = hashlib.sha256(repr(state_tokens).encode()).hexdigest()
            shared = {
                key: record[key] for key in (
                    "generation_kind", "guidance_target", "definition_support",
                    "proof_skeleton_sha256", "state_sha256",
                )
            }
            shared["tokenizer_sha256"] = fingerprint
            try:
                action_tokens = tokenizer.action_tokens(
                    theorem, database, canonical, rule_lookup=rule_lookup
                )
            except ValueError as exc:
                filtered["policy_encoding"] += 1
                if len(encoding_failures) < 20:
                    encoding_failures.append({"seed": seed, "theorem": theorem.name, "error": str(exc)})
                continue
            if (
                len(state_tokens) <= cfg.max_state_tokens
                and len(action_tokens) <= cfg.max_action_tokens
            ):
                policies[split].append({
                    **shared,
                    "run_seed": seed,
                    "theorem_id": theorem.id,
                    "theorem_name": theorem.name,
                    "proof_depth": theorem.proof_depth,
                    "state_tokens": state_tokens,
                    "action_tokens": action_tokens,
                    "state_ids": tokenizer.encode(state_tokens),
                    "action_ids": tokenizer.encode(action_tokens),
                    "value_target": 1.0 / (1.0 + theorem.proof_depth),
                    "proof_skeleton_sha256": record[
                        "proof_skeleton_sha256"
                    ],
                })
                replay_record(policies[split][-1], tokenizer, database, environment)
            else:
                filtered["policy_length"] += 1

            if len(state_tokens) > cfg.max_state_tokens:
                filtered["lemma_state_length"] += 1
                continue

            for premise_index, parent_id in enumerate(
                theorem.proof.premise_map
            ):
                if parent_id is None:
                    continue
                parent = generator.store[parent_id]
                lemma = _instantiated_parent(
                    theorem, premise_index, parent
                )
                if lemma.op != "|-" or lemma == theorem.conclusion:
                    continue
                try:
                    lemma_tokens = tokenizer.lemma_tactic_tokens(
                        lemma, canonical
                    )
                except ValueError:
                    filtered["lemma_encoding"] += 1
                    continue
                if len(lemma_tokens) > cfg.max_action_tokens:
                    filtered["lemma_action_length"] += 1
                    continue
                lemma_record = {
                    **shared,
                    "run_seed": seed,
                    "root_theorem_id": theorem.id,
                    "root_theorem_name": theorem.name,
                    "premise_index": premise_index,
                    "parent_theorem_id": parent_id,
                    "parent_rule": (
                        parent.proof.rule if parent.proof else parent.name
                    ),
                    "lemma": lemma.to_prefix(),
                    "state_tokens": state_tokens,
                    "state_ids": tokenizer.encode(state_tokens),
                    "lemma_action_tokens": lemma_tokens,
                    "lemma_action_ids": tokenizer.encode(lemma_tokens),
                    "root_proof_depth": theorem.proof_depth,
                    "parent_proof_depth": parent.proof_depth,
                    "utility_target": max(
                        0.0,
                        (theorem.proof_depth - parent.proof_depth)
                        / max(1, theorem.proof_depth),
                    ),
                }
                # Parent dependencies may introduce fresh variables, exceed the
                # kernel cut limit, or already be hypotheses. Never train on
                # a proposal which the actual inference environment rejects.
                try:
                    replay_record(lemma_record, tokenizer, database, environment, lemma=True)
                except ValueError:
                    filtered["lemma_kernel_rejected"] += 1
                    continue
                lemmas[split].append(lemma_record)

        summary = generator.summary()
        run_summaries.append({
            **asdict(summary),
            "seed": seed,
            "attempts": summary.attempts,
            "stored": summary.stored,
            "active": summary.active,
            "categories": summary.categories,
            "rejected": summary.rejected,
            "rule_usage": summary.rule_usage,
            "maximum_proof_depth": max(
                (item.proof_depth for item in generated), default=0
            ),
        })

    all_policies = [record for rows in policies.values() for record in rows]
    all_lemmas = [record for rows in lemmas.values() for record in rows]
    _group_splits(nodes, all_policies, all_lemmas, cfg)
    policy_count, lemma_count = len(all_policies), len(all_lemmas)
    all_policies = _deduplicate(all_policies, "action_tokens")
    all_lemmas = _deduplicate(all_lemmas, "lemma_action_tokens")
    filtered["duplicate_policy"] = policy_count - len(all_policies)
    filtered["duplicate_lemma"] = lemma_count - len(all_lemmas)
    _write_jsonl(output / "proof_dag_nodes.jsonl", nodes)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        policies[split] = [r for r in all_policies if r["split"] == split]
        lemmas[split] = [r for r in all_lemmas if r["split"] == split]
        _write_jsonl(output / f"policy_{split}.jsonl", policies[split])
        _write_jsonl(output / f"lemma_{split}.jsonl", lemmas[split])
        counts[split] = {
            "policy": len(policies[split]),
            "lemma": len(lemmas[split]),
        }
    manifest = {
        "format": "peano-htps-forward-dag-v2",
        "database": str(source.resolve()),
        "base_tokenizer": (
            str(Path(base_tokenizer_path).resolve())
            if base_tokenizer_path is not None else None
        ),
        "configuration": asdict(cfg),
        "vocabulary_size": len(tokenizer),
        "tokenizer_sha256": fingerprint,
        "environment": environment.configuration_record(),
        "filtered": dict(filtered),
        "encoding_failure_examples": encoding_failures,
        "generation_kinds": {
            "policy": dict(Counter(r["generation_kind"] for r in all_policies)),
            "lemma": dict(Counter(r["generation_kind"] for r in all_lemmas)),
        },
        "proof_dag_nodes": len(nodes),
        "counts": counts,
        "independent_verification": {
            "certificate_families": verified_certificates,
            "replayed_generated_declarations": verified_declarations,
            "method": "export Metamath, reparse expanded source, replay every $p",
        },
        "split_policy": "connected components of recursive proof skeleton and identical root state",
        "runs": run_summaries,
    }
    audit = audit_forward_dataset(source, output, output / "action-audit.json")
    if audit["invalid_actions"]:
        raise ValueError(f"HTPS action audit failed: {audit['failures'][:3]}")
    manifest["action_audit"] = {k: audit[k] for k in ("valid_actions", "invalid_actions", "scope")}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

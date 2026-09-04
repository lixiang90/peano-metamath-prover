from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from metamath_generator.export import export_metamath, theorem_record
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.model import Node, Theorem
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import verify
from neural_prover.tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class ForwardDAGConfig:
    """Configuration for certified forward proof-DAG generation."""

    seeds: tuple[int, ...] = (7, 11, 19, 23)
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

    def validate(self) -> None:
        if not self.seeds or self.steps_per_seed <= 0:
            raise ValueError("at least one seed and positive steps are required")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions must sum to < 1")


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
    source = Path(database_path)
    database = parse(source)
    tokenizer = (
        MetamathTokenizer.load(base_tokenizer_path)
        if base_tokenizer_path is not None
        else MetamathTokenizer.from_database(database)
    ).upgraded_for_lemma_actions()
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
            ),
        )
        generator.generate("random", cfg.steps_per_seed)
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
            })
            nodes.append(record)

            canonical = tokenizer.canonical_variables(theorem)
            state_tokens = tokenizer.state_tokens(theorem, canonical)
            try:
                action_tokens = tokenizer.action_tokens(
                    theorem, database, canonical
                )
            except ValueError:
                continue
            if (
                len(state_tokens) <= cfg.max_state_tokens
                and len(action_tokens) <= cfg.max_action_tokens
            ):
                policies[split].append({
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
                    continue
                if len(lemma_tokens) > cfg.max_action_tokens:
                    continue
                lemmas[split].append({
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
                })

        summary = generator.summary()
        run_summaries.append({
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

    _write_jsonl(output / "proof_dag_nodes.jsonl", nodes)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        _write_jsonl(output / f"policy_{split}.jsonl", policies[split])
        _write_jsonl(output / f"lemma_{split}.jsonl", lemmas[split])
        counts[split] = {
            "policy": len(policies[split]),
            "lemma": len(lemmas[split]),
        }
    manifest = {
        "format": "peano-htps-forward-dag-v1",
        "database": str(source.resolve()),
        "base_tokenizer": (
            str(Path(base_tokenizer_path).resolve())
            if base_tokenizer_path is not None else None
        ),
        "configuration": asdict(cfg),
        "vocabulary_size": len(tokenizer),
        "proof_dag_nodes": len(nodes),
        "counts": counts,
        "independent_verification": {
            "certificate_families": verified_certificates,
            "replayed_generated_declarations": verified_declarations,
            "method": "export Metamath, reparse expanded source, replay every $p",
        },
        "split_policy": "SHA-256 of recursive proof skeleton",
        "runs": run_summaries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

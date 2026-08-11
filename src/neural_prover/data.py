from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from metamath_generator.generator import (
    GenerationConfig,
    TheoremGenerator,
)
from metamath_generator.model import Node, Theorem
from metamath_generator.parser import parse
from metamath_generator.quality import semantic_profile

from .tokenizer import MetamathTokenizer, TokenizerConfig

Split = Literal["train", "validation", "test"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True, slots=True)
class CorpusBuildConfig:
    seeds: tuple[int, ...] = (7, 11, 19, 23)
    steps_per_seed: int = 5_000
    max_proof_depth: int = 7
    max_ast_depth: int = 32
    max_hypotheses: int = 8
    max_variables: int = 16
    full_discharge_probability: float = 0.72
    closed_parent_probability: float = 0.78
    max_proof_states_per_conclusion: int = 3
    depth_parent_bias: float = 0.0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    include_inference_rules: bool = True
    include_source_actions: bool = True
    max_state_tokens: int = 256
    max_action_tokens: int = 192
    tokenizer: TokenizerConfig = TokenizerConfig()


@dataclass(frozen=True, slots=True)
class ProverExample:
    example_id: str
    split: Split
    origin: Literal["synthetic", "source"]
    difficulty: Difficulty
    theorem_name: str
    conclusion: str
    hypotheses: tuple[str, ...]
    rule: str
    proof_depth: int
    quality_score: float
    value_target: float
    state_tokens: tuple[str, ...]
    action_tokens: tuple[str, ...]
    state_ids: tuple[int, ...]
    action_ids: tuple[int, ...]

    def to_record(self) -> dict:
        record = asdict(self)
        for key in (
            "hypotheses",
            "state_tokens",
            "action_tokens",
            "state_ids",
            "action_ids",
        ):
            record[key] = list(record[key])
        return record

    @classmethod
    def from_record(cls, record: dict) -> "ProverExample":
        converted = dict(record)
        converted.setdefault("origin", "synthetic")
        for key in (
            "hypotheses",
            "state_tokens",
            "action_tokens",
            "state_ids",
            "action_ids",
        ):
            converted[key] = tuple(converted[key])
        return cls(**converted)


def _key_bytes(theorem: Theorem) -> bytes:
    profile = semantic_profile(theorem)
    return repr(profile.full_key).encode("utf-8")


def _conclusion_bytes(theorem: Theorem) -> bytes:
    profile = semantic_profile(theorem)
    return repr(profile.conclusion_key).encode("utf-8")


def _split_for(
    theorem: Theorem,
    validation_fraction: float,
    test_fraction: float,
) -> Split:
    digest = hashlib.sha256(_conclusion_bytes(theorem)).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _difficulty(theorem: Theorem) -> Difficulty:
    node_count = sum(
        1
        for expression in [
            *(hypothesis.expr for hypothesis in theorem.hypotheses),
            theorem.conclusion,
        ]
        for _ in expression.walk()
    )
    if (
        theorem.proof_depth <= 2
        and node_count <= 40
        and len(theorem.hypotheses) <= 1
    ):
        return "easy"
    if (
        theorem.proof_depth <= 4
        and node_count <= 120
        and len(theorem.hypotheses) <= 3
    ):
        return "medium"
    return "hard"


def _example_id(theorem: Theorem) -> str:
    return hashlib.sha256(_key_bytes(theorem)).hexdigest()[:20]


def _examples_from_generator(
    generator: TheoremGenerator,
    tokenizer: MetamathTokenizer,
    config: CorpusBuildConfig,
) -> Iterable[ProverExample]:
    categories = ["closed_theorems"]
    if config.include_inference_rules:
        categories.append("inference_rules")
    for category in categories:
        for theorem in generator.ranked(category):
            if theorem.proof is None or theorem.id is None:
                continue
            canonical = tokenizer.canonical_variables(theorem)
            try:
                state_tokens = tuple(
                    tokenizer.state_tokens(theorem, canonical)
                )
                action_tokens = tuple(
                    tokenizer.action_tokens(
                        theorem,
                        generator.parsed,
                        canonical,
                    )
                )
            except ValueError:
                continue
            if (
                len(state_tokens) > config.max_state_tokens
                or len(action_tokens) > config.max_action_tokens
            ):
                continue
            assessment = generator.assessments[theorem.id]
            yield ProverExample(
                example_id=_example_id(theorem),
                split=_split_for(
                    theorem,
                    config.validation_fraction,
                    config.test_fraction,
                ),
                origin="synthetic",
                difficulty=_difficulty(theorem),
                theorem_name=theorem.name,
                conclusion=theorem.conclusion.to_prefix(),
                hypotheses=tuple(
                    hypothesis.expr.to_prefix()
                    for hypothesis in theorem.hypotheses
                ),
                rule=theorem.proof.rule,
                proof_depth=theorem.proof_depth,
                quality_score=assessment.score,
                # A bounded monotonic proxy for AlphaProof's remaining-return
                # value.  RL iterations can replace it with search returns.
                value_target=1.0 / (1.0 + theorem.proof_depth),
                state_tokens=state_tokens,
                action_tokens=action_tokens,
                state_ids=tuple(tokenizer.encode(state_tokens)),
                action_ids=tuple(tokenizer.encode(action_tokens)),
            )


def _source_examples(
    database,
    tokenizer: MetamathTokenizer,
    config: CorpusBuildConfig,
) -> Iterable[ProverExample]:
    for theorem in database.logical_assertions.values():
        canonical = tokenizer.canonical_variables(theorem)
        state_tokens = tuple(tokenizer.state_tokens(theorem, canonical))
        floating_order = [
            floating.expr.args[0].op for floating in theorem.floating
        ] or list(theorem.variable_types)
        substitution = {
            variable: Node(variable)
            for variable in floating_order
        }
        action_tokens = tuple(tokenizer.tactic_tokens(
            theorem.name,
            substitution,
            canonical,
            variable_order=floating_order,
        ))
        if (
            len(state_tokens) > config.max_state_tokens
            or len(action_tokens) > config.max_action_tokens
        ):
            continue
        digest = hashlib.sha256(
            f"source:{theorem.name}".encode("utf-8")
        ).hexdigest()[:20]
        yield ProverExample(
            example_id=f"source_{digest}",
            split="train",
            origin="source",
            difficulty=_difficulty(theorem),
            theorem_name=theorem.name,
            conclusion=theorem.conclusion.to_prefix(),
            hypotheses=tuple(
                hypothesis.expr.to_prefix()
                for hypothesis in theorem.hypotheses
            ),
            rule=theorem.name,
            proof_depth=1,
            quality_score=100.0,
            value_target=0.5,
            state_tokens=state_tokens,
            action_tokens=action_tokens,
            state_ids=tuple(tokenizer.encode(state_tokens)),
            action_ids=tuple(tokenizer.encode(action_tokens)),
        )


def _write_jsonl(path: Path, examples: Iterable[ProverExample]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(
                json.dumps(example.to_record(), ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def build_corpus(
    database_path: str | Path,
    output_directory: str | Path,
    config: CorpusBuildConfig | None = None,
) -> dict:
    """Generate, deduplicate, split, and serialize proof-policy data."""

    cfg = config or CorpusBuildConfig()
    if cfg.validation_fraction + cfg.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to < 1")
    database = parse(database_path)
    tokenizer = MetamathTokenizer.from_database(database, cfg.tokenizer)
    best: dict[str, ProverExample] = {}
    run_summaries: list[dict] = []
    for seed in cfg.seeds:
        generator = TheoremGenerator(
            database,
            GenerationConfig(
                seed=seed,
                max_proof_depth=cfg.max_proof_depth,
                max_ast_depth=cfg.max_ast_depth,
                max_hypotheses=cfg.max_hypotheses,
                max_variables=cfg.max_variables,
                full_discharge_probability=
                    cfg.full_discharge_probability,
                closed_parent_probability=
                    cfg.closed_parent_probability,
                max_proof_states_per_conclusion=
                    cfg.max_proof_states_per_conclusion,
                depth_parent_bias=cfg.depth_parent_bias,
            ),
        )
        generator.generate("random", cfg.steps_per_seed)
        summary = generator.summary()
        generated = [
            theorem
            for theorem in generator.store.generated()
            if theorem.proof is not None
        ]
        depth_histogram: dict[str, int] = {}
        for theorem in generated:
            depth = str(theorem.proof_depth)
            depth_histogram[depth] = depth_histogram.get(depth, 0) + 1
        run_summaries.append({
            "seed": seed,
            "stored": summary.stored,
            "active": summary.active,
            "categories": summary.categories,
            "rejected": summary.rejected,
            "maximum_proof_depth": max(
                (theorem.proof_depth for theorem in generated),
                default=0,
            ),
            "proof_depth_histogram": dict(sorted(
                depth_histogram.items(), key=lambda item: int(item[0])
            )),
        })
        for example in _examples_from_generator(
            generator,
            tokenizer,
            cfg,
        ):
            old = best.get(example.example_id)
            if old is None or (
                example.proof_depth,
                -example.quality_score,
            ) < (
                old.proof_depth,
                -old.quality_score,
            ):
                best[example.example_id] = example
    if cfg.include_source_actions:
        for example in _source_examples(database, tokenizer, cfg):
            best[example.example_id] = example

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    counts: dict[str, int] = {}
    split_depth_histograms: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        examples = sorted(
            (
                example for example in best.values()
                if example.split == split
            ),
            key=lambda item: (
                item.difficulty,
                item.proof_depth,
                item.example_id,
            ),
        )
        counts[split] = _write_jsonl(output / f"{split}.jsonl", examples)
        depth_histogram: dict[str, int] = {}
        for example in examples:
            depth = str(example.proof_depth)
            depth_histogram[depth] = depth_histogram.get(depth, 0) + 1
        split_depth_histograms[split] = dict(sorted(
            depth_histogram.items(), key=lambda item: int(item[0])
        ))

    manifest = {
        "format": "peano-neural-corpus-v2",
        "database": str(Path(database_path).resolve()),
        "configuration": {
            **asdict(cfg),
            "tokenizer": asdict(cfg.tokenizer),
        },
        "vocabulary_size": len(tokenizer),
        "counts": counts,
        "total": sum(counts.values()),
        "proof_depth_histogram": split_depth_histograms,
        "maximum_proof_depth": max(
            (
                example.proof_depth
                for example in best.values()
            ),
            default=0,
        ),
        "runs": run_summaries,
        "leakage_control": (
            "split is assigned by SHA-256 of the alpha-normalized conclusion; "
            "all premise variants of one conclusion stay in the same split"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_examples(path: str | Path) -> list[ProverExample]:
    return [
        ProverExample.from_record(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

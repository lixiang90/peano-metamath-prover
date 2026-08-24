from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from metamath_generator.generator import (
    GenerationConfig,
    TheoremGenerator,
)
from metamath_generator.model import Database, Hypothesis, Node, Theorem
from metamath_generator.parser import MetamathParser, ParseError, parse
from metamath_generator.quality import semantic_profile

from .data import load_examples
from .tokenizer import MetamathTokenizer

BenchmarkDifficulty = Literal["easy", "medium", "hard", "frontier"]
BenchmarkOrigin = Literal[
    "synthetic", "foundational", "curated", "famous"
]
BenchmarkScoreGroup = Literal["research", "sanity", "frontier"]


@dataclass(frozen=True, slots=True)
class BenchmarkBuildConfig:
    seeds: tuple[int, ...] = (101, 103, 107)
    steps_per_seed: int = 5_000
    max_proof_depth: int = 7
    synthetic_per_difficulty: int = 20
    max_synthetic_nodes: int = 180
    max_ast_depth: int = 64
    max_hypotheses: int = 12
    max_variables: int = 24
    full_discharge_probability: float = 0.8
    closed_parent_probability: float = 0.45
    max_proof_states_per_conclusion: int = 5
    depth_parent_bias: float = 0.0
    training_corpora: tuple[str, ...] = ()
    reference_output: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    title: str
    origin: BenchmarkOrigin
    difficulty: BenchmarkDifficulty
    expected_status: str
    hypotheses: tuple[str, ...]
    conclusion: str
    d_constraints: tuple[tuple[str, str], ...]
    variable_types: tuple[tuple[str, str], ...]
    reference_rule: str | None = None
    proof_depth: int = 0
    node_count: int = 0
    score_group: BenchmarkScoreGroup = "research"
    research_eligible: bool = True
    excluded_labels: tuple[str, ...] = ()
    leakage_reason: str | None = None

    def theorem(self, database: Database) -> Theorem:
        parser = MetamathParser()
        parser.database = database
        old_types = dict(database.variable_types)
        types = dict(self.variable_types)
        database.variable_types.update(types)
        try:
            hypotheses = [
                Hypothesis(
                    f"{self.case_id}_h{index}",
                    parser.parse_expression(expression.split()),
                )
                for index, expression in enumerate(self.hypotheses)
            ]
            conclusion = parser.parse_expression(self.conclusion.split())
        except ParseError as exc:
            raise ValueError(
                f"invalid benchmark {self.case_id}: {exc}"
            ) from exc
        finally:
            database.variable_types.clear()
            database.variable_types.update(old_types)
        return Theorem(
            self.case_id,
            hypotheses,
            conclusion,
            d_constraints=set(self.d_constraints),
            variable_types=types,
            kind="benchmark",
        )

    def to_record(self, *, include_reference: bool = False) -> dict:
        payload = asdict(self)
        if not include_reference:
            payload.pop("reference_rule", None)
        for key in (
            "hypotheses",
            "d_constraints",
            "variable_types",
            "excluded_labels",
        ):
            payload[key] = [list(item) if isinstance(item, tuple) else item
                            for item in payload[key]]
        return payload

    @classmethod
    def from_record(cls, payload: dict) -> "BenchmarkCase":
        record = dict(payload)
        record.setdefault("proof_depth", 0)
        record.setdefault("node_count", 0)
        record.setdefault("score_group", "research")
        record.setdefault("research_eligible", True)
        record.setdefault("excluded_labels", ())
        record.setdefault("leakage_reason", None)
        record.setdefault("reference_rule", None)
        record["hypotheses"] = tuple(record["hypotheses"])
        record["d_constraints"] = tuple(
            tuple(pair) for pair in record["d_constraints"]
        )
        record["variable_types"] = tuple(
            tuple(pair) for pair in record["variable_types"]
        )
        record["excluded_labels"] = tuple(record["excluded_labels"])
        return cls(**record)


@dataclass(slots=True)
class _LeakageIndex:
    full_keys: set[tuple]
    conclusion_keys: set[tuple]
    examples: int = 0
    sources: tuple[str, ...] = ()


def _corpus_directories(paths: tuple[str, ...]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    pending = [Path(item).resolve() for item in paths]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base = manifest.get("base_corpus")
            if base:
                pending.append(Path(base).resolve())
            if (path / "train.jsonl").is_file():
                result.append(path)
        elif path.is_file() and path.name == "train.jsonl":
            result.append(path.parent)
        else:
            raise ValueError(f"unsupported training corpus path: {path}")
    return result


def _training_leakage_index(
    paths: tuple[str, ...],
    database,
) -> _LeakageIndex:
    index = _LeakageIndex(set(), set(), sources=tuple(paths))
    for directory in _corpus_directories(paths):
        tokenizer = MetamathTokenizer.load(directory / "tokenizer.json")
        for example in load_examples(directory / "train.jsonl"):
            theorem = tokenizer.theorem_from_state_tokens(
                example.state_tokens,
                database,
                name=f"leakage_{example.example_id}",
            )
            profile = semantic_profile(theorem)
            index.full_keys.add(profile.full_key)
            index.conclusion_keys.add(profile.conclusion_key)
            index.examples += 1
    return index


def _equivalent_source_labels(theorem: Theorem, database) -> tuple[str, ...]:
    target = semantic_profile(theorem).conclusion_key
    return tuple(sorted(
        label
        for label, assertion in database.logical_assertions.items()
        if semantic_profile(assertion).conclusion_key == target
    ))


def _apply_leakage_policy(
    case: BenchmarkCase,
    theorem: Theorem,
    leakage: _LeakageIndex,
    database,
) -> BenchmarkCase:
    profile = semantic_profile(theorem)
    reason = None
    if profile.full_key in leakage.full_keys:
        reason = "exact normalized theorem occurs in training"
    elif profile.conclusion_key in leakage.conclusion_keys:
        reason = "alpha-normalized conclusion occurs in training"
    return replace(
        case,
        excluded_labels=_equivalent_source_labels(theorem, database),
        research_eligible=(
            case.score_group == "research" and reason is None
        ),
        leakage_reason=reason,
    )


def _difficulty(theorem: Theorem) -> BenchmarkDifficulty:
    nodes = _node_count(theorem)
    if (
        theorem.proof_depth <= 2
        and nodes <= 40
        and len(theorem.hypotheses) <= 1
    ):
        return "easy"
    if (
        theorem.proof_depth <= 4
        and nodes <= 120
        and len(theorem.hypotheses) <= 3
    ):
        return "medium"
    return "hard"


def _node_count(theorem: Theorem) -> int:
    return sum(
        1
        for expression in [
            *(hypothesis.expr for hypothesis in theorem.hypotheses),
            theorem.conclusion,
        ]
        for _ in expression.walk()
    )


def _case_id(prefix: str, theorem: Theorem) -> str:
    key = repr(semantic_profile(theorem).full_key).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(key).hexdigest()[:16]}"


def _depth_stratified(
    theorems: list[Theorem],
    limit: int,
) -> list[Theorem]:
    """Select across reference depths, preferring deeper cases first."""

    by_depth: dict[int, list[Theorem]] = {}
    for theorem in theorems:
        by_depth.setdefault(theorem.proof_depth, []).append(theorem)
    for bucket in by_depth.values():
        bucket.sort(key=lambda theorem: (_node_count(theorem), theorem.name))
    selected: list[Theorem] = []
    depths = sorted(by_depth, reverse=True)
    while len(selected) < limit and depths:
        remaining: list[int] = []
        for depth in depths:
            bucket = by_depth[depth]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
            if bucket:
                remaining.append(depth)
        depths = remaining
    return selected


def _from_theorem(
    theorem: Theorem,
    title: str,
    origin: BenchmarkOrigin,
    difficulty: BenchmarkDifficulty,
    expected_status: str,
    *,
    score_group: BenchmarkScoreGroup = "research",
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=_case_id(origin, theorem),
        title=title,
        origin=origin,
        difficulty=difficulty,
        expected_status=expected_status,
        hypotheses=tuple(
            hypothesis.expr.to_prefix()
            for hypothesis in theorem.hypotheses
        ),
        conclusion=theorem.conclusion.to_prefix(),
        d_constraints=tuple(sorted(theorem.d_constraints)),
        variable_types=tuple(sorted(theorem.variable_types.items())),
        reference_rule=(
            theorem.proof.rule if theorem.proof is not None else theorem.name
        ),
        proof_depth=theorem.proof_depth,
        node_count=_node_count(theorem),
        score_group=score_group,
        research_eligible=score_group == "research",
    )


FOUNDATIONAL = {
    "eq-refl": ("等式自反性", "easy"),
    "eq-sym": ("等式对称性", "easy"),
    "eq-trans": ("等式传递性", "medium"),
    "pa_ax2": ("后继函数的单射性", "medium"),
    "pa_ax4": ("加法的后继递归式", "medium"),
    "pa_ax6": ("乘法的后继递归式", "hard"),
    "induction": ("皮亚诺归纳原理", "hard"),
}

FAMOUS = {
    "flt-statement": (
        "费马大定理",
        "known theorem; formal proof absent from this library",
    ),
    "goldbach-statement": (
        "哥德巴赫猜想",
        "open conjecture",
    ),
    "pnt-statement": (
        "素数定理（有理数不等式证书形式）",
        "known theorem; formal proof absent from this library",
    ),
    "riemann-von-koch-statement": (
        "黎曼猜想的 von Koch 等价界",
        "open conjecture",
    ),
}


# Human-selected, standard logical/arithmetic results rather than random
# formulas.  They are consequences of the base axioms, not new axioms.
CURATED = (
    (
        "identity-law",
        "命题恒等律",
        "easy",
        "|- implies phi phi",
        {"phi": "wff"},
    ),
    (
        "addition-right-zero",
        "自然数加法右零元",
        "easy",
        "|- = + x 0 x",
        {"x": "var"},
    ),
    (
        "multiplication-right-zero",
        "自然数乘法右零元",
        "easy",
        "|- = * x 0 0",
        {"x": "var"},
    ),
    (
        "addition-successor-reverse",
        "加法后继递归式（反向）",
        "medium",
        "|- = + x S y S + x y",
        {"x": "var", "y": "var"},
    ),
    (
        "multiplication-successor-reverse",
        "乘法后继递归式（反向）",
        "medium",
        "|- = * x S y + * x y x",
        {"x": "var", "y": "var"},
    ),
    (
        "successor-equality-iff",
        "后继相等当且仅当原数相等",
        "hard",
        "|- iff = S x S y = x y",
        {"x": "var", "y": "var"},
    ),
)


def _curated_case(
    label: str,
    title: str,
    difficulty: BenchmarkDifficulty,
    expression: str,
    variable_types: dict[str, str],
    database: Database,
) -> BenchmarkCase:
    parser = MetamathParser()
    parser.database = database
    old_types = dict(database.variable_types)
    database.variable_types.update(variable_types)
    try:
        conclusion = parser.parse_expression(expression.split())
    except ParseError as exc:
        raise ValueError(
            f"invalid curated theorem {label}: {exc}"
        ) from exc
    finally:
        database.variable_types.clear()
        database.variable_types.update(old_types)
    theorem = Theorem(
        label,
        [],
        conclusion,
        variable_types=dict(variable_types),
        kind="benchmark",
    )
    return _from_theorem(
        theorem,
        title,
        "curated",
        difficulty,
        "known theorem; derivable from the base axioms",
        score_group="research",
    )


def _famous_case(
    label: str,
    title: str,
    status: str,
    database: Database,
) -> BenchmarkCase:
    statement = database.statements[label]
    if statement.conclusion.op != "statement":
        raise ValueError(f"{label} is not a named statement")
    target = Theorem(
        label,
        [],
        Node("|-", statement.conclusion.args),
        d_constraints=set(statement.d_constraints),
        variable_types=dict(statement.variable_types),
        kind="benchmark",
    )
    return _from_theorem(
        target,
        title,
        "famous",
        "frontier",
        status,
        score_group="frontier",
    )


def build_benchmarks(
    database_path: str | Path,
    destination: str | Path,
    config: BenchmarkBuildConfig | None = None,
) -> dict:
    cfg = config or BenchmarkBuildConfig()
    database = parse(database_path)
    leakage = _training_leakage_index(cfg.training_corpora, database)
    cases: dict[str, BenchmarkCase] = {}

    for label, (title, difficulty) in FOUNDATIONAL.items():
        theorem = database.logical_assertions[label]
        case = _from_theorem(
            theorem,
            title,
            "foundational",
            difficulty,  # type: ignore[arg-type]
            "source sanity check",
            score_group="sanity",
        )
        case = replace(case, research_eligible=False)
        cases[case.case_id] = case

    for (
        label,
        title,
        difficulty,
        expression,
        variable_types,
    ) in CURATED:
        case = _curated_case(
            label,
            title,
            difficulty,  # type: ignore[arg-type]
            expression,
            variable_types,
            database,
        )
        case = _apply_leakage_policy(
            case, case.theorem(database), leakage, database
        )
        cases[case.case_id] = case

    buckets: dict[str, list[Theorem]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    seen: set[tuple] = set()
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
        for category in ("closed_theorems", "inference_rules"):
            for theorem in generator.ranked(category):
                profile = semantic_profile(theorem)
                if profile.full_key in seen:
                    continue
                if _node_count(theorem) > cfg.max_synthetic_nodes:
                    continue
                seen.add(profile.full_key)
                buckets[_difficulty(theorem)].append(theorem)
    for difficulty, theorems in buckets.items():
        selected = _depth_stratified(
            theorems, cfg.synthetic_per_difficulty
        )
        for index, theorem in enumerate(selected):
            case = _from_theorem(
                theorem,
                f"合成 {difficulty} #{index + 1}",
                "synthetic",
                difficulty,  # type: ignore[arg-type]
                "verified synthetic theorem",
            )
            case = _apply_leakage_policy(
                case, theorem, leakage, database
            )
            if not case.research_eligible:
                continue
            cases[case.case_id] = case

    for label, (title, status) in FAMOUS.items():
        case = _famous_case(label, title, status, database)
        cases[case.case_id] = case

    ordered = sorted(
        cases.values(),
        key=lambda case: (
            ("easy", "medium", "hard", "frontier").index(
                case.difficulty
            ),
            case.origin,
            case.case_id,
        ),
    )
    manifest = {
        "format": "peano-proof-benchmark-v2",
        "database": str(Path(database_path).resolve()),
        "configuration": asdict(cfg),
        "counts": {
            difficulty: sum(
                case.difficulty == difficulty for case in ordered
            )
            for difficulty in ("easy", "medium", "hard", "frontier")
        },
        "score_group_counts": {
            group: sum(case.score_group == group for case in ordered)
            for group in ("research", "sanity", "frontier")
        },
        "research_eligible": sum(
            case.research_eligible for case in ordered
        ),
        "proof_depth_histogram": {
            str(depth): sum(
                case.proof_depth == depth for case in ordered
            )
            for depth in sorted({
                case.proof_depth for case in ordered
            })
        },
        "cases": [case.to_record() for case in ordered],
        "leakage_audit": {
            "training_corpora": list(cfg.training_corpora),
            "training_examples_indexed": leakage.examples,
            "policy": (
                "research cases exclude exact normalized theorem and "
                "alpha-normalized conclusion collisions"
            ),
        },
        "allowed_assertions": {
            "labels": sorted(database.logical_assertions),
            "sha256": hashlib.sha256("\n".join(
                sorted(database.logical_assertions)
            ).encode("utf-8")).hexdigest(),
            "per_case_exclusions": True,
        },
        "interpretation": {
            "synthetic": "held-out verified proof objects",
            "foundational": "kernel and search sanity checks",
            "curated": (
                "human-selected standard logical and arithmetic theorems "
                "that are consequences of the base axioms"
            ),
            "famous": (
                "formalization/frontier checks; failure is expected when "
                "the library contains no proof, and open conjectures are "
                "never counted as supervised targets"
            ),
        },
    }
    Path(destination).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if cfg.reference_output:
        Path(cfg.reference_output).write_text(
            json.dumps({
                "format": "peano-proof-benchmark-reference-v1",
                "benchmark": str(Path(destination).resolve()),
                "references": {
                    case.case_id: {
                        "reference_rule": case.reference_rule,
                        "reference_proof_depth": case.proof_depth,
                    }
                    for case in ordered
                },
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def load_benchmarks(path: str | Path) -> list[BenchmarkCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "peano-proof-benchmark-v1",
        "peano-proof-benchmark-v2",
    }:
        raise ValueError("unsupported benchmark format")
    return [
        BenchmarkCase.from_record(record)
        for record in payload["cases"]
    ]

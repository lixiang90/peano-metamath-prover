from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from metamath_generator.generator import (
    GenerationConfig,
    TheoremGenerator,
)
from metamath_generator.model import Database, Hypothesis, Node, Theorem
from metamath_generator.parser import MetamathParser, ParseError, parse
from metamath_generator.quality import semantic_profile

BenchmarkDifficulty = Literal["easy", "medium", "hard", "frontier"]
BenchmarkOrigin = Literal[
    "synthetic", "foundational", "curated", "famous"
]


@dataclass(frozen=True, slots=True)
class BenchmarkBuildConfig:
    seeds: tuple[int, ...] = (101, 103, 107)
    steps_per_seed: int = 5_000
    max_proof_depth: int = 7
    synthetic_per_difficulty: int = 20
    max_synthetic_nodes: int = 180


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

    def to_record(self) -> dict:
        payload = asdict(self)
        for key in (
            "hypotheses",
            "d_constraints",
            "variable_types",
        ):
            payload[key] = [list(item) if isinstance(item, tuple) else item
                            for item in payload[key]]
        return payload

    @classmethod
    def from_record(cls, payload: dict) -> "BenchmarkCase":
        record = dict(payload)
        record.setdefault("proof_depth", 0)
        record.setdefault("node_count", 0)
        record["hypotheses"] = tuple(record["hypotheses"])
        record["d_constraints"] = tuple(
            tuple(pair) for pair in record["d_constraints"]
        )
        record["variable_types"] = tuple(
            tuple(pair) for pair in record["variable_types"]
        )
        return cls(**record)


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


def _from_theorem(
    theorem: Theorem,
    title: str,
    origin: BenchmarkOrigin,
    difficulty: BenchmarkDifficulty,
    expected_status: str,
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
    )


def build_benchmarks(
    database_path: str | Path,
    destination: str | Path,
    config: BenchmarkBuildConfig | None = None,
) -> dict:
    cfg = config or BenchmarkBuildConfig()
    database = parse(database_path)
    cases: dict[str, BenchmarkCase] = {}

    for label, (title, difficulty) in FOUNDATIONAL.items():
        theorem = database.logical_assertions[label]
        case = _from_theorem(
            theorem,
            title,
            "foundational",
            difficulty,  # type: ignore[arg-type]
            "source sanity check",
        )
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
        theorems.sort(
            key=lambda theorem: (
                theorem.proof_depth,
                _node_count(theorem),
                theorem.name,
            )
        )
        for index, theorem in enumerate(
            theorems[:cfg.synthetic_per_difficulty]
        ):
            case = _from_theorem(
                theorem,
                f"合成 {difficulty} #{index + 1}",
                "synthetic",
                difficulty,  # type: ignore[arg-type]
                "verified synthetic theorem",
            )
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
        "cases": [case.to_record() for case in ordered],
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

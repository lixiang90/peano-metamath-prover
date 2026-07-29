from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .compose import CompositionError, compose
from .database import TheoremDatabase
from ._engine import Generator as _CompositionEngine
from ._engine import GeneratorConfig as _EngineConfig
from .model import Database, Node, Theorem
from .quality import (
    DatasetCategory,
    QualityAnalyzer,
    QualityAssessment,
    QualityConfig,
    SemanticProfile,
)

ALPHA_RULES = {"alpha_1", "alpha_2"}


@dataclass(slots=True)
class GenerationConfig(_EngineConfig):
    quality: QualityConfig = field(default_factory=QualityConfig)
    full_discharge_probability: float = 0.72
    closed_parent_probability: float = 0.78
    max_consecutive_alpha: int = 1
    mp_rule_weight: float = 4.0
    alpha_rule_weight: float = 0.22
    default_rule_weight: float = 1.0
    max_forward_candidates_per_premise: int = 30
    max_proof_states_per_conclusion: int = 3


@dataclass(slots=True)
class GenerationSummary:
    attempts: int
    stored: int
    active: int
    dominated: int
    categories: dict[str, int]
    rejected: dict[str, int]
    rule_usage: dict[str, int]


class TheoremGenerator(_CompositionEngine):
    """Quality-aware generator for certified theorem and proof states.

    The proof-bearing formula is immutable. Semantic normalization is used
    only for quality decisions, duplicate detection, subsumption, and
    reporting.
    """

    config: GenerationConfig

    def __init__(
        self,
        parsed: Database,
        config: GenerationConfig | None = None,
        store: TheoremDatabase | None = None,
    ) -> None:
        super().__init__(parsed, config or GenerationConfig(), store)
        self.analyzer = QualityAnalyzer(parsed, self.config.quality)
        self.assessments: dict[int, QualityAssessment] = {}
        self.profiles: dict[int, SemanticProfile] = {}
        self.source_ids: set[int] = set()
        self.active_ids: set[int] = set()
        self.dominated_ids: set[int] = set()
        self._semantic_to_id: dict[tuple, int] = {}
        self._by_conclusion: dict[tuple, set[int]] = defaultdict(set)
        self._alpha_chain_cache: dict[int, int] = {}
        self._candidate_use: Counter[int] = Counter()
        self._proof_states_by_conclusion: Counter[tuple] = Counter()
        self._new_ids: list[int] = []
        self.stats: Counter[str] = Counter()
        self.rejected: Counter[str] = Counter()
        self.rule_usage: Counter[str] = Counter()

        for theorem in self.store:
            if theorem.id is None:
                continue
            theorem_id = theorem.id
            assessment, profile = self.analyzer.assess(theorem)
            # Source assertions remain available regardless of a data-quality
            # score; quality filters govern generated objects.
            assessment.active = True
            assessment.hard_reject = False
            self.assessments[theorem_id] = assessment
            self.profiles[theorem_id] = profile
            self.source_ids.add(theorem_id)
            self.active_ids.add(theorem_id)
            self._semantic_to_id.setdefault(profile.full_key, theorem_id)
            self._by_conclusion[profile.conclusion_key].add(theorem_id)

    @staticmethod
    def _subsumes(left: SemanticProfile, right: SemanticProfile) -> bool:
        return (
            left.conclusion_key == right.conclusion_key
            and left.premise_keys <= right.premise_keys
            and left.d_constraints <= right.d_constraints
        )

    def _alpha_chain(self, theorem_id: int) -> int:
        if theorem_id in self._alpha_chain_cache:
            return self._alpha_chain_cache[theorem_id]
        theorem = self.store[theorem_id]
        if theorem.proof is None or theorem.proof.rule not in ALPHA_RULES:
            result = 0
        else:
            result = 1 + max(
                (self._alpha_chain(parent) for parent in theorem.proof.parents),
                default=0,
            )
        self._alpha_chain_cache[theorem_id] = result
        return result

    def _prospective_alpha_chain(self, theorem: Theorem) -> int:
        if theorem.proof is None or theorem.proof.rule not in ALPHA_RULES:
            return 0
        return 1 + max(
            (self._alpha_chain(parent) for parent in theorem.proof.parents),
            default=0,
        )

    def _wraps_known_closed_conclusion(self, theorem: Theorem) -> bool:
        """Detect A -> T wrappers when a stronger closed T is already known."""

        if theorem.d_constraints or len(theorem.conclusion.args) != 1:
            return False
        body = theorem.conclusion.args[0]
        while body.op == "implies" and len(body.args) == 2:
            body = body.args[1]
            suffix = Theorem(
                name="_quality_suffix",
                hypotheses=[],
                conclusion=Node("|-", (body,)),
                variable_types=theorem.variable_types,
            )
            key = self.analyzer.assess(suffix)[1].conclusion_key
            for theorem_id in self._by_conclusion.get(key, ()):
                known = self.store[theorem_id]
                if (
                    not known.hypotheses
                    and not known.d_constraints
                    and theorem_id not in self.dominated_ids
                ):
                    return True
        return False

    def _admit(self, theorem: Theorem) -> bool:
        assessment, profile = self.analyzer.assess(theorem)
        if assessment.hard_reject:
            if assessment.vacuous_quantifiers:
                self.rejected["vacuous_quantifier"] += 1
            else:
                self.rejected["trivial_conclusion"] += 1
            return False
        if self._wraps_known_closed_conclusion(theorem):
            self.rejected["known_consequence_weakening"] += 1
            return False
        if profile.full_key in self._semantic_to_id:
            self.rejected["semantic_duplicate"] += 1
            return False
        if (
            assessment.category == "proof_states"
            and self._proof_states_by_conclusion[profile.conclusion_key]
            >= self.config.max_proof_states_per_conclusion
        ):
            self.rejected["proof_state_quota"] += 1
            return False

        comparable = [
            theorem_id
            for theorem_id in self._by_conclusion.get(profile.conclusion_key, ())
            if theorem_id not in self.dominated_ids
        ]
        for theorem_id in comparable:
            if self._subsumes(self.profiles[theorem_id], profile):
                self.rejected["subsumed"] += 1
                return False

        theorem_id, added = self.store.add(theorem)
        if not added:
            self.rejected["exact_duplicate"] += 1
            return False
        stored = self.store[theorem_id]
        assessment, profile = self.analyzer.assess(stored)
        self.assessments[theorem_id] = assessment
        self.profiles[theorem_id] = profile
        self._semantic_to_id[profile.full_key] = theorem_id
        self._by_conclusion[profile.conclusion_key].add(theorem_id)
        self._new_ids.append(theorem_id)
        self.stats["stored"] += 1
        self.rule_usage[stored.proof.rule if stored.proof else stored.name] += 1

        for existing_id in comparable:
            if existing_id in self.source_ids:
                continue
            if self._subsumes(profile, self.profiles[existing_id]):
                self.dominated_ids.add(existing_id)
                old = self.assessments[existing_id]
                old.dominated = True
                old.active = False
                self.active_ids.discard(existing_id)
                self.stats["dominated"] += 1

        if assessment.active:
            self.active_ids.add(theorem_id)
            self.stats["active_admitted"] += 1
        else:
            self._proof_states_by_conclusion[profile.conclusion_key] += 1
            self.stats["proof_state_admitted"] += 1
        self.generated_count += 1
        return True

    def _try(self, rule: Theorem, matches: list[Theorem | None]) -> bool:
        self.stats["attempts"] += 1
        try:
            theorem = compose(
                rule,
                matches,
                name=f"gen{len(self.store)}_{rule.name}",
                database=self.parsed,
            )
        except CompositionError:
            self.rejected["composition_failed"] += 1
            return False
        if not self._valid(theorem):
            self.rejected["structural_limit_or_type"] += 1
            return False
        if (
            self._prospective_alpha_chain(theorem)
            > self.config.max_consecutive_alpha
        ):
            self.rejected["consecutive_alpha"] += 1
            return False
        return self._admit(theorem)

    def _active_candidates(self, conclusion: Node | None = None) -> list[Theorem]:
        result: list[Theorem] = []
        for theorem_id in self.active_ids:
            theorem = self.store[theorem_id]
            if conclusion is not None and theorem.conclusion.op != conclusion.op:
                continue
            result.append(theorem)
        return result

    def _rule_weight(self, rule: Theorem) -> float:
        if rule.name == "ax-mp":
            return self.config.mp_rule_weight
        if rule.name in ALPHA_RULES:
            return self.config.alpha_rule_weight
        return self.config.default_rule_weight

    def _choose_rule(self) -> Theorem:
        return self.random.choices(
            self.rules,
            weights=[self._rule_weight(rule) for rule in self.rules],
            k=1,
        )[0]

    def _candidate_weight(self, theorem: Theorem) -> float:
        theorem_id = theorem.id
        assessment = (
            self.assessments.get(theorem_id)
            if theorem_id is not None
            else None
        )
        closed_bonus = 7.0 if not theorem.hypotheses else 1.0
        quality_bonus = max(0.5, (assessment.score / 12.0) if assessment else 1.0)
        usage_key = theorem_id if theorem_id is not None else -1
        reuse_penalty = math.sqrt(1.0 + self._candidate_use[usage_key])
        return (closed_bonus + quality_bonus) / reuse_penalty

    def _choose_candidate(self, premise: Node) -> Theorem | None:
        candidates = self._active_candidates(premise)
        if not candidates:
            return None
        closed = [theorem for theorem in candidates if not theorem.hypotheses]
        if (
            closed
            and self.random.random() < self.config.closed_parent_probability
        ):
            candidates = closed
        chosen = self.random.choices(
            candidates,
            weights=[self._candidate_weight(item) for item in candidates],
            k=1,
        )[0]
        if chosen.id is not None:
            self._candidate_use[chosen.id] += 1
        return chosen

    def _match_indices(self, rule: Theorem) -> list[int]:
        count = len(rule.hypotheses)
        if count <= 1:
            return list(range(count))
        if self.random.random() < self.config.full_discharge_probability:
            return list(range(count))
        # With modus ponens, matching only the minor premise leaves an
        # unconstrained schematic conclusion.  Matching the implication
        # premise instead yields the useful derived rule A / B.
        if rule.name == "ax-mp" and count == 2:
            return [1]
        partial_count = self.random.randint(1, count - 1)
        return self.random.sample(range(count), partial_count)

    def random_walk(self, steps: int) -> list[Theorem]:
        start_index = len(self._new_ids)
        if not self.rules:
            return []
        for _ in range(steps):
            rule = self._choose_rule()
            matches: list[Theorem | None] = [None] * len(rule.hypotheses)
            indices = self._match_indices(rule)
            if len(indices) == len(matches):
                self.stats["full_discharge_plans"] += 1
            else:
                self.stats["partial_discharge_plans"] += 1
            for index in indices:
                matches[index] = self._choose_candidate(
                    rule.hypotheses[index].expr
                )
            if any(parent is not None for parent in matches):
                self._try(rule, matches)
            else:
                self.rejected["no_candidate"] += 1
        return [self.store[item] for item in self._new_ids[start_index:]]

    def forward_saturation(self, rounds: int = 1) -> list[Theorem]:
        start_index = len(self._new_ids)
        for _ in range(rounds):
            added_this_round = 0
            snapshot = [
                self.store[item]
                for item in sorted(self.active_ids)
            ]
            for rule in self.rules:
                choices: list[list[Theorem | None]] = []
                for premise in rule.hypotheses:
                    candidates = [
                        theorem
                        for theorem in snapshot
                        if theorem.conclusion.op == premise.expr.op
                    ]
                    candidates.sort(
                        key=self._candidate_weight,
                        reverse=True,
                    )
                    limited = candidates[
                        : self.config.max_forward_candidates_per_premise
                    ]
                    # Try fully matched products before partial variants.
                    choices.append([*limited, None])
                tried = 0
                for match_tuple in itertools.product(*choices):
                    if all(item is None for item in match_tuple):
                        continue
                    if tried >= self.config.max_combinations_per_rule:
                        break
                    tried += 1
                    if self._try(rule, list(match_tuple)):
                        added_this_round += 1
            if added_this_round == 0:
                break
        return [self.store[item] for item in self._new_ids[start_index:]]

    def generate(
        self,
        mode: Literal["random", "forward", "depth"] = "random",
        steps: int = 100,
    ) -> list[Theorem]:
        if mode == "forward":
            return self.forward_saturation(steps)
        return self.random_walk(steps)

    def categorized(
        self,
        *,
        include_dominated: bool = False,
    ) -> dict[DatasetCategory, list[Theorem]]:
        result: dict[DatasetCategory, list[Theorem]] = {
            "closed_theorems": [],
            "inference_rules": [],
            "proof_states": [],
        }
        for theorem_id in self._new_ids:
            if not include_dominated and theorem_id in self.dominated_ids:
                continue
            assessment = self.assessments[theorem_id]
            result[assessment.category].append(self.store[theorem_id])
        return result

    def ranked(
        self,
        category: DatasetCategory,
        *,
        include_dominated: bool = False,
    ) -> list[Theorem]:
        """Return one dataset category in descending quality order."""

        items = self.categorized(
            include_dominated=include_dominated
        )[category]
        return sorted(
            items,
            key=lambda theorem: (
                -self.assessments[theorem.id].score,  # type: ignore[index]
                len(theorem.hypotheses),
                theorem.proof_depth,
                theorem.conclusion.depth,
                theorem.id if theorem.id is not None else -1,
            ),
        )

    def summary(self) -> GenerationSummary:
        categories = {
            name: len(items)
            for name, items in self.categorized().items()
        }
        generated_active = len([
            theorem_id
            for theorem_id in self._new_ids
            if theorem_id in self.active_ids
        ])
        return GenerationSummary(
            attempts=self.stats["attempts"],
            stored=len(self._new_ids),
            active=generated_active,
            dominated=len(self.dominated_ids),
            categories=categories,
            rejected=dict(self.rejected),
            rule_usage=dict(self.rule_usage),
        )

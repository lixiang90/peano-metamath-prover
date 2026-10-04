from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .compose import CompositionError, compose, instantiate_assertion
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
    graph_instance_probability: float = 0.10
    graph_expression_depth: int = 2
    graph_match_candidates: int = 24
    graph_rule_bias: float = 0.5
    graph_depth_bias: float = 0.5
    graph_derived_rule_probability: float = 0.25
    quality: QualityConfig = field(default_factory=QualityConfig)
    full_discharge_probability: float = 0.72
    closed_parent_probability: float = 0.78
    max_consecutive_alpha: int = 1
    mp_rule_weight: float = 4.0
    alpha_rule_weight: float = 0.22
    default_rule_weight: float = 1.0
    max_forward_candidates_per_premise: int = 30
    max_proof_states_per_conclusion: int = 3
    # Zero preserves the quality-first sampler. Positive values increasingly
    # reuse deeper certified parents, enabling explicit depth-scaling runs.
    depth_parent_bias: float = 0.0
    # Explicit definitions are closed iff axioms, not inference rules.  In
    # guided runs, mechanically derive both implication directions through
    # bi1/bi2 and modus ponens so later search can unfold and fold them.
    bootstrap_definitions: bool = False
    focus_predicates: tuple[str, ...] = ()
    definition_coverage_weight: float = 0.0
    compatible_candidate_filter: bool = True
    max_joint_candidate_attempts: int = 8
    bounded_nat_max: int = -1
    ground_instances_per_predicate: int = 0
    target_statements: tuple[str, ...] = ()
    target_guidance_weight: float = 0.0
    max_definition_only_search_per_predicate: int = -1


@dataclass(slots=True)
class GenerationSummary:
    attempts: int
    stored: int
    active: int
    dominated: int
    categories: dict[str, int]
    rejected: dict[str, int]
    rule_usage: dict[str, int]
    definition_predicates_total: int
    definition_predicates_seen: int
    definition_coverage: float
    definition_usage: dict[str, int]
    definition_bridges: int
    bounded_ground_instances: int
    search_stored: int
    target_statements_total: int
    target_statements_touched: int
    target_mean_best_similarity: float
    target_best_similarity: dict[str, float]
    definition_only_search_admitted: int


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
        if self.config.definition_coverage_weight < 0:
            raise ValueError("definition_coverage_weight must be non-negative")
        if self.config.max_joint_candidate_attempts <= 0:
            raise ValueError("max_joint_candidate_attempts must be positive")
        if self.config.bounded_nat_max < -1:
            raise ValueError("bounded_nat_max must be at least -1")
        if self.config.ground_instances_per_predicate < 0:
            raise ValueError(
                "ground_instances_per_predicate must be non-negative"
            )
        if (
            self.config.ground_instances_per_predicate > 0
            and self.config.bounded_nat_max < 0
        ):
            raise ValueError(
                "bounded_nat_max must be non-negative when instances are enabled"
            )
        if self.config.target_guidance_weight < 0:
            raise ValueError("target_guidance_weight must be non-negative")
        if self.config.max_definition_only_search_per_predicate < -1:
            raise ValueError(
                "max_definition_only_search_per_predicate must be at least -1"
            )
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
        self._definition_sources = self._find_definition_sources(parsed)
        requested_focus = tuple(dict.fromkeys(self.config.focus_predicates))
        unknown_focus = set(requested_focus) - set(self._definition_sources)
        if unknown_focus:
            raise ValueError(
                "focus predicates are not explicit definitions: "
                f"{sorted(unknown_focus)}"
            )
        self.focus_predicates = (
            requested_focus
            if requested_focus
            else tuple(sorted(self._definition_sources))
        )
        self._focus_set = frozenset(self.focus_predicates)
        self._definition_usage: Counter[str] = Counter()
        self._definition_support: dict[int, frozenset[str]] = {}
        self._definition_only_search: Counter[str] = Counter()
        self._admission_context = "search"
        self.generation_context: dict[int, str] = {}
        self.guidance_targets: dict[int, str] = {}
        self._definitions_bootstrapped = False
        self._bounded_instances_bootstrapped = False
        self._definition_bridges: dict[str, list[int]] = defaultdict(list)
        self._bootstrap_ids: set[int] = set()
        self.target_formulas = self._load_target_formulas(
            parsed,
            self.config.target_statements,
        )
        self._target_signatures = {
            name: self._subformula_signatures(formula)
            for name, formula in self.target_formulas.items()
        }
        self._target_required_predicates = {
            name: frozenset(
                node.op for node in formula.walk()
                if node.op in self._focus_set
            )
            for name, formula in self.target_formulas.items()
        }
        self._target_selection: Counter[str] = Counter()
        self._target_best_similarity: dict[str, float] = {
            name: 0.0 for name in self.target_formulas
        }
        self._target_report_cache: dict[str, float] | None = None
        self._current_target: str | None = None
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
            support = frozenset(
                predicate
                for predicate, definition in self._definition_sources.items()
                if definition.name == theorem.name
            )
            self._definition_support[theorem_id] = support

    @staticmethod
    def _find_definition_sources(parsed: Database) -> dict[str, Theorem]:
        definitions: dict[str, Theorem] = {}
        for theorem in parsed.logical_assertions.values():
            if not theorem.name.startswith("df-") or theorem.hypotheses:
                continue
            conclusion = theorem.conclusion
            if conclusion.op != "|-" or len(conclusion.args) != 1:
                continue
            body = conclusion.args[0]
            if body.op != "iff" or len(body.args) != 2 or not body.args[0].args:
                continue
            predicate = body.args[0].op
            if theorem.name == f"df-{predicate}":
                definitions[predicate] = theorem
        return definitions

    @staticmethod
    def _load_target_formulas(
        parsed: Database,
        names: tuple[str, ...],
    ) -> dict[str, Node]:
        targets: dict[str, Node] = {}
        for name in dict.fromkeys(names):
            theorem = parsed.syntax_statements.get(name)
            if (
                theorem is None
                or theorem.conclusion.op != "statement"
                or len(theorem.conclusion.args) != 1
            ):
                raise ValueError(
                    f"target guidance requires non-logical statement {name!r}"
                )
            if name in parsed.logical_assertions:
                raise ValueError(f"target {name!r} is a logical assertion")
            targets[name] = theorem.conclusion.args[0]
        return targets

    def _fixed_symbols(self, node: Node) -> frozenset[str]:
        return frozenset(
            item.op for item in node.walk()
            if item.op not in self.parsed.variable_types
            and item.op not in {"|-", "statement"}
        )

    def _subformula_signatures(
        self,
        formula: Node,
    ) -> tuple[frozenset[str], ...]:
        signatures = {
            self._fixed_symbols(node)
            for node in formula.walk()
            if node.args
        }
        signatures.discard(frozenset())
        return tuple(
            sorted(signatures, key=lambda item: (len(item), sorted(item)))
        )

    @staticmethod
    def _subsumes(left: SemanticProfile, right: SemanticProfile) -> bool:
        return (
            left.conclusion_key == right.conclusion_key
            and left.premise_keys <= right.premise_keys
            and left.d_constraints <= right.d_constraints
        )

    def _proof_definition_support(self, theorem: Theorem) -> frozenset[str]:
        if theorem.proof is None:
            return frozenset()
        support: set[str] = set()
        for parent_id in theorem.proof.parents:
            support.update(self._definition_support.get(parent_id, ()))
        source_rule = self.parsed.logical_assertions.get(theorem.proof.rule)
        if source_rule is not None:
            for predicate, definition in self._definition_sources.items():
                if definition.name == source_rule.name:
                    support.add(predicate)
                    break
        return frozenset(support)

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
        definition_support = self._proof_definition_support(theorem)
        definition_only_predicate = (
            next(iter(definition_support))
            if len(definition_support) == 1
            else None
        )
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
        definition_only_quota = (
            self.config.max_definition_only_search_per_predicate
        )
        if (
            self._admission_context == "search"
            and definition_only_quota >= 0
            and definition_only_predicate is not None
            and definition_only_predicate in {
                node.op for node in theorem.conclusion.walk()
            }
            and self._definition_only_search[definition_only_predicate]
            >= definition_only_quota
        ):
            self.rejected["definition_only_search_quota"] += 1
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
        self._definition_support[theorem_id] = definition_support
        self.generation_context[theorem_id] = self._admission_context
        if self._current_target is not None:
            self.guidance_targets[theorem_id] = self._current_target
        self._target_report_cache = None
        self.stats["stored"] += 1
        self.rule_usage[stored.proof.rule if stored.proof else stored.name] += 1
        used_definitions = {
            node.op for node in stored.conclusion.walk()
            if node.op in self._focus_set
        }
        self._definition_usage.update(used_definitions)
        if (
            self._admission_context == "search"
            and definition_only_predicate is not None
            and definition_only_predicate in used_definitions
        ):
            self._definition_only_search[definition_only_predicate] += 1
            self.stats["definition_only_search_admitted"] += 1
        if self._current_target is not None:
            similarity = self._target_similarity(
                stored,
                self._current_target,
            )
            self._target_best_similarity[self._current_target] = max(
                self._target_best_similarity[self._current_target],
                similarity,
            )
            if similarity > 0:
                self.stats["target_guided_admissions"] += 1

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

    @staticmethod
    def _nodes_may_unify(
        left: Node,
        right: Node,
        left_types: dict[str, str],
        right_types: dict[str, str],
    ) -> bool:
        left_type = left_types.get(left.op) if not left.args else None
        right_type = right_types.get(right.op) if not right.args else None
        if left_type is not None or right_type is not None:
            return (
                left_type == right_type
                if left_type is not None and right_type is not None
                else True
            )
        return (
            left.op == right.op
            and len(left.args) == len(right.args)
            and all(
                TheoremGenerator._nodes_may_unify(
                    left_arg,
                    right_arg,
                    left_types,
                    right_types,
                )
                for left_arg, right_arg in zip(left.args, right.args)
            )
        )

    def _active_candidates(self, conclusion: Node | None = None) -> list[Theorem]:
        result: list[Theorem] = []
        for theorem_id in self.active_ids:
            theorem = self.store[theorem_id]
            if conclusion is not None and theorem.conclusion.op != conclusion.op:
                continue
            if (
                conclusion is not None
                and self.config.compatible_candidate_filter
                and not self._nodes_may_unify(
                    conclusion,
                    theorem.conclusion,
                    self.parsed.variable_types,
                    theorem.variable_types,
                )
            ):
                continue
            result.append(theorem)
        return result

    def _rule_weight(self, rule: Theorem) -> float:
        if rule.name == "ax-mp":
            return self.config.mp_rule_weight
        if rule.name in ALPHA_RULES:
            return self.config.alpha_rule_weight
        return self.config.default_rule_weight

    def _choose_target(self) -> str | None:
        if not self.target_formulas or self.config.target_guidance_weight <= 0:
            return None
        names = list(self.target_formulas)
        chosen = self.random.choices(
            names,
            weights=[
                1.0 / math.sqrt(1.0 + self._target_selection[name])
                for name in names
            ],
            k=1,
        )[0]
        self._target_selection[chosen] += 1
        return chosen

    def _target_similarity(self, theorem: Theorem, target: str) -> float:
        symbols = self._fixed_symbols(theorem.conclusion)
        if not symbols:
            return 0.0
        structural = max(
            (
                len(symbols & signature) / len(symbols | signature)
                for signature in self._target_signatures[target]
                if symbols | signature
            ),
            default=0.0,
        )
        required = self._target_required_predicates[target]
        if not required:
            return structural
        predicate_recall = len(symbols & required) / len(required)
        return 0.65 * structural + 0.35 * predicate_recall

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
        depth_bonus = (1.0 + theorem.proof_depth) ** (
            self.config.depth_parent_bias
        )
        definitions = {
            node.op for node in theorem.conclusion.walk()
            if node.op in self._focus_set
        }
        if definitions and self.config.definition_coverage_weight > 0:
            coverage_need = sum(
                1.0 / math.sqrt(1.0 + self._definition_usage[name])
                for name in definitions
            ) / len(definitions)
            coverage_bonus = (
                1.0 + self.config.definition_coverage_weight * coverage_need
            )
        else:
            coverage_bonus = 1.0
        if (
            self._current_target is not None
            and self.config.target_guidance_weight > 0
        ):
            target_bonus = 1.0 + (
                self.config.target_guidance_weight
                * self._target_similarity(theorem, self._current_target)
            )
        else:
            target_bonus = 1.0
        return (
            (closed_bonus + quality_bonus)
            * depth_bonus
            * coverage_bonus
            * target_bonus
            / reuse_penalty
        )

    def _choose_weighted_candidate(
        self,
        candidates: list[Theorem],
        *,
        record_use: bool = True,
    ) -> Theorem | None:
        if not candidates:
            return None
        chosen = self.random.choices(
            candidates,
            weights=[self._candidate_weight(item) for item in candidates],
            k=1,
        )[0]
        if record_use and chosen.id is not None:
            self._candidate_use[chosen.id] += 1
        return chosen

    def _choose_candidate(self, premise: Node) -> Theorem | None:
        candidates = self._active_candidates(premise)
        closed = [theorem for theorem in candidates if not theorem.hypotheses]
        if (
            closed
            and self.random.random() < self.config.closed_parent_probability
        ):
            candidates = closed
        return self._choose_weighted_candidate(candidates)

    def _joint_ax_mp_matches(
        self,
        rule: Theorem,
    ) -> list[Theorem | None] | None:
        """Choose a compatible minor/major pair instead of sampling blindly."""

        majors = self._active_candidates(rule.hypotheses[1].expr)
        tried = 0
        while majors and tried < self.config.max_joint_candidate_attempts:
            major = self._choose_weighted_candidate(majors, record_use=False)
            if major is None:
                return None
            majors.remove(major)
            tried += 1
            major_body = major.conclusion.args[0]
            if major_body.op != "implies" or len(major_body.args) != 2:
                continue
            antecedent = major_body.args[0]
            minors = [
                theorem
                for theorem in self._active_candidates(rule.hypotheses[0].expr)
                if theorem.conclusion.args
                and self._nodes_may_unify(
                    antecedent,
                    theorem.conclusion.args[0],
                    major.variable_types,
                    theorem.variable_types,
                )
            ]
            minor = self._choose_weighted_candidate(minors, record_use=False)
            if minor is None:
                continue
            for parent in (minor, major):
                if parent.id is not None:
                    self._candidate_use[parent.id] += 1
            self.stats["joint_match_plan"] += 1
            return [minor, major]
        self.rejected["no_joint_candidate"] += 1
        return None

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

    def bootstrap_definition_directions(self) -> list[Theorem]:
        """Derive certified unfold/fold implications for focused definitions."""

        start_index = len(self._new_ids)
        if self._definitions_bootstrapped:
            return []
        self._definitions_bootstrapped = True
        ax_mp = self.parsed.logical_assertions.get("ax-mp")
        directions = (
            ("unfold", self.parsed.logical_assertions.get("bi1")),
            ("fold", self.parsed.logical_assertions.get("bi2")),
        )
        if ax_mp is None or any(rule is None for _, rule in directions):
            self.rejected["definition_bootstrap_unavailable"] += 1
            return []

        previous_context = self._admission_context
        self._admission_context = "definition_bridge"
        try:
            for predicate in self.focus_predicates:
                definition = self._definition_sources[predicate]
                for direction, bridge in directions:
                    self.stats["definition_bootstrap_attempts"] += 1
                    try:
                        theorem = compose(
                            ax_mp,
                            [definition, bridge],
                            name=f"gen_df_{predicate}_{direction}",
                            database=self.parsed,
                        )
                    except CompositionError:
                        self.rejected[
                            "definition_bootstrap_composition_failed"
                        ] += 1
                        continue
                    if not self._valid(theorem):
                        self.rejected[
                            "definition_bootstrap_structural_limit"
                        ] += 1
                        continue
                    if self._admit(theorem):
                        self.stats["definition_bridges"] += 1
                        self._definition_bridges[predicate].append(
                            self._new_ids[-1]
                        )
                        self._bootstrap_ids.add(self._new_ids[-1])
                    else:
                        self.rejected[
                            "definition_bootstrap_not_admitted"
                        ] += 1
        finally:
            self._admission_context = previous_context
        return [self.store[item] for item in self._new_ids[start_index:]]

    @staticmethod
    def _numeral(value: int) -> Node:
        result = Node("0")
        for _ in range(value):
            result = Node("S", (result,))
        return result

    def bootstrap_bounded_term_instances(self) -> list[Theorem]:
        """Instantiate definition parameters with canonical bounded numerals."""

        start_index = len(self._new_ids)
        if self._bounded_instances_bootstrapped:
            return []
        self._bounded_instances_bootstrapped = True
        count = self.config.ground_instances_per_predicate
        if count <= 0:
            return []
        if not self._definitions_bootstrapped:
            self.bootstrap_definition_directions()
        terms = tuple(
            self._numeral(value)
            for value in range(self.config.bounded_nat_max + 1)
        )
        previous_context = self._admission_context
        self._admission_context = "bounded_instance"
        try:
            for predicate_index, predicate in enumerate(self.focus_predicates):
                for bridge_id in self._definition_bridges.get(predicate, ()):
                    bridge = self.store[bridge_id]
                    occurrence = next(
                        (
                            node for node in bridge.conclusion.walk()
                            if node.op == predicate
                        ),
                        None,
                    )
                    if occurrence is None:
                        self.rejected[
                            "bounded_instance_missing_predicate"
                        ] += 1
                        continue
                    parameters = [
                        argument.op
                        for argument in occurrence.args
                        if not argument.args
                        and bridge.variable_types.get(argument.op) == "term"
                    ]
                    assignments = list(
                        itertools.product(terms, repeat=len(parameters))
                    )
                    if not assignments:
                        continue
                    start = predicate_index % len(assignments)
                    selected_assignments = [
                        assignments[(start + index) % len(assignments)]
                        for index in range(min(count, len(assignments)))
                    ]
                    for instance_index, assignment in enumerate(
                        selected_assignments
                    ):
                        substitution = dict(zip(parameters, assignment))
                        self.stats["bounded_instance_attempts"] += 1
                        try:
                            theorem = instantiate_assertion(
                                bridge,
                                substitution,
                                name=(
                                    f"gen{len(self.store)}_{predicate}_"
                                    f"bounded_{instance_index}_"
                                    f"{bridge.name.rsplit('_', 1)[-1]}"
                                ),
                                database=self.parsed,
                            )
                        except CompositionError:
                            self.rejected[
                                "bounded_instance_composition_failed"
                            ] += 1
                            continue
                        if not self._valid(theorem):
                            self.rejected[
                                "bounded_instance_structural_limit"
                            ] += 1
                            continue
                        if self._admit(theorem):
                            self.stats["bounded_ground_instances"] += 1
                            self._bootstrap_ids.add(self._new_ids[-1])
                        else:
                            self.rejected[
                                "bounded_instance_not_admitted"
                            ] += 1
        finally:
            self._admission_context = previous_context
        return [self.store[item] for item in self._new_ids[start_index:]]

    def random_walk(self, steps: int) -> list[Theorem]:
        start_index = len(self._new_ids)
        if not self.rules:
            return []
        for _ in range(steps):
            self._current_target = self._choose_target()
            rule = self._choose_rule()
            matches: list[Theorem | None] = [None] * len(rule.hypotheses)
            indices = self._match_indices(rule)
            if len(indices) == len(matches):
                self.stats["full_discharge_plans"] += 1
            else:
                self.stats["partial_discharge_plans"] += 1
            if rule.name == "ax-mp" and set(indices) == {0, 1}:
                planned = self._joint_ax_mp_matches(rule)
                if planned is not None:
                    matches = planned
            else:
                for index in indices:
                    matches[index] = self._choose_candidate(
                        rule.hypotheses[index].expr
                    )
            if any(parent is not None for parent in matches):
                self._try(rule, matches)
            else:
                self.rejected["no_candidate"] += 1
        self._current_target = None
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
        mode: Literal["graph", "random", "forward"] = "graph",
        steps: int = 100,
    ) -> list[Theorem]:
        start_index = len(self._new_ids)
        if (
            self.config.bootstrap_definitions
            or self.config.ground_instances_per_predicate > 0
        ):
            self.bootstrap_definition_directions()
        if self.config.ground_instances_per_predicate > 0:
            self.bootstrap_bounded_term_instances()
        if mode == "graph":
            from .random_graph import generate_random_graph
            generate_random_graph(self, steps)
        elif mode == "forward":
            self.forward_saturation(steps)
        elif mode == "random":
            self.random_walk(steps)
        else:
            raise ValueError(f"unsupported generation mode: {mode}")
        return [self.store[item] for item in self._new_ids[start_index:]]

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

    def _target_similarity_report(self) -> dict[str, float]:
        if self._target_report_cache is not None:
            return dict(self._target_report_cache)
        search_theorems = [
            self.store[theorem_id]
            for theorem_id in self._new_ids
            if theorem_id not in self._bootstrap_ids
            and theorem_id not in self.dominated_ids
        ]
        report = {
            name: max(
                (
                    self._target_similarity(theorem, name)
                    for theorem in search_theorems
                ),
                default=0.0,
            )
            for name in self.target_formulas
        }
        self._target_report_cache = report
        return dict(report)

    @property
    def definition_support(self) -> dict[int, frozenset[str]]:
        return self._definition_support

    def summary(self) -> GenerationSummary:
        target_report = self._target_similarity_report()
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
            definition_predicates_total=len(self.focus_predicates),
            definition_predicates_seen=sum(
                self._definition_usage[name] > 0
                for name in self.focus_predicates
            ),
            definition_coverage=(
                sum(
                    self._definition_usage[name] > 0
                    for name in self.focus_predicates
                ) / len(self.focus_predicates)
                if self.focus_predicates
                else 1.0
            ),
            definition_usage={
                name: self._definition_usage[name]
                for name in self.focus_predicates
            },
            definition_bridges=self.stats["definition_bridges"],
            bounded_ground_instances=self.stats[
                "bounded_ground_instances"
            ],
            search_stored=len(self._new_ids) - len(self._bootstrap_ids),
            target_statements_total=len(self.target_formulas),
            target_statements_touched=sum(
                similarity >= 0.25
                for similarity in target_report.values()
            ),
            target_mean_best_similarity=(
                sum(target_report.values()) / len(target_report)
                if target_report
                else 0.0
            ),
            target_best_similarity={
                name: round(similarity, 6)
                for name, similarity in target_report.items()
            },
            definition_only_search_admitted=self.stats[
                "definition_only_search_admitted"
            ],
        )

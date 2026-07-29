from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .model import Database, Node, Theorem, normalized_pair

DatasetCategory = Literal[
    "closed_theorems",
    "inference_rules",
    "proof_states",
]

ARITHMETIC_OPERATORS = {
    "0",
    "S",
    "+",
    "*",
    "=",
    "<",
    "le",
    "divides",
    "prime",
    "even",
    "pow",
    "primecount",
    "pntat",
    "rhat",
}
QUANTIFIERS = {"forall", "exists"}


@dataclass(slots=True)
class QualityConfig:
    reject_vacuous_quantifiers: bool = True
    reject_conclusion_as_premise: bool = True
    bare_conclusion_active: bool = False
    inference_rule_min_score: float = 12.0
    closed_theorem_min_score: float = 20.0
    premise_penalty: float = 4.0
    vacuous_quantifier_penalty: float = 30.0
    bare_conclusion_penalty: float = 45.0
    source_equivalent_penalty: float = 25.0
    arithmetic_operator_reward: float = 4.0
    nonvacuous_quantifier_reward: float = 4.0
    mixed_schematic_penalty: float = 18.0
    novelty_reward: float = 12.0


@dataclass(frozen=True, slots=True)
class SemanticProfile:
    conclusion_key: tuple
    premise_keys: frozenset[tuple]
    d_constraints: frozenset[tuple[str, str]]
    full_key: tuple
    normalized_conclusion: Node
    normalized_hypotheses: tuple[Node, ...]
    vacuous_quantifiers: int


@dataclass(slots=True)
class QualityAssessment:
    score: float
    category: DatasetCategory
    active: bool
    hard_reject: bool
    reasons: list[str] = field(default_factory=list)
    vacuous_quantifiers: int = 0
    nonvacuous_quantifiers: int = 0
    schematic_wff_variables: tuple[str, ...] = ()
    bare_conclusion: bool = False
    arithmetic_operators: tuple[str, ...] = ()
    source_equivalent: bool = False
    dominated: bool = False
    semantic_key: tuple = ()
    normalized_conclusion: str = ""


def _is_quantifier(node: Node, variable_types: dict[str, str]) -> bool:
    return (
        node.op in QUANTIFIERS
        or variable_types.get(node.op) == "QUANT"
    ) and len(node.args) == 2


def _free_occurs(
    variable: str,
    node: Node,
    variable_types: dict[str, str],
    shadowed: frozenset[str] = frozenset(),
) -> bool:
    if _is_quantifier(node, variable_types):
        binder = node.args[0].op
        return _free_occurs(
            variable,
            node.args[1],
            variable_types,
            shadowed | {binder},
        )
    if node.op == variable and variable not in shadowed:
        return True
    return any(
        _free_occurs(variable, arg, variable_types, shadowed)
        for arg in node.args
    )


def remove_vacuous_quantifiers(
    node: Node,
    variable_types: dict[str, str],
) -> tuple[Node, int]:
    """Remove semantically empty quantifiers for quality keys only.

    The generator never substitutes this normalized node back into a theorem:
    doing so would require a new proof.  It is used for rejection, scoring,
    semantic duplicate detection, and reporting.
    """

    removed = 0
    args: list[Node] = []
    for arg in node.args:
        normalized, count = remove_vacuous_quantifiers(arg, variable_types)
        args.append(normalized)
        removed += count
    normalized = Node(node.op, tuple(args))
    if _is_quantifier(normalized, variable_types):
        binder, body = normalized.args
        if not _free_occurs(binder.op, body, variable_types):
            return body, removed + 1
    return normalized, removed


def _node_key(node: Node) -> tuple:
    return node.op, tuple(_node_key(arg) for arg in node.args)


def _rename_node(
    node: Node,
    known: dict[str, str],
    renaming: dict[str, str],
) -> Node:
    if node.op in known:
        replacement = renaming.setdefault(node.op, f"v{len(renaming)}")
    else:
        replacement = node.op
    return Node(
        replacement,
        tuple(_rename_node(arg, known, renaming) for arg in node.args),
    )


def _local_shape(
    node: Node,
    known: dict[str, str],
    fixed: dict[str, str],
) -> tuple:
    local: dict[str, str] = {}

    def visit(item: Node) -> tuple:
        if item.op in fixed:
            op = fixed[item.op]
        elif item.op in known:
            op = local.setdefault(
                item.op,
                f"u{len(local)}:{known[item.op]}",
            )
        else:
            op = item.op
        return op, tuple(visit(arg) for arg in item.args)

    return visit(node)


def semantic_profile(theorem: Theorem) -> SemanticProfile:
    """Build a conclusion-first, premise-order-independent semantic key."""

    known = theorem.variable_types
    normalized_conclusion, vacuous = remove_vacuous_quantifiers(
        theorem.conclusion, known
    )
    normalized_hypotheses: list[Node] = []
    for hypothesis in theorem.hypotheses:
        normalized, count = remove_vacuous_quantifiers(hypothesis.expr, known)
        normalized_hypotheses.append(normalized)
        vacuous += count

    renaming: dict[str, str] = {}
    conclusion = _rename_node(normalized_conclusion, known, renaming)
    conclusion_type_key = tuple(sorted(
        (renamed, known[original])
        for original, renamed in renaming.items()
    ))
    ordered = sorted(
        normalized_hypotheses,
        key=lambda item: repr(_local_shape(item, known, renaming)),
    )
    hypotheses = [
        _rename_node(item, known, renaming)
        for item in ordered
    ]
    hypothesis_keys = frozenset(_node_key(item) for item in hypotheses)

    # All variables in generated d constraints also occur in a hypothesis or
    # conclusion, but handle source databases defensively.
    for left, right in sorted(theorem.d_constraints):
        if left in known:
            renaming.setdefault(left, f"v{len(renaming)}")
        if right in known:
            renaming.setdefault(right, f"v{len(renaming)}")
    constraints = frozenset(
        normalized_pair(renaming.get(left, left), renaming.get(right, right))
        for left, right in theorem.d_constraints
    )
    all_type_key = tuple(sorted(
        (renamed, known[original])
        for original, renamed in renaming.items()
    ))
    conclusion_key = (_node_key(conclusion), conclusion_type_key)
    full_key = (
        conclusion_key,
        tuple(sorted(hypothesis_keys, key=repr)),
        tuple(sorted(constraints)),
        all_type_key,
    )
    return SemanticProfile(
        conclusion_key=conclusion_key,
        premise_keys=hypothesis_keys,
        d_constraints=constraints,
        full_key=full_key,
        normalized_conclusion=conclusion,
        normalized_hypotheses=tuple(hypotheses),
        vacuous_quantifiers=vacuous,
    )


def is_bare_conclusion(theorem: Theorem) -> bool:
    if len(theorem.conclusion.args) != 1:
        return False
    body = theorem.conclusion.args[0]
    return not body.args and theorem.variable_types.get(body.op) == "wff"


class QualityAnalyzer:
    def __init__(
        self,
        parsed: Database,
        config: QualityConfig | None = None,
    ) -> None:
        self.parsed = parsed
        self.config = config or QualityConfig()
        self.source_conclusions = {
            semantic_profile(theorem).conclusion_key
            for theorem in parsed.logical_assertions.values()
        }

    def assess(self, theorem: Theorem) -> tuple[QualityAssessment, SemanticProfile]:
        profile = semantic_profile(theorem)
        cfg = self.config
        reasons: list[str] = []
        closed = not theorem.hypotheses
        bare = is_bare_conclusion(theorem)
        source_equivalent = profile.conclusion_key in self.source_conclusions
        arithmetic = tuple(sorted({
            node.op
            for node in profile.normalized_conclusion.walk()
            if node.op in ARITHMETIC_OPERATORS
        }))
        nonvacuous_quantifiers = sum(
            _is_quantifier(node, theorem.variable_types)
            for node in profile.normalized_conclusion.walk()
        )
        conclusion_symbols = {
            node.op for node in profile.normalized_conclusion.walk()
        }
        schematic_wff_variables = tuple(sorted(
            variable
            for variable, typecode in theorem.variable_types.items()
            if typecode == "wff" and variable in conclusion_symbols
        ))

        score = 45.0 if closed else 14.0
        if closed:
            reasons.append("closed theorem")
        else:
            score -= cfg.premise_penalty * len(theorem.hypotheses)
            reasons.append(f"{len(theorem.hypotheses)} open premise(s)")
        if bare:
            score -= cfg.bare_conclusion_penalty
            reasons.append("bare schematic conclusion")
        if profile.vacuous_quantifiers:
            score -= (
                cfg.vacuous_quantifier_penalty
                * profile.vacuous_quantifiers
            )
            reasons.append(
                f"{profile.vacuous_quantifiers} vacuous quantifier(s)"
            )
        if source_equivalent:
            score -= cfg.source_equivalent_penalty
            reasons.append("normalized conclusion already in source")
        else:
            score += cfg.novelty_reward
            reasons.append("novel normalized conclusion")
        if arithmetic:
            score += min(
                20.0,
                cfg.arithmetic_operator_reward * len(arithmetic),
            )
            reasons.append("arithmetic structure: " + ", ".join(arithmetic))
        if nonvacuous_quantifiers:
            score += min(
                12.0,
                cfg.nonvacuous_quantifier_reward
                * nonvacuous_quantifiers,
            )
            reasons.append(
                f"{nonvacuous_quantifiers} non-vacuous quantifier(s)"
            )
        if arithmetic and schematic_wff_variables:
            score -= (
                cfg.mixed_schematic_penalty
                * len(schematic_wff_variables)
            )
            reasons.append(
                "schematic wff mixed into arithmetic conclusion: "
                + ", ".join(schematic_wff_variables)
            )

        depth = theorem.proof_depth
        if 2 <= depth <= 5:
            score += 8.0
            reasons.append("nontrivial moderate proof depth")
        elif depth == 1:
            score += 2.0
        elif depth > 6:
            score -= 3.0 * (depth - 6)
            reasons.append("deep proof penalty")

        if 3 <= theorem.conclusion.depth <= 12:
            score += 4.0
        elif theorem.conclusion.depth > 20:
            score -= float(theorem.conclusion.depth - 20)
            reasons.append("oversized conclusion")

        conclusion_is_premise = any(
            hypothesis.expr == theorem.conclusion
            for hypothesis in theorem.hypotheses
        )
        if conclusion_is_premise:
            score -= 50.0
            reasons.append("conclusion already present as premise")

        if closed:
            active = score >= cfg.closed_theorem_min_score
            category: DatasetCategory = (
                "closed_theorems" if active else "proof_states"
            )
        elif bare or score < cfg.inference_rule_min_score:
            category = "proof_states"
            active = cfg.bare_conclusion_active and score >= 0
        else:
            category = "inference_rules"
            active = True

        hard_reject = (
            cfg.reject_vacuous_quantifiers
            and profile.vacuous_quantifiers > 0
        ) or (
            cfg.reject_conclusion_as_premise
            and conclusion_is_premise
        )
        return QualityAssessment(
            score=round(score, 3),
            category=category,
            active=active and not hard_reject,
            hard_reject=hard_reject,
            reasons=reasons,
            vacuous_quantifiers=profile.vacuous_quantifiers,
            nonvacuous_quantifiers=nonvacuous_quantifiers,
            schematic_wff_variables=schematic_wff_variables,
            bare_conclusion=bare,
            arithmetic_operators=arithmetic,
            source_equivalent=source_equivalent,
            semantic_key=profile.full_key,
            normalized_conclusion=profile.normalized_conclusion.to_prefix(),
        ), profile

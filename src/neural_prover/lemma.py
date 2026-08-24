from __future__ import annotations

from dataclasses import dataclass

from metamath_generator.model import Node

from .environment import (
    LEMMA_BINDING,
    LEMMA_COMMIT_OP,
    PROPOSE_LEMMA_RULE,
    BackwardEnvironment,
    InvalidTactic,
    ProofState,
    Tactic,
    Transition,
)
from .hybrid import (
    HybridActionGenerator,
    HybridActions,
    HybridGeneratorConfig,
)


def lemma_from_tactic(tactic: Tactic) -> Node:
    if tactic.rule != PROPOSE_LEMMA_RULE:
        raise InvalidTactic("tactic is not an intermediate-lemma proposal")
    substitution = tactic.substitution_dict()
    if set(substitution) != {LEMMA_BINDING}:
        raise InvalidTactic("lemma proposal has malformed payload")
    return substitution[LEMMA_BINDING]


def propose_lemma(lemma: Node) -> Tactic:
    return Tactic.create(
        PROPOSE_LEMMA_RULE,
        {LEMMA_BINDING: lemma},
    )


@dataclass(frozen=True, slots=True)
class LemmaGeneratorConfig:
    base: HybridGeneratorConfig = HybridGeneratorConfig()
    max_lemma_candidates: int = 24
    max_lemma_nodes: int = 96
    max_pending_lemmas: int = 2
    require_bounded_proof_hint: bool = True


class LemmaBackwardEnvironment(BackwardEnvironment):
    """Backward kernel with a sound, explicitly scheduled cut action.

    Proposing ``L`` while proving ``G`` creates the ordered obligations
    ``Γ ⊢ L`` and ``Γ,L ⊢ G``.  A private marker activates ``L`` only after
    every subgoal in its proof has closed, so an unproved proposal can never
    be used as an assumption.
    """

    def __init__(self, *args, max_lemma_nodes: int = 96, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_lemma_nodes = max_lemma_nodes
        self._lemma_transition_cache: dict[
            tuple[ProofState, Tactic], Transition
        ] = {}

    @staticmethod
    def _normalize_commits(state: ProofState) -> ProofState:
        hypotheses = list(state.hypotheses)
        goals = list(state.goals)
        changed = False
        while goals and goals[0].op == LEMMA_COMMIT_OP:
            marker = goals.pop(0)
            if len(marker.args) != 1:
                raise InvalidTactic("malformed lemma commit marker")
            lemma = marker.args[0]
            if lemma not in hypotheses:
                hypotheses.append(lemma)
            changed = True
        if not changed:
            return state
        return ProofState(
            tuple(hypotheses),
            tuple(goals),
            state.d_constraints,
            state.variable_types,
        )

    def _apply_lemma(
        self,
        state: ProofState,
        tactic: Tactic,
    ) -> Transition:
        if state.solved:
            raise InvalidTactic("cannot propose a lemma in a solved state")
        lemma = lemma_from_tactic(tactic)
        if lemma.op != "|-" or len(lemma.args) != 1:
            raise InvalidTactic("intermediate lemma must be a |- assertion")
        if lemma == state.current_goal:
            raise InvalidTactic("lemma cannot duplicate the current goal")
        if lemma in state.hypotheses:
            raise InvalidTactic("lemma is already an active hypothesis")
        if sum(1 for _ in lemma.walk()) > self.max_lemma_nodes:
            raise InvalidTactic("intermediate lemma exceeds size limit")
        if any(
            goal.op == LEMMA_COMMIT_OP
            and goal.args == (lemma,)
            for goal in state.goals
        ):
            raise InvalidTactic("lemma is already pending")
        if not self._well_typed((lemma,), dict(state.variable_types)):
            raise InvalidTactic("intermediate lemma is not well typed")
        marker = Node(LEMMA_COMMIT_OP, (lemma,))
        after = ProofState(
            state.hypotheses,
            (lemma, marker, *state.goals),
            state.d_constraints,
            state.variable_types,
        )
        return Transition(
            state,
            tactic,
            after,
            (lemma, state.current_goal),
            (),
        )

    def apply(self, state: ProofState, tactic: Tactic) -> Transition:
        key = (state, tactic)
        cached = self._lemma_transition_cache.get(key)
        if cached is not None:
            self.metrics.apply_requests += 1
            self.metrics.apply_cache_hits += 1
            return cached
        if tactic.rule == PROPOSE_LEMMA_RULE:
            self.metrics.apply_requests += 1
            self.metrics.transition_evaluations += 1
            try:
                transition = self._apply_lemma(state, tactic)
            except InvalidTactic:
                self.metrics.invalid_tactics += 1
                raise
        else:
            raw = super().apply(state, tactic)
            normalized = self._normalize_commits(raw.after)
            transition = (
                raw
                if normalized is raw.after
                else Transition(
                    raw.before,
                    raw.tactic,
                    normalized,
                    raw.generated_goals,
                    raw.resolved_substitution,
                )
            )
        self._lemma_transition_cache[key] = transition
        return transition

    def lemma_candidates(
        self,
        state: ProofState,
        *,
        limit: int,
        max_nodes: int,
        require_bounded_proof_hint: bool,
    ) -> list[Node]:
        if state.solved or limit <= 0:
            return []
        formulas = self._candidate_expressions(
            state,
            "wff",
            max(limit * 3, limit),
            True,
        )
        goal_ops = {node.op for node in state.current_goal.walk()}
        ranked: list[tuple[tuple[int, int, int], Node]] = []
        seen: set[Node] = set()
        for formula in formulas:
            lemma = Node("|-", (formula,))
            if (
                lemma in seen
                or lemma == state.current_goal
                or lemma in state.hypotheses
                or sum(1 for _ in lemma.walk()) > max_nodes
            ):
                continue
            seen.add(lemma)
            direct = self.directly_closable(state, lemma)
            near = False
            if not direct and require_bounded_proof_hint:
                near = self.one_step_closable(state, lemma)
            if require_bounded_proof_hint and not (direct or near):
                continue
            overlap = len(
                goal_ops & {node.op for node in lemma.walk()}
            )
            ranked.append((
                (int(direct), int(near), overlap),
                lemma,
            ))
        ranked.sort(
            key=lambda item: (
                item[0], -sum(1 for _ in item[1].walk())
            ),
            reverse=True,
        )
        return [lemma for _, lemma in ranked[:limit]]


class LemmaActionGenerator(HybridActionGenerator):
    """Add finite, typed intermediate-lemma choices to hybrid actions."""

    def __init__(
        self,
        environment: LemmaBackwardEnvironment,
        config: LemmaGeneratorConfig | None = None,
    ) -> None:
        self.lemma_config = config or LemmaGeneratorConfig()
        super().__init__(environment, self.lemma_config.base)

    def actions(self, state: ProofState) -> HybridActions:
        base = super().actions(state)
        cfg = self.lemma_config
        pending = sum(
            goal.op == LEMMA_COMMIT_OP for goal in state.goals
        )
        if (
            pending >= cfg.max_pending_lemmas
            or any(
                self.environment.apply(state, tactic).after.solved
                for tactic in base.deterministic
            )
        ):
            return base
        lemmas = self.environment.lemma_candidates(
            state,
            limit=cfg.max_lemma_candidates,
            max_nodes=cfg.max_lemma_nodes,
            require_bounded_proof_hint=cfg.require_bounded_proof_hint,
        )
        proposals = tuple(propose_lemma(lemma) for lemma in lemmas)
        return HybridActions(
            base.deterministic,
            (*base.constructions, *proposals),
        )

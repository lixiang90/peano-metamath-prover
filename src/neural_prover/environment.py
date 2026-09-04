from __future__ import annotations

import copy
import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Iterable

from metamath_generator.model import (
    Database,
    Hypothesis,
    Node,
    Proof,
    Theorem,
    normalized_pair,
)
from metamath_generator.parser import MetamathParser, ParseError
from metamath_generator.unification import (
    UnificationError,
    substitute,
    substitute_simultaneous,
    unify,
)

from .tokenizer import (
    VARIABLE_TOKEN_PATTERN,
    CanonicalVariables,
    MetamathTokenizer,
)


class InvalidTactic(ValueError):
    pass


PROPOSE_LEMMA_RULE = "<PROPOSE_LEMMA>"
LEMMA_BINDING = "__lemma_value__"
LEMMA_COMMIT_OP = "__lemma_commit__"


@dataclass(slots=True)
class EnvironmentMetrics:
    apply_requests: int = 0
    apply_cache_hits: int = 0
    transition_evaluations: int = 0
    invalid_tactics: int = 0
    tactic_enumerations: int = 0
    tactic_cache_hits: int = 0
    candidates_returned: int = 0

    def to_record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Tactic:
    rule: str
    substitution: tuple[tuple[str, Node], ...] = ()

    @classmethod
    def create(
        cls,
        rule: str,
        substitution: dict[str, Node] | None = None,
    ) -> "Tactic":
        return cls(rule, tuple(sorted((substitution or {}).items())))

    def substitution_dict(self) -> dict[str, Node]:
        return dict(self.substitution)


@dataclass(frozen=True, slots=True)
class ProofState:
    hypotheses: tuple[Node, ...]
    goals: tuple[Node, ...]
    d_constraints: frozenset[tuple[str, str]]
    variable_types: tuple[tuple[str, str], ...]

    @classmethod
    def from_theorem(cls, theorem: Theorem) -> "ProofState":
        return cls(
            hypotheses=tuple(h.expr for h in theorem.hypotheses),
            goals=(theorem.conclusion,),
            d_constraints=frozenset(theorem.d_constraints),
            variable_types=tuple(sorted(theorem.variable_types.items())),
        )

    @property
    def solved(self) -> bool:
        return not self.goals

    @property
    def current_goal(self) -> Node:
        if not self.goals:
            raise InvalidTactic("proof state is already solved")
        return self.goals[0]

    def as_theorem(self, name: str = "_proof_state") -> Theorem:
        if not self.goals:
            conclusion = Node("|-", (Node("_solved"),))
        else:
            conclusion = self.goals[0]
        return Theorem(
            name=name,
            hypotheses=[
                Hypothesis(f"h{index}", expression)
                for index, expression in enumerate(self.hypotheses)
            ],
            conclusion=conclusion,
            d_constraints=set(self.d_constraints),
            variable_types=dict(self.variable_types),
            kind="proof_state",
        )


@dataclass(frozen=True, slots=True)
class Transition:
    before: ProofState
    tactic: Tactic
    after: ProofState
    generated_goals: tuple[Node, ...]
    resolved_substitution: tuple[tuple[str, Node], ...] = ()
    assertion: Theorem | None = None


class BackwardEnvironment:
    """Typed backward Metamath environment with kernel-like action checks."""

    def __init__(
        self,
        database: Database,
        lemmas: Iterable[Theorem] = (),
        *,
        excluded_assertions: Iterable[str] = (),
    ) -> None:
        self.database = database
        excluded = set(excluded_assertions)
        self.excluded_assertions = frozenset(excluded)
        self._pa_plus_bridges: set[str] = set()
        self.assertions = {
            label: theorem
            for label, theorem in database.logical_assertions.items()
            if label not in excluded
        }
        self.assertions.update({
            theorem.name: theorem
            for theorem in lemmas
            if theorem.name not in excluded
        })
        self.metrics = EnvironmentMetrics()
        self._parser = MetamathParser()
        self._parser.database = database
        self._type_cache: dict[
            tuple[Node, str, tuple[tuple[str, str], ...]], bool
        ] = {}
        self._local_candidate_cache: dict[
            tuple[ProofState, str, int], tuple[Node, ...]
        ] = {}
        self._candidate_cache: dict[
            tuple[ProofState, str, int, bool], tuple[Node, ...]
        ] = {}
        self._tactic_cache: dict[
            tuple[ProofState, int, int, bool], tuple[Tactic, ...]
        ] = {}
        self._transition_cache: dict[
            tuple[ProofState, Tactic], Transition
        ] = {}
        self._direct_closure_cache: dict[
            tuple[ProofState, Node], bool
        ] = {}
        self._one_step_closure_cache: dict[
            tuple[ProofState, Node], bool
        ] = {}
        self._well_typed_cache: dict[
            tuple[tuple[Node, ...], tuple[tuple[str, str], ...]], bool
        ] = {}
        self._bounded_nat_max = -1
        self._target_formulas: tuple[Node, ...] = ()

    def configure_from_tokenizer(
        self,
        tokenizer: MetamathTokenizer,
        *,
        target_guidance: bool = True,
    ) -> None:
        """Enable PA+ bounded terms and target ordering from tokenizer metadata."""

        context = tokenizer.pa_plus_context
        for name in self._pa_plus_bridges:
            self.assertions.pop(name, None)
        self._pa_plus_bridges.clear()
        self._install_pa_plus_bridges(context.definition_predicates)
        targets: list[Node] = []
        for _, body in context.target_statements if target_guidance else ():
            try:
                parsed = self._parser.parse_expression([
                    "wff", *body.split()
                ])
            except ParseError as exc:
                raise ValueError(
                    f"invalid target formula in tokenizer: {body}"
                ) from exc
            if len(parsed.args) != 1:
                raise ValueError("target formula did not parse as one wff")
            targets.append(parsed.args[0])
        self._bounded_nat_max = context.bounded_nat_max
        self._target_formulas = tuple(targets)
        self._local_candidate_cache.clear()
        self._candidate_cache.clear()
        self._tactic_cache.clear()
        self._transition_cache.clear()
        self._direct_closure_cache.clear()
        self._one_step_closure_cache.clear()

    def configuration_record(self) -> dict:
        """Stable semantic fingerprint for same-environment policy comparisons."""
        rules = tuple(
            (
                name, rule.conclusion.to_prefix(),
                tuple(h.expr.to_prefix() for h in rule.hypotheses),
                tuple(sorted(rule.variable_types.items())),
                tuple(sorted(rule.d_constraints)),
                tuple((h.label, h.expr.to_prefix()) for h in rule.floating),
                rule.proof.source_labels if rule.proof else (),
            )
            for name, rule in self.assertions.items()
        )
        targets = tuple(t.to_prefix() for t in self._target_formulas)
        fingerprint = hashlib.sha256(repr((
            rules, self._bounded_nat_max, targets,
            tuple(sorted(self.excluded_assertions)),
        )).encode("utf-8")).hexdigest()
        return {
            "fingerprint": fingerprint,
            "assertions": len(self.assertions),
            "definition_bridges": len(self._pa_plus_bridges),
            "bounded_nat_max": self._bounded_nat_max,
            "inference_target_guidance": bool(targets),
            "target_formulas": len(targets),
            "excluded_assertions": sorted(self.excluded_assertions),
        }

    def _install_pa_plus_bridges(
        self,
        definition_predicates: tuple[str, ...],
    ) -> None:
        if not definition_predicates:
            return
        from metamath_generator.export import (
            _SyntaxCompiler,
            _generated_proof,
        )
        from metamath_generator.generator import (
            GenerationConfig,
            TheoremGenerator,
        )

        generator = TheoremGenerator(
            self.database,
            GenerationConfig(
                focus_predicates=definition_predicates,
                bootstrap_definitions=True,
            ),
        )
        generator.generate("random", 0)
        from metamath_generator.verifier import verify

        verification_database = copy.copy(self.database)
        verification_database.variables = set(self.database.variables)
        verification_database.floating_hypotheses = dict(
            self.database.floating_hypotheses
        )
        for bridge in generator.store.generated():
            if (
                bridge.name in self.assertions
                or bridge.name in self.excluded_assertions
            ):
                continue
            all_types = {
                **bridge.variable_types,
                **bridge.proof_variable_types,
            }
            floating = tuple(
                Hypothesis(
                    f"{bridge.name}_f{index}",
                    Node(typecode, (Node(variable),)),
                )
                for index, (variable, typecode) in enumerate(
                    sorted(all_types.items())
                )
            )
            floating_map = {
                item.expr.args[0].op: (item.label, item.expr.op)
                for item in floating
            }
            compiler = _SyntaxCompiler(generator.store, floating_map)
            labels = tuple(
                _generated_proof(bridge, generator.store, compiler)
            )
            if self.excluded_assertions.intersection(labels):
                continue
            derived = Theorem(
                name=bridge.name,
                hypotheses=list(bridge.hypotheses),
                conclusion=bridge.conclusion,
                d_constraints=set(bridge.d_constraints),
                variable_types=dict(bridge.variable_types),
                floating=floating,
                proof=Proof(
                    rule=bridge.name,
                    source_labels=labels,
                    depth=bridge.proof_depth,
                ),
                kind="derived",
            )
            verification_database.variables.update(all_types)
            verification_database.floating_hypotheses.update(
                (h.label, h) for h in floating
            )
            verify(derived, verification_database)
            self.assertions[derived.name] = derived
            self._pa_plus_bridges.add(derived.name)

    @staticmethod
    def _numeral(value: int) -> Node:
        result = Node("0")
        for _ in range(value):
            result = Node("S", (result,))
        return result

    def _fixed_symbols(
        self,
        expression: Node,
        variable_types: dict[str, str] | None = None,
    ) -> set[str]:
        variables = set(self.database.variable_types)
        variables.update((variable_types or {}).keys())
        return {
            node.op for node in expression.walk()
            if node.op not in variables
            and node.op not in {"|-", "statement"}
        }

    def _ordered_assertions(self, state: ProofState) -> list[Theorem]:
        assertions = list(self.assertions.values())
        if not self._target_formulas:
            return assertions
        goal_symbols = self._fixed_symbols(
            state.current_goal, dict(state.variable_types)
        )
        target_ranked: list[tuple[float, set[str]]] = []
        for target in self._target_formulas:
            target_symbols = self._fixed_symbols(target)
            union = goal_symbols | target_symbols
            similarity = (
                len(goal_symbols & target_symbols) / len(union)
                if union else 0.0
            )
            target_ranked.append((similarity, target_symbols))
        target_ranked.sort(key=lambda item: item[0], reverse=True)
        context_symbols = set(goal_symbols)
        for _, symbols in target_ranked[:3]:
            context_symbols.update(symbols)

        def priority(assertion: Theorem) -> tuple[float, int, str]:
            symbols = self._fixed_symbols(
                assertion.conclusion, assertion.variable_types
            )
            for hypothesis in assertion.hypotheses:
                symbols.update(self._fixed_symbols(
                    hypothesis.expr, assertion.variable_types
                ))
            union = symbols | context_symbols
            overlap = (
                len(symbols & context_symbols) / len(union)
                if union else 0.0
            )
            return (-overlap, len(assertion.hypotheses), assertion.name)

        return sorted(assertions, key=priority)

    def reset_metrics(self) -> None:
        self.metrics = EnvironmentMetrics()

    @staticmethod
    def _unify_schema(
        pattern: Node,
        variable_types: dict[str, str],
        target: Node,
        *,
        namespace: str,
    ) -> dict[str, Node]:
        """Unify a rule schema after renaming its variables apart."""

        renamed = {
            variable: Node(f"__{namespace}_{variable}")
            for variable in variable_types
        }
        renamed_pattern = substitute_simultaneous(pattern, renamed)
        substitution = unify(
            renamed_pattern,
            target,
            {node.op for node in renamed.values()},
        )
        return {
            variable: substitution[replacement.op]
            for variable, replacement in renamed.items()
            if replacement.op in substitution
        }

    def directly_closable(
        self,
        state: ProofState,
        goal: Node,
    ) -> bool:
        """Whether an assumption or a premise-free assertion closes goal."""

        key = (state, goal)
        cached = self._direct_closure_cache.get(key)
        if cached is not None:
            return cached
        if goal in state.hypotheses:
            self._direct_closure_cache[key] = True
            return True
        for assertion in self.assertions.values():
            if assertion.hypotheses:
                continue
            try:
                substitution = self._unify_schema(
                    assertion.conclusion,
                    assertion.variable_types,
                    goal,
                    namespace=f"closure_{assertion.name}",
                )
                mandatory = {
                    node.op
                    for node in assertion.conclusion.walk()
                    if node.op in assertion.variable_types
                }
                if mandatory - set(substitution):
                    continue
                self._check_distinct(assertion, substitution, state)
                self._direct_closure_cache[key] = True
                return True
            except (UnificationError, InvalidTactic):
                continue
        self._direct_closure_cache[key] = False
        return False

    def one_step_closable(
        self,
        state: ProofState,
        goal: Node,
    ) -> bool:
        """Cheap bounded prover used only as a search-ordering feature."""

        key = (state, goal)
        cached = self._one_step_closure_cache.get(key)
        if cached is not None:
            return cached
        probe = ProofState(
            state.hypotheses,
            (goal,),
            state.d_constraints,
            state.variable_types,
        )
        tactics = self.enumerate_tactics(
            probe,
            max_candidates_per_variable=64,
            max_tactics=128,
            include_derived=True,
        )
        for tactic in tactics:
            try:
                transition = self.apply(probe, tactic)
            except InvalidTactic:
                continue
            if transition.after.solved or (
                transition.generated_goals
                and all(
                    self.directly_closable(
                        transition.after, child
                    )
                    for child in transition.generated_goals
                )
            ):
                self._one_step_closure_cache[key] = True
                return True
        self._one_step_closure_cache[key] = False
        return False

    def _well_typed(
        self,
        expressions: Iterable[Node],
        variable_types: dict[str, str],
    ) -> bool:
        expression_tuple = tuple(expressions)
        cache_key = (
            expression_tuple,
            tuple(sorted(variable_types.items())),
        )
        cached = self._well_typed_cache.get(cache_key)
        if cached is not None:
            return cached
        old_types = dict(self.database.variable_types)
        self.database.variable_types.update(variable_types)
        try:
            result = all(
                self._parser.parse_expression(
                    expression.to_prefix().split()
                ) == expression
                for expression in expression_tuple
            )
        except ParseError:
            result = False
        finally:
            self.database.variable_types.clear()
            self.database.variable_types.update(old_types)
        self._well_typed_cache[cache_key] = result
        return result

    @staticmethod
    def _expression_variables(
        expression: Node,
        variable_types: dict[str, str],
    ) -> set[str]:
        return {
            node.op
            for node in expression.walk()
            if node.op in variable_types
        }

    def _check_distinct(
        self,
        assertion: Theorem,
        substitution: dict[str, Node],
        state: ProofState,
    ) -> None:
        state_types = dict(state.variable_types)
        for left, right in assertion.d_constraints:
            left_value = substitute_simultaneous(
                Node(left), substitution
            )
            right_value = substitute_simultaneous(
                Node(right), substitution
            )
            left_variables = self._expression_variables(
                left_value, state_types
            )
            right_variables = self._expression_variables(
                right_value, state_types
            )
            if left_variables & right_variables:
                raise InvalidTactic(
                    f"$d {left} {right} collapses under substitution"
                )
            for a in left_variables:
                for b in right_variables:
                    if (
                        normalized_pair(a, b)
                        not in state.d_constraints
                    ):
                        raise InvalidTactic(
                            f"state lacks required $d {a} {b}"
                        )

    def apply(
        self,
        state: ProofState,
        tactic: Tactic,
    ) -> Transition:
        self.metrics.apply_requests += 1
        cache_key = (state, tactic)
        cached = self._transition_cache.get(cache_key)
        if cached is not None:
            self.metrics.apply_cache_hits += 1
            return cached
        self.metrics.transition_evaluations += 1
        try:
            return self._apply_uncached(state, tactic, cache_key)
        except InvalidTactic:
            self.metrics.invalid_tactics += 1
            raise

    def _apply_uncached(
        self,
        state: ProofState,
        tactic: Tactic,
        cache_key: tuple[ProofState, Tactic],
    ) -> Transition:
        if state.solved:
            raise InvalidTactic("cannot act on a solved state")
        if tactic.rule == "<ASSUMPTION>":
            if state.current_goal not in state.hypotheses:
                raise InvalidTactic("current goal is not a hypothesis")
            after = ProofState(
                state.hypotheses,
                state.goals[1:],
                state.d_constraints,
                state.variable_types,
            )
            transition = Transition(state, tactic, after, (), ())
            self._transition_cache[cache_key] = transition
            return transition

        assertion = self.assertions.get(tactic.rule)
        if assertion is None:
            raise InvalidTactic(f"unknown logical assertion {tactic.rule!r}")
        explicit = tactic.substitution_dict()
        unknown = set(explicit) - set(assertion.variable_types)
        if unknown:
            raise InvalidTactic(
                "substitution contains unknown variables: "
                + ", ".join(sorted(unknown))
            )
        renamed = {
            variable: Node(
                f"__apply_{assertion.name}_{variable}"
            )
            for variable in assertion.variable_types
        }
        renamed_conclusion = substitute_simultaneous(
            assertion.conclusion, renamed
        )
        renamed_explicit = {
            renamed[variable].op: value
            for variable, value in explicit.items()
        }
        try:
            internal_substitution = unify(
                renamed_conclusion,
                state.current_goal,
                {node.op for node in renamed.values()},
                renamed_explicit,
            )
        except UnificationError as exc:
            raise InvalidTactic(str(exc)) from exc
        substitution = {
            variable: internal_substitution[replacement.op]
            for variable, replacement in renamed.items()
            if replacement.op in internal_substitution
        }

        mandatory = {
            variable
            for expression in [
                assertion.conclusion,
                *(h.expr for h in assertion.hypotheses),
            ]
            for node in expression.walk()
            for variable in [node.op]
            if variable in assertion.variable_types
        }
        unresolved = mandatory - set(substitution)
        if unresolved:
            raise InvalidTactic(
                "unresolved rule variables: "
                + ", ".join(sorted(unresolved))
            )
        state_types = dict(state.variable_types)
        for variable, value in substitution.items():
            expected_type = assertion.variable_types[variable]
            if not self._has_type(
                value, expected_type, state_types
            ):
                raise InvalidTactic(
                    f"substitution for {variable} is not "
                    f"{expected_type}: {value}"
                )
        conclusion = substitute_simultaneous(
            assertion.conclusion, substitution
        )
        if conclusion != state.current_goal:
            raise InvalidTactic(
                f"instantiated conclusion {conclusion} does not match "
                f"goal {state.current_goal}"
            )
        new_goals = tuple(
            substitute_simultaneous(h.expr, substitution)
            for h in assertion.hypotheses
        )
        if not self._well_typed(
            [conclusion, *new_goals],
            state_types,
        ):
            raise InvalidTactic("instantiated action is not well typed")
        self._check_distinct(assertion, substitution, state)
        pending = tuple(
            goal for goal in new_goals
            if goal not in state.hypotheses
        )
        after = ProofState(
            state.hypotheses,
            (*pending, *state.goals[1:]),
            state.d_constraints,
            state.variable_types,
        )
        transition = Transition(
            state,
            tactic,
            after,
            pending,
            tuple(sorted(substitution.items())),
            assertion,
        )
        self._transition_cache[cache_key] = transition
        return transition

    def _has_type(
        self,
        node: Node,
        typecode: str,
        variable_types: dict[str, str],
    ) -> bool:
        key = (node, typecode, tuple(sorted(variable_types.items())))
        cached = self._type_cache.get(key)
        if cached is not None:
            return cached
        old_types = dict(self.database.variable_types)
        self.database.variable_types.update(variable_types)
        try:
            parsed = self._parser.parse_expression(
                [typecode, *node.to_prefix().split()]
            )
            result = parsed.args == (node,)
        except ParseError:
            result = False
        finally:
            self.database.variable_types.clear()
            self.database.variable_types.update(old_types)
        self._type_cache[key] = result
        return result

    def _candidate_expressions(
        self,
        state: ProofState,
        typecode: str,
        limit: int,
        include_derived: bool,
    ) -> list[Node]:
        cache_key = (state, typecode, limit, include_derived)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        local_limit = limit if typecode != "wff" else max(4, limit // 2)
        nodes = self._local_candidate_expressions(
            state,
            typecode,
            local_limit,
        )
        if (
            not include_derived
            or typecode != "wff"
            or len(nodes) >= limit
        ):
            result = nodes[:limit]
            self._candidate_cache[cache_key] = tuple(result)
            return result

        # MP often needs an intermediate formula that is not a subformula of
        # the goal.  Instantiate closed source assertions from typed local
        # terms/formulas to propose such grounded intermediate facts.
        types = dict(state.variable_types)
        seen = set(nodes)
        closed_assertions = [
            assertion
            for assertion in self._ordered_assertions(state)
            if (
                not assertion.hypotheses
                and assertion.conclusion.op == "|-"
                and len(assertion.conclusion.args) == 1
                and sum(
                    1 for _ in assertion.conclusion.args[0].walk()
                ) <= 64
            )
        ]

        # First recover lemma-shaped "auxiliary lines" by matching a subtree
        # of a source theorem against a subtree of the current goal.  For
        # example, matching the right side of df-an against a target
        # antecedent reconstructs the full biconditional needed by MP.
        goal_formula = (
            state.current_goal.args[0]
            if (
                state.current_goal.op == "|-"
                and len(state.current_goal.args) == 1
            )
            else state.current_goal
        )
        goal_nodes = tuple(goal_formula.walk())
        goal_node_set = set(goal_nodes)
        anchored: dict[Node, tuple[int, int, int]] = {}
        for assertion in closed_assertions:
            formula = assertion.conclusion.args[0]
            original_variables = set(assertion.variable_types)
            mandatory = {
                node.op
                for node in formula.walk()
                if node.op in original_variables
            }
            # Source schemas and benchmark states often both use names such
            # as x/phi.  Standardize the schema apart so the unifier treats a
            # target x as a constant of this episode, not as the same
            # metavariable object.
            renamed = {
                variable: Node(
                    f"__aux_{assertion.name}_{variable}"
                )
                for variable in original_variables
            }
            renamed_formula = substitute_simultaneous(
                formula, renamed
            )
            reverse = {
                replacement.op: variable
                for variable, replacement in renamed.items()
            }
            variables = set(reverse)
            renamed_mandatory = {
                renamed[variable].op for variable in mandatory
            }
            for pattern in renamed_formula.walk():
                for target in goal_nodes:
                    if (
                        pattern.op not in variables
                        and pattern.op != target.op
                    ):
                        continue
                    try:
                        substitution = unify(
                            pattern, target, variables
                        )
                    except UnificationError:
                        continue
                    if renamed_mandatory - set(substitution):
                        continue
                    original_substitution = {
                        reverse[variable]: value
                        for variable, value in substitution.items()
                        if variable in reverse
                    }
                    candidate = substitute_simultaneous(
                        formula, original_substitution
                    )
                    if candidate in seen:
                        continue
                    candidate_nodes = tuple(candidate.walk())
                    unique_candidate_nodes = set(candidate_nodes)
                    overlap = len(
                        unique_candidate_nodes & goal_node_set
                    )
                    candidate_size = len(candidate_nodes)
                    anchor_size = sum(1 for _ in pattern.walk())
                    old = anchored.get(candidate)
                    rank = (
                        1000 * anchor_size // candidate_size,
                        1000 * overlap // len(unique_candidate_nodes),
                        -candidate_size,
                    )
                    if old is None or rank > old:
                        anchored[candidate] = rank
        for candidate, _ in sorted(
            anchored.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            if not self._has_type(candidate, "wff", types):
                continue
            seen.add(candidate)
            nodes.append(candidate)
            if len(nodes) >= limit:
                self._candidate_cache[cache_key] = tuple(nodes)
                return nodes

        for assertion in closed_assertions:
            variables = sorted({
                node.op
                for node in assertion.conclusion.walk()
                if node.op in assertion.variable_types
            })
            choices: list[list[Node]] = []
            feasible = True
            for variable in variables:
                variable_type = assertion.variable_types[variable]
                candidates = self._local_candidate_expressions(
                    state,
                    variable_type,
                    4,
                )
                if not candidates:
                    feasible = False
                    break
                choices.append(candidates)
            if not feasible:
                continue
            products = itertools.product(*choices) if choices else [()]
            for values in itertools.islice(products, 32):
                substitution = dict(zip(variables, values))
                formula = substitute_simultaneous(
                    assertion.conclusion.args[0],
                    substitution,
                )
                if (
                    formula not in seen
                ):
                    seen.add(formula)
                    nodes.append(formula)
                    if len(nodes) >= limit:
                        self._candidate_cache[cache_key] = tuple(nodes)
                        return nodes
        self._candidate_cache[cache_key] = tuple(nodes)
        return nodes

    def _local_candidate_expressions(
        self,
        state: ProofState,
        typecode: str,
        limit: int,
    ) -> list[Node]:
        cache_key = (state, typecode, limit)
        cached = self._local_candidate_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        types = dict(state.variable_types)
        nodes: list[Node] = []
        seen: set[Node] = set()
        if typecode == "term" and self._bounded_nat_max >= 0:
            for value in range(self._bounded_nat_max + 1):
                numeral = self._numeral(value)
                if numeral not in seen:
                    seen.add(numeral)
                    nodes.append(numeral)
                    if len(nodes) >= limit:
                        self._local_candidate_cache[cache_key] = tuple(nodes)
                        return nodes
        for expression in [state.current_goal, *state.hypotheses]:
            roots = expression.args if len(expression.args) == 1 else (expression,)
            for root in roots:
                for node in root.walk():
                    if (
                        node not in seen
                        and self._has_type(node, typecode, types)
                    ):
                        seen.add(node)
                        nodes.append(node)
                        if len(nodes) >= limit:
                            self._local_candidate_cache[cache_key] = tuple(
                                nodes
                            )
                            return nodes
        self._local_candidate_cache[cache_key] = tuple(nodes)
        return nodes

    def enumerate_tactics(
        self,
        state: ProofState,
        *,
        max_candidates_per_variable: int = 24,
        max_tactics: int = 256,
        include_derived: bool = True,
    ) -> list[Tactic]:
        self.metrics.tactic_enumerations += 1
        cache_key = (
            state,
            max_candidates_per_variable,
            max_tactics,
            include_derived,
        )
        cached = self._tactic_cache.get(cache_key)
        if cached is not None:
            self.metrics.tactic_cache_hits += 1
            self.metrics.candidates_returned += len(cached)
            return list(cached)
        if state.solved:
            return []
        tactics: list[Tactic] = []
        if state.current_goal in state.hypotheses:
            tactics.append(Tactic.create("<ASSUMPTION>"))
        per_assertion_limit = max(
            1,
            min(16, max_tactics // 4 if max_tactics >= 4 else 1),
        )
        for assertion in self._ordered_assertions(state):
            assertion_added = 0
            try:
                base = self._unify_schema(
                    assertion.conclusion,
                    assertion.variable_types,
                    state.current_goal,
                    namespace=f"rule_{assertion.name}",
                )
            except UnificationError:
                continue
            mandatory = {
                node.op
                for expression in [
                    assertion.conclusion,
                    *(h.expr for h in assertion.hypotheses),
                ]
                for node in expression.walk()
                if node.op in assertion.variable_types
            }
            unresolved = sorted(mandatory - set(base))
            choices: list[list[Node]] = []
            feasible = True
            for variable in unresolved:
                candidates = self._candidate_expressions(
                    state,
                    assertion.variable_types[variable],
                    max_candidates_per_variable,
                    include_derived,
                )
                if not candidates:
                    feasible = False
                    break
                choices.append(candidates)
            if not feasible:
                continue
            products = itertools.product(*choices) if choices else [()]
            for values in products:
                substitution = {
                    **base,
                    **dict(zip(unresolved, values)),
                }
                tactic = Tactic.create(assertion.name, substitution)
                try:
                    self.apply(state, tactic)
                except InvalidTactic:
                    continue
                tactics.append(tactic)
                assertion_added += 1
                if len(tactics) >= max_tactics:
                    self._tactic_cache[cache_key] = tuple(tactics)
                    self.metrics.candidates_returned += len(tactics)
                    return tactics
                if assertion_added >= per_assertion_limit:
                    break
        self._tactic_cache[cache_key] = tuple(tactics)
        self.metrics.candidates_returned += len(tactics)
        return tactics


def parse_tactic_tokens(
    tokens: Iterable[str],
    state_theorem: Theorem,
    tokenizer: MetamathTokenizer,
    database: Database,
    *,
    environment: BackwardEnvironment | None = None,
) -> Tactic:
    sequence = list(tokens)
    if "<EOS>" in sequence:
        sequence = sequence[:sequence.index("<EOS>") + 1]
    if "<PROPOSE_LEMMA>" in sequence:
        try:
            start = sequence.index("<LEMMA>") + 1
            end = sequence.index("<END_LEMMA>", start)
        except ValueError as exc:
            raise InvalidTactic(
                "lemma action lacks LEMMA delimiters"
            ) from exc
        canonical = tokenizer.canonical_variables(state_theorem)
        parser = MetamathParser()
        parser.database = database
        old_types = dict(database.variable_types)
        database.variable_types.update(state_theorem.variable_types)
        try:
            lemma = parser.parse_expression([
                canonical.decode_symbol(item)
                for item in sequence[start:end]
            ])
        except ParseError as exc:
            raise InvalidTactic(str(exc)) from exc
        finally:
            database.variable_types.clear()
            database.variable_types.update(old_types)
        if lemma.op != "|-" or len(lemma.args) != 1:
            raise InvalidTactic(
                "intermediate lemma must be a |- assertion"
            )
        return Tactic.create(
            PROPOSE_LEMMA_RULE,
            {LEMMA_BINDING: lemma},
        )
    try:
        rule_index = sequence.index("<RULE>")
        subst_index = sequence.index("<SUBST>")
    except ValueError as exc:
        raise InvalidTactic("action lacks RULE or SUBST marker") from exc
    if subst_index != rule_index + 2:
        raise InvalidTactic("malformed rule header")
    rule_name = sequence[rule_index + 1]
    if (
        rule_name.startswith("gen_df_")
        and tokenizer.pa_plus_context.bridge_variable_order != "sorted-v1"
    ):
        raise InvalidTactic(
            "legacy PA+ bridge binding order is not reproducible; regenerate corpus"
        )
    if rule_name == "<ASSUMPTION>":
        return Tactic.create(rule_name)
    if environment is not None and environment.database is not database:
        raise InvalidTactic("action environment belongs to a different database")
    rules = (
        environment.assertions
        if environment is not None else database.logical_assertions
    )
    rule = rules.get(rule_name)
    if rule is None:
        raise InvalidTactic(f"unknown rule {rule_name!r}")
    canonical = tokenizer.canonical_variables(state_theorem)
    parser = MetamathParser()
    parser.database = database
    old_types = dict(database.variable_types)
    database.variable_types.update(state_theorem.variable_types)
    substitution: dict[str, Node] = {}
    cursor = subst_index + 1
    try:
        while cursor < len(sequence):
            token = sequence[cursor]
            if token in ("<END_ACTION>", "<EOS>"):
                break
            if token != "<BIND>" or cursor + 3 >= len(sequence):
                raise InvalidTactic("malformed substitution binding")
            variable = sequence[cursor + 1]
            match = VARIABLE_TOKEN_PATTERN.match(variable)
            if match is not None:
                requested_type = match.group(1)
                requested_index = int(match.group(2))
                counters: dict[str, int] = {}
                resolved = None
                order = [
                    floating.expr.args[0].op
                    for floating in rule.floating
                ] or sorted(rule.variable_types)
                for candidate in order:
                    candidate_type = rule.variable_types[candidate]
                    index = counters.get(candidate_type, 0)
                    counters[candidate_type] = index + 1
                    if (
                        candidate_type == requested_type
                        and index == requested_index
                    ):
                        resolved = candidate
                        break
                if resolved is None:
                    raise InvalidTactic(
                        f"unknown typed rule variable {variable}"
                    )
                variable = resolved
            if sequence[cursor + 2] != "<TO>":
                raise InvalidTactic("binding lacks TO marker")
            try:
                end = sequence.index("<END_BIND>", cursor + 3)
            except ValueError as exc:
                raise InvalidTactic("binding lacks END_BIND") from exc
            value_tokens = [
                canonical.decode_symbol(item)
                for item in sequence[cursor + 3:end]
            ]
            typecode = rule.variable_types.get(variable)
            if typecode is None:
                raise InvalidTactic(f"unknown rule variable {variable}")
            try:
                typed = parser.parse_expression([typecode, *value_tokens])
            except ParseError as exc:
                raise InvalidTactic(str(exc)) from exc
            if len(typed.args) != 1:
                raise InvalidTactic("binding did not parse as one expression")
            substitution[variable] = typed.args[0]
            cursor = end + 1
    finally:
        database.variable_types.clear()
        database.variable_types.update(old_types)
    return Tactic.create(rule_name, substitution)

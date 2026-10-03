from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Protocol

from neural_prover.environment import (
    LEMMA_BINDING,
    PROPOSE_LEMMA_RULE,
    BackwardEnvironment,
    InvalidTactic,
    ProofState,
    Tactic,
    Transition,
)
from neural_prover.mcts import MCTSExperience, MCTSResult
from neural_prover.search import RankedTactic, SearchResult, TacticPolicy


class ActionSet(Protocol):
    @property
    def all(self) -> tuple[Tactic, ...]: ...


class ActionGenerator(Protocol):
    def actions(self, state: ProofState) -> ActionSet: ...


@dataclass(frozen=True, slots=True)
class HTPSConfig:
    """Search limits for verified HyperTree Proof Search.

    ``expansion_budget`` counts distinct goal nodes expanded by the symbolic
    environment. One simulation selects a partial proof hypertree and may
    therefore expand several frontier goals.
    """

    simulations: int = 800
    expansion_budget: int = 4_000
    branching: int = 32
    c_puct: float = 1.5
    policy_temperature: float = 1.0
    depth_decay: float = 0.99
    virtual_loss: float = 1.0
    max_frontier: int = 16
    parallel_selections: int = 4
    critic_visit_threshold: int = 2
    randomize: bool = False
    seed: int = 7

    def validate(self) -> None:
        if self.simulations <= 0 or self.expansion_budget <= 0:
            raise ValueError("search budgets must be positive")
        if (
            self.branching <= 0
            or self.max_frontier <= 0
            or self.parallel_selections <= 0
        ):
            raise ValueError("branching and max_frontier must be positive")
        if not 0.0 < self.depth_decay <= 1.0:
            raise ValueError("depth_decay must be in (0, 1]")


def materialize_search_config(
    config: HTPSConfig,
    episode_index: int,
) -> HTPSConfig:
    """Return the fully recorded search configuration for one episode.

    HTPS-style training benefits from varying the search effort and PUCT
    constants between attempts.  The supplied budgets remain hard upper
    bounds, and sampling is deterministic from ``seed`` and episode index.
    """

    config.validate()
    episode_seed = config.seed + 1_000_003 * episode_index
    if not config.randomize:
        return replace(config, seed=episode_seed)
    rng = random.Random(episode_seed)
    min_simulations = max(1, config.simulations // 2)
    min_expansions = max(1, config.expansion_budget // 2)
    min_branching = max(1, config.branching // 2)
    return replace(
        config,
        simulations=rng.randint(min_simulations, config.simulations),
        expansion_budget=rng.randint(min_expansions, config.expansion_budget),
        branching=rng.randint(min_branching, config.branching),
        c_puct=10.0 ** rng.uniform(-1.0, 1.0),
        policy_temperature=rng.uniform(0.8, 2.0),
        depth_decay=rng.choice((0.8, 0.9, 0.95, 0.99, 1.0)),
        randomize=False,
        seed=episode_seed,
    )


@dataclass(slots=True)
class HyperEdge:
    tactic: Tactic
    prior: float
    initial_value: float
    children: tuple[int, ...]
    guarded: bool = False
    visits: int = 0
    virtual_visits: int = 0
    value_sum: float = 0.0
    solved: bool = False
    invalid: bool = False

    @property
    def q_value(self) -> float:
        return (
            self.value_sum / self.visits
            if self.visits
            else self.initial_value
        )


@dataclass(slots=True)
class GoalNode:
    state: ProofState
    edges: list[HyperEdge] = field(default_factory=list)
    parents: set[tuple[int, int]] = field(default_factory=set)
    expanded: bool = False
    solved: bool = False
    invalid: bool = False
    visits: int = 0
    prior_value: float = 0.5
    proof_edge: int | None = None


@dataclass(frozen=True, slots=True)
class HTPSMetrics:
    simulations: int
    expansions: int
    goal_nodes: int
    hyperedges: int
    transposition_hits: int
    cycles_rejected: int
    guarded_edges: int
    maximum_frontier: int


class HyperTreeProofSearch:
    """Verified AND/OR hypergraph search with guarded lemma edges.

    Ordinary Metamath actions are OR choices whose premises form an AND set.
    A lemma proposal is represented as an ordered/guarded AND edge: first
    prove ``Gamma |- L``; only then search ``Gamma,L |- G``.  The extracted
    hypertree is replayed through ``LemmaBackwardEnvironment`` before it is
    returned, so no graph-only success can bypass the symbolic kernel.
    """

    def __init__(
        self,
        environment: BackwardEnvironment,
        action_generator: ActionGenerator,
        policy: TacticPolicy,
        config: HTPSConfig | None = None,
    ) -> None:
        self.environment = environment
        self.action_generator = action_generator
        self.policy = policy
        self.config = config or HTPSConfig()
        self.config.validate()
        self.random = random.Random(self.config.seed)
        self.nodes: list[GoalNode] = []
        self._by_state: dict[ProofState, int] = {}
        self._expansions = 0
        self._transposition_hits = 0
        self._cycles_rejected = 0
        self._maximum_frontier = 0

    @staticmethod
    def _single_goal(
        state: ProofState,
        goal,
        *,
        hypotheses: tuple | None = None,
    ) -> ProofState:
        return ProofState(
            state.hypotheses if hypotheses is None else hypotheses,
            (goal,),
            state.d_constraints,
            state.variable_types,
        )

    def _intern(self, state: ProofState, prior_value: float = 0.5) -> int:
        if len(state.goals) != 1:
            raise ValueError("HTPS goal nodes must contain exactly one goal")
        existing = self._by_state.get(state)
        if existing is not None:
            self._transposition_hits += 1
            self.nodes[existing].prior_value = max(
                self.nodes[existing].prior_value,
                max(0.0, min(1.0, prior_value)),
            )
            return existing
        node_id = len(self.nodes)
        self._by_state[state] = node_id
        self.nodes.append(GoalNode(
            state=state,
            prior_value=max(0.0, min(1.0, prior_value)),
        ))
        return node_id

    def _rank(self, state: ProofState) -> list[RankedTactic]:
        actions = self.action_generator.actions(state)
        rank_hybrid = getattr(self.policy, "rank_hybrid", None)
        if callable(rank_hybrid):
            ranked = rank_hybrid(state, actions)
        else:
            ranked = self.policy.rank(state, list(actions.all))
        return ranked[: self.config.branching]

    def _priors(self, ranked: list[RankedTactic]) -> list[float]:
        if not ranked:
            return []
        temperature = max(self.config.policy_temperature, 1e-6)
        maximum = max(item.log_probability for item in ranked)
        weights = [
            math.exp((item.log_probability - maximum) / temperature)
            for item in ranked
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    def _edge_children(
        self,
        node_id: int,
        proposal: RankedTactic,
    ) -> tuple[tuple[int, ...], bool] | None:
        node = self.nodes[node_id]
        state = node.state
        try:
            transition = self.environment.apply(state, proposal.tactic)
        except InvalidTactic:
            return None
        guarded = proposal.tactic.rule == PROPOSE_LEMMA_RULE
        if guarded:
            lemma = proposal.tactic.substitution_dict().get(LEMMA_BINDING)
            if lemma is None:
                return None
            lemma_state = self._single_goal(state, lemma)
            continuation_hypotheses = (
                state.hypotheses
                if lemma in state.hypotheses
                else (*state.hypotheses, lemma)
            )
            continuation = self._single_goal(
                state,
                state.current_goal,
                hypotheses=continuation_hypotheses,
            )
            raw_children = (lemma_state, continuation)
        else:
            raw_children = tuple(
                self._single_goal(transition.after, goal)
                for goal in transition.generated_goals
            )
        if not raw_children:
            return (), guarded
        child_prior = proposal.next_value ** (1.0 / len(raw_children))
        child_ids = tuple(
            self._intern(child, child_prior) for child in raw_children
        )
        if node_id in child_ids:
            self._cycles_rejected += 1
            return None
        return child_ids, guarded

    def _expand(self, node_id: int) -> None:
        node = self.nodes[node_id]
        if node.expanded or node.solved or node.invalid:
            return
        ranked = self._rank(node.state)
        priors = self._priors(ranked)
        unique: dict[tuple[tuple[int, ...], bool], HyperEdge] = {}
        for proposal, prior in zip(ranked, priors):
            child_spec = self._edge_children(node_id, proposal)
            if child_spec is None:
                continue
            children, guarded = child_spec
            key = (children, guarded)
            edge = HyperEdge(
                tactic=proposal.tactic,
                prior=prior,
                initial_value=max(0.0, min(1.0, proposal.next_value)),
                children=children,
                guarded=guarded,
                solved=not children,
            )
            old = unique.get(key)
            if old is None or edge.prior > old.prior:
                unique[key] = edge
        node.edges = list(unique.values())
        node.expanded = True
        self._expansions += 1
        for edge_index, edge in enumerate(node.edges):
            for child_id in edge.children:
                self.nodes[child_id].parents.add((node_id, edge_index))
        self._refresh_status()

    def _refresh_status(self) -> None:
        changed = True
        while changed:
            changed = False
            for node in reversed(self.nodes):
                old = (node.solved, node.invalid, node.proof_edge)
                for edge in node.edges:
                    edge.solved = all(
                        self.nodes[child].solved for child in edge.children
                    )
                    edge.invalid = any(
                        self.nodes[child].invalid for child in edge.children
                    )
                solved_edges = [
                    index
                    for index, edge in enumerate(node.edges)
                    if edge.solved and not edge.invalid
                ]
                node.solved = bool(solved_edges)
                node.invalid = (
                    node.expanded
                    and not node.solved
                    and (
                        not node.edges
                        or all(edge.invalid for edge in node.edges)
                    )
                )
                if solved_edges:
                    node.proof_edge = min(
                        solved_edges,
                        key=lambda index: self._edge_cost(node.edges[index]),
                    )
                else:
                    node.proof_edge = None
                if old != (node.solved, node.invalid, node.proof_edge):
                    changed = True

    def _proof_cost(self, node_id: int, seen: set[int] | None = None) -> int:
        node = self.nodes[node_id]
        if node.proof_edge is None:
            return 10**9
        active = set() if seen is None else set(seen)
        if node_id in active:
            return 10**9
        active.add(node_id)
        edge = node.edges[node.proof_edge]
        return 1 + sum(self._proof_cost(child, active) for child in edge.children)

    def _edge_cost(self, edge: HyperEdge) -> int:
        return 1 + sum(self._proof_cost(child) for child in edge.children)

    def _value(self, node_id: int) -> float:
        node = self.nodes[node_id]
        if node.solved:
            return 1.0
        if node.invalid:
            return 0.0
        viable = [edge.q_value for edge in node.edges if not edge.invalid]
        return max(viable, default=node.prior_value)

    def _active_children(self, edge: HyperEdge) -> tuple[int, ...]:
        if not edge.guarded:
            return tuple(
                child
                for child in edge.children
                if not self.nodes[child].solved
            )
        for child in edge.children:
            if not self.nodes[child].solved:
                return (child,)
        return ()

    def _productive_nodes(self, forbidden: frozenset[int]) -> set[int]:
        """Find finite partial proofs without revisiting the current ancestry.

        Unexpanded and solved nodes are bases.  An AND edge is productive
        only when all its currently active children are; one such OR choice
        suffices for its parent.  The least fixed point excludes closed
        cyclic components in linear graph work instead of enumerating all
        simple paths through them.  This is a selection filter, never a
        global invalidity or an unprovability claim.
        """
        productive = {
            node_id for node_id, node in enumerate(self.nodes)
            if node_id not in forbidden and not node.invalid
            and (node.solved or not node.expanded)
        }
        ready = deque(productive)
        dependents: list[list[int]] = [[] for _ in self.nodes]
        pending: list[list[int]] = []
        for node_id, node in enumerate(self.nodes):
            if node_id in forbidden or node.invalid or node_id in productive:
                continue
            for edge in node.edges:
                if edge.invalid:
                    continue
                children = set(self._active_children(edge))
                if any(
                    child in forbidden or self.nodes[child].invalid
                    for child in children
                ):
                    continue
                if not children:
                    productive.add(node_id)
                    ready.append(node_id)
                    break
                edge_id = len(pending)
                pending.append([node_id, len(children)])
                for child in children:
                    dependents[child].append(edge_id)
        while ready:
            child = ready.popleft()
            for edge_id in dependents[child]:
                pending[edge_id][1] -= 1
                parent, remaining = pending[edge_id]
                if remaining == 0 and parent not in productive:
                    productive.add(parent)
                    ready.append(parent)
        return productive

    def _select_edge(
        self,
        node: GoalNode,
        excluded: set[int] | None = None,
    ) -> int | None:
        candidates = [
            index for index, edge in enumerate(node.edges)
            if not edge.invalid and not edge.solved
            and (excluded is None or index not in excluded)
        ]
        if not candidates:
            return None
        total = max(1, sum(node.edges[index].visits for index in candidates))
        sqrt_total = math.sqrt(total)

        def score(index: int) -> float:
            edge = node.edges[index]
            count = edge.visits + edge.virtual_visits
            return (
                edge.q_value
                + self.config.c_puct
                * edge.prior
                * sqrt_total
                / (1 + count)
                - self.config.virtual_loss * edge.virtual_visits
            )

        scored = [(index, score(index)) for index in candidates]
        maximum = max(value for _, value in scored)
        tied = [
            index for index, value in scored
            if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)
        ]
        return self.random.choice(tied)

    def _select_hypertree(
        self,
        node_id: int,
        ancestors: frozenset[int],
        selected: list[tuple[int, int]],
        frontier: list[int],
    ) -> bool:
        """Select an acyclic partial proof, backtracking locally on cycles.

        An edge that loops into this path can still be useful from another
        root or transposition.  Never turn that path-dependent obstruction
        into a permanent, global invalid edge.
        """
        if len(frontier) >= self.config.max_frontier:
            return True
        if node_id in ancestors:
            self._cycles_rejected += 1
            return False
        node = self.nodes[node_id]
        node.visits += 1
        if node.solved:
            return True
        if node.invalid:
            return False
        if not node.expanded:
            frontier.append(node_id)
            return True
        lineage = ancestors | {node_id}
        productive = self._productive_nodes(lineage)
        excluded = {
            index for index, edge in enumerate(node.edges)
            if not edge.invalid and not edge.solved
            and any(child not in productive for child in self._active_children(edge))
        }
        self._cycles_rejected += len(excluded)
        while (edge_index := self._select_edge(node, excluded)) is not None:
            edge = node.edges[edge_index]
            selection_start = len(selected)
            frontier_start = len(frontier)
            edge.virtual_visits += 1
            selected.append((node_id, edge_index))
            if all(
                self._select_hypertree(child, lineage, selected, frontier)
                for child in self._active_children(edge)
            ):
                return True
            # This AND choice has no acyclic continuation in the current
            # ancestry.  Undo its pending selection before trying another
            # OR choice, including any virtual visits below this edge.
            for parent_id, selected_edge in selected[selection_start:]:
                pending = self.nodes[parent_id].edges[selected_edge]
                pending.virtual_visits = max(0, pending.virtual_visits - 1)
            del selected[selection_start:]
            del frontier[frontier_start:]
            excluded.add(edge_index)
        return False

    def _backup(self, selected: list[tuple[int, int]]) -> None:
        for node_id, edge_index in reversed(selected):
            edge = self.nodes[node_id].edges[edge_index]
            edge.virtual_visits = max(0, edge.virtual_visits - 1)
            value = math.prod(self._value(child) for child in edge.children)
            value *= self.config.depth_decay
            edge.visits += 1
            edge.value_sum += value
        self._refresh_status()

    def _proof_actions(
        self,
        node_id: int,
        active: frozenset[int] = frozenset(),
    ) -> list[Tactic]:
        if node_id in active:
            raise RuntimeError("cycle in selected proof hypertree")
        node = self.nodes[node_id]
        if node.proof_edge is None:
            raise RuntimeError("cannot extract an unsolved goal")
        edge = node.edges[node.proof_edge]
        actions = [edge.tactic]
        lineage = active | {node_id}
        for child in edge.children:
            actions.extend(self._proof_actions(child, lineage))
        return actions

    def _replay(
        self,
        initial: ProofState,
        actions: list[Tactic],
    ) -> tuple[list[Transition], ProofState]:
        state = initial
        transitions: list[Transition] = []
        for tactic in actions:
            transition = self.environment.apply(state, tactic)
            transitions.append(transition)
            state = transition.after
        if not state.solved:
            raise RuntimeError("extracted hypertree did not replay to a solution")
        return transitions, state

    def _proof_nodes(self, root_id: int) -> set[int]:
        result: set[int] = set()
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            if node_id in result:
                continue
            result.add(node_id)
            node = self.nodes[node_id]
            if node.proof_edge is not None:
                stack.extend(node.edges[node.proof_edge].children)
        return result

    def _experiences(self, root_id: int) -> tuple[MCTSExperience, ...]:
        proof_nodes = self._proof_nodes(root_id) if self.nodes[root_id].solved else set()
        result: list[MCTSExperience] = []
        for node_id, node in enumerate(self.nodes):
            if not node.expanded:
                continue
            tactics = tuple(edge.tactic for edge in node.edges)
            if node_id in proof_nodes and node.proof_edge is not None:
                probabilities = tuple(
                    1.0 if index == node.proof_edge else 0.0
                    for index in range(len(node.edges))
                )
                policy_valid = True
            else:
                total = sum(edge.visits for edge in node.edges)
                probabilities = tuple(
                    edge.visits / total for edge in node.edges
                ) if total else tuple(0.0 for _ in node.edges)
                policy_valid = False
            value_valid = (
                node.solved
                or node.invalid
                or node.visits >= self.config.critic_visit_threshold
            )
            result.append(MCTSExperience(
                state=node.state,
                tactics=tactics,
                visit_probabilities=probabilities,
                value_target=self._value(node_id),
                policy_target_valid=policy_valid,
                value_target_valid=value_valid,
            ))
        return tuple(result)

    def prove(self, initial: ProofState) -> tuple[MCTSResult, HTPSMetrics]:
        if len(initial.goals) != 1:
            raise ValueError("HTPS requires an initial state with one root goal")
        self.nodes = []
        self._by_state = {}
        self._expansions = 0
        self._transposition_hits = 0
        self._cycles_rejected = 0
        self._maximum_frontier = 0
        root_id = self._intern(initial)
        simulations_done = 0
        for simulation in range(self.config.simulations):
            if (
                self.nodes[root_id].solved
                or self.nodes[root_id].invalid
                or self._expansions >= self.config.expansion_budget
            ):
                break
            simulations_done = simulation + 1
            selections: list[list[tuple[int, int]]] = []
            frontier: list[int] = []
            for _ in range(self.config.parallel_selections):
                selected: list[tuple[int, int]] = []
                tree_frontier: list[int] = []
                self._select_hypertree(
                    root_id, frozenset(), selected, tree_frontier
                )
                selections.append(selected)
                frontier.extend(tree_frontier)
                if not selected and tree_frontier:
                    # An unexpanded root/frontier cannot be diversified until
                    # the first policy call has created outgoing hyperedges.
                    break
            frontier = list(dict.fromkeys(frontier))
            self._maximum_frontier = max(
                self._maximum_frontier, len(frontier)
            )
            if not frontier:
                self._refresh_status()
                if not self.nodes[root_id].solved:
                    break
            for node_id in frontier:
                if self._expansions >= self.config.expansion_budget:
                    break
                self._expand(node_id)
            for selected in selections:
                self._backup(selected)

        root = self.nodes[root_id]
        if root.solved:
            actions = self._proof_actions(root_id)
            transitions, final_state = self._replay(initial, actions)
        else:
            actions = []
            transitions = []
            final_state = initial
        search = SearchResult(
            solved=root.solved,
            simulations=simulations_done,
            nodes=len(self.nodes),
            actions=tuple(actions),
            transitions=tuple(transitions),
            final_state=final_state,
        )
        result = MCTSResult(
            search=search,
            root_visits=tuple(
                (edge.tactic, edge.visits) for edge in root.edges
            ),
            experiences=self._experiences(root_id),
            outcome=(
                "certifiable_solution"
                if root.solved
                else "proven_dead_end"
                if root.invalid
                else "budget_exhausted"
            ),
        )
        metrics = HTPSMetrics(
            simulations=simulations_done,
            expansions=self._expansions,
            goal_nodes=len(self.nodes),
            hyperedges=sum(len(node.edges) for node in self.nodes),
            transposition_hits=self._transposition_hits,
            cycles_rejected=self._cycles_rejected,
            guarded_edges=sum(
                edge.guarded for node in self.nodes for edge in node.edges
            ),
            maximum_frontier=self._maximum_frontier,
        )
        return result, metrics

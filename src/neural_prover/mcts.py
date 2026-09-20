from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .environment import (
    BackwardEnvironment,
    ProofState,
    Tactic,
    Transition,
)
from .hybrid import HybridActionGenerator
from .search import (
    RankedTactic,
    SearchResult,
    TacticPolicy,
)


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 800
    c_puct: float = 1.5
    policy_temperature: float = 1.0
    max_depth: int = 24
    branching: int = 48
    step_penalty: float = 0.01
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.15
    seed: int = 7


@dataclass(slots=True)
class MCTSEdge:
    tactic: Tactic
    prior: float
    transition: Transition
    visits: int = 0
    value_sum: float = 0.0
    child: "MCTSNode | None" = None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class MCTSNode:
    state: ProofState
    depth: int
    edges: list[MCTSEdge] = field(default_factory=list)
    expanded: bool = False
    visits: int = 0


@dataclass(frozen=True, slots=True)
class MCTSExperience:
    state: ProofState
    tactics: tuple[Tactic, ...]
    visit_probabilities: tuple[float, ...]
    value_target: float
    policy_target_valid: bool
    value_target_valid: bool


@dataclass(frozen=True, slots=True)
class MCTSResult:
    search: SearchResult
    root_visits: tuple[tuple[Tactic, int], ...]
    experiences: tuple[MCTSExperience, ...]
    outcome: str


class ProofMCTS:
    """AlphaZero-style PUCT search in the verified Metamath environment."""

    def __init__(
        self,
        environment: BackwardEnvironment,
        action_generator: HybridActionGenerator,
        policy: TacticPolicy,
        config: MCTSConfig | None = None,
    ) -> None:
        self.environment = environment
        self.action_generator = action_generator
        self.policy = policy
        self.config = config or MCTSConfig()
        self.random = random.Random(self.config.seed)
        self._expanded_nodes: list[MCTSNode] = []

    def _priors(
        self,
        ranked: list[RankedTactic],
        root: bool,
    ) -> list[float]:
        if not ranked:
            return []
        temperature = max(1e-6, self.config.policy_temperature)
        maximum = max(item.log_probability for item in ranked)
        weights = [
            math.exp((item.log_probability - maximum) / temperature)
            for item in ranked
        ]
        total = sum(weights)
        priors = [weight / total for weight in weights]
        if root and len(priors) > 1 and self.config.dirichlet_fraction > 0:
            noise = [
                self.random.gammavariate(
                    self.config.dirichlet_alpha, 1.0
                )
                for _ in priors
            ]
            noise_total = sum(noise)
            noise = [value / noise_total for value in noise]
            fraction = self.config.dirichlet_fraction
            priors = [
                (1.0 - fraction) * prior + fraction * perturbation
                for prior, perturbation in zip(priors, noise)
            ]
        return priors

    def _expand(
        self,
        node: MCTSNode,
        *,
        root: bool,
    ) -> float:
        actions = self.action_generator.actions(node.state)
        rank_hybrid = getattr(self.policy, "rank_hybrid", None)
        if rank_hybrid is not None:
            ranked = rank_hybrid(node.state, actions)
        else:
            ranked = self.policy.rank(node.state, actions.all)
        ranked = ranked[:self.config.branching]
        priors = self._priors(ranked, root)
        node.edges = []
        for proposal, prior in zip(ranked, priors):
            transition = self.environment.apply(
                node.state, proposal.tactic
            )
            node.edges.append(MCTSEdge(
                proposal.tactic,
                prior,
                transition,
            ))
        node.expanded = True
        self._expanded_nodes.append(node)
        if not ranked:
            return 0.0
        return max(item.next_value for item in ranked)

    def _select(self, node: MCTSNode) -> MCTSEdge:
        total = max(1, sum(edge.visits for edge in node.edges))
        sqrt_total = math.sqrt(total)

        def puct(edge: MCTSEdge) -> float:
            exploration = (
                self.config.c_puct
                * edge.prior
                * sqrt_total
                / (1 + edge.visits)
            )
            return edge.q_value + exploration

        return max(node.edges, key=puct)

    def prove(
        self,
        initial: ProofState,
    ) -> MCTSResult:
        # A ProofMCTS object may be reused for a corpus.  Experiences belong
        # to exactly one search and must never leak between theorem episodes.
        self._expanded_nodes = []
        root = MCTSNode(initial, 0)
        solution: list[Transition] | None = None
        solution_returns: dict[int, float] = {}
        simulations_done = 0
        for simulation in range(self.config.simulations):
            simulations_done = simulation + 1
            node = root
            path: list[MCTSEdge] = []
            transitions: list[Transition] = []
            visited_nodes = [node]
            leaf_value = 0.0
            while True:
                node.visits += 1
                if node.state.solved:
                    leaf_value = 1.0
                    solution = transitions
                    break
                if node.depth >= self.config.max_depth:
                    leaf_value = 0.0
                    break
                if not node.expanded:
                    leaf_value = self._expand(
                        node,
                        root=node is root,
                    )
                    break
                if not node.edges:
                    leaf_value = 0.0
                    break
                edge = self._select(node)
                path.append(edge)
                transitions.append(edge.transition)
                if edge.child is None:
                    edge.child = MCTSNode(
                        edge.transition.after,
                        node.depth + 1,
                    )
                node = edge.child
                visited_nodes.append(node)
            value = leaf_value
            for parent, edge in zip(
                reversed(visited_nodes[:-1]), reversed(path)
            ):
                value = max(0.0, value - self.config.step_penalty)
                edge.visits += 1
                edge.value_sum += value
                if solution is not None:
                    # Supervised values use the same remaining-path return
                    # as backup, independent of depth from the search root.
                    solution_returns[id(parent)] = value
            if solution is not None:
                break

        experiences: list[MCTSExperience] = []
        for node in self._expanded_nodes:
            total = sum(edge.visits for edge in node.edges)
            proven_dead_end = node.expanded and not node.edges
            if total == 0 and not proven_dead_end:
                continue
            solved_path = id(node) in solution_returns
            experiences.append(MCTSExperience(
                state=node.state,
                tactics=tuple(edge.tactic for edge in node.edges),
                visit_probabilities=tuple(
                    edge.visits / total for edge in node.edges
                ),
                value_target=solution_returns.get(id(node), 0.0),
                policy_target_valid=solved_path,
                value_target_valid=solved_path or proven_dead_end,
            ))

        if solution is None:
            best_transitions: list[Transition] = []
            node = root
            while node.edges:
                edge = max(
                    node.edges,
                    key=lambda item: item.visits,
                )
                if edge.visits == 0:
                    break
                best_transitions.append(edge.transition)
                if edge.child is None:
                    break
                node = edge.child
            final_state = (
                best_transitions[-1].after
                if best_transitions else initial
            )
            transitions_tuple = tuple(best_transitions)
        else:
            final_state = (
                solution[-1].after if solution else initial
            )
            transitions_tuple = tuple(solution)
        search = SearchResult(
            solved=solution is not None,
            simulations=simulations_done,
            nodes=sum(1 for _ in self._walk(root)),
            actions=tuple(
                transition.tactic for transition in transitions_tuple
            ),
            transitions=transitions_tuple,
            final_state=final_state,
        )
        return MCTSResult(
            search=search,
            root_visits=tuple(
                (edge.tactic, edge.visits) for edge in root.edges
            ),
            experiences=tuple(experiences),
            outcome=("certifiable_solution" if solution is not None
                     else "budget_exhausted"),
        )

    def _walk(self, root: MCTSNode):
        stack = [root]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            yield node
            for edge in node.edges:
                if edge.child is not None:
                    stack.append(edge.child)

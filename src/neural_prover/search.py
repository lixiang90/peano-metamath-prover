from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch.nn import functional as F

from .environment import (
    BackwardEnvironment,
    InvalidTactic,
    ProofState,
    Tactic,
    Transition,
)
from .model import ProofTransformer
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class RankedTactic:
    tactic: Tactic
    log_probability: float
    next_value: float


class TacticPolicy(Protocol):
    def rank(
        self,
        state: ProofState,
        tactics: list[Tactic],
    ) -> list[RankedTactic]:
        ...


class HeuristicPolicy:
    def __init__(
        self,
        environment: BackwardEnvironment | None = None,
    ) -> None:
        self.environment = environment

    def rank(
        self,
        state: ProofState,
        tactics: list[Tactic],
    ) -> list[RankedTactic]:
        ranked = []
        for tactic in tactics:
            assumption = tactic.rule == "<ASSUMPTION>"
            complexity = len(tactic.substitution)
            score = (2.0 if assumption else 0.0) - 0.05 * complexity
            next_value = 0.5
            if self.environment is not None:
                try:
                    after = self.environment.apply(state, tactic).after
                    score += 10.0 if after.solved else 0.0
                    score -= 0.5 * len(after.goals)
                    score += 5.0 * (
                        len(state.goals) - len(after.goals)
                    )
                    direct = sum(
                        self.environment.directly_closable(after, goal)
                        for goal in after.goals
                    )
                    # A kernel-recognized direct closure is strictly more
                    # reliable than a bounded look-ahead prediction.
                    score += 4.0 * direct
                    near = 0
                    if (
                        after.goals
                        and direct + near == len(after.goals)
                    ):
                        score += 4.0
                    next_value = 1.0 / (1.0 + len(after.goals))
                    if after.goals:
                        next_value = max(
                            next_value,
                            (direct + 0.8 * near)
                            / len(after.goals),
                        )
                except InvalidTactic:
                    continue
            ranked.append(RankedTactic(tactic, score, next_value))
        return sorted(
            ranked,
            key=lambda item: item.log_probability,
            reverse=True,
        )


class TransformerPolicy:
    """Score only symbolically valid actions with the Transformer policy."""

    def __init__(
        self,
        model: ProofTransformer,
        tokenizer: MetamathTokenizer,
        environment: BackwardEnvironment,
        device: str | torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.environment = environment
        self.device = torch.device(device)

    @torch.no_grad()
    def rank(
        self,
        state: ProofState,
        tactics: list[Tactic],
    ) -> list[RankedTactic]:
        if not tactics:
            return []
        theorem = state.as_theorem()
        canonical = self.tokenizer.canonical_variables(theorem)
        serialized_state = self.tokenizer.state_tokens(
            theorem, canonical
        )
        if len(serialized_state) > self.model.config.max_state_tokens:
            # Search can create a larger conjunction of open goals than any
            # supervised sample.  Never truncate a formal state: that could
            # erase a hypothesis or goal and give a misleading neural score.
            # The kernel-backed symbolic policy is the safe OOD fallback.
            return HeuristicPolicy(self.environment).rank(state, tactics)
        state_ids = torch.tensor(
            [self.tokenizer.encode(
                serialized_state
            )],
            dtype=torch.long,
            device=self.device,
        )
        memory, memory_padding, _ = self.model.encode(state_ids)
        entries: list[tuple[Tactic, list[int], ProofState]] = []
        for tactic in tactics:
            if tactic.rule == "<ASSUMPTION>":
                tokens = [
                    "<BOS>",
                    "<ACTION>",
                    "<ASSUMPTION>",
                    "<END_ACTION>",
                    "<EOS>",
                ]
            else:
                assertion = self.environment.assertions[tactic.rule]
                order = [
                    floating.expr.args[0].op
                    for floating in assertion.floating
                ] or list(assertion.variable_types)
                tokens = self.tokenizer.tactic_tokens(
                    tactic.rule,
                    tactic.substitution_dict(),
                    canonical,
                    variable_order=order,
                )
            ids = self.tokenizer.encode(tokens)
            if len(ids) > self.model.config.max_action_tokens:
                continue
            try:
                next_state = self.environment.apply(
                    state, tactic
                ).after
            except InvalidTactic:
                continue
            entries.append((tactic, ids, next_state))
        if not entries:
            return []

        width = max(len(ids) for _, ids, _ in entries)
        action = torch.full(
            (len(entries), width),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for row, (_, ids, _) in enumerate(entries):
            action[row, :len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )
        repeated_memory = memory.expand(len(entries), -1, -1)
        repeated_padding = memory_padding.expand(len(entries), -1)
        logits = self.model.decode(
            action[:, :-1],
            repeated_memory,
            repeated_padding,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        target = action[:, 1:]
        target_mask = target.ne(self.tokenizer.pad_id)
        token_scores = log_probs.gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        policy_scores = (
            (token_scores * target_mask).sum(dim=1)
            / target_mask.sum(dim=1).clamp_min(1)
        )

        next_values = torch.ones(
            len(entries),
            device=self.device,
        )
        unsolved_rows: list[int] = []
        unsolved_ids: list[list[int]] = []
        for row, (_, _, next_state) in enumerate(entries):
            if next_state.solved:
                continue
            next_theorem = next_state.as_theorem()
            next_tokens = self.tokenizer.state_tokens(next_theorem)
            ids = self.tokenizer.encode(next_tokens)
            if len(ids) <= self.model.config.max_state_tokens:
                unsolved_rows.append(row)
                unsolved_ids.append(ids)
            else:
                next_values[row] = 0.0
        if unsolved_ids:
            state_width = max(len(ids) for ids in unsolved_ids)
            next_batch = torch.full(
                (len(unsolved_ids), state_width),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, ids in enumerate(unsolved_ids):
                next_batch[row, :len(ids)] = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.device,
                )
            _, _, values = self.model.encode(next_batch)
            for row, value in zip(unsolved_rows, values):
                next_values[row] = value

        ranked: list[RankedTactic] = []
        for row, (tactic, _, _) in enumerate(entries):
            ranked.append(RankedTactic(
                tactic,
                float(policy_scores[row].item()),
                float(next_values[row].item()),
            ))
        return sorted(
            ranked,
            key=lambda item: item.log_probability,
            reverse=True,
        )


class HybridPolicy:
    """Blend neural construction scores with deterministic proof progress."""

    def __init__(
        self,
        environment: BackwardEnvironment,
        neural: TransformerPolicy,
        *,
        neural_weight: float = 1.0,
        symbolic_weight: float = 0.8,
        lookahead_candidates: int = 4,
    ) -> None:
        self.environment = environment
        self.neural = neural
        self.heuristic = HeuristicPolicy(environment)
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.lookahead_candidates = lookahead_candidates

    def rank(
        self,
        state: ProofState,
        tactics: list[Tactic],
    ) -> list[RankedTactic]:
        neural_ranked = self.neural.rank(state, tactics)
        neural = {item.tactic: item for item in neural_ranked}
        lookahead = {
            item.tactic
            for item in neural_ranked[:self.lookahead_candidates]
        }
        symbolic = {
            item.tactic: item
            for item in self.heuristic.rank(state, tactics)
        }
        combined: list[RankedTactic] = []
        for tactic in tactics:
            neural_item = neural.get(tactic)
            symbolic_item = symbolic.get(tactic)
            if neural_item is None or symbolic_item is None:
                continue
            score = (
                self.neural_weight * neural_item.log_probability
                + self.symbolic_weight * symbolic_item.log_probability
            )
            if tactic in lookahead:
                after = self.environment.apply(state, tactic).after
                direct = sum(
                    self.environment.directly_closable(after, goal)
                    for goal in after.goals
                )
                near = 0
                if (
                    after.goals
                    and direct < len(after.goals)
                    and all(
                        sum(1 for _ in goal.walk()) <= 40
                        for goal in after.goals
                    )
                ):
                    near = sum(
                        not self.environment.directly_closable(
                            after, goal
                        )
                        and self.environment.one_step_closable(
                            after, goal
                        )
                        for goal in after.goals
                    )
                score += 1.5 * near
                if (
                    near
                    and direct + near == len(after.goals)
                ):
                    score += 4.0
            next_value = max(
                neural_item.next_value,
                symbolic_item.next_value,
            )
            combined.append(RankedTactic(tactic, score, next_value))
        return sorted(
            combined,
            key=lambda item: item.log_probability,
            reverse=True,
        )

    def rank_hybrid(self, state, actions):
        """Use deduction for routine steps and the network for constructions.

        This mirrors AlphaGeometry's division of labour: the symbolic engine
        owns ordinary deductions, while the learned policy ranks actions that
        introduce an intermediate expression not already present in the
        proof state.
        """

        deterministic = self.heuristic.rank(
            state, list(actions.deterministic)
        )
        constructions = self.rank(
            state, list(actions.constructions)
        )
        return sorted(
            [*deterministic, *constructions],
            key=lambda item: item.log_probability,
            reverse=True,
        )


@dataclass(slots=True)
class SearchNode:
    state: ProofState
    parent: int | None
    transition: Transition | None
    depth: int
    log_probability: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    solved: bool
    simulations: int
    nodes: int
    actions: tuple[Tactic, ...]
    transitions: tuple[Transition, ...]
    final_state: ProofState


@dataclass(order=True)
class _QueueItem:
    priority: float
    serial: int
    node_id: int = field(compare=False)


class NeuralBestFirstSearch:
    """Best-first AND-goal search guided by policy likelihood and value."""

    def __init__(
        self,
        environment: BackwardEnvironment,
        policy: TacticPolicy,
        *,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        max_depth: int = 16,
        branching: int = 32,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.max_depth = max_depth
        self.branching = branching

    def prove(
        self,
        initial: ProofState,
        *,
        simulations: int = 1_000,
    ) -> SearchResult:
        if initial.solved:
            return SearchResult(True, 0, 1, (), (), initial)
        nodes = [SearchNode(initial, None, None, 0, 0.0)]
        serials = itertools.count()
        queue = [_QueueItem(0.0, next(serials), 0)]
        visited = {initial: 0}
        expanded = 0
        solved_id: int | None = None
        while queue and expanded < simulations:
            item = heapq.heappop(queue)
            node = nodes[item.node_id]
            if node.state.solved:
                solved_id = item.node_id
                break
            if node.depth >= self.max_depth:
                continue
            tactics = self.environment.enumerate_tactics(node.state)
            ranked = self.policy.rank(node.state, tactics)
            expanded += 1
            for proposal in ranked[:self.branching]:
                try:
                    transition = self.environment.apply(
                        node.state, proposal.tactic
                    )
                except InvalidTactic:
                    continue
                next_state = transition.after
                new_log_probability = (
                    node.log_probability + proposal.log_probability
                )
                depth = node.depth + 1
                old_depth = visited.get(next_state)
                if old_depth is not None and old_depth <= depth:
                    continue
                visited[next_state] = depth
                child_id = len(nodes)
                nodes.append(SearchNode(
                    next_state,
                    item.node_id,
                    transition,
                    depth,
                    new_log_probability,
                ))
                # Lower is better.  The number of outstanding goals is an
                # explicit AND-node cost; policy/value refine the ordering.
                priority = (
                    depth
                    + len(next_state.goals)
                    - self.policy_weight * new_log_probability
                    - self.value_weight * proposal.next_value
                )
                heapq.heappush(
                    queue,
                    _QueueItem(priority, next(serials), child_id),
                )
                if next_state.solved:
                    solved_id = child_id
                    queue.clear()
                    break
        if solved_id is None:
            best_id = min(
                range(len(nodes)),
                key=lambda index: (
                    len(nodes[index].state.goals),
                    nodes[index].depth,
                ),
            )
            return SearchResult(
                False,
                expanded,
                len(nodes),
                tuple(self._actions(nodes, best_id)),
                tuple(self._transitions(nodes, best_id)),
                nodes[best_id].state,
            )
        return SearchResult(
            True,
            expanded,
            len(nodes),
            tuple(self._actions(nodes, solved_id)),
            tuple(self._transitions(nodes, solved_id)),
            nodes[solved_id].state,
        )

    @staticmethod
    def _actions(
        nodes: list[SearchNode],
        node_id: int,
    ) -> list[Tactic]:
        actions: list[Tactic] = []
        while nodes[node_id].parent is not None:
            transition = nodes[node_id].transition
            if transition is not None:
                actions.append(transition.tactic)
            node_id = nodes[node_id].parent  # type: ignore[assignment]
        actions.reverse()
        return actions

    @staticmethod
    def _transitions(
        nodes: list[SearchNode],
        node_id: int,
    ) -> list[Transition]:
        transitions: list[Transition] = []
        while nodes[node_id].parent is not None:
            transition = nodes[node_id].transition
            if transition is not None:
                transitions.append(transition)
            node_id = nodes[node_id].parent  # type: ignore[assignment]
        transitions.reverse()
        return transitions

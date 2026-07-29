from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Literal

from .compose import CompositionError, compose
from .database import TheoremDatabase
from .model import Database, Theorem
from .parser import MetamathParser, ParseError


@dataclass(slots=True)
class GeneratorConfig:
    max_ast_depth: int = 32
    max_hypotheses: int = 8
    max_variables: int = 16
    max_proof_depth: int = 8
    max_combinations_per_rule: int = 500
    seed: int | None = None


class Generator:
    def __init__(
        self,
        parsed: Database,
        config: GeneratorConfig | None = None,
        store: TheoremDatabase | None = None,
    ) -> None:
        self.parsed = parsed
        self.config = config or GeneratorConfig()
        self.store = store or TheoremDatabase.from_parsed(parsed)
        self._expression_parser = MetamathParser()
        self._expression_parser.database = parsed
        self.random = random.Random(self.config.seed)
        self.rules = [
            theorem for theorem in parsed.rules.values()
            if theorem.conclusion.op == "|-" and theorem.hypotheses
        ]
        self.generated_count = 0

    def _valid(self, theorem: Theorem) -> bool:
        return (
            theorem.conclusion.depth <= self.config.max_ast_depth
            and all(
                hypothesis.expr.depth <= self.config.max_ast_depth
                for hypothesis in theorem.hypotheses
            )
            and len(theorem.hypotheses) <= self.config.max_hypotheses
            and len(theorem.variable_types) <= self.config.max_variables
            and theorem.proof_depth <= self.config.max_proof_depth
            and self._well_typed(theorem)
        )

    def _well_typed(self, theorem: Theorem) -> bool:
        original_types = dict(self.parsed.variable_types)
        self.parsed.variable_types.update(theorem.variable_types)
        try:
            for expr in [*(h.expr for h in theorem.hypotheses), theorem.conclusion]:
                reparsed = self._expression_parser.parse_expression(
                    expr.to_prefix().split()
                )
                if reparsed != expr:
                    return False
            return True
        except ParseError:
            return False
        finally:
            self.parsed.variable_types.clear()
            self.parsed.variable_types.update(original_types)

    def _try(self, rule: Theorem, matches: list[Theorem | None]) -> bool:
        try:
            theorem = compose(
                rule,
                matches,
                name=f"gen{len(self.store)}_{rule.name}",
                database=self.parsed,
            )
        except CompositionError:
            return False
        if not self._valid(theorem):
            return False
        _, added = self.store.add(theorem)
        if added:
            self.generated_count += 1
        return added

    def random_walk(self, steps: int) -> list[Theorem]:
        start = len(self.store)
        if not self.rules:
            return []
        for _ in range(steps):
            rule = self.random.choice(self.rules)
            matches: list[Theorem | None] = [None] * len(rule.hypotheses)
            count = self.random.randint(1, len(matches))
            for index in self.random.sample(range(len(matches)), count):
                candidates = self.store.candidates(rule.hypotheses[index].expr)
                if candidates:
                    matches[index] = self.random.choice(candidates)
            self._try(rule, matches)
        return list(self.store)[start:]

    def forward_saturation(self, rounds: int = 1) -> list[Theorem]:
        start = len(self.store)
        for _ in range(rounds):
            added_this_round = 0
            snapshot = list(self.store)
            for rule in self.rules:
                choices: list[list[Theorem | None]] = []
                for premise in rule.hypotheses:
                    candidates = [
                        theorem for theorem in snapshot
                        if theorem.conclusion.op == premise.expr.op
                    ]
                    choices.append([None, *candidates])
                tried = 0
                for matches_tuple in itertools.product(*choices):
                    if all(item is None for item in matches_tuple):
                        continue
                    if tried >= self.config.max_combinations_per_rule:
                        break
                    tried += 1
                    if self._try(rule, list(matches_tuple)):
                        added_this_round += 1
            if added_this_round == 0:
                break
        return list(self.store)[start:]

    def generate(
        self,
        mode: Literal["random", "forward", "depth"] = "random",
        steps: int = 100,
    ) -> list[Theorem]:
        if mode == "forward":
            return self.forward_saturation(steps)
        # Depth-limited generation uses the same walker; the configured proof
        # depth is enforced by _valid.
        return self.random_walk(steps)

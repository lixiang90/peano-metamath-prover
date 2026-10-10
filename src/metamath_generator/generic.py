"""Optional, proof-carrying forward sampling over raw Metamath token strings.

This deliberately does not use the project's PA AST, grammar, or quality
heuristics. Bounded word matching proposes substitutions; proofs of every $f,
exact $e matching and $d checking establish their validity. Search is heuristic,
not a complete word-unification decision procedure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Mapping

from .token_mm import (
    Disjoints, Expression, TokenAssertion, TokenDatabase, TokenHypothesis,
    TokenMMError, make_assertion, required_disjoints, substitute, verify_tokens,
)


class SamplingFailure(ValueError):
    pass


@dataclass
class MatchBudget:
    remaining: int

    def spend(self) -> None:
        if self.remaining <= 0:
            raise SamplingFailure("search_budget")
        self.remaining -= 1


def match_tokens(pattern: Expression, target: Expression, variables: set[str],
                 initial: Mapping[str, Expression], budget: MatchBudget,
                 rng: random.Random) -> Iterator[dict[str, Expression]]:
    """Match schematic variables to contiguous strings, including empty ones.

    Target variables are rigid during this call. Repeated pattern variables
    must receive identical strings. All branches share the caller's budget.
    """
    pending = [(0, 0, dict(initial))]
    while pending:
        budget.spend()
        p, t, subst = pending.pop()
        if p == len(pattern):
            if t == len(target):
                yield subst
            continue
        token = pattern[p]
        if token not in variables or token in subst:
            value = subst.get(token, (token,)) if token in variables else (token,)
            if target[t:t + len(value)] == value:
                pending.append((p + 1, t + len(value), subst))
            continue
        minimum = sum(len(subst[x]) if x in subst else int(x not in variables)
                      for x in pattern[p + 1:])
        lengths = list(range(max(-1, len(target) - t - minimum) + 1))
        rng.shuffle(lengths)
        for length in lengths:
            pending.append((p + 1, t + length, {**subst, token: target[t:t + length]}))


@dataclass(frozen=True)
class GenericConfig:
    seed: int = 7
    max_tokens: int = 192
    max_hypotheses: int = 8
    max_variables: int = 16
    max_proof_depth: int = 8
    max_proof_labels: int = 4096
    match_candidates: int = 24
    match_budget: int = 4000
    type_search_depth: int = 6
    seed_assertions: int = 256
    variables_per_type: int = 4
    max_pool_nodes: int = 10000
    instance_probability: float = 0.10
    partial_premise_probability: float = 0.30
    derived_rule_probability: float = 0.25

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_proof_labels", "match_candidates", "match_budget",
                     "variables_per_type", "max_pool_nodes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_hypotheses", "max_variables", "max_proof_depth",
                     "type_search_depth", "seed_assertions"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("instance_probability", "partial_premise_probability",
                     "derived_rule_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class TokenFact:
    expression: Expression
    hypotheses: tuple[Expression, ...]
    disjoints: Disjoints
    # A string is a proof label; a tuple is an essential-hypothesis reference.
    proof: tuple[str | Expression, ...]
    depth: int


@dataclass(frozen=True)
class GeneratedStatement:
    assertion: TokenAssertion
    proof: Expression
    proof_disjoints: Disjoints
    depth: int
    rule: str
    matched_premises: int
    reserved_premises: int
    unmatched_premises: int


def statement_key(expr: Expression, hypotheses: tuple[Expression, ...],
                  disjoints: Disjoints, variable_types: Mapping[str, str]) -> tuple:
    names: dict[str, int] = {}

    def normalize(expression: Expression) -> tuple:
        return tuple(("v", names.setdefault(t, len(names)), variable_types[t])
                     if t in variable_types else ("c", t) for t in expression)

    hyps = tuple(normalize(h) for h in hypotheses)
    conclusion = normalize(expr)
    dv = tuple(sorted(tuple(sorted((names[x], names[y]))) for x, y in disjoints
                      if x in names and y in names))
    return hyps, conclusion, dv


class GenericGenerator:
    def __init__(self, database: TokenDatabase, config: GenericConfig | None = None):
        self.database = database
        self.config = config or GenericConfig()
        self.rng = random.Random(self.config.seed)
        self.assertions = dict(database.assertions)
        self.source_rules = list(database.assertions.values())
        self.type_rules: dict[str, list[TokenAssertion]] = defaultdict(list)
        for rule in self.source_rules:
            if not rule.essential:
                self.type_rules[rule.expression[0]].append(rule)
        self.generated: list[GeneratedStatement] = []
        self.pool: list[TokenFact] = []
        self.by_type: dict[str, list[TokenFact]] = defaultdict(list)
        self.by_expression: dict[Expression, list[TokenFact]] = defaultdict(list)
        self.pool_keys: set[tuple] = set()
        self.stats: Counter = Counter()
        self.rule_uses: Counter = Counter()
        self.parent_uses: Counter = Counter()
        self.rule_depths: dict[str, int] = {}
        prefix = "ug-"
        occupied = database.constants | database.variables | database.labels
        while any(token.startswith(prefix) for token in occupied):
            prefix += "g-"
        self.prefix = prefix
        self.floating: dict[str, TokenHypothesis] = {}
        self.leaves: dict[str, list[TokenFact]] = defaultdict(list)
        self.variable_types: dict[str, str] = {}
        type_counts: Counter = Counter()
        for rule in self.source_rules:
            for typ, count in Counter(rule.variable_types.values()).items():
                type_counts[typ] = max(type_counts[typ], count)
        for index, typ in enumerate(sorted(type_counts)):
            for n in range(max(type_counts[typ], self.config.variables_per_type)):
                var = f"{prefix}v{index}-{n}"
                label = f"{prefix}f{index}-{n}"
                h = TokenHypothesis(label, "$f", (typ, var))
                self.floating[label] = h
                self.variable_types[var] = typ
                fact = TokenFact(h.expression, (), frozenset(), (label,), 0)
                self.leaves[typ].append(fact)
                self._add_pool(fact)
        self.variables = set(self.variable_types)
        if len(self.floating) > self.config.max_pool_nodes:
            raise ValueError("max_pool_nodes is smaller than the required floating-variable pool")
        self.seen = {statement_key(r.expression, tuple(h.expression for h in r.essential),
                                   r.disjoints, r.variable_types) for r in self.source_rules}
        self._seed()

    def _add_pool(self, fact: TokenFact) -> None:
        key = (fact.expression, fact.hypotheses, fact.disjoints)
        if key in self.pool_keys or len(self.pool) >= self.config.max_pool_nodes:
            return
        self.pool_keys.add(key)
        self.pool.append(fact)
        self.by_type[fact.expression[0]].append(fact)
        self.by_expression[fact.expression].append(fact)

    def _check_limits(self, fact: TokenFact) -> None:
        cfg = self.config
        expressions = (fact.expression, *fact.hypotheses)
        used = set().union(*(set(e) for e in expressions)) & self.variables
        if (any(len(e) > cfg.max_tokens for e in expressions)
                or len(fact.hypotheses) > cfg.max_hypotheses
                or len(used) > cfg.max_variables or fact.depth > cfg.max_proof_depth
                or len(fact.proof) > cfg.max_proof_labels):
            raise SamplingFailure("structural_limit")

    def _construct(self, rule: TokenAssertion, witnesses: Mapping[str, TokenFact],
                   parents: Mapping[int, TokenFact]) -> TokenFact:
        subst = {v: f.expression[1:] for v, f in witnesses.items()}
        if set(subst) != set(rule.variable_types):
            raise SamplingFailure("missing_type_witness")
        try:
            disjoints = set(required_disjoints(rule, subst, self.variables))
        except TokenMMError as exc:
            raise SamplingFailure("disjoint_collapse") from exc
        arguments = []
        essential_index = 0
        for h in rule.hypotheses:
            if h.kind == "$f":
                fact = witnesses[h.expression[1]]
                if fact.expression[0] != h.expression[0]:
                    raise SamplingFailure("type_mismatch")
            else:
                expr = substitute(h.expression, subst)
                fact = parents.get(essential_index, TokenFact(expr, (expr,), frozenset(), (expr,), 0))
                if fact.expression != expr:
                    raise SamplingFailure("premise_mismatch")
                essential_index += 1
            arguments.append(fact)
            disjoints.update(fact.disjoints)
        hypotheses = tuple(dict.fromkeys(h for f in arguments for h in f.hypotheses))
        proof = tuple(label for f in arguments for label in f.proof) + (rule.label,)
        # Source assertions are unit steps. Generated rules keep their previous
        # depth; substituting deeper proofs adds to it, never resets it to one.
        depth = self.rule_depths.get(rule.label, 1) + max((f.depth for f in arguments), default=0)
        fact = TokenFact(substitute(rule.expression, subst), hypotheses,
                         frozenset(disjoints), proof, depth)
        self._check_limits(fact)
        return fact

    def _seed(self) -> None:
        closed = [r for r in self.source_rules if not r.essential]
        conditional = [r for r in self.source_rules if r.essential]
        self.rng.shuffle(closed)
        self.rng.shuffle(conditional)
        # Interleave both kinds so large databases do not exhaust the entire
        # startup budget on one family. All source rules remain selectable.
        ordered = []
        while closed or conditional:
            if closed:
                ordered.append(closed.pop())
            if conditional:
                ordered.append(conditional.pop())
        for rule in ordered[:self.config.seed_assertions]:
            counters: Counter = Counter()
            witnesses = {}
            for v, typ in rule.variable_types.items():
                witnesses[v] = self.leaves[typ][counters[typ]]
                counters[typ] += 1
            try:
                fact = self._construct(rule, witnesses, {})
                self._add_pool(fact)
                self.stats["seed_instances"] += 1
            except SamplingFailure:
                self.stats["seed_over_limit"] += 1

    def _candidates(self, typ: str) -> list[TokenFact]:
        nodes = list(self.by_type.get(typ, ()))
        result = []
        for _ in range(min(len(nodes), self.config.match_candidates)):
            weights = [(n.depth + 1) ** .5 / (1 + self.parent_uses[n.expression]) ** .5 for n in nodes]
            n = self.rng.choices(nodes, weights=weights, k=1)[0]
            result.append(n)
            nodes.remove(n)
        return result

    def _resolve_type(self, goal: Expression, budget: MatchBudget, depth: int,
                      trail: frozenset[Expression] = frozenset()) -> TokenFact | None:
        budget.spend()
        existing = self.by_expression.get(goal)
        if existing:
            return min(existing, key=lambda f: (len(f.hypotheses), f.depth, len(f.proof)))
        if depth <= 0 or goal in trail:
            return None
        # Any no-$e assertion with this typecode may construct a witness. This
        # also handles non-prefix/ambiguous grammars and empty replacements.
        rules = list(self.type_rules.get(goal[0], ()))
        self.rng.shuffle(rules)
        for rule in rules:
            for subst in match_tokens(rule.expression, goal, set(rule.variable_types), {}, budget, self.rng):
                witnesses = {}
                for v, typ in rule.variable_types.items():
                    witness = self._resolve_type((typ, *subst[v]), budget, depth - 1, trail | {goal})
                    if witness is None:
                        break
                    witnesses[v] = witness
                else:
                    try:
                        return self._construct(rule, witnesses, {})
                    except SamplingFailure:
                        continue
        return None

    def _witnesses(self, rule: TokenAssertion, subst: Mapping[str, Expression],
                   budget: MatchBudget) -> dict[str, TokenFact] | None:
        result = {}
        for v, value in subst.items():
            witness = self._resolve_type((rule.variable_types[v], *value), budget,
                                         self.config.type_search_depth)
            if witness is None:
                return None
            result[v] = witness
        try:
            # Only test pairs whose bindings are already known.
            partial = TokenAssertion(rule.label, rule.kind, rule.expression, rule.hypotheses,
                                     frozenset((x, y) for x, y in rule.disjoints if x in subst and y in subst))
            required_disjoints(partial, subst, self.variables)
        except TokenMMError:
            return None
        return result

    def sample(self, rule: TokenAssertion) -> tuple[TokenFact, dict[int, TokenFact], int]:
        budget = MatchBudget(self.config.match_budget)
        try:
            return self._sample(rule, budget)
        finally:
            self.stats["match_states"] += self.config.match_budget - budget.remaining

    def _sample(self, rule: TokenAssertion, budget: MatchBudget) -> tuple[TokenFact, dict[int, TokenFact], int]:
        subst: dict[str, Expression] = {}
        witnesses: dict[str, TokenFact] = {}
        parents: dict[int, TokenFact] = {}
        essential = rule.essential
        indices = sorted(range(len(essential)), key=lambda i: -sum(
            t not in rule.variable_types for t in essential[i].expression))
        reserved = 0
        if len(indices) >= 2 and self.rng.random() < self.config.partial_premise_probability:
            indices.remove(self.rng.choice(indices))
            reserved = 1
        for index in indices:
            pattern = essential[index].expression
            found = False
            for parent in self._candidates(pattern[0]):
                self.stats["parent_attempts"] += 1
                for candidate in match_tokens(pattern, parent.expression, set(rule.variable_types),
                                               subst, budget, self.rng):
                    typed = self._witnesses(rule, candidate, budget)
                    if typed is not None:
                        subst, witnesses = candidate, typed
                        parents[index] = parent
                        found = True
                        break
                if found:
                    break
        if essential and not parents:
            raise SamplingFailure("no_matched_premise")
        instantiate = self.rng.random() < self.config.instance_probability
        # Complete unbound variables with proof-carrying typed objects. Most
        # steps remain schematic; arbitrary strings are never trusted as $f.
        for v, typ in rule.variable_types.items():
            if v in witnesses:
                continue
            candidates = self._candidates(typ) if instantiate else list(self.leaves[typ])
            self.rng.shuffle(candidates)
            for witness in candidates:
                budget.spend()
                candidate = {**subst, v: witness.expression[1:]}
                partial = TokenAssertion(rule.label, rule.kind, rule.expression, rule.hypotheses,
                                         frozenset((x, y) for x, y in rule.disjoints if x in candidate and y in candidate))
                try:
                    required_disjoints(partial, candidate, self.variables)
                except TokenMMError:
                    continue
                subst = candidate
                witnesses[v] = witness
                break
            else:
                raise SamplingFailure("no_typed_substitution")
        return self._construct(rule, witnesses, parents), parents, reserved

    def _admit(self, fact: TokenFact, rule: TokenAssertion,
               parents: Mapping[int, TokenFact], reserved: int) -> bool:
        self._check_limits(fact)
        key = statement_key(fact.expression, fact.hypotheses, fact.disjoints, self.variable_types)
        if key in self.seen:
            self.stats["duplicate"] += 1
            return False
        if len(self.pool) >= self.config.max_pool_nodes:
            raise SamplingFailure("pool_limit")
        label = f"{self.prefix}g{len(self.generated)}"
        essential = [TokenHypothesis(f"{label}-h{i}", "$e", expr) for i, expr in enumerate(fact.hypotheses)]
        labels = {h.expression: h.label for h in essential}
        active = {**self.floating, **{h.label: h for h in essential}}
        assertion = make_assertion(label, "$p", fact.expression, active.values(), self.variables, fact.disjoints)
        proof = tuple(labels[p] if isinstance(p, tuple) else p for p in fact.proof)
        # Unexpected verification errors are fatal implementation bugs, not
        # silently counted as ordinary unsuccessful sampling attempts.
        verify_tokens(assertion, proof, self.assertions, active, self.variables, fact.disjoints)
        self.generated.append(GeneratedStatement(assertion, proof, fact.disjoints, fact.depth,
                                                rule.label, len(parents), reserved,
                                                len(rule.essential) - len(parents) - reserved))
        self.assertions[label] = assertion
        self.rule_depths[label] = fact.depth
        if not assertion.essential:
            self.type_rules[assertion.expression[0]].append(assertion)
        self.seen.add(key)
        # Reuse the theorem by a proper assertion application, retaining its
        # mandatory hypotheses and $d. Its proof can contain extra dummy vars.
        reuse_proof = tuple(h.label if h.kind == "$f" else h.expression
                            for h in assertion.hypotheses) + (label,)
        self._add_pool(TokenFact(fact.expression, fact.hypotheses, assertion.disjoints,
                                 reuse_proof, fact.depth))
        self.parent_uses.update(p.expression for p in parents.values())
        self.stats["admitted"] += 1
        self.stats["matched_premises"] += len(parents)
        self.stats["reserved_premises"] += reserved
        self.stats["unmatched_premises"] += len(rule.essential) - len(parents) - reserved
        return True

    def generate(self, steps: int) -> list[GeneratedStatement]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        start = len(self.generated)
        for _ in range(steps):
            self.stats["attempts"] += 1
            rules = ([g.assertion for g in self.generated]
                     if self.generated and self.rng.random() < self.config.derived_rule_probability
                     else self.source_rules)
            if not rules:
                self.stats["no_rules"] += 1
                continue
            conditional = [r for r in rules if r.essential]
            closed = [r for r in rules if not r.essential]
            rules = (conditional if conditional and (not closed or self.rng.random() < .75) else closed)
            rule = self.rng.choices(rules, weights=[1 / (1 + self.rule_uses[r.label]) ** .5 for r in rules], k=1)[0]
            self.rule_uses[rule.label] += 1
            try:
                fact, parents, reserved = self.sample(rule)
                self._admit(fact, rule, parents, reserved)
            except SamplingFailure as exc:
                self.stats[str(exc)] += 1
        return self.generated[start:]

    def export(self, directory: str | Path, typecode: str | None = None) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"metamath": directory / "generated.mm", "theorems": directory / "theorems.jsonl",
                 "summary": directory / "summary.json"}
        lines = [self.database.isolated_source(), "$( Generic token sampler: generated proofs follow. $)", "${"]
        if self.variables:
            lines.append("$v " + " ".join(self.variable_types) + " $.")
        for h in self.floating.values():
            lines.append(f"{h.label} $f {' '.join(h.expression)} $.")
        records = []
        for generated in self.generated:
            a = generated.assertion
            lines.append("${")
            for x, y in sorted(generated.proof_disjoints):
                lines.append(f"$d {x} {y} $.")
            for h in a.essential:
                lines.append(f"{h.label} $e {' '.join(h.expression)} $.")
            lines.append(f"{a.label} $p {' '.join(a.expression)} $= {' '.join(generated.proof)} $.")
            lines.append("$}")
            if typecode is None or a.expression[0] == typecode:
                records.append({"schema": "metamath-token-theorem-v1", "label": a.label,
                                "expression": a.expression,
                                "hypotheses": [h.expression for h in a.essential],
                                "variable_types": a.variable_types,
                                "disjoints": sorted(a.disjoints),
                                "proof_disjoints": sorted(generated.proof_disjoints),
                                "proof": generated.proof, "proof_depth_bound": generated.depth,
                                "rule": generated.rule, "matched_premises": generated.matched_premises,
                                "reserved_premises": generated.reserved_premises,
                                "unmatched_premises": generated.unmatched_premises})
        lines.append("$}")
        paths["metamath"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        with paths["theorems"].open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = {"algorithm": "generic-token-forward-v1", "config": asdict(self.config),
                   "source": self.database.source_path,
                   "source_sha256": hashlib.sha256(" ".join(self.database.source_tokens).encode()).hexdigest(),
                   "source_assertions": len(self.source_rules),
                   "source_proofs_verified": self.database.verified_proofs,
                   "generated": len(self.generated), "exported_records": len(records),
                   "typecode_filter": typecode, "pool_nodes": len(self.pool),
                   "statistics": dict(self.stats), "rule_attempts": dict(self.rule_uses)}
        paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return paths

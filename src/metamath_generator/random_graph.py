"""HTPS-inspired forward generation, using only the loaded Metamath theory.

The theory is fixed; proof graphs are not. Each run samples new syntax instances
and combines earlier certified nodes.  Quality scores never prune the working
graph: a simple intermediate statement can be essential to a later proof.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .compose import CompositionError, compose, instantiate_assertion
from .model import Node
from .unification import substitute_simultaneous


def premise_statistics(theorem, rule, store, reserved_premise=None):
    """Count assumptions after unification, before/after alpha renaming alike.

    An unmatched premise may coincide with a parent's assumption. Count it as
    new only when its instantiated formula is absent from inherited context.
    """
    proof = theorem.proof
    inherited = set()
    for index, parent_id in enumerate(proof.premise_map):
        if parent_id is None:
            continue
        parent = store[parent_id]
        mapping = {v: proof.substitution[f"__p{index}_{v}"] for v in parent.variable_types}
        inherited.update(substitute_simultaneous(h.expr, mapping) for h in parent.hypotheses)
    new = {h.expr for h in theorem.hypotheses} - inherited
    unmatched = proof.premise_map.count(None)
    reserved_new = False
    if reserved_premise is not None:
        mapping = {v: proof.substitution[f"__rule_{v}"] for v in rule.variable_types}
        reserved_new = substitute_simultaneous(
            rule.hypotheses[reserved_premise].expr, mapping) in new
    return {
        "graph_partial_plan_admitted": int(reserved_premise is not None),
        "graph_nodes_with_open_premises": int(unmatched > 0),
        "graph_nodes_with_new_hypotheses": int(bool(new)),
        "graph_nodes_with_inherited_hypotheses": int(bool(inherited)),
        "graph_nodes_with_reserved_new_hypothesis": int(reserved_new),
        "graph_new_hypotheses": len(new),
        "graph_inherited_hypotheses": len(inherited),
        "graph_unmatched_premises": unmatched,
        "graph_unmatched_after_search": unmatched - int(reserved_premise is not None),
    }


class RandomProofGraph:
    def __init__(self, generator):
        self.g = generator
        self.rng = generator.random
        self.cfg = generator.config
        self.rule_attempts = Counter()
        self.parent_uses = Counter()
        self.composition_statistics = {}
        self.closed = [r for r in generator.parsed.logical_assertions.values()
                       if not r.hypotheses]
        self.rules = list(generator.rules)
        self.leaves = defaultdict(list)
        for name, typ in sorted(generator.parsed.variable_types.items()):
            self.leaves[typ].append(Node(name))
        self.syntax = defaultdict(list)
        for typ, rules in generator.parsed.syntax_rules.items():
            for rule in rules:
                # This project's parser represents prefix syntax as ASTs.
                if len(rule.pattern) == 1 and not rule.variable_types:
                    self.leaves[typ].append(Node(rule.pattern[0]))
                elif rule.pattern and rule.pattern[0] not in rule.variable_types:
                    self.syntax[typ].append(rule)

    def expression(self, typ, depth):
        leaves = self.leaves[typ]
        if leaves and (depth <= 0 or not self.syntax[typ] or self.rng.random() < .35):
            return self.rng.choice(leaves)
        if depth < 0 or not self.syntax[typ]:
            raise CompositionError(f"no bounded syntax for {typ}")
        rule = self.rng.choice(self.syntax[typ])
        subst = {name: self.expression(child_type, depth - 1)
                 for name, child_type in rule.variable_types.items()}
        # Parse the instantiated syntax rule, rather than assuming an arity.
        tokens = [typ]
        for token in rule.pattern:
            tokens.extend(subst[token].to_prefix().split() if token in subst else [token])
        return self.g._expression_parser.parse_expression(tokens).args[0]

    def choose_rule(self, rules):
        weights = [(1 + self.rule_attempts[r.name]) ** -self.cfg.graph_rule_bias
                   for r in rules]
        rule = self.rng.choices(rules, weights=weights, k=1)[0]
        self.rule_attempts[rule.name] += 1
        return rule

    def instantiate(self):
        # Optional final augmentation of an already constructed statement;
        # also seed fresh source instances so composition cannot stagnate.
        derived = [n for n in self.g.store.generated()
                   if n.proof_depth < self.cfg.max_proof_depth]
        choices = derived if derived and self.rng.random() < .5 else self.closed
        if not choices:
            return None
        rule = self.choose_rule(choices)
        mapping = {name: self.expression(typ, self.cfg.graph_expression_depth)
                   for name, typ in sorted(rule.variable_types.items())}
        return instantiate_assertion(rule, mapping, database=self.g.parsed,
                                     name=f"gen{len(self.g.store)}_{rule.name}")

    def candidates(self, premise, rule):
        buckets = defaultdict(list)
        for node in self.g.store:
            if node.proof_depth >= self.cfg.max_proof_depth:
                continue
            if self.g._nodes_may_unify(premise, node.conclusion,
                                      rule.variable_types, node.variable_types):
                buckets[node.proof_depth].append(node)
        result = []
        # Balance depth buckets, then downweight repeatedly reused parents.
        # There is no special preference for closed or low-depth parents.
        while buckets and len(result) < self.cfg.graph_match_candidates:
            depths = sorted(buckets)
            depth = self.rng.choices(depths, weights=[(d + 1) ** self.cfg.graph_depth_bias
                                                    for d in depths], k=1)[0]
            nodes = buckets[depth]
            node = self.rng.choices(nodes, weights=[1 / (1 + self.parent_uses[n.id]) ** .5
                                                   for n in nodes], k=1)[0]
            result.append(node)
            nodes.remove(node)
            if not nodes:
                del buckets[depth]
        return result

    def combine(self):
        self.composition_statistics = {}
        derived = [n for n in self.g.store.generated() if n.hypotheses
                   and n.proof_depth < self.cfg.max_proof_depth]
        rules = (derived if derived and self.rng.random() < self.cfg.graph_derived_rule_probability
                 else self.rules)
        if not rules:
            return None
        rule = self.choose_rule(rules)
        matches = [None] * len(rule.hypotheses)
        # Constrain structured premises before schematic ones (e.g. major
        # premise before minor premise for modus ponens), for any source rule.
        indices = sorted(range(len(matches)), key=lambda i: -sum(
            n.op not in rule.variable_types for n in rule.hypotheses[i].expr.walk()))
        reserved = None
        if len(indices) >= 2:
            self.g.stats["graph_multi_premise_plans"] += 1
            if self.rng.random() < self.cfg.graph_partial_premise_probability:
                # Reserve exactly one uniformly chosen premise even if an old
                # node could prove it. Keep at least one slot for composition.
                reserved = self.rng.choice(indices)
                indices.remove(reserved)
                self.g.stats["graph_partial_plans"] += 1
        else:
            self.g.stats["graph_single_premise_plans"] += 1
        best = None
        for index in indices:
            for parent in self.candidates(rule.hypotheses[index].expr, rule):
                proposed = list(matches)
                proposed[index] = parent
                try:
                    theorem = compose(rule, proposed, database=self.g.parsed,
                                      name=f"gen{len(self.g.store)}_{rule.name}")
                except CompositionError:
                    continue
                if not self.g._valid(theorem):
                    continue
                matches, best = proposed, theorem
                break
        # Unmatched premises remain explicit hypotheses, never new axioms.
        if best is not None:
            self.composition_statistics = premise_statistics(best, rule, self.g.store, reserved)
        return best

    def admit(self, theorem, kind):
        if not self.g._valid(theorem):
            self.g.rejected["graph_structural_limit_or_type"] += 1
            return False
        node_id, added = self.g.store.add(theorem)
        if not added:
            self.g.rejected["graph_duplicate"] += 1
            return False
        node = self.g.store[node_id]
        assessment, profile = self.g.analyzer.assess(node)
        # Scores describe the output but cannot remove intermediate lemmas.
        assessment.active = True
        self.g.assessments[node_id] = assessment
        self.g.profiles[node_id] = profile
        self.g.active_ids.add(node_id)
        self.g._new_ids.append(node_id)
        self.g._definition_support[node_id] = self.g._proof_definition_support(node)
        self.g.generation_context[node_id] = kind
        self.g.rule_usage[node.proof.rule] += 1
        self.g.generated_count += 1
        self.g.stats[kind] += 1
        self.parent_uses.update(node.proof.parents)
        self.g._target_report_cache = None
        return True

    def run(self, attempts):
        from .parser import ParseError
        for _ in range(attempts):
            self.g.stats["attempts"] += 1
            instance = not self.rules or self.rng.random() < self.cfg.graph_instance_probability
            try:
                theorem = self.instantiate() if instance else self.combine()
                if theorem is not None:
                    added = self.admit(theorem, "random_instance" if instance else "random_graph")
                    if added and not instance:
                        self.g.stats.update(self.composition_statistics)
                else:
                    self.g.rejected["graph_no_match"] += 1
            except (CompositionError, ParseError):
                self.g.rejected["graph_invalid_instance"] += 1
        self.g.stats["graph_reused_parents"] = sum(n > 1 for n in self.parent_uses.values())
        self.g.stats["graph_parent_edges"] = sum(self.parent_uses.values())


def generate_random_graph(generator, steps):
    if steps < 0:
        raise ValueError("graph steps must be non-negative")
    cfg = generator.config
    if not 0 <= cfg.graph_instance_probability <= 1:
        raise ValueError("graph_instance_probability must be in [0, 1]")
    if not 0 <= cfg.graph_partial_premise_probability <= 1:
        raise ValueError("graph_partial_premise_probability must be in [0, 1]")
    if not 0 <= cfg.graph_derived_rule_probability <= 1:
        raise ValueError("graph_derived_rule_probability must be in [0, 1]")
    if cfg.graph_expression_depth < 0 or cfg.graph_match_candidates <= 0:
        raise ValueError("invalid graph expression/matching budget")
    if cfg.graph_rule_bias < 0 or cfg.graph_depth_bias < 0:
        raise ValueError("graph sampling biases must be non-negative")
    RandomProofGraph(generator).run(steps)

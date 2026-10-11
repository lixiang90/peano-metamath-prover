"""Replay source certificates as complete backward proof obligations."""
from __future__ import annotations

from dataclasses import dataclass, replace

from metamath_generator.model import Node
from metamath_generator.unification import substitute_simultaneous
from .environment import ProofState, Tactic


@dataclass(frozen=True)
class ProofPlan:
    conclusion: Node
    tactic: Tactic
    children: tuple["ProofPlan", ...] = ()

    @property
    def depth(self):
        return 1 + max((child.depth for child in self.children), default=0)


def certificate_plan(certificate, database):
    """Recover essential-hypothesis edges; syntax applications add no goals.

    The caller supplies a verified, flattened certificate. Do not reverse its
    stack labels: mandatory floating and essential hypotheses can interleave.
    """
    locals_ = {h.label: h.expr for h in (*certificate.floating, *certificate.hypotheses)}
    stack = []
    for label in certificate.proof.source_labels:
        if label in locals_:
            expr = locals_[label]
            plan = ProofPlan(expr, Tactic.create("<ASSUMPTION>")) if expr.op == "|-" else None
            stack.append((expr, plan))
            continue
        rule = database.statements[label]
        count = len(rule.mandatory_hypotheses)
        if len(stack) < count:
            raise ValueError("certificate stack underflow")
        actuals = stack[-count:] if count else []
        if count:
            del stack[-count:]
        bindings = dict(zip((h.label for h in rule.mandatory_hypotheses), actuals))
        subst = {h.expr.args[0].op: bindings[h.label][0].args[0] for h in rule.floating}
        expr = substitute_simultaneous(rule.conclusion, subst)
        plan = None
        if expr.op == "|-":
            children = tuple(bindings[h.label][1] for h in rule.hypotheses)
            if any(child is None for child in children):
                raise ValueError("non-logical essential premise in backward certificate")
            plan = ProofPlan(expr, Tactic.create(rule.name, subst), children)
        stack.append((expr, plan))
    if len(stack) != 1 or stack[0][0] != certificate.conclusion or stack[0][1] is None:
        raise ValueError("certificate has no unique logical root")
    return stack[0][1]


def replay_certificate(certificate, environment):
    root = certificate_plan(certificate, environment.database)
    state = ProofState.from_theorem(certificate)
    pending = [root]
    transitions = []
    while pending:
        plan = pending.pop(0)
        if state.current_goal != plan.conclusion:
            raise ValueError("certificate and pending goal order disagree")
        transition = environment.apply(state, plan.tactic)
        pending[:0] = [c for c in plan.children if c.conclusion not in state.hypotheses]
        transitions.append(transition)
        state = transition.after
        if state.goals != tuple(p.conclusion for p in pending):
            raise ValueError("certificate replay lost an obligation")
    if not state.solved:
        raise ValueError("certificate replay did not close every goal")
    return transitions


def closing_tactics(state, environment):
    """Cheap source-rule shortcuts, fully checked including types and DV."""
    if state.current_goal in state.hypotheses:
        yield Tactic.create("<ASSUMPTION>")
    for rule in environment.assertions.values():
        if rule.hypotheses:
            continue
        try:
            subst = environment._unify_schema(rule.conclusion, rule.variable_types,
                                             state.current_goal, namespace=f"shortcut_{rule.name}")
            tactic = Tactic.create(rule.name, subst)
            if environment.apply(state, tactic).after.goals == state.goals[1:]:
                yield tactic
        except ValueError:
            continue


def shorten_trace(transitions, plans, environment):
    state = transitions[0].before
    result = []
    seen = set()
    while not state.solved:
        if state in seen or len(result) >= len(transitions):
            return transitions
        seen.add(state)
        tactic = next(closing_tactics(state, environment), None)
        if tactic is None:
            plan = plans.get(state.current_goal)
            if plan is None:
                return transitions
            tactic = plan.tactic
        transition = environment.apply(state, tactic)
        result.append(transition)
        state = transition.after
    return result


def tactic_tokens(tactic, state, tokenizer, environment):
    if tactic.rule == "<ASSUMPTION>":
        return ["<BOS>", "<ACTION>", "<ASSUMPTION>", "<END_ACTION>", "<EOS>"]
    return tokenizer.assertion_tactic_tokens(
        environment.assertions[tactic.rule], tactic.substitution_dict(),
        tokenizer.proof_state_variables(state),
    )


def structure_family(state):
    body = state.current_goal.args[0] if state.current_goal.args else state.current_goal
    types = dict(state.variable_types)
    return f"variable:{types[body.op]}" if body.op in types else body.op


def plan_library(root):
    """Keep the shortest certificate witness for each expression in this scope."""
    plans = {}
    stack = [root]
    while stack:
        plan = stack.pop()
        if plan.conclusion not in plans or plan.depth < plans[plan.conclusion].depth:
            plans[plan.conclusion] = plan
        stack.extend(plan.children)
    return plans


def complete_from_library(state, plans, environment, limit=256):
    """A constructive positive witness, never an unprovability test."""
    trace = []
    seen = set()
    while not state.solved:
        if state in seen or len(trace) >= limit:
            return None
        seen.add(state)
        if state.current_goal in state.hypotheses:
            tactic = Tactic.create("<ASSUMPTION>")
        else:
            plan = plans.get(state.current_goal)
            if plan is None:
                return None
            tactic = plan.tactic
        transition = environment.apply(state, tactic)
        trace.append(transition)
        state = transition.after
    return trace


def candidate_supervision(state, gold, plans, environment, tokenizer, config):
    from .hybrid import HybridActionGenerator, HybridGeneratorConfig
    from .environment import parse_tactic_tokens
    limit = config.candidate_limit
    if limit <= 0:
        return (), ()
    generator = HybridActionGenerator(environment, HybridGeneratorConfig(
        max_deterministic_tactics=limit, max_construction_tactics=limit,
        deterministic_candidates_per_variable=4, construction_candidates_per_variable=4,
    ))
    # Gold survives bounded enumeration; other actions come from the same
    # symbolic generator as inference, including alternate closing rules.
    tactics = list(dict.fromkeys([gold, *closing_tactics(state, environment), *generator.actions(state).all]))[:limit]
    decoded = tokenizer.proof_state_from_tokens(tokenizer.proof_state_tokens(state), environment.database)
    actions, positives = [], []
    for tactic in tactics:
        try:
            tokens = tactic_tokens(tactic, state, tokenizer, environment)
            if len(tokens) > config.max_action_tokens:
                continue
            parsed = parse_tactic_tokens(tokens, decoded.as_theorem(), tokenizer,
                                         environment.database, environment=environment)
            environment.apply(decoded, parsed)
            after = environment.apply(state, tactic).after
            ids = tuple(tokenizer.encode(tokens))
            if ids in actions:
                continue
            try:
                positive = tactic == gold or complete_from_library(after, plans, environment) is not None
            except ValueError:
                # A legal alternative can invalidate a witness from the old
                # context. Keep the candidate, but do not claim it is solved.
                positive = False
            if positive:
                positives.append(len(actions))
            actions.append(ids)
        except ValueError:
            if tactic == gold:
                raise
    if not positives:
        raise ValueError("candidate supervision lost its verified action")
    return tuple(actions), tuple(positives)


def merge_positive_candidates(examples):
    """Share verified alternatives across certificates of the identical state.

    State IDs include the complete goal queue, context and distinct-variable
    constraints. Never transfer a positive based on the current goal alone.
    Candidate limits stay unchanged; only labels inside existing pools merge.
    """
    from collections import defaultdict
    witnessed = defaultdict(set)
    for e in examples:
        witnessed[e.state_ids].update(e.candidate_action_ids[i] for i in e.positive_candidate_indices)
    return [replace(e, positive_candidate_indices=tuple(
                i for i, action in enumerate(e.candidate_action_ids) if action in witnessed[e.state_ids]
            )) if e.candidate_action_ids else e for e in examples]


def trajectory_examples(certificate, environment, tokenizer, config):
    import hashlib
    from .data import ProverExample, _split_for
    from .environment import parse_tactic_tokens
    transitions = replay_certificate(certificate, environment)
    plans = plan_library(certificate_plan(certificate, environment.database))
    reference_steps = len(transitions)
    if config.shorten_trajectories:
        transitions = shorten_trace(transitions, plans, environment)
    trajectory_id = hashlib.sha256(repr((certificate.conclusion, certificate.hypotheses,
                                        certificate.proof.source_labels)).encode()).hexdigest()[:20]
    examples = []
    for index, transition in enumerate(transitions):
        state = transition.before
        tokens = tuple(tokenizer.proof_state_tokens(state))
        action = tuple(tactic_tokens(transition.tactic, state, tokenizer, environment))
        if len(tokens) > config.max_state_tokens or len(action) > config.max_action_tokens:
            raise ValueError("trajectory exceeds token budget")
        # Verify the serialized state too, including all obligations and DV.
        decoded = tokenizer.proof_state_from_tokens(tokens, environment.database)
        decoded_action = parse_tactic_tokens(action, decoded.as_theorem(), tokenizer,
                                             environment.database, environment=environment)
        environment.apply(decoded, decoded_action)
        remaining = len(transitions) - index
        theorem = state.as_theorem()
        candidates, positives = candidate_supervision(
            state, transition.tactic, plans, environment, tokenizer, config,
        )
        examples.append(ProverExample(
            example_id=hashlib.sha256(repr((tokens, action)).encode()).hexdigest()[:20],
            split=_split_for(theorem, config.validation_fraction, config.test_fraction),
            origin="synthetic", difficulty="easy" if remaining <= 2 else "medium" if remaining <= 5 else "hard",
            theorem_name=certificate.name, conclusion=state.current_goal.to_prefix(),
            hypotheses=tuple(h.to_prefix() for h in state.hypotheses), rule=transition.tactic.rule,
            proof_depth=remaining, quality_score=0.0,
            value_target=max(0.0, 1.0 - config.trajectory_step_penalty * remaining),
            state_tokens=tokens, action_tokens=action, state_ids=tuple(tokenizer.encode(tokens)),
            action_ids=tuple(tokenizer.encode(action)), generation_kind="random_graph",
            trajectory_id=trajectory_id, trajectory_step=index, remaining_steps=remaining,
            goal_count=len(state.goals),
            candidate_action_ids=candidates, positive_candidate_indices=positives,
            reference_steps=reference_steps,
            rule_premise_count=len(environment.assertions[transition.tactic.rule].hypotheses) if transition.tactic.rule != "<ASSUMPTION>" else 0,
            structure_family=structure_family(state),
        ))
    return examples

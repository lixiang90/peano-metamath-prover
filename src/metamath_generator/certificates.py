"""Source-only certificates shared by graph data generation and v2."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from .model import Hypothesis, Node, Proof, normalized_pair
from .unification import substitute_simultaneous
from .verifier import verify


def flatten_certificate(theorem, expanded_database, source_database, max_labels=4096):
    """Expand generated $p macros into original assertions, never into new axioms."""
    if max_labels <= 0:
        raise ValueError("proof label budget must be positive")
    verify(theorem, expanded_database)
    fresh_floating = []
    occupied = set(expanded_database.symbols) | set(expanded_database.variables) | set(expanded_database.label_order)

    def fresh_dummy(hypothesis):
        index = len(fresh_floating)
        while True:
            variable = f"__flat_{index}"
            label = f"{theorem.name}_flat_{index}"
            if variable not in occupied and label not in occupied:
                break
            index += 1
        occupied.update((variable, label))
        hyp = Hypothesis(label, Node(hypothesis.expr.op, (Node(variable),)))
        fresh_floating.append(hyp)
        return [label]

    def expand(labels, replacements, active):
        stack = []
        stack_labels = 0

        def push(fragment):
            nonlocal stack_labels
            if stack_labels + len(fragment) > max_labels:
                raise ValueError("proof label budget exceeded")
            stack.append(fragment)
            stack_labels += len(fragment)

        for label in labels:
            if label in replacements:
                push(list(replacements[label]))
                continue
            if label in expanded_database.floating_hypotheses:
                # A macro's non-mandatory $f belongs to its own proof scope.
                # Freshen it for each application instead of leaking the old
                # label or accidentally identifying it with a caller variable.
                if len(active) > 1:
                    replacements[label] = fresh_dummy(expanded_database.floating_hypotheses[label])
                    push(replacements[label])
                else:
                    push([label])
                continue
            if label in expanded_database.essential_hypotheses:
                if len(active) > 1:
                    raise ValueError("unbound essential hypothesis in generated macro")
                push([label])
                continue
            rule = expanded_database.statements[label]
            count = len(rule.mandatory_hypotheses)
            if len(stack) < count:
                raise ValueError("malformed macro proof stack")
            actuals = stack[-count:] if count else []
            if count:
                del stack[-count:]
                stack_labels -= sum(map(len, actuals))
            if label in source_database.statements:
                fragment = [token for item in actuals for token in item] + [label]
            else:
                if label in active or rule.proof is None:
                    raise ValueError("cyclic or unproved generated dependency")
                bindings = {h.label: item for h, item in zip(rule.mandatory_hypotheses, actuals)}
                fragment = expand(rule.proof.source_labels, bindings, active | {label})
            push(fragment)
        if len(stack) != 1:
            raise ValueError("macro proof did not leave one result")
        return stack[0]

    labels = expand(theorem.proof.source_labels, {}, {theorem.name})
    result = copy.deepcopy(theorem)
    result.declaration_index = -1
    result.active_hypothesis_labels = frozenset()
    # Proof-only variables must be available to replay the explicit certificate.
    used_labels = set(labels)
    result.floating = tuple(h for label, h in expanded_database.floating_hypotheses.items()
                            if label in used_labels and label in theorem.active_hypothesis_labels)
    for h in theorem.floating:
        if h not in result.floating:
            result.floating += (h,)
    result.floating += tuple(h for h in fresh_floating if h.label in used_labels)
    result.variable_types = {h.expr.args[0].op: h.expr.op for h in result.floating}
    result.proof_variable_types = {}
    result.d_constraints |= result.proof_d_constraints
    result.d_constraints = {pair for pair in result.d_constraints if set(pair) <= set(result.variable_types)}
    result.proof_d_constraints = set()
    result.hypothesis_order = ()
    result.proof = Proof(result.name, source_labels=tuple(labels))
    # Macro-local dummy variables may require fresh $d conditions. Reconstruct
    # these from actual source applications; never strengthen the caller's
    # constraints between its existing variables.
    fresh_variables = {h.expr.args[0].op for h in fresh_floating}
    locals_by_label = {h.label: h.expr for h in (*result.floating, *result.hypotheses)}
    stack = []
    for label in labels:
        if label in locals_by_label:
            stack.append(locals_by_label[label])
            continue
        rule = source_database.statements[label]
        count = len(rule.mandatory_hypotheses)
        actuals = stack[-count:] if count else []
        if count:
            del stack[-count:]
        bindings = dict(zip((h.label for h in rule.mandatory_hypotheses), actuals))
        subst = {h.expr.args[0].op: bindings[h.label].args[0] for h in rule.floating}
        for left, right in rule.d_constraints:
            a_vars = {n.op for n in subst[left].walk() if n.op in result.variable_types}
            b_vars = {n.op for n in subst[right].walk() if n.op in result.variable_types}
            if a_vars & b_vars:
                raise ValueError("distinct variables collapse in expanded certificate")
            for a in a_vars:
                for b in b_vars:
                    pair = normalized_pair(a, b)
                    if pair not in result.d_constraints and not ({a, b} & fresh_variables):
                        raise ValueError("macro expansion requires an unavailable distinct-variable condition")
                    result.d_constraints.add(pair)
        stack.append(substitute_simultaneous(rule.conclusion, subst))
    augmented = copy.copy(source_database)
    augmented.variables = source_database.variables | set(result.variable_types)
    augmented.floating_hypotheses = {**source_database.floating_hypotheses,
                                     **{h.label: h for h in result.floating}}
    augmented.essential_hypotheses = {**source_database.essential_hypotheses,
                                      **{h.label: h for h in result.hypotheses}}
    verify(result, augmented)
    return result


def certify_generated(theorem, store, source_text, max_labels=16384):
    """Export all dependencies, replay every declaration, and erase macros."""
    from .export import export_metamath
    from .parser import MetamathParser
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "proof.mm"
        export_metamath(theorem, store, path)
        expanded = MetamathParser().parse_text(source_text + "\n" + path.read_text(encoding="utf-8"))
    for name, item in expanded.proved_theorems.items():
        if name not in store.parsed.statements:
            verify(item, expanded)
    return flatten_certificate(expanded.proved_theorems[theorem.name], expanded,
                               store.parsed, max_labels)


def replay_graph(store, source_text):
    """Serialize and verify a whole graph once, including generated rules."""
    from .export import export_metamath
    from .parser import MetamathParser
    roots = list(store.generated())
    if not roots:
        return store.parsed
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "graph.mm"
        export_metamath(roots[0], store, path, additional_roots=roots[1:])
        expanded = MetamathParser().parse_text(source_text + "\n" + path.read_text(encoding="utf-8"))
    for node in roots:
        verify(expanded.proved_theorems[node.name], expanded)
    return expanded


def certificate_steps(certificate, database):
    """Yield every logical source-rule application in its actual proof context.

    Depth counts logical source applications after macro expansion, not syntax
    labels or stored DAG edges. A caller must supply a verified certificate.
    """
    from .model import Theorem
    from .unification import substitute_simultaneous
    hypotheses = {h.label: h for h in (*certificate.floating, *certificate.hypotheses)}
    stack = []
    for index, label in enumerate(certificate.proof.source_labels):
        if label in hypotheses:
            stack.append((hypotheses[label].expr, 0))
            continue
        rule = database.statements[label]
        count = len(rule.mandatory_hypotheses)
        actuals = stack[-count:] if count else []
        if count:
            del stack[-count:]
        bindings = dict(zip((h.label for h in rule.mandatory_hypotheses), actuals))
        subst = {h.expr.args[0].op: bindings[h.label][0].args[0] for h in rule.floating}
        conclusion = substitute_simultaneous(rule.conclusion, subst)
        depth = (1 + max((bindings[h.label][1] for h in rule.hypotheses), default=0)
                 if conclusion.op == "|-" else 0)
        stack.append((conclusion, depth))
        if conclusion.op == "|-":
            yield Theorem(
                name=f"{certificate.name}_step{index}", hypotheses=list(certificate.hypotheses),
                conclusion=conclusion, variable_types=dict(certificate.variable_types),
                d_constraints=set(certificate.d_constraints),
                proof=Proof(rule.name, substitution={f"__rule_{v}": n for v, n in subst.items()},
                            depth=depth),
            )

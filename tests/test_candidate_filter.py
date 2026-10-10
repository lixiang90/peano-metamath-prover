"""Structural pruning must not reject candidates accepted by typed composition."""
from pathlib import Path

import pytest

from metamath_generator.certificates import replay_graph
from metamath_generator.compose import CompositionError, compose, instantiate_assertion
from metamath_generator.database import TheoremDatabase
from metamath_generator.generator import GenerationConfig, TheoremGenerator
from metamath_generator.model import Hypothesis, Node, Theorem
from metamath_generator.parser import MetamathParser, parse
from metamath_generator.random_graph import RandomProofGraph
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import verify

PA = Path(__file__).resolve().parents[1] / "formal/peano-pa-plus.mm"
may_unify = TheoremGenerator._nodes_may_unify


@pytest.mark.parametrize("typecode,constant,args,child_types", [
    ("QUANT", "forall", (Node("x"), Node("phi")), {"x": "var", "phi": "wff"}),
    ("BINPRED", "=", (Node("s"), Node("t")), {"s": "term", "t": "term"}),
    ("BINOP", "+", (Node("s"), Node("t")), {"s": "term", "t": "term"}),
    ("LOGBINOP", "implies", (Node("phi"), Node("psi")), {"phi": "wff", "psi": "wff"}),
])
def test_operator_variables_match_constants_and_each_other(typecode, constant, args, child_types):
    schematic = Node("op", args)
    concrete = Node(constant, args)
    renamed = Node("other_op", args)
    types = {"op": typecode, **child_types}
    other_types = {"other_op": typecode, **child_types}
    assert may_unify(schematic, concrete, types, child_types)
    assert may_unify(concrete, schematic, child_types, types)
    assert may_unify(schematic, renamed, types, other_types)


def test_whole_expression_variable_matches_applied_operator_variable():
    expression = Node("q", (Node("x"), Node("phi")))
    types = {"q": "QUANT", "x": "var", "phi": "wff"}
    assert may_unify(Node("psi"), expression, {"psi": "wff"}, types)
    assert may_unify(expression, Node("psi"), types, {"psi": "wff"})


@pytest.mark.parametrize("left,right,left_types,right_types", [
    (Node("forall", (Node("x"), Node("phi"))),
     Node("exists", (Node("x"), Node("phi"))), {}, {}),
    (Node("q", (Node("x"), Node("phi"))),
     Node("forall", (Node("x"),)), {"q": "QUANT"}, {}),
    (Node("op", (Node("0"), Node("0"))),
     Node("=", (Node("0"), Node("S", (Node("0"),)))), {"op": "BINPRED"}, {}),
])
def test_operator_wildcard_still_checks_arity_and_children(left, right, left_types, right_types):
    assert not may_unify(left, right, left_types, right_types)
    assert not may_unify(right, left, right_types, left_types)


def test_different_typecodes_can_be_compatible_in_pa_grammar():
    database = parse(PA)
    left = Node("|-", (Node("=", (Node("s"), Node("s"))),))
    right = Node("|-", (Node("=", (Node("x"), Node("x"))),))
    rule = Theorem("probe", [Hypothesis("h", left)], left,
                   variable_types={"s": "term"}, kind="axiom")
    parent = Theorem("parent", [], right, variable_types={"x": "var"}, kind="axiom")
    assert may_unify(left, right, rule.variable_types, parent.variable_types)
    assert may_unify(right, left, parent.variable_types, rule.variable_types)
    compose(rule, [parent], database=database)


def test_type_conflicts_are_still_rejected_by_composition():
    database = parse(PA)
    quantified = Node("|-", (Node("q", (Node("x"), Node("phi"))),))
    atom = Node("|-", (Node("p", (Node("s"), Node("t"))),))
    rule = Theorem("probe", [Hypothesis("h", quantified)], quantified,
                   variable_types={"q": "QUANT", "x": "var", "phi": "wff"}, kind="axiom")
    parent = Theorem("parent", [], atom,
                     variable_types={"p": "BINPRED", "s": "term", "t": "term"}, kind="axiom")
    assert may_unify(quantified, atom, rule.variable_types, parent.variable_types)
    with pytest.raises(CompositionError, match="ill-typed"):
        compose(rule, [parent], database=database)


def test_pa_assertions_and_typed_operator_instances_are_never_pruned():
    database = parse(PA)
    operators = {"QUANT": "forall", "BINPRED": "=", "BINOP": "+", "LOGBINOP": "and"}
    checked = 0
    for assertion in database.logical_assertions.values():
        mapping = {v: Node(operators[t]) for v, t in assertion.variable_types.items() if t in operators}
        if not mapping:
            continue
        instance = instantiate_assertion(assertion, mapping, database=database)
        pairs = [(assertion.conclusion, instance.conclusion)]
        pairs.extend((a.expr, b.expr) for a, b in zip(assertion.hypotheses, instance.hypotheses))
        for left, right in pairs:
            assert may_unify(left, right, assertion.variable_types, instance.variable_types), assertion.name
            assert may_unify(right, left, instance.variable_types, assertion.variable_types), assertion.name
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("operator", ["forall", "probe_quant"])
def test_pa_quantified_parent_reaches_both_candidate_pools_and_replays(operator):
    source = " ".join(MetamathParser()._tokens_with_includes(PA, set()))
    original = parse(PA)
    expression = substitute_simultaneous(original.logical_assertions["alpha_1"].conclusion,
                                         {"quant": Node(operator)}).to_prefix()
    # Add only a proved identity rule H |- H; no additional axioms are needed.
    source += ("\n${ $v probe_quant $. probe_f $f QUANT probe_quant $.\n"
               f"probe_h $e {expression} $.\n"
               f"probe_rule $p {expression} $= probe_h $.\n$}}\n")
    database = MetamathParser().parse_text(source)
    rule = database.proved_theorems["probe_rule"]
    verify(rule, database)
    store = TheoremDatabase(database)
    parent_id, _ = store.add(database.logical_assertions["alpha_1"], canonicalize=False)
    generator = TheoremGenerator(database, GenerationConfig(), store)
    graph = RandomProofGraph(generator)
    graph.rules = [rule]
    assert parent_id in [n.id for n in graph.candidates(rule.hypotheses[0].expr, rule)]
    assert parent_id in [n.id for n in generator._active_candidates(rule.hypotheses[0].expr)]
    result = graph.combine()
    assert result is not None
    assert generator._valid(result)
    assert result.proof.premise_map == [parent_id]
    if operator == "forall":
        assert graph.admit(result, "random_graph")
        generated = next(store.generated())
        expanded = replay_graph(store, source)
        verify(expanded.proved_theorems[generated.name], expanded)
    else:
        # Renaming the operator alone reproduces the parent's statement.
        assert not graph.admit(result, "random_graph")

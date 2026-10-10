"""Grammar-independent generation, verifier boundaries, and external replay."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from metamath_generator.cli import build_parser, main
from metamath_generator.generic import (
    GenericConfig, GenericGenerator, MatchBudget, SamplingFailure, match_tokens,
)
from metamath_generator.token_mm import TokenDatabase, TokenMMError, substitute

ROOT = Path(__file__).resolve().parents[1]
STRINGS = """
$c seq theorem a b $.
$v x y $.
sx $f seq x $.
sy $f seq y $.
empty $a seq $.
sa $a seq a $.
sb $a seq b $.
cat $a seq x y $.
base $a theorem a $.
${ h $e theorem x $. grow $a theorem x a $. $}
${ h1 $e theorem x $. h2 $e theorem y $. join $a theorem x y $. $}
copy $p theorem a $= ( base ) A $.
double $p seq a a $= ( sa cat ) AZCB $.
"""

INTERLEAVED = """
$c sort yes $.
$v x y $.
fx $f sort x $.
${ e $e yes x $. fy $f sort y $. r $a yes y $.
   good $p yes y $= fx e fy r $.
   compressed $p yes y $= ( r ) ABCD $.
$}
"""

DUMMY_DV = """
$c sort premise result $.
$v x y $.
fx $f sort x $. fy $f sort y $.
base $a premise y $.
${ $d x y $. h $e premise y $. drop $a result x $. $}
"""


def test_raw_kernel_compressed_saved_steps_and_empty_substitution():
    db = TokenDatabase.from_text(STRINGS + "zero $p seq $= empty empty cat $.\n")
    assert db.verified_proofs == 3
    assert db.assertions["zero"].expression == ("seq",)
    assert substitute(("x", "y"), {"x": ("y",), "y": ()}) == ("y",)


@pytest.mark.parametrize("proof", ["?", "( sa cat ) D", "( sa ) U", "( sa ) Z",
                                    "( sa ) AZ?", "sa sb", "empty", "future", "bad"])
def test_rejects_invalid_or_incomplete_proofs(proof):
    with pytest.raises(TokenMMError):
        TokenDatabase.from_text(STRINGS + f"bad $p seq a $= {proof} $.\n")


def test_interleaved_hypotheses_and_inactive_labels():
    source = INTERLEAVED
    db = TokenDatabase.from_text(source)
    assert [h.label for h in db.assertions["r"].hypotheses] == ["fx", "e", "fy"]
    with pytest.raises(TokenMMError, match="type mismatch"):
        TokenDatabase.from_text(source.replace("fx e fy r", "fx fy e r"))
    with pytest.raises(TokenMMError, match="inactive"):
        TokenDatabase.from_text(source + "escape $p yes x $= e $.")


def test_dv_pairwise_cross_product_and_scope():
    source = """
    $c seq yes $.
    $v x y z $.
    fx $f seq x $. fy $f seq y $. fz $f seq z $.
    cat $a seq x y $.
    ${ $d x y $. r $a yes x y $. $}
    """
    valid = "${ $d x z $. $d y z $. good $p yes x y z $= fx fy cat fz r $. $}"
    TokenDatabase.from_text(source + valid)
    with pytest.raises(TokenMMError, match="missing .d"):
        TokenDatabase.from_text(source + valid.replace("$d y z $.", ""))
    with pytest.raises(TokenMMError, match="collapse"):
        TokenDatabase.from_text(source + "bad $p yes x x $= fx fx r $.")


@pytest.mark.parametrize("source", [
    "$c sort $. ${ $c other $. $}",
    "$c sort $. $v x $. r $a sort x $.",
    "$c sort $. ${ $v x $. f $f sort x $. $} r $a sort x $.",
    "$c sort $. $v x $. f $f sort x $. g $f sort x $.",
    "$c sort $. $v x $. $d x x $.",
    "$c sort $. r $a sort $. r $a sort $.",
    "$( missing comment end", "${", "$}",
])
def test_invalid_declarations_fail_closed(source):
    with pytest.raises(TokenMMError):
        TokenDatabase.from_text(source)


def test_include_once_self_include_and_snapshot(tmp_path):
    (tmp_path / "part.mm").write_text(STRINGS + "$[ main.mm $]", encoding="utf-8")
    (tmp_path / "main.mm").write_text(
        "$( $[ missing.mm $] $) $[ part.mm $] $[ part.mm $]", encoding="utf-8")
    db = TokenDatabase.from_file(tmp_path / "main.mm")
    assert db.verified_proofs == 2
    assert TokenDatabase.from_text(db.isolated_source()).verified_proofs == 2


def test_word_matcher_repeated_empty_conditional_and_bounded():
    pattern = ("x", "a", "x", "y")
    solutions = list(match_tokens(pattern, ("a", "b"), {"x", "y"}, {},
                                 MatchBudget(100), random.Random(1)))
    assert {"x": (), "y": ("b",)} in solutions
    assert all(substitute(pattern, s) == ("a", "b") for s in solutions)
    assert list(match_tokens(("x", "x"), ("a", "b"), {"x"}, {},
                            MatchBudget(100), random.Random(1))) == []
    assert list(match_tokens(("x", "y"), ("a", "b"), {"x", "y"}, {"x": ("a",)},
                            MatchBudget(100), random.Random(1))) == [{"x": ("a",), "y": ("b",)}]
    with pytest.raises(SamplingFailure, match="search_budget"):
        list(match_tokens(("x", "y"), ("a", "b"), {"x", "y"}, {}, MatchBudget(1), random.Random(1)))


def test_generic_generation_without_pa_symbols_and_full_replay(tmp_path):
    generator = GenericGenerator(TokenDatabase.from_text(STRINGS), GenericConfig(seed=7))
    generator.generate(160)
    assert len(generator.generated) >= 5
    assert any(g.assertion.expression[0] == "theorem" and not g.assertion.essential
               and len(g.assertion.expression) > 2 for g in generator.generated)
    assert any(g.matched_premises for g in generator.generated)
    assert any(g.reserved_premises for g in generator.generated)
    assert any(g.rule.startswith(generator.prefix) for g in generator.generated)
    paths = generator.export(tmp_path, typecode="theorem")
    replay = TokenDatabase.from_file(paths["metamath"])
    assert replay.verified_proofs == len(generator.generated) + 2
    records = [json.loads(line) for line in paths["theorems"].read_text().splitlines()]
    assert records and all(r["expression"][0] == "theorem" for r in records)
    assert all(len(r["expression"]) <= generator.config.max_tokens for r in records)


def test_repeated_generation_preserves_rng_and_counts(tmp_path):
    db = TokenDatabase.from_text(STRINGS)
    first, second = GenericGenerator(db), GenericGenerator(db)
    first.generate(60)
    second.generate(20)
    second.generate(40)
    assert first.generated == second.generated
    assert first.stats == second.stats
    a, b = first.export(tmp_path / "a"), second.export(tmp_path / "b")
    assert a["metamath"].read_bytes() == b["metamath"].read_bytes()


def test_global_essential_scope_does_not_leak_into_generated_assertions(tmp_path):
    source = STRINGS + "ambient $e theorem b $. last $a theorem a b $."
    generator = GenericGenerator(TokenDatabase.from_text(source))
    generator.generate(40)
    replay = TokenDatabase.from_file(generator.export(tmp_path)["metamath"])
    assert generator.generated
    for generated in generator.generated:
        assert len(replay.assertions[generated.assertion.label].essential) == len(generated.assertion.essential)


def test_dummy_disjoints_are_exported_but_not_required_by_lemma_callers(tmp_path):
    generator = GenericGenerator(TokenDatabase.from_text(DUMMY_DV))
    generator.generate(60)
    results = [g for g in generator.generated if g.assertion.expression[0] == "result"
               and not g.assertion.essential]
    assert results
    assert any(g.proof_disjoints and not g.assertion.disjoints for g in results)
    TokenDatabase.from_file(generator.export(tmp_path)["metamath"])


def test_inferred_replacement_requires_a_floating_proof():
    source = """
    $c seq yes bad $.
    $v x $. fx $f seq x $.
    untyped $a yes bad $.
    ${ h $e yes x $. rule $a yes x x $. $}
    """
    generator = GenericGenerator(TokenDatabase.from_text(source))
    # There is a proof of 'yes bad' but no proof of 'seq bad'. Token matching
    # alone must never turn this into an application of rule.
    for _ in range(10):
        try:
            fact, _, _ = generator.sample(generator.assertions["rule"])
        except SamplingFailure:
            continue
        assert fact.expression != ("yes", "bad", "bad")


def test_cli_default_unchanged_and_generic_bypasses_ast(tmp_path):
    assert build_parser().parse_args(["formal/peano.mm"]).mode == "graph"
    source = tmp_path / "strings.mm"
    source.write_text(STRINGS, encoding="utf-8")
    assert main([str(source), "--mode", "generic", "--steps", "30",
                 "--output-dir", str(tmp_path / "out")]) == 0
    assert TokenDatabase.from_file(tmp_path / "out/generated.mm").verified_proofs > 2


@pytest.mark.parametrize("option,value", [("match_budget", 0), ("instance_probability", 1.1),
                                          ("max_hypotheses", -1), ("seed_assertions", -1)])
def test_invalid_budgets(option, value):
    with pytest.raises(ValueError):
        replace(GenericConfig(), **{option: value})


@pytest.mark.parametrize("theory", ["peano.mm", "peano-pa-plus.mm"])
def test_pa_databases_replay_through_independent_token_path(theory, tmp_path):
    generator = GenericGenerator(TokenDatabase.from_file(ROOT / "formal" / theory))
    generator.generate(120)
    assert generator.generated
    TokenDatabase.from_file(generator.export(tmp_path)["metamath"])


@pytest.mark.skipif(not os.environ.get("METAMATH_EXECUTABLE"), reason="set METAMATH_EXECUTABLE for official replay")
@pytest.mark.parametrize("source", [STRINGS, INTERLEAVED, DUMMY_DV])
def test_official_verifier_accepts_generic_proofs(tmp_path, source):
    generator = GenericGenerator(TokenDatabase.from_text(source))
    generator.generate(100)
    path = generator.export(tmp_path)["metamath"]
    result = subprocess.run([os.environ["METAMATH_EXECUTABLE"]],
                            input=f'read "{path.as_posix()}"\nverify proof *\nexit\n',
                            text=True, capture_output=True, timeout=30)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    assert "All proofs in the database were verified" in log, log
    assert "?Error" not in log, log

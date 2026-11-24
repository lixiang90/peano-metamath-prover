import hashlib
import json
import time
from pathlib import Path

import pytest

from metamath_generator.model import Theorem
from metamath_generator.parser import parse
from pa_prover_v21.interaction import Interaction, command
from pa_prover_v21.kernel import N, eq
from pa_prover_v21.search import Classical, SearchConfig
from pa_prover_v21.strategy import Action, Session

from v3.proof_files import FileBudgetExceeded, ProofFiles


@pytest.fixture(scope="module")
def database():
    return parse(Path(__file__).resolve().parents[1] / "formal/peano-pa-plus.mm")


@pytest.fixture
def episode(database, tmp_path):
    n, zero = N("x0"), N("0")
    goal = eq(N("+", N("+", zero, n), N("S", zero)), N("S", n))
    target = Theorem("v3_control_probe", [], N("|-", goal), variable_types={"x0": "var"})
    # This fixture checks replayed proof states and certificates, independently of machine speed.
    cfg = SearchConfig(reasoning_mode="scratchpad", seconds=60, closure_seconds=10, trial_seconds=3)
    session = Session(database, target)
    flow = Interaction(session, session.initial, cfg.reasoning_mode)
    end = time.perf_counter() + cfg.seconds
    remaining = lambda: end - time.perf_counter()
    backend = Classical(session, cfg, end, tmp_path)
    flow.initialize(backend, remaining)
    return ProofFiles(flow), (backend, cfg, remaining)


def test_restore_read_and_use_note(episode):
    files, engine = episode
    initial = files.flow.state
    note = "wff = + 0 x0 x0"
    assert files.file("FAPPEND FILE_SCRATCH " + note) == {"index": 0}
    assert files.flow.state is initial
    action = Action("HAVE", 0, statement="= 0 x0")
    files.advance(command("TRY", action), *engine)
    files.file("FWRITE FILE_SCRATCH 0 wff = 0 x0")
    assert files.file("FREAD FILE_SCRATCH 0 1")["rows"] == ["wff = 0 x0"]
    tokens_left = files.tokens_left
    calls_left = files.calls_left
    files.advance(command("BACKTRACK"), *engine)
    assert files.flow.state is initial
    assert (files.tokens_left, files.calls_left) == (tokens_left, calls_left)
    trial = files.file("FREAD FILE_TRIALS 0 1")["rows"][0]
    assert trial["action"] == action.record() and trial["outcome"] == "UNKNOWN"
    assert trial["decision"] == "BACKTRACK" and trial["state"] == 0
    assert set(trial) == {"state", "action", "outcome", "closed_count", "decision"}
    recovered, = files.file("FREAD FILE_SCRATCH 0 1")["rows"]
    assert recovered == note
    files.advance(command("APPLY", Action("HAVE", 0, statement=recovered.removeprefix("wff "))), *engine)
    assert files.flow.state.frontier == (1,) and files.flow.state.proofs[1] is None
    files.advance(command("APPLY", Action("INDUCT", 1, variable="x0")), *engine)
    assert files.flow.state.solved


def test_published_file_trace_and_certificate(episode, tmp_path):
    files, engine = episode
    results = json.loads(Path(__file__).with_name("results.json").read_text(encoding="utf-8"))
    trace = results["manually_supplied_lemma_and_file_commands"]
    tokens, calls = files.tokens_left, files.calls_left
    for step in trace["commands_and_replies"]:
        if "file_command" in step:
            assert files.file(step["file_command"]) == step["reply"]
        else:
            files.advance(json.dumps(step["proof_command"]), *engine)
            content = files.flow.events[-1]["content"]
            assert (content["event"], content["status"]) == (step["event"], step["status"])

    assert calls - files.calls_left == trace["file_calls"]
    assert tokens - files.tokens_left == trace["file_io_tokens"]
    assert files.flow.state.solved
    fragment = tmp_path / "fragment.mm"
    session = files.flow.session
    session.b.export(files.flow.state.proofs[0], session.target, fragment)
    portable_bytes = fragment.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(portable_bytes).hexdigest() == trace["fragment_lf_sha256"]


def test_pages_commit_and_read_only_views(episode):
    files, engine = episode
    row = files.file("FREAD FILE_PROBLEM 0 1")
    assert row["next"] == 1 and row["rows"][0]["goal"] == str(files.flow.session.target.conclusion)
    row["rows"][0]["goal"] = "modified"
    assert files.file("FREAD FILE_PROBLEM 0 1")["rows"][0]["goal"] != "modified"
    assert files.file("FREAD FILE_LIBRARY 0 1") == {"rows": [], "next": None}
    for name in ("FILE_PROBLEM", "FILE_LIBRARY", "FILE_TRIALS"):
        for op in ("FAPPEND", "FWRITE"):
            with pytest.raises(ValueError, match="written by the engine"):
                files.file(f"{op} {name} 0 wff = 0 0")
    files.advance(command("TRY", Action("HAVE", 0, statement="= 0 x0")), *engine)
    files.file("FAPPEND FILE_PLAN wff = + 0 x0 x0")
    assert files.file("FREAD FILE_TRIALS 0 1")["rows"][0]["decision"] is None
    files.advance(command("COMMIT"), *engine)
    assert files.file("FREAD FILE_PLAN 0 1")["rows"] == ["wff = + 0 x0 x0"]
    assert files.file("FREAD FILE_TRIALS 0 1")["rows"][0]["decision"] == "COMMIT"


def test_same_parent_keeps_one_state_id_across_trial_pages(episode):
    files, engine = episode
    for statement in ("= 0 x0", "= 0 S 0"):
        files.advance(command("TRY", Action("HAVE", 0, statement=statement)), *engine)
        files.advance(command("BACKTRACK"), *engine)
    page1 = files.file("FREAD FILE_TRIALS 0 1")
    page2 = files.file(f"FREAD FILE_TRIALS {page1['next']} 1")
    assert page1["rows"][0]["state"] == page2["rows"][0]["state"] == 0
    assert page1["rows"][0]["action"] != page2["rows"][0]["action"] and page2["next"] is None


def test_library_reads_only_public_statements(episode, monkeypatch):
    files, _ = episode
    visible = files.flow.events[-1]["content"]["state"]
    entry = {"id": "l5", "statement": "= + 0 x0 x0", "archive": "private"}
    monkeypatch.setitem(visible, "library", [entry])
    assert files.file("FREAD FILE_LIBRARY 0 1") == {
        "rows": [{"id": "l5", "statement": "= + 0 x0 x0"}], "next": None,
    }


def test_bad_commands_and_exhausted_reply_budget_are_atomic(episode):
    files, _ = episode
    files.file("FAPPEND FILE_SCRATCH term + 0 x0")
    for text in ("FWRITE FILE_SCRATCH 99 wff = 0 0", "FWRITE FILE_SCRATCH 0 wff = x8 x8",
                 "FAPPEND FILE_PLAN term 0", "FWRITE FILE_SCRATCH 0 wff = 0 0\nCOMMIT",
                 "FREAD ../FILE_PLAN 0 1", "FREAD FILE_PLAN 0 9"):
        with pytest.raises(ValueError):
            files.file(text)
    assert files.file("FREAD FILE_SCRATCH 0 8")["rows"] == ["term + 0 x0"]
    text = "FAPPEND FILE_SCRATCH term S 0"
    # Enough tokens for the command but not its acknowledgement.
    cost = len(files.tokenizer.encode(text))
    limited = ProofFiles(files.flow, max_io_tokens=cost, max_calls=1)
    with pytest.raises(FileBudgetExceeded):
        limited.file(text)
    assert limited.calls_left == 0 and limited.tokens_left == 0
    assert limited._pages["FILE_SCRATCH"] == ()

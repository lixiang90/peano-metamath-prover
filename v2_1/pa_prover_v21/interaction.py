"""Executable symbolic scratchpads. Engine receipts never enter model prompts.

TRY opens one isolated macro-action trial with bounded classical closure.
COMMIT keeps that branch; BACKTRACK restores its exact parent. A memory entry
is an observation about a search attempt, never an additional logical fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import time

from .library import archive, replay
from .solver import BudgetExpired
from .strategy import Action, Session

PROTOCOL = "v21-interaction-1"


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def command(kind, action=None):
    r = {"op": kind}
    if action is not None:
        r["action"] = action.record()
    return compact(r)


def parse_command(text):
    r = json.loads(text)
    if r.get("op") in {"APPLY", "TRY"} and set(r) == {"op", "action"}:
        return r["op"], Action.parse(compact(r["action"]))
    if r.get("op") in {"COMMIT", "BACKTRACK"} and set(r) == {"op"}:
        return r["op"], None
    raise ValueError("invalid interaction command")


def segments(events, *, awaiting_model=False):
    """One role-delimited continuous stream, with byte-aligned model spans."""
    result = []
    for event in events:
        role = event["role"]
        if role not in {"engine", "model"}:
            raise ValueError("invalid interaction role")
        result.append((role.upper()+"\n", False))
        result.append((compact(event["content"])+"\n", role == "model"))
    if awaiting_model:
        result.append(("MODEL\n", False))
    return result


def render(events, *, awaiting_model=False):
    return "".join(s for s, _ in segments(events, awaiting_model=awaiting_model))


def context(builder):
    return {"types": dict(builder.types), "distinct": [list(p) for p in sorted(builder.distinct)]}


def extend_fresh_context(builder, record):
    """Replay may only add the same fresh variables, not weaken source $d's."""
    desired = record["types"]
    if any(desired.get(v) != t for v, t in builder.types.items()):
        raise ValueError("receipt changed existing variable types")
    while len(builder.types) < len(desired):
        v = builder.fresh()
        if desired.get(v.op) != "var":
            raise ValueError("receipt contains a non-fresh context extension")
    if builder.types != desired or builder.distinct != set(map(tuple, record["distinct"])):
        raise ValueError("receipt changed distinct-variable constraints")


def close_recorded(session, state, backend):
    operations = []
    attempts = []
    started = time.perf_counter()
    def finish(exhausted):
        return state, {"operations": operations, "context": context(session.b), "exhausted": exhausted,
                       "attempts": attempts, "elapsed_seconds": round(time.perf_counter()-started, 6)}
    changed = True
    while changed and not state.solved:
        changed = False
        for i in tuple(state.frontier):
            if time.perf_counter() >= backend.deadline:
                return finish(True)
            try:
                p = backend.solve(state.tasks[i])
            except BudgetExpired:
                attempts.append({"goal": i, "status": "UNKNOWN_WITHIN_BUDGET"})
                return finish(True)
            attempts.append({"goal": i, "status": "PROOF_CONSTRUCTED" if p is not None else "UNKNOWN"})
            if p is not None:
                operations.append({"kind": "close", "goal": i, "proof": archive(session.b, p)})
                state = session.close(state, i, p)
                changed = True
                break
            before = context(session.b)
            child = session.structural(state, i)
            if child is not None:
                operations.append({"kind": "structural", "goal": i, "context": before})
                state = child
                changed = True
                break
    return finish(False)


def replay_closure(session, state, receipt):
    for op in receipt["operations"]:
        if op["kind"] == "close":
            extend_fresh_context(session.b, op["proof"])
            proof, _ = replay(session.b, op["proof"])
            state = session.close(state, op["goal"], proof)
        elif op["kind"] == "structural":
            extend_fresh_context(session.b, op["context"])
            child = session.structural(state, op["goal"])
            if child is None:
                raise ValueError("invalid structural receipt")
            state = child
        else:
            raise ValueError("invalid closure receipt")
    extend_fresh_context(session.b, receipt["context"])
    return state


@dataclass
class Interaction:
    session: Session
    state: object
    mode: str
    events: list = field(default_factory=list)
    receipts: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    pending: object = None
    depth: int = 0
    trials: int = 0

    def fork(self):
        other = copy.copy(self)
        other.session = self.session.fork()
        other.events = list(self.events)
        other.receipts = list(self.receipts)
        other.memory = list(self.memory)
        return other

    def observation(self):
        # Receipts, proof archives, future actions, and sibling branches are absent.
        return {"protocol": PROTOCOL, "mode": self.mode, "events": copy.deepcopy(self.events)}

    def feedback(self, event, remaining, receipt, action=None):
        state = self.session.observation(self.state, remaining)
        closed = [op["goal"] for op in receipt.get("operations", ()) if op["kind"] == "close"]
        content = {"event": event, "mode": self.mode, "state": state, "closed": closed,
                   "status": "ROOT_READY" if self.state.solved else "OPEN",
                   "pending_trial": self.pending is not None,
                   "memory": self.memory[-8:], "depth": self.depth, "trials": self.trials}
        content["execution"] = {"attempts": receipt.get("attempts", []),
                                "seconds": receipt.get("elapsed_seconds", 0.)}
        if action is not None:
            content["action"] = action.record()
        if receipt.get("exhausted"):
            content["search_status"] = "UNKNOWN_WITHIN_BUDGET"
        self.events.append({"role": "engine", "content": json.loads(compact(content))})
        self.receipts.append(receipt)

    def initialize(self, backend, remaining, receipt=None):
        if receipt is None:
            self.state, receipt = close_recorded(self.session, self.state, backend)
        else:
            self.state = replay_closure(self.session, self.state, receipt)
        self.feedback("STATE", remaining(), receipt)

    def options(self, cfg):
        from .search import legal_candidates
        if self.pending is not None:
            commit, back = command("COMMIT"), command("BACKTRACK")
            useful = self.state.solved or bool(self.events[-1]["content"]["closed"])
            return [commit, back] if useful else [back, commit]
        if self.depth >= cfg.max_depth or self.state.solved:
            return []
        actions = [a for a, _, _ in legal_candidates(self.session, self.state, cfg.max_candidates)]
        normal = [command("APPLY", a) for a in actions]
        if self.mode != "scratchpad" or self.trials >= cfg.max_trials:
            return normal
        key = self.session.key(self.state)
        tried = {(m["state_key"], m["action"]) for m in self.memory}
        trials = [command("TRY", a) for a in actions if (key, a.text()) not in tried]
        return trials + normal

    def execute(self, text, backend, cfg, remaining, receipt=None):
        op, action = parse_command(text)
        if op == "TRY" and (self.mode != "scratchpad" or self.trials >= cfg.max_trials):
            raise ValueError("trial limit or mode forbids TRY")
        if op in {"COMMIT", "BACKTRACK"}:
            if self.pending is None:
                raise ValueError("no trial to commit or restore")
            if op == "BACKTRACK":
                self.session, self.state, self.depth = self.pending[:3]
                self.session = self.session.fork()
            self.pending = None
            if op == "COMMIT":
                # Resume normal closure after a deliberately short trial. An
                # unresolved trial is not a permanent failure of this branch.
                backend.session = self.session
                backend.cache = {}
                if receipt is None:
                    self.state, closure = close_recorded(self.session, self.state, backend)
                else:
                    closure = receipt
                    self.state = replay_closure(self.session, self.state, receipt)
            else:
                closure = {"operations": [], "context": context(self.session.b), "exhausted": False}
                if receipt is not None and receipt != closure:
                    raise ValueError("restore cannot inject proof facts")
        else:
            if self.pending is not None or self.depth >= cfg.max_depth:
                raise ValueError("commit or restore pending trial before another action")
            old = (self.session.fork(), self.state, self.depth, self.session.key(self.state))
            self.session = self.session.fork()
            self.state = self.session.apply(self.state, action)
            self.depth += 1
            if op == "TRY":
                self.pending = old
                self.trials += 1
            backend.session = self.session
            # Classical proofs can contain auxiliary variables allocated by
            # their originating branch. Never reuse that cache across forks.
            backend.cache = {}
            previous_deadline = backend.deadline
            if op == "TRY":
                backend.deadline = min(previous_deadline, time.perf_counter()+cfg.trial_seconds)
            try:
                if receipt is None:
                    self.state, closure = close_recorded(self.session, self.state, backend)
                else:
                    closure = receipt
                    self.state = replay_closure(self.session, self.state, receipt)
            finally:
                backend.deadline = previous_deadline
            if op == "TRY":
                self.memory.append({"state_key": old[3], "action": action.text(),
                    "outcome": "ROOT_READY" if self.state.solved else "UNKNOWN",
                    "closed": [x["goal"] for x in closure["operations"] if x["kind"] == "close"]})
        self.events.append({"role": "model", "content": json.loads(text)})
        self.feedback(op+"_RESULT", remaining(), closure, action)


def replay_interaction(database, target, library, result, cfg):
    """Validate all visible engine states against proof-producing receipts.

    Timed failures are recorded observations, not replayed mathematical claims.
    Successful root certification is independently checked by corpus auditing.
    """
    from .search import Classical
    events = result["interaction"]
    receipts = result["receipts"]
    if not events or len(events) != 2*len(receipts)-1:
        raise ValueError("malformed interaction transcript")
    s = Session(database, target, library)
    flow = Interaction(s, s.initial, cfg.reasoning_mode)
    backend = Classical(s, cfg, time.perf_counter()+1, ".")
    def budget(event):
        return lambda: event["content"]["state"]["budget"]
    flow.initialize(backend, budget(events[0]), receipts[0])
    if flow.events[-1] != events[0]:
        raise ValueError("initial engine feedback mismatch")
    for i, receipt in enumerate(receipts[1:], 1):
        model, engine = events[2*i-1:2*i+1]
        if model["role"] != "model" or engine["role"] != "engine":
            raise ValueError("invalid event roles")
        text = compact(model["content"])
        if text not in flow.options(cfg):
            raise ValueError("transcript command outside legal candidate grammar")
        if result["certified"] or result.get("trace"):
            trace=result["trace"]
            if len(trace)!=len(receipts)-1 or trace[i-1]["observation"]!=flow.observation() or trace[i-1]["action"]!=text:
                raise ValueError("policy trace does not match causal interaction prefix")
            if trace[i-1]["candidates"]!=flow.options(cfg) or trace[i-1]["candidates"][trace[i-1]["index"]]!=text:
                raise ValueError("policy trace candidates differ from executor")
        flow.execute(text, backend, cfg, budget(engine), receipt)
        if flow.events[-1] != engine:
            raise ValueError("engine feedback mismatch")
    if result["certified"] and (not flow.state.solved or flow.pending is not None):
        raise ValueError("certified transcript has uncommitted or open obligations")
    return flow

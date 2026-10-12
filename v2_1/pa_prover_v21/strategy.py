"""Typed high-level actions and proof-producing, scoped AND obligations.

No neural prediction creates a fact. Persistent states share immutable proof
DAG nodes; a HAVE continuation is inaccessible until its lemma is proved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as changed
import hashlib
import json

from metamath_generator.model import Theorem, Hypothesis
from metamath_generator.parser import MetamathParser
from .kernel import Builder, Evidence, N, eq, imp, replace
from .atp import Translator, DEFINITIONS


def formula(text, builder):
    parser = MetamathParser()
    # Parsing must not mutate the shared source database.
    import copy
    parser.database = copy.copy(builder.db)
    parser.database.variable_types = {**builder.db.variable_types, **builder.types}
    node = parser.parse_expression(["wff", *text.split()]).args[0]
    builder.check_type(node)
    Translator(builder.types).formula(node)
    return node


@dataclass(frozen=True)
class Action:
    kind: str
    goal: int
    variable: str = ""
    generalize: tuple[str, ...] = ()
    statement: str = ""
    lemma: str = ""

    def record(self):
        result = {"kind": self.kind, "goal": self.goal}
        for key in ("variable", "generalize", "statement", "lemma"):
            if getattr(self, key): result[key] = getattr(self, key)
        return result

    def text(self):
        return json.dumps(self.record(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def parse(cls, text):
        record = json.loads(text)
        if set(record) - {"kind", "goal", "variable", "generalize", "statement", "lemma"}:
            raise ValueError("unknown action fields")
        if type(record.get("goal")) is not int: raise ValueError("integer goal required")
        if record.get("kind") not in {"INDUCT", "STRENGTHEN", "HAVE", "USE", "UNFOLD"}:
            raise ValueError("unknown high-level action")
        if any(not isinstance(record.get(k,""),str) for k in ("variable","statement","lemma")):
            raise ValueError("action expressions and names must be strings")
        if not isinstance(record.get("generalize",()),(list,tuple)):
            raise ValueError("generalization requires a list of names")
        record["generalize"] = tuple(record.get("generalize", ()))
        if any(not isinstance(x, str) for x in record["generalize"]):
            raise ValueError("generalization variables must be names")
        return cls(**record)


@dataclass(frozen=True)
class Task:
    goal: object
    facts: tuple[Evidence, ...]
    role: str = "goal"


@dataclass(frozen=True)
class Frame:
    parent: int
    children: tuple[int, ...]
    kind: str
    payload: object = None


@dataclass(frozen=True)
class State:
    tasks: tuple[Task, ...]
    proofs: tuple[Evidence | None, ...]
    frames: tuple[Frame, ...]
    frontier: tuple[int, ...]
    history: tuple[str, ...] = ()

    @property
    def solved(self):
        return self.proofs[0] is not None and not self.frontier


@dataclass(frozen=True)
class InductionPlan:
    goal: object
    binders: tuple
    n: object
    x: object
    y: object
    z: object
    phi: object
    generalized: tuple
    hypothesis: object
    base: object
    step: object
    step_fact: Evidence

    def finish(self, b, base, step):
        def guarded(value, proof):
            h = b.hyp(eq(self.y, value))
            bridge = b.formula_subst(self.phi, self.y, value, h)
            p = b.mp(proof, b.iff_dir(bridge, 1))
            return b.forall_intro(self.y, b.deduct(h.formula, p))
        base_guard = guarded(N("0"), base)
        step_imp = b.deduct(self.hypothesis, guarded(N("S", self.x), step))
        schema_ih = N("forall", self.y, imp(eq(self.y, self.x), self.phi))
        renamed = b.forall_intro(self.z, b.forall_elim(b.hyp(schema_ih), self.z))
        step_all = b.forall_intro(self.x, b.deduct(schema_ih, b.mp(renamed, step_imp)))
        schema = b.ax("induction", {"x": self.x, "y": self.y, "phi": self.phi})
        full = b.mp(b.and_intro(base_guard, step_all), schema)
        at_n = b.forall_elim(full, self.n)
        result = b.mp(b.refl(self.n), b.forall_elim(at_n, self.n))
        for v in self.generalized:
            # Rename the binder before instantiating it at its old free name.
            fresh = b.fresh()
            result = b.forall_elim(b.forall_intro(fresh, b.forall_elim(result, fresh)), v)
        for v in reversed(self.binders): result = b.forall_intro(v, result)
        if result.formula != self.goal: raise ValueError("induction target mismatch")
        return result


def induction_plan(b, task, variable, generalize=()):
    body, binders = task.goal, []
    while body.op == "forall":
        v, body = body.args
        if v in binders: raise ValueError("shadowed induction binders are unsupported")
        binders.append(v)
    n = N(variable)
    if b.types.get(variable) != "var" or n not in set(body.walk()):
        raise ValueError("induction variable must occur in the predicate")
    if len(set(generalize)) != len(generalize) or variable in generalize:
        raise ValueError("invalid generalization variables")
    gs = tuple(N(v) for v in generalize)
    for v in (n, *gs):
        if b.types.get(v.op) != "var" or v not in set(body.walk()):
            raise ValueError("invalid parameter")
        if any(v in set(a.walk()) for p in task.facts for a in p.assumptions):
            raise ValueError("cannot generalize a variable in an open assumption")
    predicate = body
    for v in reversed(gs): predicate = N("forall", v, predicate)
    x, y, z = b.fresh(), b.fresh(), b.fresh()
    phi = replace(predicate, n, y)
    ih = N("forall", z, imp(eq(z, x), replace(phi, y, z)))
    px = b.mp(b.refl(x), b.forall_elim(b.hyp(ih), x))
    return InductionPlan(task.goal, tuple(binders), n, x, y, z, phi, gs, ih,
                         replace(predicate, n, N("0")), replace(predicate, n, N("S", x)), px)


class Session:
    def __init__(self, database, target, library=None):
        self.target, self.library = target, library
        self.b = Builder(database, target.variable_types, target.d_constraints)
        Translator(self.b.types).formula(target.conclusion.args[0])
        self.b.check_type(target.conclusion.args[0])
        facts = tuple(self.b.hyp(h.expr.args[0]) for h in target.hypotheses)
        for p in facts: self.b.check_type(p.formula)
        self.initial = State((Task(target.conclusion.args[0], facts),), (None,), (), (0,))

    def fork(self):
        # Each branch owns fresh-name allocation. Exploration of a sibling
        # must not change the serialization or replay of this trajectory.
        import copy
        other=copy.copy(self)
        other.b=Builder(self.b.db,self.b.types,self.b.distinct)
        other.b._cache=dict(self.b._cache)
        other.b._deductions=dict(self.b._deductions)
        other.b._hypotheses=dict(self.b._hypotheses)
        other.b._renamed_quantifiers=dict(getattr(self.b,"_renamed_quantifiers",{}))
        return other

    def spawn(self, state, parent, tasks, kind, payload=None, blocked=()):
        children = tuple(range(len(state.tasks), len(state.tasks)+len(tasks)))
        return State(state.tasks+tuple(tasks), state.proofs+(None,)*len(tasks),
                     state.frames+(Frame(parent, children, kind, payload),),
                     tuple(i for i in state.frontier if i != parent)+tuple(c for j,c in enumerate(children) if j not in blocked),
                     state.history)

    def close(self, state, index, proof):
        if index not in state.frontier: raise ValueError("goal is not active")
        task = state.tasks[index]
        allowed = frozenset().union(*(p.assumptions for p in task.facts))
        if proof.formula != task.goal or not proof.assumptions <= allowed:
            raise ValueError("proof does not discharge the scoped goal")
        proofs = list(state.proofs); proofs[index] = proof
        tasks = list(state.tasks)
        frontier = [i for i in state.frontier if i != index]
        changed_any = True
        while changed_any:
            changed_any = False
            for frame in reversed(state.frames):
                if proofs[frame.parent] is not None: continue
                ps = [proofs[i] for i in frame.children]
                if frame.kind == "HAVE" and ps[0] is not None and ps[1] is None:
                    continuation = frame.children[1]
                    if continuation not in frontier and not any(f.parent == continuation for f in state.frames):
                        tasks[continuation] = changed(tasks[continuation], facts=tasks[frame.parent].facts+(ps[0],))
                        frontier.append(continuation)
                if not all(p is not None for p in ps): continue
                b = self.b
                if frame.kind == "INDUCT": p = frame.payload.finish(b, *ps)
                elif frame.kind == "HAVE": p = ps[1]
                elif frame.kind == "STRENGTHEN": p = b.mp(ps[0], ps[1])
                elif frame.kind == "and": p = b.and_intro(*ps)
                elif frame.kind == "implies": p = b.deduct(frame.payload, ps[0])
                elif frame.kind == "forall": p = b.forall_intro(frame.payload, ps[0])
                elif frame.kind == "UNFOLD": p = b.mp(ps[0], b.iff_dir(frame.payload, 1))
                elif frame.kind == "USE":
                    p = frame.payload
                    for minor in ps: p = b.mp(minor, p)
                else: raise ValueError("unknown proof frame")
                parent = tasks[frame.parent]
                allowed = frozenset().union(*(f.assumptions for f in parent.facts))
                if p.formula != parent.goal or not p.assumptions <= allowed:
                    raise ValueError("proof frame escaped its scope")
                proofs[frame.parent] = p
                changed_any = True
        return State(tuple(tasks), tuple(proofs), state.frames, tuple(frontier), state.history)

    def structural(self, state, index):
        if index not in state.frontier:raise ValueError("inactive structural goal")
        task = state.tasks[index]; op, args = task.goal.op, task.goal.args
        if op == "and": return self.spawn(state, index, [Task(g, task.facts) for g in args], op)
        if op == "implies":
            return self.spawn(state, index, [Task(args[1], task.facts+(self.b.hyp(args[0]),))], op, args[0])
        if op == "forall" and not any(args[0] in set(a.walk()) for p in task.facts for a in p.assumptions):
            return self.spawn(state, index, [Task(args[1], task.facts)], op, args[0])
        if op == "forall":
            bridge=self.b.rename_forall(task.goal,self.b.fresh())
            return self.spawn(state,index,[Task(bridge.formula.args[1],task.facts)],"UNFOLD",bridge)
        return None

    def apply(self, state, action):
        if action.goal not in state.frontier: raise ValueError("inactive action target")
        task = state.tasks[action.goal]
        if action.kind == "INDUCT":
            plan = induction_plan(self.b, task, action.variable, action.generalize)
            result = self.spawn(state, action.goal,
                [Task(plan.base, task.facts, "base"), Task(plan.step, task.facts+(plan.step_fact,), "step")], "INDUCT", plan)
        elif action.kind in {"HAVE", "STRENGTHEN"}:
            lemma = formula(action.statement, self.b)
            if lemma == task.goal or any(lemma == p.formula for p in task.facts):
                raise ValueError("circular or redundant lemma")
            next_goal = task.goal if action.kind == "HAVE" else imp(lemma, task.goal)
            result = self.spawn(state, action.goal,
                [Task(lemma, task.facts, "lemma"), Task(next_goal, task.facts, "continuation")],
                action.kind, blocked=(1,) if action.kind == "HAVE" else ())
        elif action.kind == "UNFOLD":
            label = "df-"+task.goal.op
            if label not in DEFINITIONS: raise ValueError("definition outside supported fragment")
            ax = self.b.db.logical_assertions[label]
            subst = {v.op: t for v,t in zip(ax.conclusion.args[0].args[0].args, task.goal.args)}
            bridge = self.b.ax(label, subst)
            result = self.spawn(state, action.goal, [Task(bridge.formula.args[1], task.facts)], "UNFOLD", bridge)
        elif action.kind == "USE" and self.library is not None:
            p, premises = self.library.instantiate(action.lemma, task, self.b)
            if not premises: result = self.close(state, action.goal, p)
            else: result = self.spawn(state, action.goal, [Task(g, task.facts) for g in premises], "USE", p)
        else: raise ValueError("unsupported action")
        return changed(result, history=state.history+(action.text(),))

    def observation(self, state, remaining, library_limit=8):
        # Include blocked obligations; never truncate away an unsolved goal.
        unresolved = [i for i,p in enumerate(state.proofs) if p is None
                      and not any(f.parent == i for f in state.frames)]
        tasks = [{"id": i, "goal": str(state.tasks[i].goal), "role": state.tasks[i].role,
                  "active": i in state.frontier,
                  "facts": [{"formula": str(p.formula), "assumptions": sorted(map(str,p.assumptions))}
                            for p in state.tasks[i].facts]} for i in unresolved]
        nodes = [n for i in unresolved for f in [state.tasks[i].goal,
                 *(p.formula for p in state.tasks[i].facts),
                 *(a for p in state.tasks[i].facts for a in p.assumptions)] for n in f.walk()]
        used = {n.op for n in nodes if n.op in self.b.types}
        return {"goals": tasks, "types": {v:self.b.types[v] for v in sorted(used)},
                "distinct": sorted(p for p in self.b.distinct if set(p)<=used),
                "history": list(state.history), "budget": round(max(0.,remaining),3),
                "library_snapshot": self.library.fingerprint if self.library else "empty",
                "library": self.library.retrieve([state.tasks[i].goal for i in state.frontier],library_limit) if self.library else []}

    def key(self, state):
        record = self.observation(state, 0)
        record.pop("history")
        # Pending proof frames are significant: same leaves can have different
        # required continuations and scoped induction assumptions.
        record["frames"] = [(f.parent,f.children,f.kind,str(state.tasks[f.parent].goal)) for f in state.frames]
        return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()

    def task_theorem(self, task):
        return Theorem("macro_goal", [Hypothesis(f"f{i}",N("|-",p.formula)) for i,p in enumerate(task.facts)],
                       N("|-",task.goal), variable_types=dict(self.b.types), d_constraints=set(self.b.distinct))

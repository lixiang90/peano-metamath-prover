"""Bounded proof reconstruction and conventional induction for PA v2.1.

This is deliberately incomplete. A Vampire claim is accepted only when this
independent proof-producing reconstruction succeeds and Metamath checks it.
"""
from __future__ import annotations

import time
from .kernel import N, eq, imp, replace, Builder


class BudgetExpired(Exception):
    pass


class Reconstructor:
    def __init__(self, builder, deadline, *, max_nodes=3000, ranker=None):
        self.b = builder
        self.deadline = deadline
        self.max_nodes = max_nodes
        self.calls = 0
        self.ranker = ranker

    def tick(self):
        self.calls += 1
        if time.perf_counter() >= self.deadline or self.calls > self.max_nodes:
            raise BudgetExpired()

    def normalize(self, t, facts, active=None):
        self.tick()
        active = set() if active is None else set(active)
        if t in active:
            return self.b.refl(t)
        active.add(t)
        p = self.b.refl(t)
        if t.op in {"S", "+", "*"}:
            children = [self.normalize(c, facts, active) for c in t.args]
            if any(q.formula.args[0] != q.formula.args[1] for q in children):
                p = self.b.congr(t.op, children)
        current = p.formula.args[1]
        next_proof = None
        if current.op in {"+", "*"}:
            a, c = current.args
            if c == N("0"):
                rule = "pa_ax3" if current.op == "+" else "pa_ax5"
                next_proof = self.b.sym(self.b.arithmetic(rule, {"x": a}))
            elif c.op == "S":
                rule = "pa_ax4" if current.op == "+" else "pa_ax6"
                next_proof = self.b.sym(self.b.arithmetic(rule, {"x": a, "y": c.args[0]}))
        if next_proof is None:
            for f in facts:
                if f.formula.op != "=":
                    continue
                a, c = f.formula.args
                key = lambda v: (sum(1 for _ in v.walk()), str(v))
                if key(a) < key(c):
                    a, c = c, a
                    f = self.b.sym(f)
                if current == a and a != c:
                    next_proof = f
                    break
        if next_proof is not None:
            q = self.normalize(next_proof.formula.args[1], facts, active)
            p = self.b.trans(p, self.b.trans(next_proof, q))
        return p

    def solve(self, goal, facts=(), depth=0):
        self.tick()
        if depth > 20:
            return None
        for p in facts:
            if p.formula == goal:
                return p
        from .atp import DEFINITIONS
        label = "df-" + goal.op
        if label in DEFINITIONS:
            definition = self.b.db.logical_assertions[label]
            pattern = definition.conclusion.args[0].args[0]
            subst = {v.op: actual for v, actual in zip(pattern.args, goal.args)}
            bridge = self.b.ax(label, subst)
            p = self.solve(bridge.formula.args[1], facts, depth+1)
            return self.b.mp(p, self.b.iff_dir(bridge, 1)) if p else None
        if goal.op == "implies":
            a, c = goal.args
            p = self.solve(c, (*facts, self.b.hyp(a)), depth+1)
            return self.b.deduct(a, p) if p else None
        if goal.op == "and":
            a = self.solve(goal.args[0], facts, depth+1)
            c = self.solve(goal.args[1], facts, depth+1)
            return self.b.and_intro(a, c) if a and c else None
        if goal.op == "forall":
            x, body = goal.args
            if any(x in set(a.walk()) for p in facts for a in p.assumptions):
                fresh=self.b.fresh()
                bridge=self.b.rename_forall(goal,fresh)
                p=self.solve(bridge.formula.args[1],facts,depth+1)
                return self.b.mp(p,self.b.iff_dir(bridge,1)) if p else None
            p = self.solve(body, facts, depth+1)
            return self.b.forall_intro(x, p) if p else None
        expanded = list(facts)
        known = {f.formula for f in facts}
        terms = {n for f in [goal, *known] for n in f.walk()
                 if n.op in {"0", "S", "+", "*"} or self.b.types.get(n.op) in {"var", "term"}}
        # ATP equality resolution can discharge t=t without listing it as an
        # input premise. Reconstruct that evidence rather than requiring a
        # redundant reflexivity hypothesis to survive the ATP proof core.
        for t in sorted(terms, key=str)[:16]:
            reflexive = self.b.refl(t)
            if reflexive.formula not in known:
                expanded.append(reflexive)
                known.add(reflexive.formula)
        for _ in range(5):
            changed = False
            for p in list(expanded):
                self.tick()
                candidates = []
                if p.formula.op == "and":
                    candidates = [self.b.and_elim(p, i) for i in (0, 1)]
                elif p.formula.op == "implies" and p.formula.args[0] in known:
                    minor = next(q for q in expanded if q.formula == p.formula.args[0])
                    candidates = [self.b.mp(minor, p)]
                elif p.formula.op == "forall" and p.formula.args[1].op != "forall":
                    for t in sorted(terms, key=str)[:16]:
                        try:
                            universal=p
                            if p.formula.args[0] in set(t.walk()):
                                # Source $d forbids direct instantiation using
                                # the binder itself. Prove an alpha-renamed
                                # universal first; this is needed when a
                                # generalized IH is specialized at its old
                                # parameter name. No textual capture is used.
                                cache=getattr(self.b,"_renamed_quantifiers",None)
                                if cache is None:cache={};self.b._renamed_quantifiers=cache
                                if p not in cache:
                                    fresh=self.b.fresh()
                                    cache[p]=self.b.forall_intro(fresh,self.b.forall_elim(p,fresh))
                                universal=cache[p]
                            candidates.append(self.b.forall_elim(universal, t))
                        except ValueError:
                            pass
                for q in candidates:
                    if q.formula == goal:
                        return q
                    if q.formula not in known:
                        known.add(q.formula)
                        expanded.append(q)
                        changed = True
            if not changed:
                break
        if goal.op == "=":
            left = self.normalize(goal.args[0], expanded)
            right = self.normalize(goal.args[1], expanded)
            if left.formula.args[1] == right.formula.args[1]:
                return self.b.trans(left, self.b.sym(right))
            # Bidirectional transitive equality closure over existing facts.
            edges = [p for p in expanded if p.formula.op == "="]
            frontier = [(goal.args[0], self.b.refl(goal.args[0]))]
            seen = {goal.args[0]}
            for node, path in frontier:
                self.tick()
                for p in edges:
                    a, c = p.formula.args
                    if node == c:
                        a, c, p = c, a, self.b.sym(p)
                    if node != a or c in seen:
                        continue
                    path2 = self.b.trans(path, p)
                    if c == goal.args[1]:
                        return path2
                    seen.add(c)
                    frontier.append((c, path2))
        return None

    def induction(self, goal, facts=()):
        """One ordinary PA induction, with no conjectured strengthening."""
        self.tick()
        binders = []
        body = goal
        while body.op == "forall":
            x, body = body.args
            binders.append(x)
        if body.op not in {"=", "implies", "and"}:
            return None
        candidates = binders or [N(v) for v, typ in self.b.types.items()
                                 if typ == "var" and N(v) in set(body.walk())]
        if self.ranker:
            candidates = self.ranker(goal, candidates)
        for n in candidates:
            if any(n in set(a.walk()) for f in facts for a in f.assumptions):
                continue
            self.tick()
            # Fresh x,y are necessary for the special source induction schema.
            x, y = self.b.fresh(), self.b.fresh()
            phi = replace(body, n, y)
            base = self.solve(replace(body, n, N("0")), facts)
            if base is None:
                continue
            def guarded(value, proof):
                h = self.b.hyp(eq(y, value))
                equivalent = self.b.formula_subst(phi, y, value, h)
                result = self.b.mp(proof, self.b.iff_dir(equivalent, 1))
                return self.b.forall_intro(y, self.b.deduct(h.formula, result))
            base_guard = guarded(N("0"), base)
            # Use a different bound name in the working induction hypothesis;
            # PA's syntactic $d conditions are stronger than free-variable tests.
            z = self.b.fresh()
            ih_formula = N("forall", z, imp(eq(z, x), replace(phi, y, z)))
            ih = self.b.hyp(ih_formula)
            px = self.b.mp(self.b.refl(x), self.b.forall_elim(ih, x))
            step = self.solve(replace(body, n, N("S", x)), (*facts, px))
            if step is None:
                continue
            step_guard = guarded(N("S", x), step)
            step_imp = self.b.deduct(ih_formula, step_guard)
            schema_ih = N("forall", y, imp(eq(y, x), phi))
            renamed_ih = self.b.forall_intro(z, self.b.forall_elim(self.b.hyp(schema_ih), z))
            step_all = self.b.forall_intro(x, self.b.deduct(schema_ih, self.b.mp(renamed_ih, step_imp)))
            schema = self.b.ax("induction", {"x": x, "y": y, "phi": phi})
            all_guarded = self.b.mp(self.b.and_intro(base_guard, step_all), schema)
            # Nested universal substitution uses the source all_elim2 rule.
            at_n = self.b.forall_elim(all_guarded, n)
            result = self.b.mp(self.b.refl(n), self.b.forall_elim(at_n, n))
            for v in reversed(binders):
                result = self.b.forall_intro(v, result)
            return result
        return None

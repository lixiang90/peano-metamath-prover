"""Proof-producing logical plumbing, derived only from the PA source axioms.

Proof DAGs are exported as ordinary $p statements, never trusted $a macros.
Every dependency, including logical helper steps, is independently replayed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools

from metamath_generator.model import Node, Hypothesis, Theorem
from metamath_generator.parser import parse, MetamathParser
from metamath_generator.database import TheoremDatabase
from metamath_generator.export import _SyntaxCompiler, _ordered_application_proof
from metamath_generator.unification import substitute_simultaneous
from metamath_generator.verifier import verify


def N(op, *args):
    return Node(op, tuple(args))


def imp(a, b):
    return N("implies", a, b)


def neg(a):
    return N("not", a)


def eq(a, b):
    return N("=", a, b)


def replace(node, variable, term):
    if node == variable:
        return term
    if node.op in {"forall", "exists"} and node.args[0] == variable:
        return node
    return N(node.op, *(replace(c, variable, term) for c in node.args))


@dataclass(eq=False)
class Evidence:
    formula: Node
    rule: str
    subst: dict
    parents: tuple
    assumptions: frozenset


class Builder:
    def __init__(self, database, variable_types, d_constraints=()):
        self.db = database
        self.types = dict(variable_types)
        self.distinct = set(d_constraints)
        self._cache = {}
        self._deductions = {}
        self._hypotheses = {}
        self._floating = {v: (v, t) for v, t in self.types.items()}
        self._syntax = _SyntaxCompiler(TheoremDatabase(database), self._floating)
        self._syntax.compile = lru_cache(maxsize=65536)(self._syntax.compile)

    def fresh(self):
        i = 0
        while f"zzv{i}" in self.types or f"zzv{i}" in self.db.symbols:
            i += 1
        name = f"zzv{i}"
        self.distinct.update(tuple(sorted((name, old))) for old in self.types)
        self.types[name] = "var"
        self._floating[name] = (name, "var")
        return N(name)

    def check_type(self, formula, typecode="wff"):
        self._syntax.compile(typecode, formula)

    def hyp(self, formula):
        if formula not in self._hypotheses:
            self._hypotheses[formula] = Evidence(formula, "hyp", {}, (), frozenset({formula}))
        return self._hypotheses[formula]

    def ax(self, rule, subst, *parents):
        assertion = self.db.logical_assertions[rule]
        if assertion.kind == "theorem":
            raise ValueError("source human theorem libraries are not accepted")
        key = (rule, tuple(sorted(subst.items())), parents)
        if key in self._cache:
            return self._cache[key]
        if set(subst) != set(assertion.variable_types):
            raise ValueError(f"incomplete substitution for {rule}")
        for variable, typ in assertion.variable_types.items():
            self.check_type(subst[variable], typ)
        from metamath_generator.compose import _instantiate_constraints_simultaneous
        needed = _instantiate_constraints_simultaneous(assertion.d_constraints, subst, set(self.types))
        if not needed <= self.distinct:
            raise ValueError(f"missing distinct-variable conditions for {rule}")
        expected = tuple(substitute_simultaneous(h.expr, subst).args[0]
                         for h in assertion.hypotheses)
        if expected != tuple(p.formula for p in parents):
            raise ValueError(f"premise mismatch for {rule}")
        formula = substitute_simultaneous(assertion.conclusion, subst).args[0]
        if key not in self._cache:
            self._cache[key] = Evidence(formula, rule, subst, parents,
                                       frozenset().union(*(p.assumptions for p in parents)))
        return self._cache[key]

    def mp(self, premise, implication):
        if implication.formula.op != "implies" or implication.formula.args[0] != premise.formula:
            raise ValueError("modus ponens mismatch")
        return self.ax("ax-mp", {"phi": premise.formula,
                               "psi": implication.formula.args[1]}, premise, implication)

    def lift(self, p, a):
        return self.mp(p, self.ax("ax-1", {"phi": p.formula, "psi": a}))

    def identity(self, a):
        ab = imp(a, a)
        p = self.ax("ax-1", {"phi": a, "psi": ab})
        q = self.ax("ax-1", {"phi": a, "psi": a})
        return self.mp(q, self.mp(p, self.ax("ax-2", {"phi": a, "psi": ab, "chi": a})))

    def deduct(self, a, p):
        key = (a, p)
        if key in self._deductions:
            return self._deductions[key]
        if a not in p.assumptions:
            result = self.lift(p, a)
        elif p.rule == "hyp":
            result = self.identity(a)
        elif p.rule == "ax-mp":
            minor, major = p.parents
            first, second = self.deduct(a, minor), self.deduct(a, major)
            result = self.mp(first, self.mp(second, self.ax("ax-2", {
                "phi": a, "psi": minor.formula, "chi": p.formula})))
        elif p.rule == "alpha_2":
            s = p.subst
            if any(n == s["x"] for n in a.walk()):
                raise ValueError("cannot discharge an assumption containing the eigenvariable")
            parent = self.deduct(a, p.parents[0])
            both = N("and", a, s["phi"])
            hb = self.hyp(both)
            q = self.mp(self.and_elim(hb, 1), self.mp(self.and_elim(hb, 0), parent))
            lifted = self.ax("alpha_2", {"phi": both, "chi": s["chi"], "x": s["x"]},
                             self.deduct(both, q))
            joined = self.and_intro(self.hyp(a), self.hyp(s["phi"]))
            result = self.deduct(a, self.deduct(s["phi"], self.mp(joined, lifted)))
        else:
            raise ValueError(f"unsupported deduction through {p.rule}")
        self._deductions[key] = result
        return result

    def explosion(self, positive, negative, target):
        if negative.formula != neg(positive.formula):
            raise ValueError("not a contradiction")
        return self.mp(positive, self.mp(self.lift(negative, neg(target)),
            self.ax("ax-3", {"phi": target, "psi": positive.formula})))

    def raa(self, target, positive, negative):
        """Discharge not(target) using ax-3 and an explicitly proved identity."""
        k = imp(target, target)
        bad = self.explosion(positive, negative, neg(k))
        implication = self.deduct(neg(target), bad)
        return self.mp(self.identity(target), self.mp(implication,
            self.ax("ax-3", {"phi": target, "psi": k})))

    def dne(self, p):
        if p.formula.op != "not" or p.formula.args[0].op != "not":
            raise ValueError("double negation expected")
        a = p.formula.args[0].args[0]
        return self.raa(a, self.hyp(neg(a)), p)

    def and_intro(self, p, q):
        a, b = p.formula, q.formula
        c = imp(a, neg(b))
        hc = self.dne(self.hyp(neg(neg(c))))
        nc = self.raa(neg(c), q, self.mp(p, hc))
        definition = self.ax("df-an", {"phi": a, "psi": b})
        return self.mp(nc, self.iff_dir(definition, 1))

    def iff_dir(self, p, direction):
        a, b = p.formula.args
        return self.mp(p, self.ax("bi1" if direction == 0 else "bi2", {"phi": a, "psi": b}))

    def and_elim(self, p, index):
        a, b = p.formula.args
        nc = self.mp(p, self.iff_dir(self.ax("df-an", {"phi": a, "psi": b}), 0))
        target = (a, b)[index]
        hn = self.hyp(neg(target))
        if index == 0:
            nb = self.explosion(self.hyp(a), hn, neg(b))
            c = self.deduct(a, nb)
        else:
            c = self.lift(hn, a)
        return self.raa(target, c, nc)

    def iff_intro(self, ab, ba):
        a, b = ab.formula.args
        return self.mp(ba, self.mp(ab, self.ax("bi3", {"phi": a, "psi": b})))

    def refl(self, a):
        return self.ax("eq-refl", {"t": a})

    def sym(self, p):
        a, b = p.formula.args
        return self.mp(p, self.ax("eq-sym", {"s": a, "t": b}))

    def trans(self, p, q):
        a, b = p.formula.args
        if q.formula.args[0] != b:
            raise ValueError("equality chain mismatch")
        return self.mp(self.and_intro(p, q), self.ax("eq-trans", {
            "s": a, "t": b, "u": q.formula.args[1]}))

    def congr(self, op, proofs):
        if op == "S":
            a, b = proofs[0].formula.args
            return self.mp(proofs[0], self.ax("eq-suc", {"s": a, "t": b}))
        if op not in {"+", "*"} or len(proofs) != 2:
            raise ValueError("unsupported function congruence")
        a, b = proofs[0].formula.args
        c, d = proofs[1].formula.args
        return self.mp(self.and_intro(*proofs), self.ax("eq-binop", {
            "binop": N(op), "s0": a, "s1": b, "t0": c, "t1": d}))

    def term_subst(self, term, x, t, equality):
        if term == x:
            return equality
        if not any(n == x for n in term.walk()):
            return self.refl(term)
        return self.congr(term.op, tuple(self.term_subst(c, x, t, equality) for c in term.args))

    def formula_subst(self, formula, x, t, equality):
        """Quantifier-free equality/ordering/implication/conjunction equivalence."""
        changed = replace(formula, x, t)
        if formula == changed:
            return self.iff_intro(self.identity(formula), self.identity(formula))
        if formula.op in {"=", "<"}:
            p, q = [self.term_subst(c, x, t, equality) for c in formula.args]
            a, b = p.formula.args
            c, d = q.formula.args
            return self.mp(self.and_intro(p, q), self.ax("eq-congr", {
                "binpred": N(formula.op), "s0": a, "s1": b, "t0": c, "t1": d}))
        if formula.op == "forall":
            y, body = formula.args
            if y == x or y in set(t.walk()):
                raise ValueError("capture in quantified substitution")
            h = self.hyp(eq(x, t))
            inner = self.formula_subst(body, x, t, h)
            lifted = self.ax("all_elim2", {"x": x, "y": y, "t": t,
                "quant": N("forall"), "phi": body, "chi": replace(body, x, t)},
                self.deduct(h.formula, inner))
            return self.mp(equality, lifted)
        if formula.op not in {"implies", "and"}:
            raise ValueError("quantifier substitution outside supported fragment")
        parts = [self.formula_subst(c, x, t, equality) for c in formula.args]
        directions = []
        for reverse in (False, True):
            before, after = (changed, formula) if reverse else (formula, changed)
            h = self.hyp(before)
            if formula.op == "and":
                result = self.and_intro(*(self.mp(self.and_elim(h, i), self.iff_dir(parts[i], int(reverse)))
                                          for i in (0, 1)))
            else:
                ha = self.hyp(after.args[0])
                arg = self.mp(ha, self.iff_dir(parts[0], int(not reverse)))
                result = self.deduct(after.args[0], self.mp(self.mp(arg, h), self.iff_dir(parts[1], int(reverse))))
            directions.append(self.deduct(before, result))
        return self.iff_intro(*directions)

    def forall_elim(self, p, term):
        if p.formula.op != "forall":
            raise ValueError("universal formula expected")
        x, body = p.formula.args
        if any(n == x for n in term.walk()):
            raise ValueError("fresh binder required for instantiation")
        result = replace(body, x, term)
        equivalence = self.deduct(eq(x, term), self.formula_subst(body, x, term, self.hyp(eq(x, term))))
        first = self.mp(p, self.ax("all_elim", {"x": x, "t": term, "phi": body}))
        return self.mp(first, self.ax("all_elim3", {
            "x": x, "t": term, "phi": body, "chi": result}, equivalence))

    def forall_intro(self, x, p):
        if any(x in set(a.walk()) for a in p.assumptions):
            raise ValueError("eigenvariable occurs in an open assumption")
        truth = self.refl(N("0"))
        return self.mp(truth, self.ax("alpha_2", {
            "phi": truth.formula, "chi": p.formula, "x": x}, self.lift(p, truth.formula)))

    def rename_forall(self, formula, fresh):
        """Closed alpha-equivalence, safe even under assumptions containing x."""
        if formula.op != "forall": raise ValueError("universal formula expected")
        x,body=formula.args
        if fresh in set(formula.walk()):raise ValueError("alpha-renaming requires a fresh binder")
        h=self.hyp(eq(x,fresh))
        equivalent=self.deduct(h.formula,self.formula_subst(body,x,fresh,h))
        premise=self.forall_intro(x,self.forall_intro(fresh,equivalent))
        return self.mp(premise,self.ax("alpha_3",{"x":x,"y":fresh,"phi":body,
            "psi":replace(body,x,fresh),"quant":N("forall")}))

    def arithmetic(self, rule, subst):
        """Instantiate var-typed PA axioms via proved universal elimination."""
        ax = self.db.logical_assertions[rule]
        variables = [h.expr.args[0].op for h in ax.floating]
        if all(not subst[v].args and self.types.get(subst[v].op) == "var" for v in variables):
            return self.ax(rule, subst)
        fresh = {v: self.fresh() for v in variables}
        p = self.ax(rule, fresh)
        # Multiple quantifier elimination is intentionally avoided: generalize
        # and instantiate one independent binder at a time.
        for v in variables:
            p = self.forall_elim(self.forall_intro(fresh[v], p), subst[v])
        return p

    def export(self, root, target, destination):
        if root.formula != target.conclusion.args[0]:
            raise ValueError("proof conclusion differs from requested target")
        allowed = {h.expr.args[0] for h in target.hypotheses}
        if not root.assumptions <= allowed:
            raise ValueError("proof has undischarged assumptions")
        ordered, seen = [], set()
        def visit(p):
            if p in seen or p.rule == "hyp":
                return
            seen.add(p)
            for child in p.parents:
                visit(child)
            ordered.append(p)
        visit(root)
        indices = {p: i for i, p in enumerate(ordered)}
        store = TheoremDatabase(self.db)
        floating = {v: (f"v21_f{i}", typ) for i, (v, typ) in enumerate(sorted(self.types.items()))}
        compiler = _SyntaxCompiler(store, floating)
        # This compiler's type environment stays fixed for the whole export.
        # Repeated Hilbert helper formulas otherwise dominate ATP wall time.
        compiler.compile = lru_cache(maxsize=65536)(compiler.compile)
        lines = ["$( v2.1 generated certificate: no additional logical axioms. $)", "${"]
        # Reuse ambient $f declarations; fresh proof variables are local.
        ambient = {h.expr: label for label, h in self.db.active_floating_hypotheses.items()}
        for v, (label, typ) in list(floating.items()):
            existing = ambient.get(N(typ, N(v)))
            if existing:
                floating[v] = (existing, typ)
            else:
                if v not in self.db.variables:
                    lines.append(f"$v {v} $.")
                lines.append(f"{label} $f {typ} {v} $.")
        for a, b in sorted(self.distinct):
            lines.append(f"$d {a} {b} $.")
        deps = {p: sorted(p.assumptions, key=str) for p in ordered}
        def variables(p):
            return {n.op for f in [p.formula, *p.assumptions] for n in f.walk() if n.op in self.types}
        varsets = {p: variables(p) for p in ordered}
        vorder = [h.expr.args[0].op for h in self.db.active_floating_hypotheses.values()
                  if h.expr.args[0].op in self.types]
        vorder += [v for v in sorted(self.types) if v not in vorder]
        for p in ordered:
            i = indices[p]
            local = {f: f"v21_p{i}_h{j}" for j, f in enumerate(deps[p])}
            lines.append("${")
            for f, label in local.items():
                lines.append(f"{label} $e |- {f} $.")
            def parent_labels(q):
                if q.rule == "hyp":
                    return [local[q.formula]]
                # Metamath orders mandatory $f by declaration order, not name.
                return [floating[v][0] for v in vorder if v in varsets[q]] + [local[f] for f in deps[q]] + [f"v21_p{indices[q]}"]
            ax = self.db.logical_assertions[p.rule]
            fs = [compiler.compile(h.expr.op, p.subst[h.expr.args[0].op]) for h in ax.floating]
            labels = _ordered_application_proof(ax, fs, [parent_labels(q) for q in p.parents])
            lines.append(f"v21_p{i} $p |- {p.formula} $= {' '.join(labels)} $.")
            lines.append("$}")
        # Exact public target retains every original assumption and constraint.
        lines.append("${")
        hlabels = {}
        for i, h in enumerate(target.hypotheses):
            hlabels[h.expr.args[0]] = f"v21_target_h{i}"
            lines.append(f"v21_target_h{i} $e {h.expr} $.")
        if root.rule == "hyp":
            labels = [hlabels[root.formula]]
        else:
            labels = [floating[v][0] for v in vorder if v in varsets[root]]
            labels += [hlabels[f] for f in deps[root]] + [f"v21_p{indices[root]}"]
        lines += [f"v21_result $p {target.conclusion} $= {' '.join(labels)} $.", "$}", "$}"]
        Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return len(ordered)

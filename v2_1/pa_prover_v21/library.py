"""Frozen, proof-carrying lemma memory; archived formulas are never axioms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .kernel import N, imp
from .atp import sha256, verify_bundle


def node_record(n):
    return [n.op, *[node_record(c) for c in n.args]]


def node_read(row, rename=None):
    if not isinstance(row, list) or not row or not isinstance(row[0], str):
        raise ValueError("invalid expression record")
    if len(row) == 1 and rename and row[0] in rename: return rename[row[0]]
    return N(row[0], *(node_read(c, rename) for c in row[1:]))


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest(source):
    # Include recursively referenced files, not just the entry point.
    # Source folders in this project consist of .mm files. Hash relative paths
    # as well as contents; the parser independently resolves all includes.
    root = Path(source).resolve().parent
    return digest({"entry":Path(source).name,
                   "files":{p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.mm"))}})


def archive(builder, proof):
    rows, seen = [], {}
    def visit(p):
        if id(p) in seen: return seen[id(p)]
        parents = [visit(q) for q in p.parents]
        i = len(rows); seen[id(p)] = i
        rows.append({"rule": p.rule, "formula": node_record(p.formula),
                     "subst": {k: node_record(v) for k,v in p.subst.items()}, "parents": parents})
        return i
    root = visit(proof)
    return {"nodes": rows, "root": root, "types": dict(builder.types),
            "distinct": sorted(builder.distinct)}


def replay(builder, record, *, rename=False):
    mapping = {}
    if rename:
        if any(t not in {"var","term"} for t in record["types"].values()):
            raise ValueError("lemma memory currently supports object-variable parameters")
        mapping = {v: builder.fresh() for v in record["types"]}
    elif any(builder.types.get(v) != t for v,t in record["types"].items()):
        raise ValueError("archive variable types differ")
    if not rename and not set(map(tuple,record["distinct"])) <= builder.distinct:
        raise ValueError("archive distinct-variable context differs")
    proofs = []
    for row in record["nodes"]:
        f = node_read(row["formula"], mapping)
        if any(type(i) is not int or i < 0 or i >= len(proofs) for i in row["parents"]):
            raise ValueError("cyclic or future proof dependency")
        if row["rule"] == "hyp":
            if row["parents"] or row["subst"]: raise ValueError("invalid hypothesis record")
            builder.check_type(f); p = builder.hyp(f)
        else:
            p = builder.ax(row["rule"], {k:node_read(v,mapping) for k,v in row["subst"].items()},
                           *(proofs[i] for i in row["parents"]))
        if p.formula != f: raise ValueError("archive formula does not match replay")
        proofs.append(p)
    root = record["root"]
    if type(root) is not int or not 0 <= root < len(proofs): raise ValueError("invalid proof root")
    return proofs[root], mapping


class Library:
    """Immutable snapshots. Only explicit admission creates a new snapshot."""
    def __init__(self, entries=()):
        self.entries = {e["id"]: e for e in json.loads(json.dumps(list(entries)))}
        self.fingerprint = digest(self.entries)
        self._symbols={key:{n.op for n in node_read(e["formula"]).walk()} for key,e in self.entries.items()}
        self._index={}
        for key,symbols in self._symbols.items():
            for symbol in symbols:self._index.setdefault(symbol,set()).add(key)

    @classmethod
    def load(cls, path, database, source):
        from .kernel import Builder
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["source"] != source_digest(source): raise ValueError("lemma source changed")
        for e in data["entries"]:
            if e["id"] != digest(e["proof"]): raise ValueError("lemma archive checksum mismatch")
            r=e["proof"]; b=Builder(database,r["types"],map(tuple,r["distinct"]))
            p,_=replay(b,r)
            if p.assumptions or node_record(p.formula) != e["formula"]:
                raise ValueError("library lemma must replay as a closed theorem")
            if e.get("verification",{}).get("status") != "passed":
                raise ValueError("lemma lacks admission verification")
        return cls(data["entries"])

    def save(self, path, source):
        Path(path).write_text(json.dumps({"source":source_digest(source),"entries":list(self.entries.values())},indent=2),encoding="utf-8")

    def admit(self, builder, proof, target, source, metamath, directory, *, origin):
        # Conditional lemmas are stored as closed implication chains.
        for h in reversed(target.hypotheses): proof=builder.deduct(h.expr.args[0],proof)
        if proof.assumptions: raise ValueError("unclosed lemma assumptions")
        from metamath_generator.model import Theorem
        closed=Theorem("memory",[],N("|-",proof.formula),variable_types=builder.types.copy(),d_constraints=builder.distinct.copy())
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        builder.export(proof,closed,out/"fragment.mm")
        verification=verify_bundle(out/"fragment.mm",source,metamath,30)
        if verification["status"] != "passed": raise ValueError("lemma admission failed official verification")
        r=archive(builder,proof)
        e={"id":digest(r),"proof":r,"formula":node_record(proof.formula),"text":str(proof.formula),
           "origin":origin,"verification":verification}
        return Library([*self.entries.values(),e])

    def retrieve(self, goals, limit=8):
        symbols={n.op for g in goals for n in g.walk()}
        pool=set().union(*(self._index.get(s,set()) for s in symbols)) or set(self.entries)
        def score(key):return -len(symbols & self._symbols[key]),key
        return [{"id":key,"statement":self.entries[key]["text"]} for key in sorted(pool,key=score)[:limit]]

    def instantiate(self, identity, task, builder):
        e=self.entries[identity]
        # Match object parameters only. Quantifier-bearing statements can be
        # used exactly; arbitrary higher-order unification is not claimed.
        original=node_read(e["formula"]); matrix=original;quantifiers=[]
        while matrix.op=="forall":
            quantifiers.append(matrix.args[0]);matrix=matrix.args[1]
        tail=matrix
        while tail.op == "implies": tail=tail.args[1]
        bindings={}
        def match(a,b):
            if not a.args and e["proof"]["types"].get(a.op) in {"var","term"}:
                if a.op in bindings:return bindings[a.op]==b
                builder.check_type(b,"term");bindings[a.op]=b;return True
            return a.op==b.op and len(a.args)==len(b.args) and all(match(x,y) for x,y in zip(a.args,b.args))
        if not match(tail,task.goal): raise ValueError("lemma conclusion does not match goal")
        premise=matrix
        while premise.op=="implies":
            for fact in task.facts:
                saved=dict(bindings)
                if match(premise.args[0],fact.formula):break
                bindings.clear();bindings.update(saved)
            premise=premise.args[1]
        p,names=replay(builder,e["proof"],rename=True)
        for variable in quantifiers:
            p=builder.forall_elim(p,bindings.get(variable.op) or builder.fresh())
        # Universal closure followed by elimination supports compound terms
        # without illegally replacing a Metamath var-typed floating premise.
        for name,term in bindings.items():
            if names[name] in set(p.formula.walk()):
                p=builder.forall_elim(builder.forall_intro(names[name],p),term)
        premises=[];tail=p.formula
        while tail.op=="implies":premises.append(tail.args[0]);tail=tail.args[1]
        if tail!=task.goal:raise ValueError("lemma instantiation left unmatched binders")
        return p,premises

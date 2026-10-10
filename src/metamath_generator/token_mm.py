"""Grammar-free Metamath parsing and proof checking.

The stack/substitution design follows Raph Levien's mmverify.py (GPL),
https://us.metamath.org/downloads/mmverify.py. This is an independent
implementation, including declaration-ordered hypotheses, strict scope checks,
and compressed-proof replay without expanding saved subproofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import re
from typing import Iterable, Mapping

Expression = tuple[str, ...]
Disjoints = frozenset[tuple[str, str]]


class TokenMMError(ValueError):
    """Malformed database or invalid proof; never a sampling failure."""


def pair(x: str, y: str) -> tuple[str, str]:
    return tuple(sorted((x, y)))


def substitute(expr: Expression, subst: Mapping[str, Expression]) -> Expression:
    # Simultaneous string substitution, including the empty string. Variables
    # in replacement strings must NOT themselves be recursively substituted.
    return tuple(part for token in expr for part in subst.get(token, (token,)))


@dataclass(frozen=True)
class TokenHypothesis:
    label: str
    kind: str
    expression: Expression


@dataclass(frozen=True)
class TokenAssertion:
    label: str
    kind: str
    expression: Expression
    hypotheses: tuple[TokenHypothesis, ...]
    disjoints: Disjoints

    @property
    def floating(self) -> tuple[TokenHypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.kind == "$f")

    @property
    def essential(self) -> tuple[TokenHypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.kind == "$e")

    @property
    def variable_types(self) -> dict[str, str]:
        return {h.expression[1]: h.expression[0] for h in self.floating}


def make_assertion(label: str, kind: str, expr: Expression,
                   active: Iterable[TokenHypothesis], variables: set[str],
                   disjoints: Iterable[tuple[str, str]]) -> TokenAssertion:
    active = tuple(active)
    used = set(expr) & variables
    for h in active:
        if h.kind == "$e":
            used.update(set(h.expression) & variables)
    typed = {h.expression[1] for h in active if h.kind == "$f"}
    if used - typed:
        raise TokenMMError(f"{label}: variables without active $f: {sorted(used - typed)}")
    return TokenAssertion(
        label, kind, expr,
        tuple(h for h in active if h.kind == "$e" or h.expression[1] in used),
        frozenset((x, y) for x, y in disjoints if x in used and y in used),
    )


def required_disjoints(assertion: TokenAssertion, subst: Mapping[str, Expression],
                       variables: set[str]) -> Disjoints:
    result = set()
    for x, y in assertion.disjoints:
        xs, ys = set(subst[x]) & variables, set(subst[y]) & variables
        if xs & ys:
            raise TokenMMError(f"{assertion.label}: disjoint variables collapse")
        result.update(pair(a, b) for a, b in product(xs, ys))
    return frozenset(result)


def verify_tokens(assertion: TokenAssertion, proof: Expression,
                  assertions: Mapping[str, TokenAssertion],
                  active: Mapping[str, TokenHypothesis], variables: set[str],
                  disjoints: Iterable[tuple[str, str]]) -> None:
    """Replay a proof using only earlier assertions and currently active $f/$e."""
    stack: list[Expression] = []
    allowed_d = set(disjoints)

    def step(label: str) -> None:
        if label in active:
            stack.append(active[label].expression)
            return
        rule = assertions.get(label)
        if rule is None:
            raise TokenMMError(f"{assertion.label}: unknown or inactive proof label {label!r}")
        size = len(rule.hypotheses)
        if len(stack) < size:
            raise TokenMMError(f"{assertion.label}: stack underflow at {label}")
        actual = stack[-size:] if size else []
        subst = {}
        # All floating entries are collected first, even when $f follows $e.
        for h, value in zip(rule.hypotheses, actual):
            if h.kind == "$f":
                if not value or value[0] != h.expression[0]:
                    raise TokenMMError(f"{assertion.label}: type mismatch at {label}")
                subst[h.expression[1]] = value[1:]
        for h, value in zip(rule.hypotheses, actual):
            if h.kind == "$e" and substitute(h.expression, subst) != value:
                raise TokenMMError(f"{assertion.label}: essential mismatch at {label}")
        if not required_disjoints(rule, subst, variables) <= allowed_d:
            raise TokenMMError(f"{assertion.label}: missing $d at {label}")
        if size:
            del stack[-size:]
        stack.append(substitute(rule.expression, subst))

    if not proof:
        raise TokenMMError(f"{assertion.label}: empty proof")
    if proof[0] != "(":
        for label in proof:
            step(label)
    else:
        try:
            end = proof.index(")")
        except ValueError as exc:
            raise TokenMMError(f"{assertion.label}: unterminated compressed label list") from exc
        labels = [h.label for h in assertion.hypotheses] + list(proof[1:end])
        for label in labels:
            if label not in assertions and label not in active:
                raise TokenMMError(f"{assertion.label}: unknown compressed label {label!r}")
        saved: list[Expression] = []
        accumulator = 0
        for char in "".join(proof[end + 1:]):
            if "U" <= char <= "Y":
                accumulator = accumulator * 5 + ord(char) - ord("U") + 1
            elif "A" <= char <= "T":
                number = accumulator * 20 + ord(char) - ord("A")
                accumulator = 0
                if number < len(labels):
                    step(labels[number])
                elif number - len(labels) < len(saved):
                    stack.append(saved[number - len(labels)])
                else:
                    raise TokenMMError(f"{assertion.label}: invalid saved-subproof index")
            elif char == "Z" and not accumulator and stack:
                saved.append(stack[-1])
            else:
                raise TokenMMError(f"{assertion.label}: invalid/incomplete compressed proof {char!r}")
        if accumulator:
            raise TokenMMError(f"{assertion.label}: unfinished compressed proof number")
    if stack != [assertion.expression]:
        raise TokenMMError(f"{assertion.label}: final proof stack does not match assertion")


def _uncomment(tokens: list[str]) -> list[str]:
    result = []
    comment = False
    for token in tokens:
        if token == "$(" and not comment:
            comment = True
        elif token == "$)" and comment:
            comment = False
        elif token in {"$(", "$)"}:
            raise TokenMMError("nested comment or unmatched $)")
        elif not comment:
            result.append(token)
    if comment:
        raise TokenMMError("unterminated comment")
    return result


def _read_includes(path: Path, seen: set[Path]) -> list[str]:
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    raw = path.read_text(encoding="utf-8-sig").split()
    _uncomment(raw)  # Comments cannot cross file boundaries.
    result: list[str] = []
    i, comment = 0, False
    while i < len(raw):
        token = raw[i]
        if token == "$(":
            comment = True
        elif token == "$)":
            comment = False
        if token == "$[" and not comment:
            if i + 2 >= len(raw) or raw[i + 2] != "$]" or "$" in raw[i + 1]:
                raise TokenMMError(f"{path}: malformed include")
            # The spec leaves directory conventions unspecified. Use the
            # including file's directory and export a flattened snapshot.
            result.extend(_read_includes(path.parent / raw[i + 1], seen))
            i += 3
        else:
            result.append(token)
            i += 1
    return result


class TokenDatabase:
    """No AST, distinguished judgment, parser grammar, or trusted $p shortcuts."""

    def __init__(self) -> None:
        self.assertions: dict[str, TokenAssertion] = {}
        self.constants: set[str] = set()
        self.variables: set[str] = set()
        self.labels: set[str] = set()
        self.source_tokens: list[str] = []
        self.source_path = "<string>"
        self.verified_proofs = 0

    @classmethod
    def from_file(cls, path: str | Path) -> TokenDatabase:
        db = cls()
        db.source_path = str(Path(path).resolve())
        db.source_tokens = _read_includes(Path(path), set())
        db._parse()
        return db

    @classmethod
    def from_text(cls, source: str) -> TokenDatabase:
        db = cls()
        db.source_tokens = source.split()
        db._parse()
        return db

    def _parse(self) -> None:
        tokens = _uncomment(self.source_tokens)
        active_v: set[str] = set()
        active: dict[str, TokenHypothesis] = {}
        active_d: set[tuple[str, str]] = set()
        scopes = []
        i = 0

        def expression(body: Expression) -> None:
            if not body or body[0] not in self.constants:
                raise TokenMMError("expression must start with a declared constant")
            if set(body) - self.constants - active_v:
                raise TokenMMError(f"inactive/undeclared symbol in {' '.join(body)}")

        while i < len(tokens):
            token = tokens[i]
            i += 1
            if token == "${":
                scopes.append((set(active_v), dict(active), set(active_d)))
                continue
            if token == "$}":
                if not scopes:
                    raise TokenMMError("unmatched $}")
                active_v, active, active_d = scopes.pop()
                continue
            label = None
            if not token.startswith("$"):
                label = token
                if (not re.fullmatch(r"[A-Za-z0-9_.-]+", label)
                        or label in self.labels or label in self.constants or label in self.variables):
                    raise TokenMMError(f"invalid/duplicate label {label!r}")
                if i == len(tokens):
                    raise TokenMMError(f"missing statement after {label}")
                token = tokens[i]
                i += 1
                if token not in {"$f", "$e", "$a", "$p"}:
                    raise TokenMMError(f"invalid labeled statement {token}")
                self.labels.add(label)
            elif token not in {"$c", "$v", "$d"}:
                raise TokenMMError(f"unexpected or unlabeled statement {token}")
            try:
                end = tokens.index("$.", i)
            except ValueError as exc:
                raise TokenMMError("missing $.") from exc
            body = tuple(tokens[i:end])
            i = end + 1
            if token in {"$c", "$v"}:
                if not body or len(set(body)) != len(body):
                    raise TokenMMError("empty/repeated symbol declaration")
                for symbol in body:
                    if ("$" in symbol or symbol in self.labels or symbol in self.constants or symbol in active_v
                            or (token == "$c" and symbol in self.variables)):
                        raise TokenMMError(f"invalid/redeclared symbol {symbol!r}")
                if token == "$c":
                    if scopes:
                        raise TokenMMError("$c is only legal in outermost scope")
                    self.constants.update(body)
                else:
                    active_v.update(body)
                    self.variables.update(body)
            elif token == "$d":
                if len(body) < 2 or len(set(body)) != len(body) or set(body) - active_v:
                    raise TokenMMError("invalid $d declaration")
                active_d.update(pair(x, y) for x, y in combinations(body, 2))
            elif token in {"$f", "$e"}:
                expression(body)
                if token == "$f":
                    if len(body) != 2 or body[1] not in active_v:
                        raise TokenMMError("$f must contain a constant and an active variable")
                    if any(h.kind == "$f" and h.expression[1] == body[1] for h in active.values()):
                        raise TokenMMError("variable already has an active $f")
                else:
                    make_assertion(label, token, body, active.values(), active_v, active_d)
                active[label] = TokenHypothesis(label, token, body)
            else:
                proof = ()
                if token == "$p":
                    if body.count("$=") != 1:
                        raise TokenMMError(f"{label}: $p needs exactly one $=")
                    split = body.index("$=")
                    body, proof = body[:split], body[split + 1:]
                expression(body)
                assertion = make_assertion(label, token, body, active.values(), active_v, active_d)
                if token == "$p":
                    verify_tokens(assertion, proof, self.assertions, active, active_v, active_d)
                    self.verified_proofs += 1
                self.assertions[label] = assertion
        if scopes:
            raise TokenMMError("unterminated ${ block")

    def isolated_source(self) -> str:
        """Preserve comments; close source hypotheses before new declarations.

        Constants must remain outermost, so hoist only $c declarations and wrap
        the remaining source. Assertions remain globally accessible in Metamath.
        """
        body = []
        comment, declaration = False, False
        for token in self.source_tokens:
            if token == "$(":
                comment = True
            if comment:
                body.append(token)
                if token == "$)":
                    comment = False
            elif token == "$c":
                declaration = True
            elif declaration:
                if token == "$.":
                    declaration = False
            else:
                body.append(token)
        constants = "$c " + " ".join(sorted(self.constants)) + " $.\n" if self.constants else ""
        # Line breaks avoid enormous lines in external verifier diagnostics.
        text = " ".join(body).replace(" $.", " $.\n")
        return constants + "${\n" + text + "\n$}\n"

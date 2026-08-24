from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .model import (
    Database,
    Hypothesis,
    Node,
    Proof,
    SyntaxRule,
    Theorem,
    distinct_pairs,
)


class ParseError(ValueError):
    pass


def _lex(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text.startswith("$(", i):
            end = text.find("$)", i + 2)
            if end < 0:
                raise ParseError("unterminated Metamath comment")
            i = end + 2
            continue
        start = i
        while i < len(text) and not text[i].isspace():
            if text.startswith("$(", i):
                break
            i += 1
        if start == i:
            continue
        tokens.append(text[start:i])
    return tokens


@dataclass
class _Scope:
    floating_len: int
    essential_len: int
    d_constraints: set[tuple[str, str]]


class MetamathParser:
    def __init__(self) -> None:
        self.database = Database()
        self._floating: list[tuple[str, str, str]] = []
        self._essential: list[tuple[str, tuple[str, ...], Node]] = []
        self._d_constraints: set[tuple[str, str]] = set()
        self._scopes: list[_Scope] = []
        self._declaration_index = 0

    def _declare(self, label: str) -> int:
        index = self._declaration_index
        self._declaration_index += 1
        self.database.label_order[label] = index
        return index

    def parse_file(self, path: str | Path) -> Database:
        source = Path(path)
        self.database.source_path = str(source.resolve())
        tokens = self._tokens_with_includes(source.resolve(), set())
        self._parse_tokens(list(tokens))
        self._record_active_hypotheses()
        return self.database

    def parse_text(self, text: str, source_name: str = "<string>") -> Database:
        self.database.source_path = source_name
        self._parse_tokens(_lex(text))
        self._record_active_hypotheses()
        return self.database

    def _record_active_hypotheses(self) -> None:
        self.database.active_floating_hypotheses = {
            label: Hypothesis(label, Node(typecode, (Node(variable),)))
            for label, typecode, variable in self._floating
        }

    def _tokens_with_includes(
        self,
        path: Path,
        active: set[Path],
    ) -> Iterator[str]:
        if path in active:
            raise ParseError(f"recursive include: {path}")
        active.add(path)
        raw = _lex(path.read_text(encoding="utf-8-sig"))
        i = 0
        while i < len(raw):
            if raw[i] == "$[":
                try:
                    end = raw.index("$]", i + 1)
                except ValueError as exc:
                    raise ParseError(f"unterminated include in {path}") from exc
                if end != i + 2:
                    raise ParseError("an include must contain exactly one filename")
                include = (path.parent / raw[i + 1]).resolve()
                yield from self._tokens_with_includes(include, active)
                i = end + 1
            else:
                yield raw[i]
                i += 1
        active.remove(path)

    @staticmethod
    def _until(tokens: list[str], index: int, marker: str = "$.") -> tuple[list[str], int]:
        try:
            end = tokens.index(marker, index)
        except ValueError as exc:
            raise ParseError(f"missing {marker}") from exc
        return tokens[index:end], end + 1

    def _parse_tokens(self, tokens: list[str]) -> None:
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "${":
                self._scopes.append(
                    _Scope(
                        len(self._floating),
                        len(self._essential),
                        set(self._d_constraints),
                    )
                )
                i += 1
                continue
            if token == "$}":
                if not self._scopes:
                    raise ParseError("unmatched $}")
                scope = self._scopes.pop()
                del self._floating[scope.floating_len:]
                del self._essential[scope.essential_len:]
                self._d_constraints = scope.d_constraints
                i += 1
                continue
            if token in {"$c", "$v", "$d"}:
                body, i = self._until(tokens, i + 1)
                if token == "$c":
                    self.database.symbols.update(body)
                elif token == "$v":
                    self.database.variables.update(body)
                else:
                    unknown = set(body) - self.database.variables
                    if unknown:
                        raise ParseError(f"$d contains undeclared variables: {unknown}")
                    declaration = distinct_pairs(body)
                    self.database.d_declarations.append(declaration)
                    self._d_constraints.update(declaration)
                continue
            if token.startswith("$"):
                raise ParseError(f"unexpected token {token!r}")

            label = token
            if (
                label in self.database.statements
                or label in self.database.floating_hypotheses
                or label in self.database.essential_hypotheses
            ):
                raise ParseError(f"duplicate label {label!r}")
            if i + 1 >= len(tokens):
                raise ParseError(f"missing statement kind after {label!r}")
            kind = tokens[i + 1]
            if kind == "$f":
                body, i = self._until(tokens, i + 2)
                if len(body) != 2:
                    raise ParseError(f"$f {label} must contain a type and variable")
                typecode, variable = body
                if variable not in self.database.variables:
                    raise ParseError(f"undeclared variable {variable!r}")
                self.database.types.add(typecode)
                self.database.variable_types[variable] = typecode
                self._floating.append((label, typecode, variable))
                self._declare(label)
                self.database.floating_hypotheses[label] = Hypothesis(
                    label, Node(typecode, (Node(variable),))
                )
                continue
            if kind == "$e":
                body, i = self._until(tokens, i + 2)
                expr = self.parse_expression(body)
                self._essential.append((label, tuple(body), expr))
                self._declare(label)
                self.database.essential_hypotheses[label] = Hypothesis(label, expr)
                continue
            if kind not in {"$a", "$p"}:
                raise ParseError(f"unknown statement kind {kind!r} for {label}")

            if kind == "$a":
                body, i = self._until(tokens, i + 2)
                proof_tokens: tuple[str, ...] = ()
            else:
                try:
                    equal = tokens.index("$=", i + 2)
                except ValueError as exc:
                    raise ParseError(f"$p {label} has no $=") from exc
                body = tokens[i + 2:equal]
                proof_body, i = self._until(tokens, equal + 1)
                proof_tokens = tuple(proof_body)

            if not body:
                raise ParseError(f"empty assertion {label!r}")
            output_type = body[0]
            if output_type != "|-":
                floating_map = {var: typ for _, typ, var in self._floating}
                syntax = SyntaxRule(
                    label,
                    output_type,
                    tuple(body[1:]),
                    {var: floating_map[var] for var in body[1:] if var in floating_map},
                )
                self.database.add_syntax_rule(syntax)

            conclusion = self.parse_expression(body)
            declaration_index = self._declare(label)
            theorem = self._make_theorem(
                label,
                kind,
                tuple(body),
                conclusion,
                proof_tokens,
                declaration_index,
            )
            self.database.statements[label] = theorem
            if output_type != "|-":
                theorem.kind = "syntax"
                self.database.syntax_statements[label] = theorem
            else:
                self.database.logical_assertions[label] = theorem
                if theorem.hypotheses:
                    self.database.rules[label] = theorem
                if kind == "$p":
                    self.database.proved_theorems[label] = theorem
                else:
                    self.database.axioms[label] = theorem

        if self._scopes:
            raise ParseError("unterminated ${ block")

    def _make_theorem(
        self,
        label: str,
        kind: str,
        source_tokens: tuple[str, ...],
        conclusion: Node,
        proof_tokens: tuple[str, ...],
        declaration_index: int,
    ) -> Theorem:
        active_f = {var: (flabel, typ) for flabel, typ, var in self._floating}
        used = self._variables_in_tokens(source_tokens)
        for _, raw, _ in self._essential:
            used.update(self._variables_in_tokens(raw))
        missing = used - set(active_f)
        if missing:
            raise ParseError(f"{label}: variables without active $f: {sorted(missing)}")

        floating = tuple(
            Hypothesis(flabel, Node(typ, (Node(var),)))
            for flabel, typ, var in self._floating
            if var in used
        )
        hypotheses = [
            Hypothesis(elabel, expr)
            for elabel, _, expr in self._essential
        ]
        d_constraints = {
            pair for pair in self._d_constraints if pair[0] in used and pair[1] in used
        }
        proof = None
        if kind == "$p":
            proof = Proof(label, source_labels=proof_tokens)
        return Theorem(
            name=label,
            hypotheses=hypotheses,
            conclusion=conclusion,
            d_constraints=d_constraints,
            proof=proof,
            variable_types={var: active_f[var][1] for var in used},
            proof_variable_types=(
                {var: typ for var, (_, typ) in active_f.items()}
                if kind == "$p" else {}
            ),
            proof_d_constraints=(
                set(self._d_constraints) if kind == "$p" else set()
            ),
            floating=floating,
            kind="theorem" if kind == "$p" else "axiom",
            source_tokens=source_tokens,
            declaration_index=declaration_index,
            active_hypothesis_labels=frozenset([
                *(flabel for flabel, _, _ in self._floating),
                *(elabel for elabel, _, _ in self._essential),
            ]),
        )

    def _variables_in_tokens(self, tokens: Iterable[str]) -> set[str]:
        return set(tokens) & self.database.variables

    def parse_expression(self, tokens: Iterable[str]) -> Node:
        raw = tuple(tokens)
        if not raw:
            raise ParseError("empty expression")
        typecode, body = raw[0], raw[1:]
        if not body:
            return Node(typecode)

        # In peano.mm ``|-`` is the provability judgment and its payload must
        # be a formula.  Trying term/var as fallbacks would accept malformed
        # assertions such as ``|- 0``.
        expected_types = ["wff"] if typecode == "|-" else [typecode]
        errors: list[str] = []
        for expected in expected_types:
            matches = self._parse_type(expected, body, 0, {})
            complete = [node for node, end in matches if end == len(body)]
            if complete:
                return Node(typecode, (complete[0],))
            errors.append(expected)
        raise ParseError(
            f"cannot parse expression {' '.join(raw)!r}; tried {', '.join(errors)}"
        )

    def _parse_type(
        self,
        expected: str,
        tokens: tuple[str, ...],
        pos: int,
        memo: dict[tuple[str, int], list[tuple[Node, int]]],
    ) -> list[tuple[Node, int]]:
        key = (expected, pos)
        if key in memo:
            return memo[key]
        # Mark in progress to stop accidental left-recursive syntax rules.
        memo[key] = []
        results: list[tuple[Node, int]] = []
        if pos < len(tokens):
            token = tokens[pos]
            if self.database.variable_types.get(token) == expected:
                results.append((Node(token), pos + 1))

        for rule in self.database.syntax_rules.get(expected, ()):
            states: list[tuple[list[tuple[str, Node | str]], int]] = [([], pos)]
            for part in rule.pattern:
                next_states: list[tuple[list[tuple[str, Node | str]], int]] = []
                variable_type = rule.variable_types.get(part)
                for pieces, cursor in states:
                    if variable_type is not None:
                        for child, end in self._parse_type(
                            variable_type, tokens, cursor, memo
                        ):
                            next_states.append((pieces + [("var", child)], end))
                    elif cursor < len(tokens) and tokens[cursor] == part:
                        next_states.append((pieces + [("const", part)], cursor + 1))
                states = next_states
                if not states:
                    break
            for pieces, end in states:
                results.append((self._build_node(rule, pieces), end))
        # Preserve grammar order but remove identical parses.
        memo[key] = list(dict.fromkeys(results))
        return memo[key]

    @staticmethod
    def _build_node(
        rule: SyntaxRule,
        pieces: list[tuple[str, Node | str]],
    ) -> Node:
        children = [value for kind, value in pieces if kind == "var"]
        constants = [value for kind, value in pieces if kind == "const"]
        if len(pieces) == 1:
            kind, value = pieces[0]
            return value if kind == "var" else Node(str(value))
        first_kind, first_value = pieces[0]
        if first_kind == "const":
            return Node(str(first_value), tuple(v for v in children if isinstance(v, Node)))
        if (
            isinstance(first_value, Node)
            and not first_value.args
            and children
            and children[0] is first_value
        ):
            return Node(first_value.op, tuple(v for v in children[1:] if isinstance(v, Node)))
        fallback = [
            value if isinstance(value, Node) else Node(value)
            for _, value in pieces
        ]
        return Node(rule.label, tuple(fallback))


def parse(path: str | Path) -> Database:
    return MetamathParser().parse_file(path)

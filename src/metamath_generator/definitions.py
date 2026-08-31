from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .model import Database, Node, Theorem, normalized_pair
from .parser import parse


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_PARAMETER = re.compile(r"[a-r]\Z")
_BINDER = re.compile(r"x(?:[0-9]|1[0-5])\Z")
_QUANTIFIERS = frozenset({"forall", "exists"})


class DefinitionCatalogError(ValueError):
    """Raised when a declarative PA+ catalog is malformed or non-conservative."""


@dataclass(frozen=True, slots=True)
class PredicateDefinition:
    name: str
    parameters: tuple[str, ...]
    body: str
    area: str
    summary: str
    theorem_families: tuple[str, ...] = ()
    allow_unused_parameters: tuple[str, ...] = ()

    @property
    def binders(self) -> tuple[str, ...]:
        tokens = self.body.split()
        result: list[str] = []
        for index, token in enumerate(tokens):
            if token not in _QUANTIFIERS:
                continue
            if index + 1 >= len(tokens):
                raise DefinitionCatalogError(
                    f"{self.name}: quantifier has no bound variable"
                )
            binder = tokens[index + 1]
            if not _BINDER.fullmatch(binder):
                raise DefinitionCatalogError(
                    f"{self.name}: invalid bound variable {binder!r}"
                )
            if binder not in result:
                result.append(binder)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class StatementTemplate:
    name: str
    body: str
    area: str
    summary: str
    requires: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DefinitionCatalog:
    format: str
    include: str
    definitions: tuple[PredicateDefinition, ...]
    statements: tuple[StatementTemplate, ...]

    @property
    def definition_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)

    @property
    def statement_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.statements)

    @property
    def allowed_unused_parameters(self) -> dict[str, tuple[str, ...]]:
        return {
            item.name: item.allow_unused_parameters
            for item in self.definitions
            if item.allow_unused_parameters
        }


@dataclass(frozen=True, slots=True)
class AuditedDefinition:
    name: str
    label: str
    arity: int
    dependencies: tuple[str, ...]
    body_nodes: int
    body_depth: int


@dataclass(frozen=True, slots=True)
class DefinitionAuditReport:
    definitions: tuple[AuditedDefinition, ...]
    statements: tuple[str, ...]

    @property
    def definition_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)


def _read_string(item: Mapping[str, object], key: str, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DefinitionCatalogError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _read_string_list(
    item: Mapping[str, object], key: str, context: str
) -> tuple[str, ...]:
    value = item.get(key, [])
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        raise DefinitionCatalogError(f"{context}: {key} must be a string list")
    return tuple(part.strip() for part in value if part.strip())


def load_definition_catalog(path: str | Path) -> DefinitionCatalog:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DefinitionCatalogError("catalog root must be an object")
    format_name = _read_string(payload, "format", "catalog")
    if format_name != "pa-plus-definition-catalog-v1":
        raise DefinitionCatalogError(f"unsupported catalog format {format_name!r}")
    include = _read_string(payload, "include", "catalog")

    raw_definitions = payload.get("definitions")
    if not isinstance(raw_definitions, list):
        raise DefinitionCatalogError("catalog: definitions must be a list")
    definitions: list[PredicateDefinition] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_definitions):
        context = f"definitions[{index}]"
        if not isinstance(raw, dict):
            raise DefinitionCatalogError(f"{context} must be an object")
        name = _read_string(raw, "name", context)
        if not _NAME.fullmatch(name):
            raise DefinitionCatalogError(f"{context}: invalid name {name!r}")
        if name in seen:
            raise DefinitionCatalogError(f"duplicate definition {name!r}")
        seen.add(name)
        parameters = _read_string_list(raw, "parameters", context)
        if not parameters or len(set(parameters)) != len(parameters):
            raise DefinitionCatalogError(
                f"{context}: parameters must be non-empty and unique"
            )
        invalid = [name for name in parameters if not _PARAMETER.fullmatch(name)]
        if invalid:
            raise DefinitionCatalogError(
                f"{context}: invalid parameters {invalid}; use a through r"
            )
        allow_unused = _read_string_list(
            raw, "allow_unused_parameters", context
        )
        invalid_unused = set(allow_unused) - set(parameters)
        if invalid_unused:
            raise DefinitionCatalogError(
                f"{context}: unused-parameter whitelist is not a parameter: "
                f"{sorted(invalid_unused)}"
            )
        definition = PredicateDefinition(
            name=name,
            parameters=parameters,
            body=_read_string(raw, "body", context),
            area=_read_string(raw, "area", context),
            summary=_read_string(raw, "summary", context),
            theorem_families=_read_string_list(raw, "theorem_families", context),
            allow_unused_parameters=allow_unused,
        )
        overlap = set(definition.binders) & set(parameters)
        if overlap:
            raise DefinitionCatalogError(
                f"{context}: bound variables overlap parameters: {sorted(overlap)}"
            )
        definitions.append(definition)

    raw_statements = payload.get("statements", [])
    if not isinstance(raw_statements, list):
        raise DefinitionCatalogError("catalog: statements must be a list")
    statements: list[StatementTemplate] = []
    statement_names: set[str] = set()
    for index, raw in enumerate(raw_statements):
        context = f"statements[{index}]"
        if not isinstance(raw, dict):
            raise DefinitionCatalogError(f"{context} must be an object")
        name = _read_string(raw, "name", context)
        if not _NAME.fullmatch(name) or not name.endswith("-statement"):
            raise DefinitionCatalogError(
                f"{context}: statement name must end in '-statement'"
            )
        if name in statement_names:
            raise DefinitionCatalogError(f"duplicate statement {name!r}")
        statement_names.add(name)
        requires = _read_string_list(raw, "requires", context)
        unknown = set(requires) - seen
        if unknown:
            raise DefinitionCatalogError(
                f"{context}: unknown required definitions {sorted(unknown)}"
            )
        statements.append(
            StatementTemplate(
                name=name,
                body=_read_string(raw, "body", context),
                area=_read_string(raw, "area", context),
                summary=_read_string(raw, "summary", context),
                requires=requires,
            )
        )
    return DefinitionCatalog(
        format=format_name,
        include=include,
        definitions=tuple(definitions),
        statements=tuple(statements),
    )


def _wrap_constants(names: Sequence[str], width: int = 88) -> list[str]:
    lines: list[str] = []
    current = "$c"
    for name in names:
        candidate = f"{current} {name}"
        if len(candidate) > width and current != "$c":
            lines.append(f"{current} $.")
            current = f"$c {name}"
        else:
            current = candidate
    lines.append(f"{current} $.")
    return lines


def render_definition_catalog(catalog: DefinitionCatalog) -> str:
    """Compile a compact JSON catalog into a replayable Metamath extension.

    Freshness declarations are derived mechanically from quantified variables.
    The resulting file must still pass :func:`audit_conservative_extension`;
    rendering alone is not a soundness check.
    """

    lines = [
        "$(",
        "  GENERATED FILE: edit formal/pa-plus-definitions.json and regenerate.",
        "  Every logical assertion below is an acyclic explicit definition.",
        "  Named theorem formulas have type statement and assert no theorem.",
        "$)",
        "",
        f"$[ {catalog.include} $]",
        "",
        "$( Fresh high-level predicate syntax. $)",
    ]
    lines.extend(_wrap_constants(catalog.definition_names))
    for item in catalog.definitions:
        lines.append(
            f"wff_{item.name} $a wff {item.name} {' '.join(item.parameters)} $."
        )

    lines.extend(["", "$( Conservative explicit definitions. $)"])
    for item in catalog.definitions:
        binders = item.binders
        if binders:
            lines.extend(["", "${"])
            for binder in binders:
                for parameter in item.parameters:
                    lines.append(f"  $d {binder} {parameter} $.")
            for left, right in combinations(binders, 2):
                lines.append(f"  $d {left} {right} $.")
            lines.append(
                f"  df-{item.name} $a |- iff {item.name} "
                f"{' '.join(item.parameters)} {item.body} $."
            )
            lines.append("$}")
        else:
            lines.extend(
                [
                    "",
                    f"df-{item.name} $a |- iff {item.name} "
                    f"{' '.join(item.parameters)} {item.body} $.",
                ]
            )

    if catalog.statements:
        lines.extend(
            [
                "",
                "$( Closed target formulas only; these are not logical assertions. $)",
            ]
        )
        for item in catalog.statements:
            lines.extend(
                [
                    "",
                    f"$( {item.summary} $)",
                    f"{item.name} $a statement {item.body} $.",
                ]
            )
    return "\n".join(lines) + "\n"


def _quantified_variables(node: Node) -> set[str]:
    return {
        item.args[0].op
        for item in node.walk()
        if item.op in _QUANTIFIERS and len(item.args) == 2
    }


def _free_variables(
    node: Node,
    known_variables: frozenset[str],
    bound: frozenset[str] = frozenset(),
) -> set[str]:
    if node.op in _QUANTIFIERS and len(node.args) == 2:
        variable = node.args[0].op
        return _free_variables(
            node.args[1], known_variables, bound | frozenset({variable})
        )
    result: set[str] = set()
    if not node.args and node.op in known_variables and node.op not in bound:
        result.add(node.op)
    for child in node.args:
        result.update(_free_variables(child, known_variables, bound))
    return result


def _definition_body(theorem: Theorem) -> tuple[Node, Node]:
    if theorem.hypotheses:
        raise DefinitionCatalogError(f"{theorem.name}: definitions cannot have hypotheses")
    conclusion = theorem.conclusion
    if conclusion.op != "|-" or len(conclusion.args) != 1:
        raise DefinitionCatalogError(f"{theorem.name}: definition must assert a formula")
    body = conclusion.args[0]
    if body.op != "iff" or len(body.args) != 2:
        raise DefinitionCatalogError(f"{theorem.name}: definition must have iff at root")
    return body.args


def audit_conservative_extension(
    base: Database,
    extension: Database,
    *,
    expected_definitions: Iterable[str] | None = None,
    expected_statements: Iterable[str] | None = None,
    allowed_unused_parameters: Mapping[str, Iterable[str]] | None = None,
) -> DefinitionAuditReport:
    """Check the syntactic conservativity contract used by PA+ definitions.

    This proves that each fresh predicate is merely an acyclic abbreviation of
    earlier vocabulary with no leaked free variables.  It does not prove the
    intended informal mathematical interpretation of the right-hand side.
    """

    missing_base_labels = set(base.statements) - set(extension.statements)
    if missing_base_labels:
        raise DefinitionCatalogError(
            f"extension omits base declarations: {sorted(missing_base_labels)}"
        )
    for label, original in base.statements.items():
        inherited = extension.statements[label]
        if (
            inherited.conclusion != original.conclusion
            or inherited.hypotheses != original.hypotheses
            or inherited.d_constraints != original.d_constraints
            or inherited.variable_types != original.variable_types
            or inherited.kind != original.kind
            or inherited.source_tokens != original.source_tokens
        ):
            raise DefinitionCatalogError(
                f"extension changes inherited base declaration {label!r}"
            )
    fresh_variables = set(extension.variables) - set(base.variables)
    if fresh_variables:
        raise DefinitionCatalogError(
            f"extension introduces object/metavariables: {sorted(fresh_variables)}"
        )

    fresh_labels = set(extension.logical_assertions) - set(base.logical_assertions)
    ordered = sorted(
        (extension.logical_assertions[label] for label in fresh_labels),
        key=lambda theorem: theorem.declaration_index,
    )
    fresh_predicates: set[str] = set()
    for theorem in ordered:
        if not theorem.name.startswith("df-"):
            raise DefinitionCatalogError(
                f"fresh logical assertion {theorem.name!r} is not a definition"
            )
        lhs, _ = _definition_body(theorem)
        fresh_predicates.add(lhs.op)

    expected = None if expected_definitions is None else tuple(expected_definitions)
    if expected is not None and set(expected) != fresh_predicates:
        raise DefinitionCatalogError(
            "definition mismatch: "
            f"missing={sorted(set(expected) - fresh_predicates)}, "
            f"extra={sorted(fresh_predicates - set(expected))}"
        )
    collisions = fresh_predicates & (set(base.symbols) | set(base.variables))
    if collisions:
        raise DefinitionCatalogError(
            f"fresh predicates collide with base vocabulary: {sorted(collisions)}"
        )
    fresh_constants = set(extension.symbols) - set(base.symbols)
    if fresh_constants != fresh_predicates:
        raise DefinitionCatalogError(
            "fresh constant mismatch: "
            f"missing={sorted(fresh_predicates - fresh_constants)}, "
            f"extra={sorted(fresh_constants - fresh_predicates)}"
        )

    allowed_unused = {
        name: frozenset(parameters)
        for name, parameters in (allowed_unused_parameters or {}).items()
    }
    unknown_whitelists = set(allowed_unused) - fresh_predicates
    if unknown_whitelists:
        raise DefinitionCatalogError(
            f"unused-parameter whitelist names unknown predicates: "
            f"{sorted(unknown_whitelists)}"
        )

    defined: set[str] = set()
    audited: list[AuditedDefinition] = []
    for theorem in ordered:
        lhs, rhs = _definition_body(theorem)
        if theorem.name != f"df-{lhs.op}":
            raise DefinitionCatalogError(
                f"{theorem.name}: label must be df-{lhs.op}"
            )
        if not lhs.args or any(argument.args for argument in lhs.args):
            raise DefinitionCatalogError(
                f"{theorem.name}: left side must apply predicate to variables"
            )
        parameters = tuple(argument.op for argument in lhs.args)
        if len(parameters) != len(set(parameters)):
            raise DefinitionCatalogError(f"{theorem.name}: repeated left parameter")
        incorrectly_typed = {
            parameter: theorem.variable_types.get(parameter)
            for parameter in parameters
            if theorem.variable_types.get(parameter) != "term"
        }
        if incorrectly_typed:
            raise DefinitionCatalogError(
                f"{theorem.name}: left parameters must all have type term: "
                f"{incorrectly_typed}"
            )
        constrained_parameters = {
            normalized_pair(left, right)
            for left, right in combinations(parameters, 2)
            if normalized_pair(left, right) in theorem.d_constraints
        }
        if constrained_parameters:
            raise DefinitionCatalogError(
                f"{theorem.name}: left parameters must not have $d constraints: "
                f"{sorted(constrained_parameters)}"
            )
        if lhs.op in {item.op for item in rhs.walk()}:
            raise DefinitionCatalogError(f"{theorem.name}: recursive definition")

        dependencies = {item.op for item in rhs.walk()} & fresh_predicates
        unavailable = dependencies - defined
        if unavailable:
            raise DefinitionCatalogError(
                f"{theorem.name}: forward/cyclic dependencies {sorted(unavailable)}"
            )
        known = frozenset(theorem.variable_types)
        body_free_variables = _free_variables(rhs, known)
        leaked = body_free_variables - set(parameters)
        if leaked:
            raise DefinitionCatalogError(
                f"{theorem.name}: free variables leak from body: {sorted(leaked)}"
            )
        binders = _quantified_variables(rhs)
        if binders & set(parameters):
            raise DefinitionCatalogError(
                f"{theorem.name}: parameters rebound in definition"
            )
        unused = set(parameters) - body_free_variables
        permitted_unused = set(allowed_unused.get(lhs.op, frozenset()))
        invalid_permitted = permitted_unused - set(parameters)
        if invalid_permitted:
            raise DefinitionCatalogError(
                f"{theorem.name}: invalid unused-parameter whitelist: "
                f"{sorted(invalid_permitted)}"
            )
        if unused != permitted_unused:
            raise DefinitionCatalogError(
                f"{theorem.name}: unused parameters mismatch: "
                f"unapproved={sorted(unused - permitted_unused)}, "
                f"unnecessary_whitelist={sorted(permitted_unused - unused)}"
            )
        for binder in binders:
            for parameter in parameters:
                pair = normalized_pair(binder, parameter)
                if pair not in theorem.d_constraints:
                    raise DefinitionCatalogError(
                        f"{theorem.name}: missing freshness constraint {pair}"
                    )
        for left, right in combinations(sorted(binders), 2):
            pair = normalized_pair(left, right)
            if pair not in theorem.d_constraints:
                raise DefinitionCatalogError(
                    f"{theorem.name}: missing binder freshness constraint {pair}"
                )

        syntax_rules = [
            rule
            for rule in extension.syntax_rules.get("wff", ())
            if rule.label == f"wff_{lhs.op}"
        ]
        if len(syntax_rules) != 1:
            raise DefinitionCatalogError(
                f"{theorem.name}: expected exactly one syntax declaration "
                f"wff_{lhs.op}, got {len(syntax_rules)}"
            )
        syntax_rule = syntax_rules[0]
        expected_pattern = (lhs.op, *parameters)
        expected_types = {parameter: "term" for parameter in parameters}
        if (
            syntax_rule.output_type != "wff"
            or syntax_rule.pattern != expected_pattern
            or dict(syntax_rule.variable_types) != expected_types
        ):
            raise DefinitionCatalogError(
                f"{theorem.name}: syntax declaration does not match left side"
            )
        audited.append(
            AuditedDefinition(
                name=lhs.op,
                label=theorem.name,
                arity=len(parameters),
                dependencies=tuple(sorted(dependencies)),
                body_nodes=sum(1 for _ in rhs.walk()),
                body_depth=rhs.depth,
            )
        )
        defined.add(lhs.op)

    expected_targets = () if expected_statements is None else tuple(expected_statements)
    if expected is not None and expected_statements is not None:
        expected_fresh_syntax = {
            *(f"wff_{name}" for name in expected),
            *expected_targets,
        }
        actual_fresh_syntax = set(extension.syntax_statements) - set(
            base.syntax_statements
        )
        if actual_fresh_syntax != expected_fresh_syntax:
            raise DefinitionCatalogError(
                "fresh non-logical declaration mismatch: "
                f"missing={sorted(expected_fresh_syntax - actual_fresh_syntax)}, "
                f"extra={sorted(actual_fresh_syntax - expected_fresh_syntax)}"
            )
        base_rule_labels = {
            rule.label
            for rules in base.syntax_rules.values()
            for rule in rules
        }
        fresh_rule_labels = {
            rule.label
            for rules in extension.syntax_rules.values()
            for rule in rules
            if rule.label not in base_rule_labels
        }
        expected_rule_labels = {f"wff_{name}" for name in expected}
        if fresh_rule_labels != expected_rule_labels:
            raise DefinitionCatalogError(
                "fresh syntax-rule mismatch: "
                f"missing={sorted(expected_rule_labels - fresh_rule_labels)}, "
                f"extra={sorted(fresh_rule_labels - expected_rule_labels)}"
            )
    verified_statements: list[str] = []
    for name in expected_targets:
        theorem = extension.syntax_statements.get(name)
        if theorem is None or theorem.conclusion.op != "statement":
            raise DefinitionCatalogError(f"missing non-logical statement {name!r}")
        if any(
            rule.label == name
            for rules in extension.syntax_rules.values()
            for rule in rules
        ):
            raise DefinitionCatalogError(
                f"{name}: target must not register itself as a syntax rule"
            )
        formula = theorem.conclusion.args[0]
        # Non-logical ``statement`` declarations are syntax assertions.  Their
        # mandatory-variable map is intentionally empty, so closedness must be
        # checked against the database's global typed-variable environment.
        free = _free_variables(formula, frozenset(extension.variable_types))
        if free:
            raise DefinitionCatalogError(
                f"{name}: target formula is not closed: {sorted(free)}"
            )
        verified_statements.append(name)

    return DefinitionAuditReport(tuple(audited), tuple(verified_statements))


def audit_catalog_requirements(
    catalog: DefinitionCatalog,
    extension: Database,
) -> None:
    """Require statement dependency metadata to match its formal AST exactly."""

    fresh = set(catalog.definition_names)
    for item in catalog.statements:
        theorem = extension.syntax_statements[item.name]
        used = {node.op for node in theorem.conclusion.walk()} & fresh
        declared = set(item.requires)
        if used != declared:
            raise DefinitionCatalogError(
                f"{item.name}: requirement mismatch: "
                f"missing={sorted(used - declared)}, "
                f"unused={sorted(declared - used)}"
            )


def compile_and_audit(
    catalog_path: str | Path,
    output_path: str | Path,
) -> DefinitionAuditReport:
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    catalog = load_definition_catalog(catalog_path)
    rendered = render_definition_catalog(catalog)
    base = parse(catalog_path.parent / catalog.include)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8", newline="\n")
        extension = parse(temporary)
        report = audit_conservative_extension(
            base,
            extension,
            expected_definitions=catalog.definition_names,
            expected_statements=catalog.statement_names,
            allowed_unused_parameters=catalog.allowed_unused_parameters,
        )
        audit_catalog_requirements(catalog, extension)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="compile and audit a declarative conservative PA+ extension"
    )
    parser.add_argument("catalog")
    parser.add_argument("output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compile_and_audit(args.catalog, args.output)
    print(
        json.dumps(
            {
                "definitions": len(report.definitions),
                "statements": len(report.statements),
                "max_body_nodes": max(
                    (item.body_nodes for item in report.definitions), default=0
                ),
                "max_body_depth": max(
                    (item.body_depth for item in report.definitions), default=0
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import export_datasets
from .definitions import load_definition_catalog
from .export import export_graphviz, export_metamath
from .generator import GenerationConfig, TheoremGenerator
from .parser import parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate certified theorem/proof states from a Metamath "
            "database with quality filtering and premise subsumption."
        )
    )
    parser.add_argument("database", help="input .mm file")
    parser.add_argument(
        "--mode",
        choices=["random", "forward"],
        default="random",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-proof-depth", type=int, default=8)
    parser.add_argument("--max-hypotheses", type=int, default=8)
    parser.add_argument("--max-variables", type=int, default=16)
    parser.add_argument("--output-dir", default="generated")
    parser.add_argument(
        "--full-discharge-probability",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--closed-parent-probability",
        type=float,
        default=0.78,
    )
    parser.add_argument("--max-consecutive-alpha", type=int, default=1)
    parser.add_argument(
        "--max-proof-states-per-conclusion",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--definition-catalog",
        help="JSON definition catalog whose predicates define coverage",
    )
    parser.add_argument(
        "--bootstrap-definitions",
        action="store_true",
        help="derive certified unfold/fold implications before random search",
    )
    parser.add_argument(
        "--definition-coverage-weight",
        type=float,
        default=0.0,
        help="candidate bonus for underrepresented focused predicates",
    )
    parser.add_argument(
        "--disable-compatible-candidate-filter",
        action="store_true",
        help="restore legacy output-type-only candidate sampling",
    )
    parser.add_argument(
        "--max-joint-candidate-attempts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--bounded-nat-max",
        type=int,
        default=-1,
        help="largest canonical numeral used for definition instances",
    )
    parser.add_argument(
        "--ground-instances-per-predicate",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--target-guidance-weight",
        type=float,
        default=0.0,
        help="bias candidates toward subformulas of catalog target statements",
    )
    parser.add_argument(
        "--max-definition-only-search-per-predicate",
        type=int,
        default=-1,
        help="cap search results that only repackage one definition; -1 disables",
    )
    parser.add_argument("--dot", help="Graphviz output for the last result")
    parser.add_argument(
        "--metamath",
        help="replayable Metamath proof for the last result",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for option, value in (
        ("--full-discharge-probability", args.full_discharge_probability),
        ("--closed-parent-probability", args.closed_parent_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{option} must be in [0, 1]")
    if args.max_consecutive_alpha < 0:
        raise SystemExit("--max-consecutive-alpha must be non-negative")
    if args.max_proof_states_per_conclusion < 0:
        raise SystemExit(
            "--max-proof-states-per-conclusion must be non-negative"
        )
    if args.definition_coverage_weight < 0:
        raise SystemExit("--definition-coverage-weight must be non-negative")
    if args.max_joint_candidate_attempts <= 0:
        raise SystemExit("--max-joint-candidate-attempts must be positive")
    if args.bounded_nat_max < -1:
        raise SystemExit("--bounded-nat-max must be at least -1")
    if args.ground_instances_per_predicate < 0:
        raise SystemExit(
            "--ground-instances-per-predicate must be non-negative"
        )
    if args.ground_instances_per_predicate and args.bounded_nat_max < 0:
        raise SystemExit(
            "--bounded-nat-max is required when ground instances are enabled"
        )
    if args.target_guidance_weight < 0:
        raise SystemExit("--target-guidance-weight must be non-negative")
    if args.max_definition_only_search_per_predicate < -1:
        raise SystemExit(
            "--max-definition-only-search-per-predicate must be at least -1"
        )

    database = parse(args.database)
    focus_predicates: tuple[str, ...] = ()
    target_statements: tuple[str, ...] = ()
    if args.definition_catalog:
        catalog = load_definition_catalog(args.definition_catalog)
        focus_predicates = catalog.definition_names
        target_statements = catalog.statement_names
    elif args.target_guidance_weight > 0:
        raise SystemExit(
            "--target-guidance-weight requires --definition-catalog"
        )
    generator = TheoremGenerator(
        database,
        GenerationConfig(
            max_ast_depth=args.max_depth,
            max_hypotheses=args.max_hypotheses,
            max_variables=args.max_variables,
            max_proof_depth=args.max_proof_depth,
            seed=args.seed,
            full_discharge_probability=
                args.full_discharge_probability,
            closed_parent_probability=args.closed_parent_probability,
            max_consecutive_alpha=args.max_consecutive_alpha,
            max_proof_states_per_conclusion=
                args.max_proof_states_per_conclusion,
            bootstrap_definitions=args.bootstrap_definitions,
            focus_predicates=focus_predicates,
            definition_coverage_weight=args.definition_coverage_weight,
            compatible_candidate_filter=
                not args.disable_compatible_candidate_filter,
            max_joint_candidate_attempts=args.max_joint_candidate_attempts,
            bounded_nat_max=args.bounded_nat_max,
            ground_instances_per_predicate=
                args.ground_instances_per_predicate,
            target_statements=target_statements,
            target_guidance_weight=args.target_guidance_weight,
            max_definition_only_search_per_predicate=
                args.max_definition_only_search_per_predicate,
        ),
    )
    generated = generator.generate(args.mode, args.steps)
    paths = export_datasets(generator, args.output_dir)
    categorized = generator.categorized()
    preferred = [
        *categorized["closed_theorems"],
        *categorized["inference_rules"],
    ]
    last = preferred[-1] if preferred else (
        generated[-1] if generated else None
    )
    if last is not None and args.dot:
        export_graphviz(last, generator.store, args.dot)
    if last is not None and args.metamath:
        export_metamath(last, generator.store, args.metamath)

    summary = generator.summary()
    print(
        f"parsed {len(database.statements)} assertions; stored "
        f"{summary.stored} proof objects ({summary.active} active, "
        f"{summary.definition_bridges} definition bridges, "
        f"{summary.bounded_ground_instances} bounded instances, "
        f"{summary.search_stored} search-generated); "
        f"categories={summary.categories}; output="
        f"{Path(args.output_dir).resolve()}"
    )
    print(f"rejections={summary.rejected}; summary={paths['summary']}")
    print(
        "definition coverage="
        f"{summary.definition_predicates_seen}/"
        f"{summary.definition_predicates_total} "
        f"({summary.definition_coverage:.1%})"
    )
    if summary.target_statements_total:
        print(
            "search/target similarity touched="
            f"{summary.target_statements_touched}/"
            f"{summary.target_statements_total} at similarity >= 0.25; "
            f"mean best={summary.target_mean_best_similarity:.3f}"
        )
    print(
        "single-definition search objects admitted="
        f"{summary.definition_only_search_admitted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

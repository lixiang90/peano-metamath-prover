from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import export_datasets
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
        choices=["random", "forward", "depth"],
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

    database = parse(args.database)
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
        f"{summary.stored} proof objects ({summary.active} active); "
        f"categories={summary.categories}; output="
        f"{Path(args.output_dir).resolve()}"
    )
    print(f"rejections={summary.rejected}; summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

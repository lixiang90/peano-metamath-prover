from __future__ import annotations

import json
from pathlib import Path

from .export import theorem_record
from .generator import TheoremGenerator
from .model import Theorem


DATASET_FILENAMES = {
    "closed_theorems": "closed_theorems.jsonl",
    "inference_rules": "inference_rules.jsonl",
    "proof_states": "proof_states.jsonl",
}


def quality_theorem_record(
    theorem: Theorem,
    generator: TheoremGenerator,
) -> dict:
    """Return a proof-bearing record with quality metadata."""

    if theorem.id is None:
        raise ValueError("the theorem must belong to the generator database")
    assessment = generator.assessments[theorem.id]
    record = theorem_record(theorem, generator.store)
    record["quality"] = {
        "score": assessment.score,
        "category": assessment.category,
        "active_candidate": theorem.id in generator.active_ids,
        "dominated": theorem.id in generator.dominated_ids,
        "hard_reject": assessment.hard_reject,
        "reasons": assessment.reasons,
        "vacuous_quantifiers": assessment.vacuous_quantifiers,
        "nonvacuous_quantifiers": assessment.nonvacuous_quantifiers,
        "schematic_wff_variables": list(
            assessment.schematic_wff_variables
        ),
        "bare_conclusion": assessment.bare_conclusion,
        "arithmetic_operators": list(assessment.arithmetic_operators),
        "defined_predicates": list(assessment.defined_predicates),
        "source_equivalent": assessment.source_equivalent,
        # This is a comparison/reporting key, not a replacement theorem.
        "normalized_conclusion": assessment.normalized_conclusion,
    }
    return record


def export_datasets(
    generator: TheoremGenerator,
    output_directory: str | Path,
    *,
    include_dominated: bool = False,
) -> dict[str, Path]:
    """Write the three datasets and a machine-readable run summary."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for category, filename in DATASET_FILENAMES.items():
        path = directory / filename
        with path.open("w", encoding="utf-8") as stream:
            for theorem in generator.ranked(
                category,
                include_dominated=include_dominated,
            ):
                stream.write(
                    json.dumps(
                        quality_theorem_record(theorem, generator),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        paths[category] = path

    summary = generator.summary()
    summary_path = directory / "quality_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "attempts": summary.attempts,
                "stored": summary.stored,
                "search_stored": summary.search_stored,
                "definition_bridges": summary.definition_bridges,
                "bounded_ground_instances":
                    summary.bounded_ground_instances,
                "definition_only_search_admitted":
                    summary.definition_only_search_admitted,
                "active": summary.active,
                "dominated": summary.dominated,
                "categories": summary.categories,
                "rejected": summary.rejected,
                "graph_statistics": {
                    k: v for k, v in generator.stats.items()
                    if k.startswith("graph_") or k in {"random_graph", "random_instance"}
                },
                "rule_usage": summary.rule_usage,
                "definition_coverage": {
                    "total": summary.definition_predicates_total,
                    "seen": summary.definition_predicates_seen,
                    "ratio": summary.definition_coverage,
                    "usage": summary.definition_usage,
                },
                "target_guidance": {
                    "total": summary.target_statements_total,
                    "touched_at_similarity_0_25":
                        summary.target_statements_touched,
                    "mean_best_similarity":
                        summary.target_mean_best_similarity,
                    "best_similarity": summary.target_best_similarity,
                },
                "configuration": {
                    "graph_instance_probability": generator.config.graph_instance_probability,
                    "graph_partial_premise_probability": generator.config.graph_partial_premise_probability,
                    "full_discharge_probability":
                        generator.config.full_discharge_probability,
                    "closed_parent_probability":
                        generator.config.closed_parent_probability,
                    "max_consecutive_alpha":
                        generator.config.max_consecutive_alpha,
                    "reject_vacuous_quantifiers":
                        generator.config.quality.reject_vacuous_quantifiers,
                    "bootstrap_definitions":
                        generator.config.bootstrap_definitions,
                    "definition_coverage_weight":
                        generator.config.definition_coverage_weight,
                    "compatible_candidate_filter":
                        generator.config.compatible_candidate_filter,
                    "bounded_nat_max": generator.config.bounded_nat_max,
                    "ground_instances_per_predicate":
                        generator.config.ground_instances_per_predicate,
                    "target_guidance_weight":
                        generator.config.target_guidance_weight,
                    "max_definition_only_search_per_predicate": (
                        generator.config.max_definition_only_search_per_predicate
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    return paths

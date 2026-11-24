"""Count decision supervision and first policy rankings in saved v2.1 runs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit_corpora(directories):
    cases = set()
    policy_cases = set()
    kinds = Counter()
    positive_ranks = Counter()
    trajectories_with_decisions = 0
    trajectories_with_backtrack = 0
    for directory in directories:
        # IDs are local to a shard. Use the normalized statement across shards.
        by_id = {
            row["id"]: row["full_key"]
            for row in read_rows(directory / "cases.jsonl")
            if row["split"] == "train"
        }
        cases.update(by_id.values())
        for row in read_rows(directory / "examples.jsonl"):
            if row["split"] != "train":
                continue
            kind = row["kind"]
            kinds[kind] += 1
            if kind == "policy":
                candidates = row["candidates"]
                positives = set(row["positives"])
                if not positives or not positives.issubset(candidates):
                    raise ValueError(f"Invalid positive candidates in {directory}, case {row['case']}")
                first = next(i for i, action in enumerate(candidates, 1) if action in positives)
                positive_ranks[first] += 1
                policy_cases.add(by_id[row["case"]])
            elif kind == "trajectory":
                commands = [event["content"] for event in row["events"] if event["role"] == "model"]
                trajectories_with_decisions += bool(commands)
                trajectories_with_backtrack += any(cmd["op"] == "BACKTRACK" for cmd in commands)
    return {
        "training_cases": len(cases),
        "rows_by_kind": dict(kinds),
        "cases_with_policy_rows": len(policy_cases),
        "first_known_positive_rank": dict(sorted(positive_ranks.items())),
        "trajectories_with_decisions": trajectories_with_decisions,
        "trajectories_with_backtrack": trajectories_with_backtrack,
    }


def audit_first_ranking(evaluation):
    results = [r for r in read_rows(evaluation / "results.jsonl") if r["method"] == "learned-a"]
    if not results:
        raise ValueError(f"No learned-a results in {evaluation}")
    ranks = Counter()
    actions = Counter()
    missing = no_ranking = ties = 0
    for row in results:
        detail = evaluation / "learned-a" / row["id"] / "result.json"
        if not detail.exists():
            missing += 1
            continue
        attempts = json.loads(detail.read_text(encoding="utf-8"))["attempts"]
        entry = next((a for a in attempts if "distribution" in a), None)
        if entry is None:
            no_ranking += 1
            continue
        weights = entry["distribution"]
        best = max(weights)
        if weights.count(best) > 1:
            ties += 1
            continue
        index = weights.index(best)
        ranks[index + 1] += 1
        actions[entry["candidates"][index]] += 1
    return {
        "cases": len(results),
        "missing_detail": missing,
        "no_recorded_ranking": no_ranking,
        "tied_for_first": ties,
        "unique_top_choice_rank_in_candidate_order": dict(sorted(ranks.items())),
        "unique_top_choice_actions": dict(actions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", type=Path, nargs="+", help="v2.1 strategy corpus directories")
    parser.add_argument("--evaluation", type=Path, help="evaluation directory containing a learned-a arm")
    args = parser.parse_args()
    groups = defaultdict(list)
    for directory in dict.fromkeys(p.resolve() for p in args.corpora):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        mode = manifest["search"].get("reasoning_mode", "baseline")
        groups[mode].append(directory)
    report = {"training": {mode: audit_corpora(paths) for mode, paths in groups.items()}}
    if args.evaluation:
        report["learned_a_first_ranking"] = audit_first_ranking(args.evaluation)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Training-only balance over known proof cost, structure and assumptions."""
from collections import Counter, defaultdict


def stratum(record):
    if not isinstance(record, dict):
        cost = record.remaining_steps or record.proof_depth
        rule, structure = record.rule, record.structure_family
        hypotheses, goals = len(record.hypotheses), record.goal_count
        premises = record.rule_premise_count
    else:
        cost = record.get('remaining_steps') or record.get('proof_depth', 1)
        rule = record.get('rule', 'legacy')
        structure = record.get('structure_family', 'legacy')
        hypotheses, goals = record.get('hypothesis_count', 0), record.get('goal_count', 1)
        premises = record.get('rule_premise_count', 0)
    return (1 if cost <= 1 else 2 if cost <= 3 else 3 if cost <= 8 else 4,
            rule, structure, min(hypotheses, 4), min(goals, 3), min(premises, 3))


def supervision_histograms(records, histograms=None):
    histograms = {} if histograms is None else histograms
    for r in records:
        if not isinstance(r, dict):
            fields = dict(goals=r.goal_count, hypotheses=len(r.hypotheses), remaining_steps=r.remaining_steps,
                rule_premises=r.rule_premise_count, candidates=len(r.candidate_action_ids),
                positive_candidates=len(r.positive_candidate_indices), value=format(r.value_target, '.6g'))
        else:
            fields = dict(goals=r['goal_count'], hypotheses=r['hypothesis_count'], remaining_steps=r['remaining_steps'],
                rule_premises=r['rule_premise_count'], candidates=len(r['candidate_action_ids']),
                positive_candidates=len(r['positive_candidate_indices']), value=format(r['value'], '.6g'))
        for key, value in fields.items():
            histogram = histograms.setdefault(key, {})
            histogram[str(value)] = histogram.get(str(value), 0) + 1
    return histograms


def balanced_weights(records, exploration=.25):
    counts = Counter(stratum(r) for r in records)
    n = len(records)
    return [exploration / n + (1 - exploration) / (len(counts) * counts[stratum(r)]) for r in records]


def balanced_window(records, rng):
    # Fixed output count and local RNG make resume by consumed offset exact.
    return rng.choices(records, weights=balanced_weights(records), k=len(records))


def balanced_order(records, seed):
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[stratum(record)].append(record)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while groups:
        keys = list(groups)
        rng.shuffle(keys)
        for key in keys:
            result.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
    return result

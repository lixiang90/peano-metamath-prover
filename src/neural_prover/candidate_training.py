"""Same-state, multi-positive ranking with compatibility for old corpora."""
import torch


def candidate_batch(actions, groups, positives, pad_id):
    pool = list(dict.fromkeys(tuple(a) for a in actions))
    memberships = []
    targets = []
    for gold, group, positive in zip(actions, groups, positives):
        if not group:
            # Legacy corpora only contain the recorded action.
            group = actions
            positive = [i for i, a in enumerate(actions) if tuple(a) == tuple(gold)]
        if not positive or any(i < 0 or i >= len(group) for i in positive):
            raise ValueError("invalid positive candidate indices")
        if tuple(gold) not in {tuple(group[i]) for i in positive}:
            raise ValueError("gold action must be a positive candidate")
        ids = []
        for action in group:
            key = tuple(action)
            if not key:
                raise ValueError("candidate action cannot be empty")
            if key not in pool:
                pool.append(key)
            ids.append(pool.index(key))
        memberships.append(set(ids))
        targets.append({ids[i] for i in positive})
    candidate_ids = torch.full((len(pool), max(map(len, pool))), pad_id, dtype=torch.long)
    for i, action in enumerate(pool):
        candidate_ids[i, :len(action)] = torch.tensor(action)
    allowed = torch.zeros((len(actions), len(pool)), dtype=torch.bool)
    positive_mask = torch.zeros_like(allowed)
    for i, (members, target) in enumerate(zip(memberships, targets)):
        allowed[i, list(members)] = True
        positive_mask[i, list(target)] = True
    return {"candidate_ids": candidate_ids, "candidate_allowed": allowed, "candidate_positive": positive_mask}


def candidate_objective(model, state, batch):
    scores, _ = model.score_candidate_matrix(state, batch["candidate_ids"].to(state.device))
    scores = scores.float().masked_fill(~batch["candidate_allowed"].to(state.device), float("-inf"))
    positives = batch["candidate_positive"].to(state.device)
    positive_scores = scores.masked_fill(~positives, float("-inf"))
    loss = (torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)).mean()
    return loss, scores, positives

"""B/C search driver; A remains in search.py without transcript overhead."""
import heapq
import json
import math
from pathlib import Path
import random
import time

from .atp import verify_bundle
from .interaction import Interaction
from .library import digest, source_digest
from .strategy import Session


def interactive_search(database, source, target, output, metamath, *, config, policy,
                       model=None, library=None, vampire=None, forced=None, return_session=False):
    from .search import Classical
    cfg = config
    if policy not in {"heuristic", "random", "learned"}:
        raise ValueError("unsupported interaction policy")
    if policy == "learned" and model is None:
        raise ValueError("learned policy requires model")
    if model is not None and model.provenance.get("source", source_digest(source)) != source_digest(source):
        raise ValueError("checkpoint formal source changed")
    started = time.perf_counter()
    end = started+cfg.seconds
    deadline = end-min(cfg.verification_reserve, cfg.seconds*.2)
    remaining = lambda: max(0., deadline-time.perf_counter())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(database, target, library)
    backend = Classical(session, cfg, deadline, out, vampire)
    flow = Interaction(session, session.initial, cfg.reasoning_mode)
    flow.initialize(backend, remaining)
    initial_observation = flow.observation()
    queue = [(0., 0, flow, [])]
    rng = random.Random(cfg.seed)
    serial = expanded = 0
    scored_bytes = policy_calls = 0
    seen, attempts = set(), []
    winner = None
    result = {"id": target.name, "method": policy, "reasoning_mode": cfg.reasoning_mode,
              "certified": False, "status": "unknown_exhausted", "budget_seconds": cfg.seconds,
              "trace": [], "attempts": attempts, "initial_observation": initial_observation}
    last_flow = flow
    while queue:
        if time.perf_counter() >= deadline:
            result["status"] = "unknown_timeout"
            break
        cost, _, flow, trace = heapq.heappop(queue)
        if isinstance(flow, tuple):
            parent, entry = flow
            flow = parent.fork()
            flow.execute(entry["action"], backend, cfg, remaining)
            attempts.append({**entry, "status": "expanded"})
            trace = trace+[entry]
            if cfg.stochastic:
                result["trace"] = trace
        last_flow = flow
        if flow.state.solved and flow.pending is None:
            fragment = out/"fragment.mm"
            flow.session.b.export(flow.state.proofs[0], target, fragment)
            left = end-time.perf_counter()
            if left <= 0:
                result["status"] = "unknown_timeout"
                break
            verification = verify_bundle(fragment, source, metamath, left)
            result["verification"] = verification
            if verification["status"] == "passed" and time.perf_counter() <= end:
                result.update(certified=True, status="proved", trace=trace, certificate=str(fragment.resolve()))
                winner = flow.state
            else:
                result["status"] = "unknown_verification" if verification["status"] != "passed" else "unknown_timeout"
            break
        if len(trace) >= cfg.max_interactions or expanded >= cfg.max_expansions:
            continue
        key = digest([flow.session.key(flow.state), flow.memory, flow.pending is not None, flow.trials])
        if key in seen:
            continue
        seen.add(key)
        expanded += 1
        texts = flow.options(cfg)
        if not texts:
            continue
        obs = flow.observation()
        if policy == "learned":
            from .interaction import render
            policy_calls += 1
            scored_bytes += len(render(obs["events"], awaiting_model=True).encode())+sum(len(t.encode())+1 for t in texts)
            try:
                logps, value = model.distribution(obs, texts, cfg.temperature)
            except ValueError as exc:
                if "context" not in str(exc):
                    raise
                attempts.append({"status": "unknown_context", "error": str(exc)})
                continue
            weights = logps.exp().detach().cpu().tolist()
        else:
            weights = [1. if policy == "random" else 1./(i+1) for i in range(len(texts))]
            total = sum(weights)
            weights = [w/total for w in weights]
            value = 0.
        if forced is not None and not trace:
            indices = [texts.index(forced)] if forced in texts else []
        elif cfg.stochastic:
            indices = [rng.choices(range(len(texts)), weights=weights, k=1)[0]]
        elif policy == "random":
            indices = rng.sample(range(len(texts)), len(texts))
        else:
            indices = sorted(range(len(texts)), key=lambda i: -weights[i])
        # Lazy children: classical execution happens when an edge is selected,
        # not eagerly for every sibling, so a wide candidate set cannot eat the
        # entire budget before its best branch is considered.
        for i in indices:
            serial += 1
            entry = {"observation": obs, "candidates": texts, "action": texts[i], "index": i,
                     "log_prob": math.log(max(weights[i], 1e-38)), "distribution": weights,
                     "log_distribution": [math.log(max(w, 1e-38)) for w in weights],
                     "temperature": cfg.temperature, "stochastic": cfg.stochastic, "state_key": key}
            uncertainty = -.1*math.log(max(float(value), .01)) if policy == "learned" else 0.
            increment = .01 if flow.pending is not None else 1.
            priority = cost+increment-math.log(max(weights[i], 1e-38))*.1+uncertainty
            heapq.heappush(queue, (priority, serial, (flow, entry), trace))
    result.update(seconds=time.perf_counter()-started, expansions=expanded, classical_calls=backend.calls,
                  external_calls=len(backend.events), library_snapshot=library.fingerprint if library else "empty",
                  interaction=last_flow.events, receipts=last_flow.receipts,
                  macro_depth=last_flow.depth,
                  path_trials=last_flow.trials,
                  trials=sum(json.loads(e["action"])["op"]=="TRY" for e in attempts if "action" in e),
                  backtracks=sum(json.loads(e["action"])["op"]=="BACKTRACK" for e in attempts if "action" in e),
                  policy_calls=policy_calls,policy_scored_bytes=scored_bytes,
                  executed_command_bytes=sum(len(e["action"].encode())+1 for e in attempts if "action" in e))
    (out/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return (result, last_flow.session, winner) if return_session else result

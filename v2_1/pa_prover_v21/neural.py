"""Frozen v1 checkpoint adapter; no optimizers or training entry points."""
import heapq
import itertools
import time

from .solver import BudgetExpired


class FrozenModel:
    def __init__(self, checkpoint, tokenizer, database, device="cuda"):
        import torch
        from neural_prover.model import ProofTransformer
        from neural_prover.tokenizer import MetamathTokenizer
        from neural_prover.data_contract import validate_checkpoint_tokenizer
        torch.set_num_threads(1)
        self.model, payload = ProofTransformer.load_checkpoint(checkpoint, map_location="cpu")
        original = MetamathTokenizer.load(tokenizer)
        validate_checkpoint_tokenizer(self.model, payload, original)
        self.tokenizer = original.with_target_hints(False)
        self.model.to(device).eval()
        self.database, self.device = database, device
        # Warm the full scoring path on a fixed axiom, not an evaluation target.
        from neural_prover.environment import BackwardEnvironment, ProofState
        from neural_prover.search import TransformerPolicy
        from metamath_generator.model import Theorem
        from .kernel import N, eq
        env = BackwardEnvironment(database)
        goal = Theorem("warmup", [], N("|-",eq(N("0"),N("0"))), variable_types={})
        state = ProofState.from_theorem(goal)
        warm = TransformerPolicy(self.model,self.tokenizer,env,device,configure_environment=False)
        warm.rank(state,env.enumerate_tactics(state,max_candidates_per_variable=1,max_tactics=1))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def prove(self, target, deadline):
        import torch
        from neural_prover.environment import BackwardEnvironment, ProofState
        from neural_prover.search import TransformerPolicy, SearchResult
        from neural_prover.certificate import compile_certificate, verify_certificate
        env = BackwardEnvironment(self.database)
        env.configure_from_tokenizer(self.tokenizer, target_guidance=False)
        policy = TransformerPolicy(self.model, self.tokenizer, env, self.device, configure_environment=False)
        initial = ProofState.from_theorem(target)
        serial = itertools.count()
        queue = [(0., next(serial), initial, (), 0.)]
        visited = {initial}
        expanded = 0
        def check():
            if time.perf_counter() >= deadline:
                raise BudgetExpired()
        while queue and expanded < 512:
            check()
            _, _, state, trace, cost = heapq.heappop(queue)
            try:
                state_tokens = self.tokenizer.proof_state_tokens(state)
                self.tokenizer.encode(state_tokens)
            except ValueError:
                continue
            if len(state_tokens)>self.model.config.max_state_tokens:
                continue  # Do not silently count a heuristic fallback as model inference.
            if len(trace) >= 24:
                continue
            candidates = env.enumerate_tactics(state, max_candidates_per_variable=4,
                                               max_tactics=24, include_derived=True)
            check()
            ranked = []
            for i in range(0, len(candidates), 4):
                ranked.extend(policy.rank(state, candidates[i:i+4]))
                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()
                check()
            expanded += 1
            for candidate in sorted(ranked, key=lambda r: r.log_probability, reverse=True)[:16]:
                check()
                tr = env.apply(state, candidate.tactic)
                next_trace = (*trace, tr)
                if tr.after.solved:
                    result = SearchResult(True, expanded, len(visited),
                        tuple(t.tactic for t in next_trace), next_trace, tr.after)
                    certificate = compile_certificate(target, result, self.database, name="v21_model_result")
                    verify_certificate(certificate, self.database)
                    return certificate, {"expansions": expanded, **policy.metrics()}
                if tr.after in visited:
                    continue
                visited.add(tr.after)
                new_cost = cost - candidate.log_probability
                priority = len(next_trace) + len(tr.after.goals) + new_cost - candidate.next_value
                heapq.heappush(queue, (priority, next(serial), tr.after, next_trace, new_cost))
        return None, {"expansions": expanded, **policy.metrics()}

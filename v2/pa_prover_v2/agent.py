"""A bounded, auditable proof workspace; observations contain no private reasoning.

Only the kernel can create trusted facts.  Proposals are untrusted scheduling
notes, and read memories never become premises until a verified USE succeeds.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from metamath_generator.model import Node, Theorem
from metamath_generator.parser import MetamathParser
from metamath_generator.unification import substitute_simultaneous
from neural_prover.environment import BackwardEnvironment, ProofState

from .kernel import ProofBudgetError, ProofKernel, node_from_data, node_to_data, statement_fingerprint
from .library import TheoremLibrary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Node):
        return node_to_data(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class AgentAction:
    op: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"op": self.op.upper(), "args": _jsonable(self.args)}

    @classmethod
    def from_dict(cls, value: Mapping) -> "AgentAction":
        if not isinstance(value.get("op"), str) or not isinstance(value.get("args", {}), dict):
            raise ValueError("action requires an op string and args object")
        return cls(value["op"].upper(), dict(value.get("args", {})))


@dataclass(frozen=True)
class AgentBudget:
    max_steps: int = 128
    max_reads: int = 16
    max_memory_tokens: int = 4096
    max_facts: int = 128
    max_notes: int = 32
    max_backtracks: int = 8
    max_proof_labels: int = 65536
    max_context_chars: int = 64000
    required_external: bool = False

    def __post_init__(self):
        for key, value in vars(self).items():
            if key != "required_external" and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.max_proof_labels == 0:
            raise ValueError("max_proof_labels must be a positive integer")


@dataclass(frozen=True)
class AgentObservation:
    status: str
    op: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": self.status, "op": self.op, "message": self.message,
                "data": _jsonable(self.data)}


class _BudgetExceeded(ValueError):
    pass


def _statement(theorem: Theorem) -> dict:
    return {
        "hypotheses": [h.expr.to_prefix() for h in theorem.hypotheses],
        "conclusion": theorem.conclusion.to_prefix(),
        "variable_types": dict(sorted(theorem.variable_types.items())),
        "d_constraints": [list(pair) for pair in sorted(theorem.d_constraints)],
    }


class AgentSession:
    """Forward proof episode with fixed assumptions and transactional actions.

    Facts are named h0, h1, ... for initial assumptions and f0, f1, ... for
    constructed proofs.  Every successful action creates a snapshot s<step>.
    Snapshot s0 is the initial workspace.  Backtracking revokes later branch
    snapshots, facts, pending proposals, scratch entries and read memories;
    resource consumption and the explicit action/observation trace persist.
    """

    def __init__(
        self, kernel: ProofKernel, target: Theorem,
        library: TheoremLibrary | None = None,
        budget: AgentBudget | dict | None = None,
    ) -> None:
        self.kernel = kernel
        self.target = copy.deepcopy(target)
        self.library = library
        self.budget = AgentBudget(**budget) if isinstance(budget, dict) else (budget or AgentBudget())
        if len(target.hypotheses) > self.budget.max_facts:
            raise ValueError("initial hypotheses exceed fact budget")
        self.facts: dict[str, Theorem] = {
            f"h{index}": kernel.assumption(self.target, index)
            for index in range(len(target.hypotheses))
        }
        self.scratch: dict[str, str] = {}
        self.pending_focus: list[Node] = []
        self.search_results: list[str] = []
        self.memory_occurrences: list[dict] = []
        self.read_records: list[dict] = []
        self.trace: list[dict] = []
        self.status = "running"
        self.steps = 0
        self.reads = 0
        self.memory_tokens = 0
        self.backtracks = 0
        self.certificate: Theorem | None = None
        self.certification: dict | None = None
        self._read_rules: dict[str, Theorem] = {}
        self._read_sources: dict[str, str] = {}
        self._target_fingerprint = statement_fingerprint(self.target)
        self._searched_queries: set[str] = set()
        self._next_fact = 0
        self._next_occurrence = 0
        self._backtracked_ids: set[str] = set()
        self._snapshots: dict[str, dict] = {"s0": self._capture()}
        self._context_budget_notice: dict | None = None
        self._enforce_context_budget()

    def _capture(self) -> dict:
        return {
            "facts": dict(self.facts), "scratch": dict(self.scratch),
            "pending_focus": list(self.pending_focus),
            "search_results": list(self.search_results),
            "memory_occurrences": list(self.memory_occurrences),
            "_read_rules": dict(self._read_rules),
            "_read_sources": dict(self._read_sources),
            "_searched_queries": set(self._searched_queries),
        }

    def _restore(self, snapshot: dict) -> None:
        for key, value in snapshot.items():
            setattr(self, key, copy.copy(value))

    def _parse(self, value: Any, *, typecode: str | None = None) -> Node:
        if isinstance(value, Node):
            result = value
        elif isinstance(value, dict):
            result = node_from_data(value)
        elif isinstance(value, str):
            parser = MetamathParser()
            parser.database = copy.copy(self.kernel.database)
            parser.database.variable_types = {
                **self.kernel.database.variable_types, **self.target.variable_types,
            }
            tokens = value.split()
            if typecode is not None:
                parsed = parser.parse_expression([typecode, *tokens])
                result = parsed.args[0]
            else:
                result = parser.parse_expression(tokens)
        else:
            raise ValueError("formula must be an AST object or prefix string")
        return result

    def _substitution(self, value: Any, rule: Theorem) -> dict[str, Node]:
        if not isinstance(value, dict):
            raise ValueError("substitution must be an object")
        if set(value) - set(rule.variable_types):
            raise ValueError("substitution contains unknown rule variables")
        return {name: self._parse(node, typecode=rule.variable_types[name])
                for name, node in value.items()}

    def _premises(self, args: dict) -> list[Theorem]:
        refs = args.get("premises", args.get("premise_refs", []))
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError("premises must be a list of fact references")
        return [self.facts[ref] for ref in refs]

    def _add_fact(self, fact: Theorem) -> dict:
        if len(self.facts) >= self.budget.max_facts:
            raise _BudgetExceeded("fact budget exhausted")
        self._check_proof_budget(fact)
        # compose verifies as well, but this boundary also protects future
        # adapters from adding an unverified certificate to the workspace.
        self.kernel.verify(fact)
        ref = f"f{self._next_fact}"
        self._next_fact += 1
        self.facts[ref] = fact
        self.pending_focus = [goal for goal in self.pending_focus if goal != fact.conclusion]
        return {"fact_id": ref, "statement": _statement(fact)}

    def _check_proof_budget(self, theorem: Theorem) -> None:
        if theorem.proof is not None and len(theorem.proof.source_labels) > self.budget.max_proof_labels:
            raise ProofBudgetError("proof label budget exhausted")

    def _execute(self, action: AgentAction) -> dict:
        op, args = action.op.upper(), action.args
        if op == "APPLY":
            rule = self.kernel.database.logical_assertions[str(args["rule"])]
            fact = self.kernel.compose(
                rule, self._substitution(args.get("substitution", {}), rule),
                self._premises(args), self.target, max_proof_labels=self.budget.max_proof_labels,
            )
            return self._add_fact(fact)
        if op == "PROPOSE":
            formula = self._parse(args["formula"])
            environment = BackwardEnvironment(self.kernel.database)
            if formula.op != "|-" or not environment._well_typed((formula,), self.target.variable_types):
                raise ValueError("proposal must be a well-typed logical assertion")
            if formula in self.pending_focus or any(f.conclusion == formula for f in self.facts.values()):
                raise ValueError("proposal is already pending or proved")
            if len(self.pending_focus) + len(self.scratch) >= self.budget.max_notes:
                raise _BudgetExceeded("note budget exhausted")
            self.pending_focus.append(formula)
            return {"formula": formula.to_prefix(), "trusted": False}
        if op == "SAVE":
            ref, name = str(args["fact_id"]), str(args["name"])
            if not name or name in self.scratch:
                raise ValueError("scratch name must be new and nonempty")
            fact = self.facts[ref]
            self.kernel.verify(fact)
            if len(self.pending_focus) + len(self.scratch) >= self.budget.max_notes:
                raise _BudgetExceeded("note budget exhausted")
            self.scratch[name] = ref
            return {"name": name, "fact_id": ref, "statement": _statement(fact)}
        if op == "SEARCH":
            if self.library is None:
                raise ValueError("no theorem library configured")
            query = args.get("query", self.target.conclusion.to_prefix())
            limit = int(args.get("limit", 8))
            if not 1 <= limit <= 64:
                raise ValueError("search limit must be between 1 and 64")
            records = self.library.search(
                query, limit=limit,
                exclude_fingerprints={self._target_fingerprint},
            )
            self.search_results = [record.id for record in records]
            self._searched_queries.add(str(query))
            return {"results": [
                {"lemma_id": record.id, "statement": _statement(record.theorem)}
                for record in records
            ]}
        if op == "READ":
            ref = str(args.get("lemma_id", args.get("name", "")))
            if ref in self.scratch:
                theorem = self.facts[self.scratch[ref]]
                source = "scratch"
            elif self.library is not None:
                theorem = self.library.get(ref)
                source = "library"
                if statement_fingerprint(theorem) == self._target_fingerprint:
                    raise ValueError("the episode target cannot be read from its library")
            else:
                raise ValueError("unknown memory reference")
            self._check_proof_budget(theorem)
            self.kernel.verify(theorem)
            text = json.dumps(_statement(theorem), sort_keys=True, ensure_ascii=False)
            # Formal symbols and JSON controls use a deterministic accounting
            # unit.  The neural tokenizer may impose an additional limit.
            cost = sum(len(node.to_prefix().split()) for node in [
                *(h.expr for h in theorem.hypotheses), theorem.conclusion,
            ]) + 2 * len(theorem.variable_types) + 2 * len(theorem.d_constraints)
            if self.reads >= self.budget.max_reads:
                raise _BudgetExceeded("read budget exhausted")
            if self.memory_tokens + cost > self.budget.max_memory_tokens:
                raise _BudgetExceeded("memory token budget exhausted")
            record = {
                "occurrence": self._next_occurrence, "lemma_id": ref,
                "source": source, "step": self.steps, "trace_index": len(self.trace),
                "text": text, "token_cost": cost,
                "statement_fingerprint": statement_fingerprint(theorem),
            }
            self.reads += 1
            self.memory_tokens += cost
            self._next_occurrence += 1
            self.memory_occurrences.append(record)
            self.read_records.append(record)
            self._read_rules[ref] = theorem
            self._read_sources[ref] = source
            return {"memory": record, "statement": _statement(theorem)}
        if op == "USE":
            ref = str(args.get("lemma_id", args.get("name", "")))
            if ref not in self._read_rules:
                raise ValueError("lemma must be READ in the current branch before USE")
            rule = self._read_rules[ref]
            if (self._read_sources[ref] == "library"
                    and statement_fingerprint(rule) == self._target_fingerprint):
                raise ValueError("the episode target cannot be used from its library")
            fact = self.kernel.compose(
                rule, self._substitution(args.get("substitution", {}), rule),
                self._premises(args), self.target, max_proof_labels=self.budget.max_proof_labels,
            )
            return self._add_fact(fact)
        if op == "BACKTRACK":
            ref = str(args["snapshot_id"])
            if ref not in self._snapshots:
                raise ValueError("unknown or revoked snapshot")
            if self.backtracks >= self.budget.max_backtracks:
                raise _BudgetExceeded("backtrack budget exhausted")
            self._restore(self._snapshots[ref])
            self._snapshots = {key: value for key, value in self._snapshots.items()
                               if int(key[1:]) <= int(ref[1:])}
            self.backtracks += 1
            self._backtracked_ids.add(ref)
            return {"snapshot_id": ref, "facts": list(self.facts)}
        if op == "SNAPSHOT":
            return {"snapshot_id": f"s{self.steps}"}
        if op == "FINISH":
            ref = str(args["fact_id"])
            fact = self.facts[ref]
            if fact.conclusion != self.target.conclusion:
                raise ValueError("fact does not prove the target")
            self._check_proof_budget(fact)
            self.kernel.verify(fact)
            certification = self.kernel.certify(
                fact, required_external=self.budget.required_external,
            )
            self.certificate = fact
            self.certification = certification
            self.status = "success"
            return {"fact_id": ref, "certification": certification}
        raise ValueError(f"unknown action {op!r}")

    def step(self, action: AgentAction | Mapping) -> AgentObservation:
        if self.status in {"success", "unknown"}:
            return AgentObservation(
                self.status, "STOP",
                "context budget exhausted" if self._context_budget_notice else "episode is already finished",
                dict(self._context_budget_notice or {}),
            )
        if not self._enforce_context_budget():
            return AgentObservation("unknown", "STOP", "context budget exhausted",
                                    dict(self._context_budget_notice))
        if self.steps >= self.budget.max_steps:
            self.status = "unknown"
            return AgentObservation("unknown", "STOP", "step budget exhausted")
        self.steps += 1
        before = self._capture()
        snapshots_before = dict(self._snapshots)
        next_fact = self._next_fact
        try:
            action = action if isinstance(action, AgentAction) else AgentAction.from_dict(action)
            if not isinstance(action.op, str) or not isinstance(action.args, dict):
                raise ValueError("action requires an op string and args object")
            data = self._execute(action)
            if self.status != "success" and self.steps >= self.budget.max_steps:
                self.status = "unknown"
            observation = AgentObservation(self.status, action.op.upper(), "action completed", data)
            self._snapshots[f"s{self.steps}"] = self._capture()
        except (ValueError, KeyError, TypeError, IndexError) as error:
            self._restore(before)
            self._next_fact = next_fact
            exhausted = isinstance(error, (_BudgetExceeded, ProofBudgetError)) or self.steps >= self.budget.max_steps
            self.status = "unknown" if exhausted else "running"
            observation = AgentObservation(
                "unknown" if exhausted else "invalid",
                action.op.upper() if isinstance(action, AgentAction) and isinstance(action.op, str) else "INVALID",
                str(error),
            )
        self.trace.append({
            "step": self.steps,
            "action": action.to_dict()
            if isinstance(action, AgentAction) and isinstance(action.op, str)
            else {"op": "INVALID", "args": {"error": "malformed action"}},
            "observation": observation.to_dict(),
        })
        if not self._enforce_context_budget():
            # Never commit a proof or memory that the actor cannot observe in
            # full.  An over-budget FINISH must not leave a success certificate.
            self._restore(before)
            self._snapshots = snapshots_before
            self._next_fact = next_fact
            self.certificate = None
            self.certification = None
            observation = AgentObservation("unknown", observation.op, "context budget exhausted",
                                           dict(self._context_budget_notice))
            self.trace[-1]["observation"] = observation.to_dict()
        return observation

    def _enforce_context_budget(self) -> bool:
        if self._context_budget_notice is not None:
            return False
        segments = self._full_context_segments()
        size = sum(len(segment) if isinstance(segment, str) else
                   len(json.dumps(segment, sort_keys=True, ensure_ascii=False))
                   for segment in segments)
        if size <= self.budget.max_context_chars:
            return True
        self.status = "unknown"
        self._context_budget_notice = {
            "status": "unknown", "reason": "context_budget_exceeded",
            "required_chars": size, "max_context_chars": self.budget.max_context_chars,
            "full_context_omitted": True,
        }
        return False

    def context(self) -> str:
        """Explicit observations only; old trace stays available in ``trace``.

        Over-budget actor context returns an explicit terminal notice, never
        an apparently usable partial set of assumptions, facts or memories.
        """
        if not self._enforce_context_budget():
            return json.dumps(self._context_budget_notice, sort_keys=True)
        record = {
            "target": _statement(self.target), "status": self.status,
            "facts": {ref: fact.conclusion.to_prefix() for ref, fact in self.facts.items()},
            "scratch": dict(self.scratch),
            "pending_untrusted": [goal.to_prefix() for goal in self.pending_focus],
            "memories": list(self.memory_occurrences),
            "snapshots": list(self._snapshots),
            "resources": {"steps": self.steps, "reads": self.reads,
                          "memory_tokens": self.memory_tokens, "backtracks": self.backtracks},
            "recent_trace": [], "omitted_trace_entries": 0,
        }
        for item in reversed(self.trace):
            record["recent_trace"].insert(0, item)
            encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if len(encoded) > self.budget.max_context_chars:
                record["recent_trace"].pop(0)
                break
        record["omitted_trace_entries"] = len(self.trace) - len(record["recent_trace"])
        result = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if len(result) > self.budget.max_context_chars:
            record["context_over_budget"] = True
            result = json.dumps(record, sort_keys=True, ensure_ascii=False)
        return result

    def context_segments(self, memory_mode: str = "vectors") -> list[str | dict]:
        """Render a causal policy view without bypassing the lemma encoder.

        ``vectors`` keeps full retrieved statements only in dictionaries sent
        to the independent lemma encoder; READ and SEARCH text contain IDs and
        metadata.  ``text`` supplies declarations as text without memory slots;
        ``both`` is the explicit text-plus-vector ablation.  The immutable
        audit trace remains complete in every mode.  Current target, facts and
        scope are always exact, and historical memories never activate facts.
        """
        if memory_mode not in {"vectors", "text", "both"}:
            raise ValueError("memory_mode must be vectors, text or both")
        if not self._enforce_context_budget():
            return [json.dumps(self._context_budget_notice, sort_keys=True)]
        return self._full_context_segments(memory_mode)

    def _full_context_segments(self, memory_mode: str = "both") -> list[str | dict]:
        current = {
            "target": _statement(self.target), "status": self.status,
            "facts": {ref: fact.conclusion.to_prefix() for ref, fact in self.facts.items()},
            "scratch": dict(self.scratch),
            "pending_untrusted": [goal.to_prefix() for goal in self.pending_focus],
            "active_memories": list(self._read_rules),
            "snapshots": list(self._snapshots),
            "resources": {"steps": self.steps, "reads": self.reads,
                          "memory_tokens": self.memory_tokens, "backtracks": self.backtracks},
        }
        initial = {
            "episode_start": True, "target": _statement(self.target),
            "facts": {f"h{index}": hypothesis.expr.to_prefix()
                      for index, hypothesis in enumerate(self.target.hypotheses)},
            "resources": {"steps": 0, "reads": 0, "memory_tokens": 0, "backtracks": 0},
        }
        segments: list[str | dict] = [json.dumps(initial, sort_keys=True, ensure_ascii=False)]
        for event in self.trace:
            memory = event["observation"].get("data", {}).get("memory")
            visible = event
            op = event["action"].get("op")
            if memory_mode == "vectors" and op in {"READ", "SEARCH"}:
                visible = copy.deepcopy(event)
                data = visible["observation"].get("data", {})
                if op == "READ" and memory is not None:
                    visible["observation"]["data"] = {
                        "memory": {key: value for key, value in memory.items() if key != "text"},
                        "verified": True,
                    }
                elif op == "SEARCH" and "results" in data:
                    data["results"] = [
                        {"lemma_id": result["lemma_id"], "rank": rank,
                         "source": "library", "verified": True}
                        for rank, result in enumerate(data["results"], start=1)
                    ]
            segments.append(json.dumps(visible, sort_keys=True, ensure_ascii=False))
            if op == "READ" and memory is not None and memory_mode != "text":
                segments.append(dict(memory))
        # The current state must follow its historical derivation.  Putting a
        # final workspace at the beginning leaks future facts into NTP labels.
        segments.append(json.dumps({"current_workspace": current}, sort_keys=True, ensure_ascii=False))
        return segments

    def candidate_actions(self, limit: int = 64) -> list[AgentAction]:
        """Finite typed actions for constrained neural autoregressive scoring.

        Backward matching is used only to propose constructions.  An APPLY
        or USE is emitted only when every premise already has a fact proof;
        missing premises can instead become explicit untrusted proposals.
        """
        if self.status != "running" or limit <= 0:
            return []
        if not self._enforce_context_budget():
            return []
        candidates: list[AgentAction] = []
        seen: set[str] = set()

        def add(action):
            signature = json.dumps(action.to_dict(), sort_keys=True)
            if signature not in seen and len(candidates) < limit:
                seen.add(signature)
                candidates.append(action)

        by_formula = {fact.conclusion: ref for ref, fact in self.facts.items()}
        if self.target.conclusion in by_formula:
            return [AgentAction("FINISH", {"fact_id": by_formula[self.target.conclusion]})]
        goals = list(dict.fromkeys([*reversed(self.pending_focus), self.target.conclusion]))
        memory_names: dict[str, str] = {}
        memory_rules = []
        for index, (ref, theorem) in enumerate(self._read_rules.items()):
            rule = copy.deepcopy(theorem)
            rule.name = f"__agent_memory_{index}"
            memory_names[rule.name] = ref
            memory_rules.append(rule)
        environment = BackwardEnvironment(self.kernel.database, lemmas=memory_rules)
        missing: list[Node] = []
        for goal in goals:
            state = ProofState(
                tuple(by_formula), (goal,), frozenset(self.target.d_constraints),
                tuple(sorted(self.target.variable_types.items())),
            )
            tactics = environment.enumerate_tactics(
                state, max_candidates_per_variable=4, max_tactics=max(32, limit * 2),
                include_derived=False,
            )
            for tactic in tactics:
                if tactic.rule == "<ASSUMPTION>":
                    continue
                try:
                    transition = environment.apply(state, tactic)
                except ValueError:
                    continue
                rule = environment.assertions[tactic.rule]
                substitution = dict(transition.resolved_substitution)
                premises = [substitute_simultaneous(h.expr, substitution) for h in rule.hypotheses]
                if goal in by_formula:
                    continue
                if all(premise in by_formula for premise in premises):
                    args = {"substitution": substitution,
                            "premises": [by_formula[premise] for premise in premises]}
                    if tactic.rule in memory_names:
                        args["lemma_id"] = memory_names[tactic.rule]
                        add(AgentAction("USE", args))
                    else:
                        args["rule"] = tactic.rule
                        add(AgentAction("APPLY", args))
                else:
                    missing.extend(premise for premise in premises if premise not in by_formula)
        for ref in self.search_results:
            if ref not in self._read_rules and self.reads < self.budget.max_reads:
                add(AgentAction("READ", {"lemma_id": ref}))
        if self.library is not None:
            for goal in goals:
                query = goal.to_prefix()
                if query not in self._searched_queries:
                    add(AgentAction("SEARCH", {"query": query, "limit": min(8, limit)}))
        for goal in dict.fromkeys(missing):
            if goal not in goals and len(self.pending_focus) + len(self.scratch) < self.budget.max_notes:
                add(AgentAction("PROPOSE", {"formula": goal}))
        for ref in self.facts:
            if ref not in self.scratch.values() and len(self.pending_focus) + len(self.scratch) < self.budget.max_notes:
                add(AgentAction("SAVE", {"fact_id": ref, "name": f"saved_{ref}"}))
        for name in self.scratch:
            if name not in self._read_rules and self.reads < self.budget.max_reads:
                add(AgentAction("READ", {"name": name}))
        # FINISH is an auditable attempt, not an oracle token.  A wrong fact
        # receives an invalid observation and can supply an actual RL penalty.
        if self.facts:
            add(AgentAction("FINISH", {"fact_id": next(iter(self.facts))}))
        if self.backtracks < self.budget.max_backtracks:
            current = self._capture()
            snapshots = [ref for ref, snapshot in self._snapshots.items()
                         if snapshot != current and ref not in self._backtracked_ids]
            # One recent recovery point and the oldest live ancestor are
            # enough for a bounded candidate list; never offer no-op loops.
            for ref in dict.fromkeys(snapshots[-1:] + snapshots[:1]):
                action = AgentAction("BACKTRACK", {"snapshot_id": ref})
                if len(candidates) >= limit:
                    candidates.pop()
                add(action)
        return candidates


def certificate_to_actions(certificate: Theorem, kernel: ProofKernel) -> list[AgentAction]:
    """Compile an independently replayable postfix proof into teacher actions.

    Syntax assertions are evaluated to reconstruct AST substitutions but do
    not become logical actions.  Essential hypothesis references become hN;
    each logical assertion emits one APPLY and a fresh fN reference.
    """
    kernel.verify(certificate)
    if certificate.proof is None:
        raise ValueError("certificate has no proof")
    floating = {h.label: h for h in certificate.floating}
    essential = {h.label: (index, h) for index, h in enumerate(certificate.hypotheses)}
    stack: list[tuple[Node, str | None]] = []
    actions: list[AgentAction] = []
    for label in certificate.proof.source_labels:
        if label in floating:
            stack.append((floating[label].expr, None))
            continue
        if label in essential:
            index, hypothesis = essential[label]
            stack.append((hypothesis.expr, f"h{index}"))
            continue
        # Parsed certificates may reference source-scope floating hypotheses
        # beyond the mandatory statement variables.  Verification above has
        # already checked visibility and source order.
        if label in kernel.database.floating_hypotheses:
            stack.append((kernel.database.floating_hypotheses[label].expr, None))
            continue
        rule = kernel.database.statements.get(label)
        if rule is None:
            raise ValueError(f"unknown or unsupported proof label {label!r}")
        mandatory = rule.mandatory_hypotheses
        if len(stack) < len(mandatory):
            raise ValueError(f"proof stack underflow at {label}")
        actuals = stack[-len(mandatory):] if mandatory else []
        if mandatory:
            del stack[-len(mandatory):]
        actual_by_label = {expected.label: actual for expected, actual in zip(mandatory, actuals)}
        substitution = {
            hyp.expr.args[0].op: actual_by_label[hyp.label][0].args[0]
            for hyp in rule.floating
        }
        result = substitute_simultaneous(rule.conclusion, substitution)
        if result.op == "|-":
            premises = [actual_by_label[hyp.label][1] for hyp in rule.hypotheses]
            if any(ref is None for ref in premises):
                raise ValueError("logical premise has no fact reference")
            ref = f"f{len(actions)}"
            actions.append(AgentAction("APPLY", {
                "rule": label, "substitution": substitution, "premises": premises,
            }))
            stack.append((result, ref))
        else:
            stack.append((result, None))
    if len(stack) != 1 or stack[0][0] != certificate.conclusion or stack[0][1] is None:
        raise ValueError("teacher proof does not end in the target logical fact")
    actions.append(AgentAction("FINISH", {"fact_id": stack[0][1]}))
    return actions


def augment_teacher_actions(
    certificate: Theorem, kernel: ProofKernel,
    library: TheoremLibrary | None = None,
    *, stats: dict | None = None,
) -> list[AgentAction]:
    """Replay a certified teacher with explicit recovery and local-memory use.

    All fact references are remapped from actual observations.  The only
    intentionally invalid action reads a nonexistent scratch entry; it cannot
    introduce an untrusted theorem.  A non-final fact used later is saved,
    read and invoked as a proof macro, and its consumer uses the macro result.
    A bounded lookup can replace a non-root logical application by a verified
    library instance whose premises are already present.  The target statement
    is excluded from all retrieval.  Statistics count actual successful USEs,
    never merely similar retrieval results.
    """
    base = certificate_to_actions(certificate, kernel)
    session = AgentSession(kernel, certificate, library, AgentBudget(
        max_steps=len(base) + 24,
        max_facts=len(certificate.hypotheses) + len(base) + 8,
        max_reads=4, max_memory_tokens=100000, max_notes=8,
    ))
    augmented: list[AgentAction] = []
    counts = {"searches": 0, "library_reads": 0, "library_uses": 0,
              "scratch_uses": 0, "library_candidates_checked": 0}

    def emit(action: AgentAction, *, invalid=False) -> AgentObservation:
        observation = session.step(action)
        if invalid:
            if observation.status != "invalid":
                raise ValueError("teacher recovery action unexpectedly succeeded")
        elif observation.status not in {"running", "success"}:
            raise ValueError(f"teacher augmentation failed: {observation.message}")
        augmented.append(action)
        if action.op == "SEARCH":
            counts["searches"] += 1
        elif action.op == "READ" and "lemma_id" in action.args:
            counts["library_reads"] += 1
        elif action.op == "USE":
            counts["library_uses" if "lemma_id" in action.args else "scratch_uses"] += 1
        return observation

    def matching_library_use(goal: Node) -> AgentAction | None:
        if library is None or goal == certificate.conclusion or counts["library_uses"] >= 1:
            return None
        records = library.search(goal, limit=4,
                                 exclude_fingerprints={statement_fingerprint(certificate)})
        facts = {fact.conclusion: ref for ref, fact in session.facts.items()}
        state = ProofState(tuple(facts), (goal,), frozenset(certificate.d_constraints),
                           tuple(sorted(certificate.variable_types.items())))
        for record in records:
            counts["library_candidates_checked"] += 1
            rule = record.theorem
            environment = BackwardEnvironment(kernel.database)
            environment.assertions = {rule.name: rule}
            for tactic in environment.enumerate_tactics(
                state, max_candidates_per_variable=4, max_tactics=8, include_derived=False,
            ):
                if tactic.rule == "<ASSUMPTION>":
                    continue
                transition = environment.apply(state, tactic)
                substitution = dict(transition.resolved_substitution)
                premises = [substitute_simultaneous(h.expr, substitution) for h in rule.hypotheses]
                if not all(premise in facts for premise in premises):
                    continue
                refs = [facts[premise] for premise in premises]
                try:
                    check = kernel.compose(rule, substitution, [session.facts[ref] for ref in refs], certificate)
                except ValueError:
                    continue
                if check.conclusion == goal:
                    return AgentAction("USE", {
                        "lemma_id": record.id, "substitution": substitution, "premises": refs,
                    })
        return None

    if certificate.conclusion not in [h.expr for h in certificate.hypotheses]:
        emit(AgentAction("PROPOSE", {"formula": certificate.conclusion}))
    emit(AgentAction("READ", {"name": "__missing_teacher_scratch__"}), invalid=True)
    emit(AgentAction("BACKTRACK", {"snapshot_id": "s0"}))
    if library is not None:
        emit(AgentAction("SEARCH", {"query": certificate.conclusion.to_prefix(), "limit": 4}))
    consumed = {
        ref for action in base
        for ref in action.args.get("premises", [])
    }
    mapping: dict[str, str] = {f"h{i}": f"h{i}" for i in range(len(certificate.hypotheses))}
    saved = False
    for index, action in enumerate(base):
        args = dict(action.args)
        if "premises" in args:
            args["premises"] = [mapping[ref] for ref in args["premises"]]
        if "fact_id" in args:
            args["fact_id"] = mapping[args["fact_id"]]
        replacement = None
        if action.op == "APPLY" and library is not None:
            source_rule = kernel.database.logical_assertions[args["rule"]]
            goal = substitute_simultaneous(source_rule.conclusion, args["substitution"])
            replacement = matching_library_use(goal)
            if replacement is not None:
                emit(AgentAction("SEARCH", {"query": goal.to_prefix(), "limit": 4}))
                emit(AgentAction("READ", {"lemma_id": replacement.args["lemma_id"]}))
        observation = emit(replacement or AgentAction(action.op, args))
        if action.op != "APPLY":
            continue
        original = f"f{index}"
        mapping[original] = observation.data["fact_id"]
        if original in consumed and not saved:
            ref = observation.data["fact_id"]
            name = "teacher_local_lemma"
            emit(AgentAction("SAVE", {"fact_id": ref, "name": name}))
            emit(AgentAction("READ", {"name": name}))
            fact = session.facts[ref]
            premise_refs = []
            for hypothesis in fact.hypotheses:
                premise_ref = next((key for key, item in session.facts.items()
                                    if item.conclusion == hypothesis.expr), None)
                if premise_ref is None:
                    raise ValueError("saved local fact has an unavailable context premise")
                premise_refs.append(premise_ref)
            reused = emit(AgentAction("USE", {
                "name": name,
                "substitution": {variable: Node(variable) for variable in fact.variable_types},
                "premises": premise_refs,
            }))
            mapping[original] = reused.data["fact_id"]
            saved = True
    if session.status != "success":
        raise ValueError("augmented teacher did not prove its target")
    if stats is not None:
        stats.update(counts)
    return augmented


# Readable alias retained for corpus builders using the participle form.
augmented_teacher_actions = augment_teacher_actions

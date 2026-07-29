from __future__ import annotations

from dataclasses import dataclass

from .environment import BackwardEnvironment, ProofState, Tactic


@dataclass(frozen=True, slots=True)
class HybridGeneratorConfig:
    deterministic_candidates_per_variable: int = 8
    construction_candidates_per_variable: int = 48
    max_deterministic_tactics: int = 96
    max_construction_tactics: int = 256


@dataclass(frozen=True, slots=True)
class HybridActions:
    deterministic: tuple[Tactic, ...]
    constructions: tuple[Tactic, ...]

    @property
    def all(self) -> list[Tactic]:
        return [*self.deterministic, *self.constructions]


class HybridActionGenerator:
    """DD-like routine tactics plus neural auxiliary constructions.

    Deterministic actions only instantiate missing variables from the current
    proof state.  Construction actions additionally introduce grounded
    formulas synthesized from source theorem schemas, analogous to proposing
    an auxiliary line/cut before the symbolic engine resumes deduction.
    """

    def __init__(
        self,
        environment: BackwardEnvironment,
        config: HybridGeneratorConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or HybridGeneratorConfig()

    def actions(self, state: ProofState) -> HybridActions:
        cfg = self.config
        deterministic = self.environment.enumerate_tactics(
            state,
            max_candidates_per_variable=
                cfg.deterministic_candidates_per_variable,
            max_tactics=cfg.max_deterministic_tactics,
            include_derived=False,
        )
        # A deterministic one-step closure should never be displaced by a
        # creative proposal.
        if any(
            self.environment.apply(state, tactic).after.solved
            for tactic in deterministic
        ):
            return HybridActions(tuple(deterministic), ())
        full = self.environment.enumerate_tactics(
            state,
            max_candidates_per_variable=
                cfg.construction_candidates_per_variable,
            max_tactics=cfg.max_construction_tactics,
            include_derived=True,
        )
        known = set(deterministic)
        constructions = [
            tactic for tactic in full if tactic not in known
        ]
        return HybridActions(
            tuple(deterministic),
            tuple(constructions),
        )

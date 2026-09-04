from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from metamath_generator.model import Database, Theorem

from .environment import PROPOSE_LEMMA_RULE, ProofState
from .latent_model import (
    LatentProofTransformer,
    LatentReasoningConfig,
)
from .lemma import (
    LemmaActionGenerator,
    LemmaBackwardEnvironment,
    LemmaGeneratorConfig,
)
from .mcts import MCTSConfig, MCTSResult, ProofMCTS
from .search import HybridPolicy, TransformerPolicy
from .tokenizer import MetamathTokenizer
from .data_contract import tokenizer_fingerprint, validate_checkpoint_tokenizer


@dataclass(frozen=True, slots=True)
class DecomposedProverConfig:
    simulations: int = 100
    max_depth: int = 20
    branching: int = 24
    seed: int = 7
    device: str = "auto"


def initialize_latent_checkpoint(
    base_checkpoint: str | Path,
    base_tokenizer_path: str | Path,
    output_checkpoint: str | Path,
    output_tokenizer_path: str | Path,
    reasoning_config: LatentReasoningConfig | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Create a latent model and append lemma tokens compatibly."""

    tokenizer = MetamathTokenizer.load(base_tokenizer_path)
    from .model import ProofTransformer
    base, payload = ProofTransformer.load_checkpoint(base_checkpoint, map_location=map_location)
    validate_checkpoint_tokenizer(base, payload, tokenizer)
    del base, payload
    if Path(base_checkpoint).resolve() == Path(output_checkpoint).resolve() or Path(base_tokenizer_path).resolve() == Path(output_tokenizer_path).resolve():
        raise ValueError("choose new output paths for the latent checkpoint and tokenizer")
    upgraded = tokenizer.upgraded_for_lemma_actions()
    model, upgrade = LatentProofTransformer.from_base_checkpoint(
        base_checkpoint,
        upgraded,
        reasoning_config,
        map_location=map_location,
    )
    checkpoint_path = Path(output_checkpoint)
    tokenizer_path = Path(output_tokenizer_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    upgraded.save(tokenizer_path)
    model.save_checkpoint(
        checkpoint_path,
        metadata={
            "initialization": upgrade["vocabulary_expansion"],
            "tokenizer_sha256": tokenizer_fingerprint(upgraded),
        },
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "tokenizer": str(tokenizer_path.resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "vocabulary_expansion": upgrade["vocabulary_expansion"],
        "reasoning": asdict(model.reasoning_config),
        "parameter_count": model.parameter_count(),
    }


class DecomposedProver:
    """Closed-loop prover with lemma decomposition and latent thought."""

    def __init__(
        self,
        database: Database,
        model: LatentProofTransformer,
        tokenizer: MetamathTokenizer,
        config: DecomposedProverConfig | None = None,
        *,
        excluded_assertions: tuple[str, ...] = (),
        lemma_config: LemmaGeneratorConfig | None = None,
    ) -> None:
        self.config = config or DecomposedProverConfig()
        device = (
            "cuda"
            if self.config.device == "auto" and torch.cuda.is_available()
            else (
                "cpu"
                if self.config.device == "auto"
                else self.config.device
            )
        )
        self.device = torch.device(device)
        self.environment = LemmaBackwardEnvironment(
            database,
            excluded_assertions=excluded_assertions,
        )
        self.generator = LemmaActionGenerator(
            self.environment, lemma_config
        )
        self.neural = TransformerPolicy(
            model, tokenizer, self.environment, self.device
        )
        self.policy = HybridPolicy(self.environment, self.neural)
        self.search = ProofMCTS(
            self.environment,
            self.generator,
            self.policy,
            MCTSConfig(
                simulations=self.config.simulations,
                max_depth=self.config.max_depth,
                branching=self.config.branching,
                seed=self.config.seed,
            ),
        )

    def prove(self, theorem: Theorem) -> MCTSResult:
        return self.search.prove(ProofState.from_theorem(theorem))

    def metrics(self, result: MCTSResult) -> dict:
        lemma_actions = [
            tactic
            for tactic in result.search.actions
            if tactic.rule == PROPOSE_LEMMA_RULE
        ]
        return {
            "device": str(self.device),
            "solved": result.search.solved,
            "outcome": result.outcome,
            "simulations": result.search.simulations,
            "nodes": result.search.nodes,
            "lemma_actions": len(lemma_actions),
            "environment": self.environment.metrics.to_record(),
            "policy": self.neural.metrics(),
        }

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import ProofTransformer, ProofTransformerConfig
from .tokenizer import MetamathTokenizer


@dataclass(frozen=True, slots=True)
class LatentReasoningConfig:
    """Configuration for an implicit continuous chain of thought."""

    max_thought_steps: int = 6
    min_thought_steps: int = 1
    halt_threshold: float = 0.85
    ponder_cost: float = 1.0e-3
    lemma_token_id: int | None = None

    def validate(self) -> None:
        if self.max_thought_steps <= 0:
            raise ValueError("max_thought_steps must be positive")
        if not 1 <= self.min_thought_steps <= self.max_thought_steps:
            raise ValueError(
                "min_thought_steps must be within the thought budget"
            )
        if not 0.0 < self.halt_threshold <= 1.0:
            raise ValueError("halt_threshold must be in (0, 1]")
        if self.ponder_cost < 0.0:
            raise ValueError("ponder_cost must be non-negative")


class LatentProofTransformer(ProofTransformer):
    """Proof Transformer with recurrent, non-linguistic latent reasoning.

    The recurrent states are continuous vectors.  They are never emitted as
    tokens and never mutate a formal proof state.  After reasoning halts, the
    model scores only discrete actions that the symbolic kernel has already
    constructed and checked, including intermediate-lemma proposals.
    """

    def __init__(
        self,
        config: ProofTransformerConfig,
        reasoning_config: LatentReasoningConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.reasoning_config = (
            reasoning_config or LatentReasoningConfig()
        )
        self.reasoning_config.validate()
        width = config.d_model
        self.thought_context = nn.Linear(width, width)
        self.thought_recurrent = nn.Linear(width, width, bias=False)
        self.thought_gate = nn.Linear(width * 2, width)
        self.thought_norm = nn.LayerNorm(width)
        self.halt_head = nn.Linear(width, 1)
        self.thought_to_memory = nn.Linear(width, width)
        self.lemma_gate = nn.Linear(width, 1)
        for module in (
            self.thought_context,
            self.thought_recurrent,
            self.thought_gate,
            self.halt_head,
            self.thought_to_memory,
            self.lemma_gate,
        ):
            module.apply(self._initialize)
        self.last_reasoning_steps = 0
        self.last_halt_probability = 0.0
        self._last_ponder_loss: Tensor | None = None

    @staticmethod
    def _pool(memory: Tensor, padding: Tensor) -> Tensor:
        weights = (~padding).unsqueeze(-1).to(memory.dtype)
        return (memory * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def _reason(self, context: Tensor) -> tuple[Tensor, Tensor]:
        thought = context
        halt_probabilities: list[Tensor] = []
        mixed = torch.zeros_like(context)
        survival = torch.ones(
            context.shape[0], device=context.device, dtype=context.dtype
        )
        expected_steps = torch.zeros_like(survival)
        steps = 0
        cfg = self.reasoning_config
        for index in range(cfg.max_thought_steps):
            proposal = torch.tanh(
                self.thought_context(context)
                + self.thought_recurrent(thought)
            )
            gate = torch.sigmoid(
                self.thought_gate(torch.cat((context, thought), dim=-1))
            )
            thought = self.thought_norm(
                gate * proposal + (1.0 - gate) * thought
            )
            halt = torch.sigmoid(self.halt_head(thought)).squeeze(-1)
            halt_probabilities.append(halt)
            steps = index + 1
            allowed_halt = (
                halt if steps >= cfg.min_thought_steps
                else torch.zeros_like(halt)
            )
            halt_weight = survival * allowed_halt
            mixed = mixed + halt_weight.unsqueeze(-1) * thought
            expected_steps = expected_steps + halt_weight * steps
            survival = survival * (1.0 - allowed_halt)
            if (
                not self.training
                and steps >= cfg.min_thought_steps
                and bool(torch.all(
                    1.0 - survival >= cfg.halt_threshold
                ).item())
            ):
                break
        # Any probability mass that has not halted acts at the hard budget.
        # The weighted state makes the stop head receive task gradients, while
        # the ponder term mildly prefers solutions using fewer recurrent steps.
        mixed = mixed + survival.unsqueeze(-1) * thought
        expected_steps = expected_steps + survival * steps
        stacked = torch.stack(halt_probabilities, dim=1)
        self.last_reasoning_steps = steps
        self.last_halt_probability = float(
            (1.0 - survival).mean().detach().item()
        )
        # Penalize expected internal compute, never formal proof length.
        self._last_ponder_loss = expected_steps.mean() * cfg.ponder_cost
        return mixed, stacked

    def encode(self, state_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        memory, padding, _ = super().encode(state_ids)
        context = self._pool(memory, padding)
        thought, _ = self._reason(context)
        refined = memory + self.thought_to_memory(thought).unsqueeze(1)
        value = torch.sigmoid(self.value_head(thought).squeeze(-1))
        return refined, padding, value

    def score_candidates_from_memory(
        self,
        candidate_ids: Tensor,
        memory: Tensor,
        padding: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scores, value = super().score_candidates_from_memory(
            candidate_ids, memory, padding, value
        )
        lemma_token_id = self.reasoning_config.lemma_token_id
        if lemma_token_id is None:
            return scores, value
        state = self._pool(memory, padding)
        gate = self.lemma_gate(state).squeeze(-1)[0]
        is_lemma = candidate_ids.eq(lemma_token_id).any(dim=1)
        mode_log_prior = torch.where(
            is_lemma,
            F.logsigmoid(gate),
            F.logsigmoid(-gate),
        )
        return scores + mode_log_prior, value

    def ponder_loss(self) -> Tensor:
        if self._last_ponder_loss is None:
            return next(self.parameters()).sum() * 0.0
        return self._last_ponder_loss

    def reasoning_metrics(self) -> dict[str, int | float | str]:
        return {
            "reasoning_mode": "continuous_latent",
            "thought_steps": self.last_reasoning_steps,
            "halt_probability": self.last_halt_probability,
            "max_thought_steps": (
                self.reasoning_config.max_thought_steps
            ),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer_state: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        checkpoint_metadata = dict(metadata or {})
        checkpoint_metadata["candidate_policy"] = {
            "trained": bool(self.candidate_head_trained),
            "mode": "finite-kernel-candidate-index",
        }
        checkpoint_metadata["latent_reasoning"] = {
            "mode": "continuous-vector-chain",
            "formal_state_mutation": False,
            "configuration": asdict(self.reasoning_config),
        }
        torch.save({
            "format": "peano-latent-proof-transformer-v1",
            "config": asdict(self.config),
            "reasoning_config": asdict(self.reasoning_config),
            "model_state": self.state_dict(),
            "optimizer_state": optimizer_state,
            "metadata": checkpoint_metadata,
        }, Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["LatentProofTransformer", dict]:
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=False
        )
        if payload.get("format") != "peano-latent-proof-transformer-v1":
            raise ValueError("unsupported latent proof checkpoint")
        model = cls(
            ProofTransformerConfig(**payload["config"]),
            LatentReasoningConfig(**payload["reasoning_config"]),
        )
        model.load_state_dict(payload["model_state"], strict=True)
        model.candidate_head_trained = bool(
            payload.get("metadata", {})
            .get("candidate_policy", {})
            .get("trained", False)
        )
        return model, payload

    @classmethod
    def from_base_checkpoint(
        cls,
        checkpoint: str | Path,
        tokenizer: MetamathTokenizer,
        reasoning_config: LatentReasoningConfig | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["LatentProofTransformer", dict]:
        """Upgrade a v1 model without changing any existing token ID."""

        if not tokenizer.supports_lemma_actions:
            raise ValueError("tokenizer must first be upgraded for lemmas")
        base, payload = ProofTransformer.load_checkpoint(
            checkpoint, map_location=map_location
        )
        if len(tokenizer) < base.config.vocab_size:
            raise ValueError("upgraded tokenizer is smaller than checkpoint")
        lemma_token_id = tokenizer.token_to_id["<PROPOSE_LEMMA>"]
        latent_cfg = reasoning_config or LatentReasoningConfig(
            lemma_token_id=lemma_token_id
        )
        if latent_cfg.lemma_token_id is None:
            latent_cfg = replace(
                latent_cfg, lemma_token_id=lemma_token_id
            )
        config = replace(base.config, vocab_size=len(tokenizer))
        model = cls(config, latent_cfg)
        source = base.state_dict()
        target = model.state_dict()
        with torch.no_grad():
            for name, value in source.items():
                if name not in target:
                    continue
                if target[name].shape == value.shape:
                    target[name].copy_(value)
                    continue
                if name in {
                    "token_embedding.weight",
                    "policy_head.weight",
                    "policy_head.bias",
                } and target[name].shape[0] >= value.shape[0]:
                    target[name][:value.shape[0]].copy_(value)
                    continue
                raise ValueError(
                    f"cannot upgrade checkpoint tensor {name}: "
                    f"{tuple(value.shape)} -> {tuple(target[name].shape)}"
                )
        model.load_state_dict(target, strict=True)
        model.candidate_head_trained = base.candidate_head_trained
        return model, {
            "base_checkpoint": str(Path(checkpoint).resolve()),
            "base_payload": payload,
            "vocabulary_expansion": {
                "old_size": base.config.vocab_size,
                "new_size": len(tokenizer),
                "existing_token_ids_preserved": True,
            },
        }

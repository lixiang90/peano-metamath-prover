from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True, slots=True)
class ProofTransformerConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_state_tokens: int = 256
    max_action_tokens: int = 192
    gradient_checkpointing: bool = False


class ProofTransformer(nn.Module):
    """Encoder-decoder tactic policy with a shared proof-state value head."""

    def __init__(self, config: ProofTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.state_position = nn.Embedding(
            config.max_state_tokens,
            config.d_model,
        )
        self.action_position = nn.Embedding(
            config.max_action_tokens,
            config.d_model,
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.policy_head = nn.Linear(config.d_model, config.vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        # A candidate-index head chooses among symbolically enumerated tactics.
        # It never generates a substitution token-by-token.  Existing v1
        # checkpoints can load without these weights and keep using the
        # autoregressive scorer until MCTS replay trains the head.
        self.candidate_state = nn.Linear(config.d_model, config.d_model)
        self.candidate_action = nn.Linear(config.d_model, config.d_model)
        self.candidate_score = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.candidate_head_trained = False
        self.embedding_scale = math.sqrt(config.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def _embed(
        self,
        token_ids: Tensor,
        positions: nn.Embedding,
    ) -> Tensor:
        length = token_ids.shape[1]
        if length > positions.num_embeddings:
            raise ValueError(
                f"sequence length {length} exceeds configured maximum "
                f"{positions.num_embeddings}"
            )
        indices = torch.arange(length, device=token_ids.device)
        return (
            self.token_embedding(token_ids) * self.embedding_scale
            + positions(indices).unsqueeze(0)
        )

    def encode(self, state_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        padding = state_ids.eq(self.config.pad_id)
        state = self._embed(state_ids, self.state_position)
        if self.config.gradient_checkpointing and self.training:
            memory = state
            for layer in self.transformer.encoder.layers:
                memory = checkpoint(
                    lambda hidden, module=layer: module(
                        hidden,
                        src_key_padding_mask=padding,
                    ),
                    memory,
                    use_reentrant=False,
                )
            if self.transformer.encoder.norm is not None:
                memory = self.transformer.encoder.norm(memory)
        else:
            memory = self.transformer.encoder(
                state,
                src_key_padding_mask=padding,
            )
        weights = (~padding).unsqueeze(-1).to(memory.dtype)
        pooled = (memory * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        value = torch.sigmoid(self.value_head(pooled).squeeze(-1))
        return memory, padding, value

    def score_candidates(
        self,
        state_ids: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score a finite candidate set and return index logits plus value.

        ``state_ids`` must contain one proof state and ``candidate_ids`` has
        shape ``[candidate, token]``.  Candidate tokens are only an encoding
        of already kernel-valid structured tactics; no token is generated.
        """

        if state_ids.shape[0] != 1:
            raise ValueError("candidate scoring currently accepts one state")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] == 0:
            raise ValueError("candidate_ids must contain at least one action")
        memory, padding, value = self.encode(state_ids)
        return self.score_candidates_from_memory(
            candidate_ids, memory, padding, value
        )

    def score_candidates_from_memory(
        self,
        candidate_ids: Tensor,
        memory: Tensor,
        padding: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score candidates using a state encoding already computed once."""

        state_weights = (~padding).unsqueeze(-1).to(memory.dtype)
        state_pooled = (memory * state_weights).sum(dim=1) / (
            state_weights.sum(dim=1).clamp_min(1.0)
        )
        candidate_padding = candidate_ids.eq(self.config.pad_id)
        candidate = self._embed(candidate_ids, self.action_position)
        candidate_weights = (~candidate_padding).unsqueeze(-1).to(
            candidate.dtype
        )
        candidate_pooled = (
            (candidate * candidate_weights).sum(dim=1)
            / candidate_weights.sum(dim=1).clamp_min(1.0)
        )
        hidden = (
            self.candidate_state(state_pooled).expand_as(candidate_pooled)
            + self.candidate_action(candidate_pooled)
        )
        return self.candidate_score(hidden).squeeze(-1), value

    def score_candidate_matrix(
        self,
        state_ids: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score each batch state against every batch candidate action."""

        memory, padding, value = self.encode(state_ids)
        state_weights = (~padding).unsqueeze(-1).to(memory.dtype)
        state_pooled = (memory * state_weights).sum(dim=1) / (
            state_weights.sum(dim=1).clamp_min(1.0)
        )
        candidate_padding = candidate_ids.eq(self.config.pad_id)
        candidate = self._embed(candidate_ids, self.action_position)
        candidate_weights = (~candidate_padding).unsqueeze(-1).to(
            candidate.dtype
        )
        candidate_pooled = (
            (candidate * candidate_weights).sum(dim=1)
            / candidate_weights.sum(dim=1).clamp_min(1.0)
        )
        hidden = (
            self.candidate_state(state_pooled)[:, None, :]
            + self.candidate_action(candidate_pooled)[None, :, :]
        )
        return self.candidate_score(hidden).squeeze(-1), value

    def resize_vocabulary(self, vocabulary_size: int) -> dict[str, int]:
        """Append randomly initialized vocabulary rows in-place."""

        old_size = self.config.vocab_size
        if vocabulary_size < old_size:
            raise ValueError("vocabulary shrinking is not supported")
        if vocabulary_size == old_size:
            return {"old_size": old_size, "new_size": old_size, "added": 0}
        device = self.token_embedding.weight.device
        dtype = self.token_embedding.weight.dtype
        embedding = nn.Embedding(
            vocabulary_size,
            self.config.d_model,
            padding_idx=self.config.pad_id,
            device=device,
            dtype=dtype,
        )
        policy = nn.Linear(
            self.config.d_model,
            vocabulary_size,
            device=device,
            dtype=dtype,
        )
        self._initialize(embedding)
        self._initialize(policy)
        with torch.no_grad():
            embedding.weight[:old_size].copy_(self.token_embedding.weight)
            policy.weight[:old_size].copy_(self.policy_head.weight)
            policy.bias[:old_size].copy_(self.policy_head.bias)
        self.token_embedding = embedding
        self.policy_head = policy
        self.config = replace(self.config, vocab_size=vocabulary_size)
        return {
            "old_size": old_size,
            "new_size": vocabulary_size,
            "added": vocabulary_size - old_size,
        }

    def decode(
        self,
        action_input_ids: Tensor,
        memory: Tensor,
        memory_padding: Tensor,
    ) -> Tensor:
        action_padding = action_input_ids.eq(self.config.pad_id)
        action = self._embed(action_input_ids, self.action_position)
        causal_mask = self._causal_mask(
            action_input_ids.shape[1],
            action_input_ids.device,
        )
        if self.config.gradient_checkpointing and self.training:
            decoded = action
            for layer in self.transformer.decoder.layers:
                decoded = checkpoint(
                    lambda hidden, encoder_memory, module=layer: module(
                        hidden,
                        encoder_memory,
                        tgt_mask=causal_mask,
                        tgt_key_padding_mask=action_padding,
                        memory_key_padding_mask=memory_padding,
                    ),
                    decoded,
                    memory,
                    use_reentrant=False,
                )
            if self.transformer.decoder.norm is not None:
                decoded = self.transformer.decoder.norm(decoded)
        else:
            decoded = self.transformer.decoder(
                action,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=action_padding,
                memory_key_padding_mask=memory_padding,
            )
        return self.policy_head(decoded)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        state_ids: Tensor,
        action_input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        memory, memory_padding, value = self.encode(state_ids)
        logits = self.decode(
            action_input_ids,
            memory,
            memory_padding,
        )
        return logits, value

    @torch.no_grad()
    def beam_search(
        self,
        state_ids: Tensor,
        *,
        beam_size: int = 4,
        max_new_tokens: int | None = None,
        length_penalty: float = 0.7,
    ) -> tuple[list[list[int]], list[float], float]:
        if state_ids.shape[0] != 1:
            raise ValueError("beam_search currently accepts one state")
        self.eval()
        limit = max_new_tokens or self.config.max_action_tokens - 1
        memory, memory_padding, value = self.encode(state_ids)
        beams: list[tuple[list[int], float, bool]] = [
            ([self.config.bos_id], 0.0, False)
        ]
        for _ in range(limit):
            candidates: list[tuple[list[int], float, bool]] = []
            for sequence, score, finished in beams:
                if finished:
                    candidates.append((sequence, score, True))
                    continue
                decoder_ids = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=state_ids.device,
                )
                logits = self.decode(
                    decoder_ids,
                    memory,
                    memory_padding,
                )[0, -1]
                log_probs = F.log_softmax(logits, dim=-1)
                values, indices = torch.topk(log_probs, beam_size)
                for token_score, token_id in zip(
                    values.tolist(),
                    indices.tolist(),
                ):
                    extended = [*sequence, int(token_id)]
                    candidates.append((
                        extended,
                        score + float(token_score),
                        token_id == self.config.eos_id,
                    ))

            def rank(item: tuple[list[int], float, bool]) -> float:
                sequence, score, _ = item
                length = max(1, len(sequence) - 1)
                return score / (length ** length_penalty)

            beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
            if all(finished for _, _, finished in beams):
                break
        return (
            [sequence for sequence, _, _ in beams],
            [score for _, score, _ in beams],
            float(value.item()),
        )

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
        torch.save(
            {
                "format": "peano-proof-transformer-v1",
                "config": asdict(self.config),
                "model_state": self.state_dict(),
                "optimizer_state": optimizer_state,
                "metadata": checkpoint_metadata,
            },
            Path(path),
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["ProofTransformer", dict]:
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        if payload.get("format") != "peano-proof-transformer-v1":
            raise ValueError("unsupported proof Transformer checkpoint")
        model = cls(ProofTransformerConfig(**payload["config"]))
        missing, unexpected = model.load_state_dict(
            payload["model_state"], strict=False
        )
        allowed_missing = {
            name
            for name in model.state_dict()
            if name.startswith("candidate_")
        }
        if unexpected or set(missing) - allowed_missing:
            raise ValueError(
                "checkpoint state is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.candidate_head_trained = bool(
            payload.get("metadata", {})
            .get("candidate_policy", {})
            .get("trained", False)
        )
        return model, payload

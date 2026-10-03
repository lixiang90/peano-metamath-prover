"""Two causal decoders with explicit, position-preserving lemma memory slots.

Memory interface: reserve ``config.lemma_slots`` consecutive MEM tokens in the
main sequence and set ``memories[(batch_index, first_slot)] = lemma_token_ids``.
The lemma decoder appends learned summary slots after the lemma. Their causal
hidden states replace the reserved main embeddings in occurrence order. There
is no length expansion, cross-attention, encoder-decoder, or latent recurrence.
Logits and labels retain exactly the input shape; all MEM/PAD/BOS targets are
ignored. The final memory position may predict the first following text token.

For ordinary language-model training pass ``labels=input_ids``. Agent SFT may
replace prompt labels with -100. Labels are always unshifted: the model performs
the conventional next-token shift internally. ``labels=None`` requests logits
without a loss. Memory content must be available at its occurrence; callers
must not insert future solutions into an earlier memory slot.

Version 1 encodes only the complete lemma assertion (statement, premises,
variable types and disjoint-variable constraints), serialized by the caller.
It never reads or recursively encodes a proof DAG or dependent lemma memories.
Both decoders execute a fixed stack once, and max_seq_len bounds every input;
neither nested dependencies nor runtime loops can expand computation silently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import BOS_ID, MEM_ID, PAD_ID, VOCAB_SIZE, ByteTokenizer


ARCHITECTURE = "causal-rope-rmsnorm-swiglu-lemma-slots-v1"
CHECKPOINT_FORMAT = "pa-prover-v2-causal-memory-v1"
MemoryMap = Mapping[tuple[int, int], Sequence[int]]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_multiplier: float = 8 / 3
    max_seq_len: int = 2048
    lemma_layers: int = 2
    lemma_slots: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError("vocab_size must match the fixed 260-token byte vocabulary")
        for name in ("d_model", "n_heads", "n_layers", "max_seq_len", "lemma_layers", "lemma_slots"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer for RoPE")
        if self.lemma_slots >= self.max_seq_len:
            raise ValueError("max_seq_len must leave room for lemma text and summary slots")
        if not math.isfinite(self.ffn_multiplier) or self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be finite and positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(hidden.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inverse, persistent=False)

    def forward(self, hidden: Tensor, positions: Tensor) -> Tensor:
        # hidden: [batch, heads, time, width], positions: [batch, time].
        angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
        cosine = angles.cos().unsqueeze(1).to(hidden.dtype)
        sine = angles.sin().unsqueeze(1).to(hidden.dtype)
        even, odd = hidden[..., 0::2], hidden[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine,
                               even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden: Tensor, valid: Tensor, positions: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = self.rope(query.transpose(1, 2), positions)
        key = self.rope(key.transpose(1, 2), positions)
        value = value.transpose(1, 2)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended) * valid.unsqueeze(-1).to(hidden.dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        inner = max(1, int(config.d_model * config.ffn_multiplier))
        self.gate = nn.Linear(config.d_model, inner, bias=False)
        self.up = nn.Linear(config.d_model, inner, bias=False)
        self.down = nn.Linear(inner, config.d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, valid: Tensor, positions: Tensor) -> Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), valid, positions))
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden * valid.unsqueeze(-1).to(hidden.dtype)


class CausalDecoder(nn.Module):
    def __init__(self, config: ModelConfig, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(layers))
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, hidden: Tensor, valid: Tensor) -> Tensor:
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        for layer in self.layers:
            hidden = layer(hidden, valid, positions)
        return self.norm(hidden)


class LemmaDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.decoder = CausalDecoder(config, config.n_layers)
        self.lemma_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.lemma_decoder = CausalDecoder(config, config.lemma_layers)
        self.summary_slots = nn.Parameter(torch.empty(config.lemma_slots, config.d_model))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.summary_slots, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode_lemmas(self, lemmas: Sequence[Sequence[int]]) -> Tensor:
        """Return [lemma, slot, width]; gradients reach both lemma and main models."""
        if not lemmas:
            return self.summary_slots.new_empty((0, self.config.lemma_slots, self.config.d_model))
        device = self.summary_slots.device
        lengths = [len(lemma) for lemma in lemmas]
        if min(lengths) == 0 or max(lengths) + self.config.lemma_slots > self.config.max_seq_len:
            raise ValueError("lemma length plus summary slots must be within max_seq_len")
        width = max(lengths) + self.config.lemma_slots
        ids = torch.full((len(lemmas), width), PAD_ID, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row, lemma in enumerate(lemmas):
            tokens = torch.as_tensor(lemma, dtype=torch.long, device=device)
            if tokens.ndim != 1 or bool(((tokens <= PAD_ID) | (tokens >= VOCAB_SIZE) | (tokens == MEM_ID)).any()):
                raise ValueError("lemma tokens must be non-padding byte/BOS/EOS IDs without MEM")
            ids[row, :lengths[row]] = tokens
            valid[row, :lengths[row] + self.config.lemma_slots] = True
        embedded = self.lemma_embedding(ids)
        positions = torch.tensor(lengths, device=device)[:, None] + torch.arange(self.config.lemma_slots, device=device)
        rows = torch.arange(len(lemmas), device=device)[:, None]
        embedded[rows, positions] = self.summary_slots.unsqueeze(0)
        encoded = self.lemma_decoder(embedded, valid)
        return encoded[rows, positions]

    def _prepare_embeddings(self, input_ids: Tensor, valid: Tensor, memories: MemoryMap | None) -> Tensor:
        embedded = self.token_embedding(input_ids)
        covered = torch.zeros_like(input_ids, dtype=torch.bool)
        entries = sorted((memories or {}).items())
        slots = self.config.lemma_slots
        for (batch, start), _ in entries:
            if not (0 <= batch < input_ids.shape[0] and 0 <= start and start + slots <= input_ids.shape[1]):
                raise ValueError("memory slot coordinates outside input sequence")
            block = input_ids[batch, start:start + slots]
            if not bool((block == MEM_ID).all()) or not bool(valid[batch, start:start + slots].all()):
                raise ValueError("each memory must occupy lemma_slots consecutive unmasked MEM tokens")
            if bool(covered[batch, start:start + slots].any()):
                raise ValueError("memory slot ranges must not overlap")
            covered[batch, start:start + slots] = True
        if not torch.equal(covered, input_ids.eq(MEM_ID)):
            raise ValueError("every MEM token must be covered by exactly one lemma memory")
        if entries:
            summaries = self.encode_lemmas([lemma for _, lemma in entries])
            for row, ((batch, start), _) in enumerate(entries):
                embedded[batch, start:start + slots] = summaries[row]
        return embedded

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if input_ids.ndim != 2 or not input_ids.shape[0] or not input_ids.shape[1]:
            raise ValueError("input_ids must be a nonempty [batch, time] tensor")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("input token outside fixed byte vocabulary")
        valid = input_ids.ne(PAD_ID)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            valid = valid & attention_mask.to(device=input_ids.device, dtype=torch.bool)
        return valid

    @staticmethod
    def _targets(input_ids: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
        if labels.shape != input_ids.shape or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("labels must be integer token IDs with the input shape")
        labels = labels.to(device=input_ids.device, dtype=torch.long).clone()
        if bool(((labels < 0) & labels.ne(-100) | (labels >= VOCAB_SIZE)).any()):
            raise ValueError("labels must be byte vocabulary IDs or -100")
        ignore = ~valid | input_ids.eq(MEM_ID) | input_ids.eq(BOS_ID)
        ignore |= labels.eq(PAD_ID) | labels.eq(MEM_ID) | labels.eq(BOS_ID)
        labels.masked_fill_(ignore, -100)
        shifted = labels[:, 1:].clone()
        # A padding query cannot predict the first valid token after left pads.
        shifted.masked_fill_(~valid[:, :-1], -100)
        return shifted

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                labels: Tensor | None = None, memories: MemoryMap | None = None) -> ModelOutput:
        valid = self._validate_inputs(input_ids, attention_mask)
        hidden = self._prepare_embeddings(input_ids, valid, memories)
        logits = self.lm_head(self.decoder(hidden, valid))
        loss = None
        if labels is not None:
            targets = self._targets(input_ids, labels, valid)
            if bool(targets.ne(-100).any()):
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                                       targets.reshape(-1), ignore_index=-100)
            else:
                loss = logits.sum() * 0.0
        return ModelOutput(logits, loss)

    def sequence_log_probs(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                           labels: Tensor | None = None, memories: MemoryMap | None = None,
                           *, reduction: str = "sum") -> Tensor:
        """Differentiable selected-token log probabilities, excluding ignored targets.

        sum/mean return [batch]; none returns [batch, time-1] with ignored values
        zeroed. In RLVR supply action-only labels to exclude prompt likelihoods.
        """
        valid = self._validate_inputs(input_ids, attention_mask)
        targets = self._targets(input_ids, input_ids if labels is None else labels, valid)
        logits = self(input_ids, attention_mask=attention_mask, memories=memories).logits
        active = targets.ne(-100)
        selected = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(~active, 0.0)
        if reduction == "none":
            return selected
        if reduction == "sum":
            return selected.sum(-1)
        if reduction == "mean":
            return selected.sum(-1) / active.sum(-1).clamp_min(1)
        raise ValueError("reduction must be sum, mean, or none")


def sequence_log_probs(model: LemmaDecoderLM, input_ids: Tensor, **kwargs) -> Tensor:
    return model.sequence_log_probs(input_ids, **kwargs)


def save_checkpoint(path: str | Path, model: LemmaDecoderLM, metadata: dict | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    tokenizer = ByteTokenizer()
    torch.save({
        "format": CHECKPOINT_FORMAT, "architecture": ARCHITECTURE,
        "config": asdict(model.config), "tokenizer": tokenizer.specification(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "model_state": model.state_dict(), "metadata": dict(metadata or {}),
    }, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[LemmaDecoderLM, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    tokenizer = ByteTokenizer()
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported V2 architecture/checkpoint format")
    if payload.get("tokenizer") != tokenizer.specification() or payload.get("tokenizer_fingerprint") != tokenizer.fingerprint():
        raise ValueError("checkpoint tokenizer contract does not match fixed UTF-8 bytes")
    config = payload.get("config", {})
    if set(config) != {field.name for field in fields(ModelConfig)}:
        raise ValueError("checkpoint model configuration fields do not match architecture")
    model = LemmaDecoderLM(ModelConfig(**config)).to(device)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"checkpoint model weights do not match architecture: {error}") from error
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("checkpoint metadata must be an object")
    return model, payload["metadata"]

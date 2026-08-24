from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from .latent_model import LatentProofTransformer


@dataclass(frozen=True, slots=True)
class LatentLossConfig:
    value_weight: float = 0.25


def latent_policy_value_loss(
    model: LatentProofTransformer,
    state_ids: Tensor,
    action_ids: Tensor,
    value_targets: Tensor,
    config: LatentLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Differentiable SFT/RL warm-start loss including the stop head.

    Policy targets remain formal action tokens.  Continuous thought vectors
    are trained only through policy/value gradients and the small ponder cost;
    no textual chain-of-thought target is introduced.
    """

    cfg = config or LatentLossConfig()
    logits, values = model(state_ids, action_ids[:, :-1])
    targets = action_ids[:, 1:]
    policy = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=model.config.pad_id,
    )
    value = F.mse_loss(values, value_targets)
    ponder = model.ponder_loss()
    total = policy + cfg.value_weight * value + ponder
    return total, {
        "policy_loss": policy,
        "value_loss": value,
        "ponder_loss": ponder,
    }

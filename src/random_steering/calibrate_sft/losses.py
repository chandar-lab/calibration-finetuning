from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from random_steering.calibrate_sft.data import PrefixTarget


def masked_soft_target_kl(
    logits: torch.Tensor,
    next_token_ids: Sequence[int],
    target_probs: Sequence[float],
    *,
    tau: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    token_ids = torch.as_tensor(list(next_token_ids), dtype=torch.long, device=logits.device)
    target = torch.as_tensor(list(target_probs), dtype=logits.dtype, device=logits.device)
    target = torch.clamp(target, min=float(epsilon))
    target = target / target.sum()

    full_log_probs = F.log_softmax(logits / float(tau), dim=-1)
    selected_log_probs = full_log_probs.index_select(dim=-1, index=token_ids)
    return torch.sum(target * (torch.log(target) - selected_log_probs))


def sequence_calibration_loss(
    step_logits: Sequence[torch.Tensor],
    prefix_targets: Sequence[PrefixTarget],
    *,
    tau: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if len(step_logits) != len(prefix_targets):
        raise ValueError("step_logits and prefix_targets must have the same length.")
    losses = [
        masked_soft_target_kl(
            logits,
            prefix_target.next_token_ids,
            prefix_target.next_token_probs,
            tau=tau,
            epsilon=epsilon,
        )
        for logits, prefix_target in zip(step_logits, prefix_targets, strict=True)
    ]
    if not losses:
        raise ValueError("At least one prefix target is required to compute calibration loss.")
    return torch.stack(losses).mean()

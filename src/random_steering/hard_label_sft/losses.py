from __future__ import annotations

import torch
import torch.nn.functional as F


def completion_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab_size].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq_len].")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions.")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

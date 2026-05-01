from __future__ import annotations

from collections import Counter
import re

import torch

from random_steering.inference.chat_format import strip_thinking_trace

_WHITESPACE_PATTERN = re.compile(r"\s+")
_THINK_TAG_PATTERN = re.compile(r"</?think>", re.IGNORECASE)


def top_p_support_size(probs: torch.Tensor, threshold: float) -> int:
    return top_p_support_sizes(probs, [threshold])[float(threshold)]


def top_p_support_sizes(probs: torch.Tensor, thresholds: list[float]) -> dict[float, int]:
    if probs.ndim != 1:
        raise ValueError("top_p_support_size expects a 1D probability tensor")
    for threshold in thresholds:
        if not 0.0 < float(threshold) <= 1.0:
            raise ValueError(f"top_p threshold must be in (0, 1], got {threshold}")

    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    support_sizes: dict[float, int] = {}
    for threshold in thresholds:
        indices = torch.nonzero(cumulative >= float(threshold), as_tuple=False)
        if indices.numel() == 0:
            support_sizes[float(threshold)] = int(sorted_probs.numel())
        else:
            support_sizes[float(threshold)] = int(indices[0].item()) + 1
    return support_sizes


def normalize_open_random_output(text: str) -> str:
    stripped = strip_thinking_trace(text)
    stripped = _THINK_TAG_PATTERN.sub("", stripped).strip()
    if not stripped:
        return ""

    first_non_empty_line = ""
    for line in stripped.splitlines():
        candidate = _THINK_TAG_PATTERN.sub("", line).strip()
        if candidate:
            first_non_empty_line = candidate
            break
    if not first_non_empty_line:
        return ""

    normalized = _WHITESPACE_PATTERN.sub(" ", first_non_empty_line)
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1].strip()
    return normalized


def has_thinking_tags(text: str) -> bool:
    return _THINK_TAG_PATTERN.search(text) is not None


def count_unique_outputs(outputs: list[str]) -> tuple[int, float, dict[str, int]]:
    non_empty_outputs = [output for output in outputs if output]
    counts = dict(Counter(non_empty_outputs))
    num_unique = len(counts)
    unique_fraction = float(num_unique / len(outputs)) if outputs else 0.0
    return num_unique, unique_fraction, counts

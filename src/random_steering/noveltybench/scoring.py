from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from random_steering.utils.hf import ensure_hf_home


REWARD_THRESHOLDS = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]


def transform_raw_reward(reward: float) -> int:
    return bisect.bisect_left(REWARD_THRESHOLDS, reward) + 1


def _conversation_text(tokenizer: Any, prompt: str, generation: str) -> str:
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generation},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return str(tokenizer.apply_chat_template(conversation, tokenize=False))
        except TypeError:
            pass
    return f"User:\n{prompt}\n\nAssistant:\n{generation}"


@dataclass
class RewardModelScorer:
    tokenizer: Any
    model: Any
    batch_size: int

    @classmethod
    def from_config(cls, reward_cfg: Any) -> "RewardModelScorer":
        ensure_hf_home()
        local_files_only = bool(getattr(reward_cfg, "local_files_only", False))
        checkpoint = str(reward_cfg.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=local_files_only)
        dtype_name = str(getattr(reward_cfg, "dtype", "bfloat16")).lower()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
        if not torch.cuda.is_available():
            dtype = torch.float32
        kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "num_labels": 1,
        }
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"] = getattr(reward_cfg, "device_map", "auto")
            kwargs["attn_implementation"] = str(getattr(reward_cfg, "attn_implementation", "eager"))
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)
        if not torch.cuda.is_available():
            model.to(torch.device("cpu"))
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            batch_size=max(int(getattr(reward_cfg, "batch_size", 4)), 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def score_generations(self, prompt: str, generations: list[str]) -> list[float]:
        if not generations:
            return []
        raw_rewards: list[float] = []
        for start in range(0, len(generations), self.batch_size):
            batch_generations = generations[start : start + self.batch_size]
            texts = [_conversation_text(self.tokenizer, prompt, generation) for generation in batch_generations]
            batch = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            batch = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                logits = self.model(**batch).logits[:, 0]
            raw_rewards.extend(float(value.item()) for value in logits)
        return raw_rewards


def score_partition(
    *,
    prompt: str,
    generations: list[str],
    partition: list[int],
    patience: float,
    scorer: RewardModelScorer,
) -> tuple[list[int], list[int], list[float], float]:
    raw_rewards = scorer.score_generations(prompt, generations)
    rating_scores = [transform_raw_reward(reward) for reward in raw_rewards]
    generation_scores: list[int] = []
    partition_scores: list[int] = []
    for rating_score, partition_id in zip(rating_scores, partition, strict=True):
        if partition_id == len(partition_scores):
            partition_scores.append(rating_score)
            generation_scores.append(rating_score)
        else:
            generation_scores.append(0)
    if len(partition_scores) != (max(partition) + 1 if partition else 0):
        raise RuntimeError("NoveltyBench reward scoring expected contiguous greedy partition ids.")
    if not generation_scores:
        utility = 0.0
    else:
        weights = [float(patience) ** generation_index for generation_index in range(len(generation_scores))]
        utility = float(sum(score * weight for score, weight in zip(generation_scores, weights, strict=True)) / sum(weights))
    return generation_scores, partition_scores, raw_rewards, utility


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    num_prompts = len(rows)
    mean_distinct = float(sum(len(row["partition_scores"]) for row in rows) / num_prompts) if rows else 0.0
    mean_utility = float(sum(float(row["utility"]) for row in rows) / num_prompts) if rows else 0.0
    return {
        "num_prompts": num_prompts,
        "mean_distinct": mean_distinct,
        "mean_utility": mean_utility,
    }

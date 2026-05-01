from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from random_steering.utils.hf import ensure_hf_home


def maybe_test_equality(response_0: str, response_1: str) -> bool | None:
    unigram_0 = response_0.strip().lower().split()
    unigram_1 = response_1.strip().lower().split()
    max_len = max(len(unigram_0), len(unigram_1))
    if max_len <= 5:
        common_unigrams = set(unigram_0) & set(unigram_1)
        return (len(common_unigrams) * 2) >= max_len
    return None


@dataclass
class GenerationSimilarityClassifier:
    tokenizer: Any
    model: Any
    threshold: float
    max_length: int
    batch_size: int

    @classmethod
    def from_config(cls, classifier_cfg: Any) -> "GenerationSimilarityClassifier":
        ensure_hf_home()
        device_name = str(getattr(classifier_cfg, "device", "cuda"))
        use_cuda = device_name == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        local_files_only = bool(getattr(classifier_cfg, "local_files_only", False))
        tokenizer = AutoTokenizer.from_pretrained(
            str(classifier_cfg.tokenizer_checkpoint),
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(classifier_cfg.checkpoint),
            local_files_only=local_files_only,
        ).to(device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            threshold=float(getattr(classifier_cfg, "threshold", 0.102)),
            max_length=int(getattr(classifier_cfg, "max_length", 128)),
            batch_size=max(int(getattr(classifier_cfg, "batch_size", 16)), 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def score_pairs(self, response_pairs: list[tuple[str, str]]) -> list[float]:
        if not response_pairs:
            return []
        scores: list[float] = []
        for start in range(0, len(response_pairs), self.batch_size):
            pair_batch = response_pairs[start : start + self.batch_size]
            encoded_batch: list[tuple[list[int], list[int]]] = []
            max_width = 0
            for response_0, response_1 in pair_batch:
                first_ids = self.tokenizer.encode(
                    response_0,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )
                second_ids = self.tokenizer.encode(
                    response_1,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )
                input_ids = [
                    int(self.tokenizer.cls_token_id),
                    *first_ids,
                    int(self.tokenizer.sep_token_id),
                    *second_ids,
                    int(self.tokenizer.sep_token_id),
                ]
                prompt_len = len(first_ids) + 2
                token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
                encoded_batch.append((input_ids, token_type_ids))
                max_width = max(max_width, len(input_ids))

            pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
            input_tensor = torch.full(
                (len(encoded_batch), max_width),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            token_type_tensor = torch.zeros((len(encoded_batch), max_width), dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(encoded_batch), max_width), dtype=torch.long, device=self.device)
            for row_index, (input_ids, token_type_ids) in enumerate(encoded_batch):
                width = len(input_ids)
                input_tensor[row_index, :width] = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                token_type_tensor[row_index, :width] = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
                attention_mask[row_index, :width] = 1

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    token_type_ids=token_type_tensor,
                    attention_mask=attention_mask,
                )
                probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
            scores.extend(float(score.item()) for score in probabilities)
        return scores

    def are_equivalent(self, response_0: str, response_1: str) -> bool:
        equality = maybe_test_equality(response_0, response_1)
        if equality is not None:
            return equality
        return self.score_pairs([(response_0, response_1)])[0] > self.threshold


def partition_responses(
    *,
    prompt: str,
    responses: list[str],
    classifier: GenerationSimilarityClassifier,
) -> list[int]:
    _ = prompt
    partition = [-1] * len(responses)
    num_partitions = 0

    for response_index, response_text in enumerate(responses):
        if partition[response_index] >= 0:
            continue
        partition[response_index] = num_partitions
        unresolved_indices: list[int] = []
        unresolved_pairs: list[tuple[str, str]] = []
        for candidate_index in range(response_index + 1, len(responses)):
            if partition[candidate_index] >= 0:
                continue
            equality = maybe_test_equality(response_text, responses[candidate_index])
            if equality is True:
                partition[candidate_index] = num_partitions
                continue
            if equality is False:
                continue
            unresolved_indices.append(candidate_index)
            unresolved_pairs.append((response_text, responses[candidate_index]))

        if unresolved_pairs:
            scores = classifier.score_pairs(unresolved_pairs)
            for candidate_index, score in zip(unresolved_indices, scores, strict=True):
                if score > classifier.threshold:
                    partition[candidate_index] = num_partitions
        num_partitions += 1

    if any(partition_id < 0 for partition_id in partition):
        raise RuntimeError("NoveltyBench partitioning failed to assign all responses.")
    return partition

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

from tqdm import tqdm

from random_steering.inference.base import GenerationBackend
from random_steering.retention.tinybenchmarks_data import GenerationExample, MultipleChoiceExample


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_STRICT_ANSWER_PATTERN = re.compile(r"####\s*([^\n\r]+)")


@dataclass(frozen=True)
class TaskRunResult:
    task_name: str
    benchmark_name: str
    summary_row: dict[str, Any]
    records: list[dict[str, Any]]
    example_scores: list[float]


def _iter_batches(items: list[Any], batch_size: int):
    safe_batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), safe_batch_size):
        yield items[start : start + safe_batch_size]


def _summary_row(task_name: str, benchmark_name: str, example_scores: list[float], **extra: Any) -> dict[str, Any]:
    num_examples = len(example_scores)
    score_sum = float(sum(example_scores))
    binary_only = all(score in (0, 1, 0.0, 1.0, False, True) for score in example_scores)
    row = {
        "task_name": task_name,
        "benchmark_name": benchmark_name,
        "num_examples": num_examples,
        "score_sum": score_sum,
        "num_correct": int(score_sum) if binary_only else None,
        "accuracy": float(score_sum / num_examples) if num_examples else 0.0,
    }
    row.update(extra)
    return row


def _choice_lengths(example: MultipleChoiceExample) -> list[float]:
    return [float(max(len(choice), 1)) for choice in example.choices]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exponentials = [math.exp(value - max_value) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def run_multiple_choice_task(
    *,
    backend: GenerationBackend,
    retention_cfg: Any,
    task_name: str,
    benchmark_name: str,
    examples: list[MultipleChoiceExample],
) -> TaskRunResult:
    records: list[dict[str, Any]] = []
    example_scores: list[float] = []
    batch_size = int(getattr(retention_cfg, "batch_size", 1))
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    with tqdm(total=len(examples), desc=task_name, leave=False) as progress:
        for example_batch in _iter_batches(examples, batch_size):
            score_lists = backend.score_prompt_continuation_pairs_batch(
                [
                    list(example.choice_contexts)
                    if example.choice_contexts is not None
                    else [example.prompt] * len(example.continuations)
                    for example in example_batch
                ],
                [list(example.continuations) for example in example_batch],
            )

            for example, score_list in zip(example_batch, score_lists, strict=True):
                labels = list(choice_labels[: len(example.choices)])
                raw_scores = [float(score) for score in score_list]
                normalized_scores = [
                    score / choice_length for score, choice_length in zip(raw_scores, _choice_lengths(example), strict=True)
                ]
                probabilities = _softmax(raw_scores)

                decision_scores = normalized_scores if example.metric_name == "acc_norm" else raw_scores
                predicted_index = max(range(len(decision_scores)), key=lambda index: decision_scores[index])
                predicted_label = labels[predicted_index]
                raw_argmax_index = max(range(len(raw_scores)), key=lambda index: raw_scores[index])

                if example.metric_name == "mc2":
                    if example.target_scores is None:
                        raise ValueError(f"{example.example_id} missing target_scores for mc2 evaluation")
                    example_score = float(
                        sum(
                            probability
                            for probability, label in zip(probabilities, example.target_scores, strict=True)
                            if label
                        )
                    )
                    raw_accuracy = float(raw_argmax_index == example.correct_choice_index)
                else:
                    example_score = float(predicted_index == example.correct_choice_index)
                    raw_accuracy = float(raw_argmax_index == example.correct_choice_index)

                example_scores.append(example_score)
                records.append(
                    {
                        "task_name": task_name,
                        "example_id": example.example_id,
                        "prompt": example.prompt,
                        "formatted_prompt": backend.format_prompt(example.prompt),
                        "metric_name": example.metric_name,
                        "prediction": predicted_label,
                        "gold": labels[example.correct_choice_index],
                        "prediction_text": example.choices[predicted_index],
                        "gold_text": example.choices[example.correct_choice_index],
                        "example_score": example_score,
                        "is_correct": bool(example_score == 1.0) if example.metric_name != "mc2" else None,
                        "raw_accuracy": raw_accuracy,
                        "choice_scores": {label: score for label, score in zip(labels, raw_scores, strict=True)},
                        "choice_scores_norm": {
                            label: score for label, score in zip(labels, normalized_scores, strict=True)
                        },
                        "choice_probabilities": {
                            label: probability for label, probability in zip(labels, probabilities, strict=True)
                        },
                        "gold_labels": {
                            label: target
                            for label, target in zip(
                                labels,
                                example.target_scores
                                or tuple(
                                    1.0 if index == example.correct_choice_index else 0.0
                                    for index in range(len(example.choices))
                                ),
                                strict=True,
                            )
                        },
                        "choices": {label: choice for label, choice in zip(labels, example.choices, strict=True)},
                        "continuations": {
                            label: continuation for label, continuation in zip(labels, example.continuations, strict=True)
                        },
                        "choice_contexts": (
                            {
                                label: context
                                for label, context in zip(labels, example.choice_contexts, strict=True)
                            }
                            if example.choice_contexts is not None
                            else None
                        ),
                        "metadata": example.metadata,
                    }
                )
            progress.update(len(example_batch))

    summary_extra: dict[str, Any] = {"metric_name": examples[0].metric_name} if examples else {}
    if examples and examples[0].metric_name == "mc2":
        summary_extra["argmax_accuracy"] = float(
            sum(float(record["raw_accuracy"]) for record in records) / len(records)
        )
    if examples and examples[0].metric_name == "acc_norm":
        summary_extra["raw_accuracy"] = float(
            sum(float(record["raw_accuracy"]) for record in records) / len(records)
        )

    return TaskRunResult(
        task_name=task_name,
        benchmark_name=benchmark_name,
        summary_row=_summary_row(task_name, benchmark_name, example_scores, **summary_extra),
        records=records,
        example_scores=example_scores,
    )


def _normalize_numeric_text(value: str) -> str | None:
    candidate = value.strip().replace(",", "")
    if not candidate:
        return None
    try:
        decimal = Decimal(candidate)
    except InvalidOperation:
        return None
    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")


def extract_strict_gsm8k_answer(text: str) -> str | None:
    match = _STRICT_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    numbers = _NUMBER_PATTERN.findall(value)
    if len(numbers) != 1:
        return None
    return _normalize_numeric_text(numbers[0])


def extract_last_numeric_answer(text: str) -> str | None:
    numbers = _NUMBER_PATTERN.findall(text)
    if not numbers:
        return None
    return _normalize_numeric_text(numbers[-1])


def _build_gsm8k_record(
    *,
    task_name: str,
    example: GenerationExample,
    raw_response: str,
    prediction: str | None,
    gold: str | None,
    is_correct: bool,
    extraction_ok: bool,
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "example_id": example.example_id,
        "prompt": example.prompt,
        "prediction": prediction,
        "gold": gold,
        "is_correct": is_correct,
        "raw_response": raw_response,
        "metadata": example.metadata,
        "extraction_ok": extraction_ok,
    }


def run_gsm8k_tasks(
    *,
    generation_backend: GenerationBackend,
    retention_cfg: Any,
    examples: list[GenerationExample],
    base_seed: int,
) -> list[TaskRunResult]:
    strict_records: list[dict[str, Any]] = []
    flexible_records: list[dict[str, Any]] = []
    strict_scores: list[float] = []
    flexible_scores: list[float] = []
    extraction_failures = 0
    batch_size = int(getattr(retention_cfg, "batch_size", 1))

    with tqdm(total=len(examples), desc="tiny_gsm8k", leave=False) as progress:
        for batch_index, example_batch in enumerate(_iter_batches(examples, batch_size)):
            raw_responses = generation_backend.generate_text_batch(
                [example.prompt for example in example_batch],
                [int(base_seed) + batch_index + offset for offset in range(len(example_batch))],
                stop_strings=["Question:", "</s>", "<|im_end|>"],
            )

            for example, raw_output in zip(example_batch, raw_responses, strict=True):
                raw_response = generation_backend.strip_response(raw_output)
                gold = _normalize_numeric_text(example.target_answer)
                strict_prediction = extract_strict_gsm8k_answer(raw_response)
                flexible_prediction = extract_last_numeric_answer(raw_response)
                strict_ok = strict_prediction is not None and gold is not None and strict_prediction == gold
                flexible_ok = flexible_prediction is not None and gold is not None and flexible_prediction == gold
                extraction_ok = strict_prediction is not None
                if not extraction_ok:
                    extraction_failures += 1

                strict_scores.append(float(strict_ok))
                flexible_scores.append(float(flexible_ok))
                strict_records.append(
                    _build_gsm8k_record(
                        task_name="tiny_gsm8k_strict",
                        example=example,
                        raw_response=raw_response,
                        prediction=strict_prediction,
                        gold=gold,
                        is_correct=strict_ok,
                        extraction_ok=extraction_ok,
                    )
                )
                flexible_records.append(
                    _build_gsm8k_record(
                        task_name="tiny_gsm8k_flexible",
                        example=example,
                        raw_response=raw_response,
                        prediction=flexible_prediction,
                        gold=gold,
                        is_correct=flexible_ok,
                        extraction_ok=flexible_prediction is not None,
                    )
                )
            progress.update(len(example_batch))

    strict_summary = _summary_row(
        "tiny_gsm8k_strict",
        "gsm8k",
        strict_scores,
        extraction_failure_rate=float(extraction_failures / len(examples)) if examples else 0.0,
    )
    flexible_summary = _summary_row(
        "tiny_gsm8k_flexible",
        "gsm8k",
        flexible_scores,
        extraction_failure_rate=float(
            sum(1 for record in flexible_records if not record["extraction_ok"]) / len(examples)
        )
        if examples
        else 0.0,
    )
    return [
        TaskRunResult(
            task_name="tiny_gsm8k_strict",
            benchmark_name="gsm8k",
            summary_row=strict_summary,
            records=strict_records,
            example_scores=strict_scores,
        ),
        TaskRunResult(
            task_name="tiny_gsm8k_flexible",
            benchmark_name="gsm8k",
            summary_row=flexible_summary,
            records=flexible_records,
            example_scores=flexible_scores,
        ),
    ]

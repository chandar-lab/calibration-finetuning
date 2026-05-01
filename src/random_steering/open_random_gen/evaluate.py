from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from random_steering.inference.base import GenerationBackend
from random_steering.inference.chat_format import format_prompt
from random_steering.inference.hf_backend import HFGenerationBackend
from random_steering.open_random_gen.metrics import (
    count_unique_outputs,
    has_thinking_tags,
    normalize_open_random_output,
    top_p_support_sizes,
)
from random_steering.utils.io import ensure_dir, write_csv, write_json, write_jsonl


def _compat_backend_model_cfg(model_cfg: Any, generation_cfg: Any, *, batch_size: int, use_chat_template: bool) -> Any:
    return SimpleNamespace(
        enable_thinking=getattr(model_cfg, "enable_thinking", None),
        reasoning_effort=getattr(model_cfg, "reasoning_effort", None),
        generation_prefix=getattr(model_cfg, "generation_prefix", None),
        max_new_tokens=int(getattr(generation_cfg, "max_new_tokens", 16)),
        do_sample=bool(getattr(generation_cfg, "do_sample", True)),
        temperature=float(getattr(generation_cfg, "temperature", 1.0)),
        top_p=float(getattr(generation_cfg, "top_p", 1.0)),
        batch_size=int(batch_size),
        use_chat_template=bool(use_chat_template),
    )


def _iter_batches(items: list[Any], batch_size: int):
    safe_batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), safe_batch_size):
        yield items[start : start + safe_batch_size]


def _support_thresholds(open_random_gen_cfg: Any) -> list[float]:
    configured = getattr(open_random_gen_cfg, "top_p_thresholds", None)
    if configured is None:
        return [float(getattr(open_random_gen_cfg, "top_p_threshold", 0.9))]
    thresholds = sorted({float(value) for value in configured})
    primary_threshold = float(getattr(open_random_gen_cfg, "top_p_threshold", 0.9))
    if primary_threshold not in thresholds:
        thresholds.append(primary_threshold)
        thresholds.sort()
    return thresholds


def _threshold_key(threshold: float) -> str:
    return f"p_{threshold:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_threshold_key(key: str) -> float:
    if not key.startswith("p_"):
        raise ValueError(f"Invalid top-p threshold key: {key}")
    return float(key[2:].replace("p", "."))


def _load_existing_per_prompt_rows(run_dir: str | Path) -> dict[int, dict[str, Any]]:
    run_path = Path(run_dir)
    rows_by_prompt_id: dict[int, dict[str, Any]] = {}

    csv_path = run_path / "metrics" / "per_prompt.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                prompt_id = int(row["prompt_id"])
                existing_row = rows_by_prompt_id.setdefault(prompt_id, {"prompt_id": prompt_id})
                if row.get("prompt"):
                    existing_row["prompt"] = row["prompt"]
                if row.get("top_p_support_size"):
                    existing_row["top_p_support_size"] = int(float(row["top_p_support_size"]))
                if row.get("num_unique_outputs"):
                    existing_row["num_unique_outputs"] = int(float(row["num_unique_outputs"]))
                if row.get("unique_fraction"):
                    existing_row["unique_fraction"] = float(row["unique_fraction"])
                top_p_support_sizes = existing_row.setdefault("top_p_support_sizes", {})
                for key, value in row.items():
                    if not key.startswith("p_") or value in {None, ""}:
                        continue
                    top_p_support_sizes[key] = int(float(value))

    jsonl_path = run_path / "samples" / "per_prompt.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_id = int(row["prompt_id"])
                existing_row = rows_by_prompt_id.setdefault(prompt_id, {"prompt_id": prompt_id})
                existing_row.update(row)

    return rows_by_prompt_id


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=True))
    encoded = tokenizer(text)
    return list(encoded["input_ids"])


def _tokenize_text_batch(tokenizer: Any, texts: list[str], *, padding_side: str = "right") -> dict[str, torch.Tensor]:
    original_padding_side = getattr(tokenizer, "padding_side", None)
    restore_padding_side = original_padding_side is not None and original_padding_side != padding_side
    if restore_padding_side:
        tokenizer.padding_side = padding_side
    try:
        try:
            encoded = tokenizer(texts, return_tensors="pt", padding=True)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return encoded
        except TypeError:
            pass
    finally:
        if restore_padding_side:
            tokenizer.padding_side = original_padding_side

    tokenized = [_token_ids(tokenizer, text) for text in texts]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    max_length = max(len(token_ids) for token_ids in tokenized)
    input_ids = []
    attention_mask = []
    for token_ids in tokenized:
        pad_length = max_length - len(token_ids)
        if padding_side == "left":
            input_ids.append(([int(pad_token_id)] * pad_length) + token_ids)
            attention_mask.append(([0] * pad_length) + ([1] * len(token_ids)))
        else:
            input_ids.append(token_ids + ([int(pad_token_id)] * pad_length))
            attention_mask.append(([1] * len(token_ids)) + ([0] * pad_length))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _move_model_inputs(model_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }


def _formatted_prompts(tokenizer: Any, prompts: list[str], enable_thinking: bool | None, reasoning_effort: str | None = None, generation_prefix: str | None = None) -> list[str]:
    return [
        format_prompt(tokenizer, prompt, enable_thinking=enable_thinking, reasoning_effort=reasoning_effort, generation_prefix=generation_prefix)
        for prompt in prompts
    ]


def compute_top_p_support_metrics(
    *,
    model: Any,
    tokenizer: Any,
    model_cfg: Any,
    prompts: list[str],
    open_random_gen_cfg: Any,
) -> tuple[list[dict[float, int]], list[float]]:
    batch_size = int(getattr(open_random_gen_cfg, "batch_size", 1))
    temperature = float(getattr(open_random_gen_cfg.generation, "temperature", 1.0))
    thresholds = _support_thresholds(open_random_gen_cfg)
    if temperature <= 0.0:
        raise ValueError("Open random generation top-p support metric requires generation.temperature > 0")
    enable_thinking = getattr(model_cfg, "enable_thinking", None)
    reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
    generation_prefix = getattr(model_cfg, "generation_prefix", None)
    formatted_prompts = _formatted_prompts(tokenizer, prompts, enable_thinking, reasoning_effort=reasoning_effort, generation_prefix=generation_prefix)
    device = next(model.parameters()).device
    support_sizes_by_prompt: list[dict[float, int]] = []

    for prompt_batch in tqdm(_iter_batches(formatted_prompts, batch_size), desc="Top-p support", leave=False):
        model_inputs = _tokenize_text_batch(tokenizer, prompt_batch)
        input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
        model_inputs = _move_model_inputs(model_inputs, device)
        with torch.no_grad():
            logits = model(**model_inputs).logits

        for row_index, prompt_length in enumerate(input_lengths):
            final_logits = logits[row_index, int(prompt_length) - 1]
            probs = torch.softmax(final_logits / temperature, dim=-1)
            support_sizes_by_prompt.append(top_p_support_sizes(probs, thresholds))
    return support_sizes_by_prompt, thresholds


def sample_prompt_outputs(
    *,
    generation_backend: GenerationBackend | None = None,
    prompts: list[str],
    open_random_gen_cfg: Any,
    base_seed: int,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_cfg: Any | None = None,
) -> list[dict[str, Any]]:
    if generation_backend is None:
        if model is None or tokenizer is None or model_cfg is None:
            raise ValueError("Either generation_backend or model/tokenizer/model_cfg must be provided.")
        generation_backend = HFGenerationBackend(
            model=model,
            tokenizer=tokenizer,
            model_cfg=_compat_backend_model_cfg(
                model_cfg,
                open_random_gen_cfg.generation,
                batch_size=int(getattr(open_random_gen_cfg, "batch_size", 1)),
                use_chat_template=True,
            ),
        )
    batch_size = int(getattr(open_random_gen_cfg, "batch_size", 1))
    num_samples_per_prompt = int(getattr(open_random_gen_cfg, "num_samples_per_prompt", 100))
    raw_outputs_by_prompt: dict[int, list[str]] = defaultdict(list)
    outputs_by_prompt: dict[int, list[str]] = defaultdict(list)
    had_thinking_tags_by_prompt: dict[int, list[bool]] = defaultdict(list)
    prompt_id_batches = list(_iter_batches(list(range(len(prompts))), batch_size))

    progress = tqdm(total=len(prompts) * num_samples_per_prompt, desc="Repeated sampling", leave=False)
    try:
        for sample_index in range(num_samples_per_prompt):
            seed = int(base_seed) + sample_index
            for prompt_ids in prompt_id_batches:
                prompt_batch = [prompts[prompt_id] for prompt_id in prompt_ids]
                seed_batch = [seed] * len(prompt_ids)
                raw_batch = generation_backend.generate_text_batch(prompt_batch, seed_batch)
                for prompt_id, raw_output in zip(prompt_ids, raw_batch, strict=True):
                    raw_outputs_by_prompt[prompt_id].append(raw_output)
                    outputs_by_prompt[prompt_id].append(normalize_open_random_output(raw_output))
                    had_thinking_tags_by_prompt[prompt_id].append(has_thinking_tags(raw_output))
                progress.update(len(prompt_ids))
    finally:
        progress.close()

    return [
        {
            "raw_outputs": raw_outputs_by_prompt[prompt_id],
            "normalized_outputs": outputs_by_prompt[prompt_id],
            "had_thinking_tags": had_thinking_tags_by_prompt[prompt_id],
        }
        for prompt_id in range(len(prompts))
    ]


def _aggregate_summary(
    *,
    prompts: list[str],
    support_sizes_by_prompt: list[dict[float, int]] | None,
    per_prompt_samples: list[dict[str, Any]] | None,
    top_p_threshold: float,
    top_p_thresholds: list[float],
    existing_per_prompt_rows: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_prompt_rows: list[dict[str, Any]] = []
    existing_per_prompt_rows = existing_per_prompt_rows or {}
    primary_support_sizes: list[int] = []
    support_sizes_by_threshold: dict[str, list[int]] = defaultdict(list)
    unique_counts: list[int] = []
    unique_fractions: list[float] = []
    empty_output_counts: list[int] = []
    thinking_tag_counts: list[int] = []

    for prompt_id, prompt in enumerate(prompts):
        prompt_row = dict(existing_per_prompt_rows.get(prompt_id, {}))
        prompt_row["prompt_id"] = prompt_id
        prompt_row["prompt"] = prompt

        if support_sizes_by_prompt is not None:
            support_sizes = support_sizes_by_prompt[prompt_id]
            prompt_row["top_p_support_size"] = int(support_sizes[float(top_p_threshold)])
            prompt_row["top_p_support_sizes"] = {_threshold_key(key): int(value) for key, value in support_sizes.items()}

        top_p_support_sizes = dict(prompt_row.get("top_p_support_sizes", {}))
        if "top_p_support_size" in prompt_row and _threshold_key(top_p_threshold) not in top_p_support_sizes:
            top_p_support_sizes[_threshold_key(top_p_threshold)] = int(prompt_row["top_p_support_size"])
        if top_p_support_sizes:
            prompt_row["top_p_support_sizes"] = top_p_support_sizes
            primary_value = top_p_support_sizes.get(_threshold_key(top_p_threshold))
            if primary_value is not None:
                prompt_row["top_p_support_size"] = int(primary_value)
                primary_support_sizes.append(int(primary_value))
            for threshold_key, value in top_p_support_sizes.items():
                support_sizes_by_threshold[threshold_key].append(int(value))

        if per_prompt_samples is not None:
            prompt_samples = per_prompt_samples[prompt_id]
            prompt_row["raw_outputs"] = list(prompt_samples["raw_outputs"])
            prompt_row["normalized_outputs"] = list(prompt_samples["normalized_outputs"])
            prompt_row["had_thinking_tags"] = list(prompt_samples["had_thinking_tags"])

        normalized_outputs = list(prompt_row.get("normalized_outputs", []))
        had_thinking_tags = list(prompt_row.get("had_thinking_tags", []))
        if normalized_outputs:
            num_unique_outputs, unique_fraction, output_counts = count_unique_outputs(normalized_outputs)
            empty_output_count = sum(1 for output in normalized_outputs if not output)
            had_thinking_tags_count = sum(1 for flag in had_thinking_tags if flag)
            unique_counts.append(num_unique_outputs)
            unique_fractions.append(unique_fraction)
            empty_output_counts.append(empty_output_count)
            thinking_tag_counts.append(had_thinking_tags_count)
            prompt_row["num_unique_outputs"] = int(num_unique_outputs)
            prompt_row["unique_fraction"] = float(unique_fraction)
            prompt_row["output_counts"] = output_counts
            prompt_row["empty_output_count"] = int(empty_output_count)
            prompt_row["had_thinking_tags_count"] = int(had_thinking_tags_count)

        per_prompt_rows.append(prompt_row)

    unique_count_tensor = torch.tensor(unique_counts, dtype=torch.float32) if unique_counts else torch.tensor([], dtype=torch.float32)
    unique_fraction_tensor = (
        torch.tensor(unique_fractions, dtype=torch.float32) if unique_fractions else torch.tensor([], dtype=torch.float32)
    )
    available_threshold_keys = sorted(support_sizes_by_threshold.keys(), key=_parse_threshold_key)

    summary = {
        "num_prompts": len(prompts),
        "num_samples_per_prompt": len(per_prompt_rows[0].get("normalized_outputs", [])) if per_prompt_rows else 0,
        "top_p_threshold": float(top_p_threshold),
        "top_p_thresholds": [_parse_threshold_key(key) for key in available_threshold_keys],
        "avg_top_p_support_size": float(sum(primary_support_sizes) / len(primary_support_sizes)) if primary_support_sizes else None,
        "median_top_p_support_size": float(statistics.median(primary_support_sizes)) if primary_support_sizes else None,
        "min_top_p_support_size": int(min(primary_support_sizes)) if primary_support_sizes else None,
        "max_top_p_support_size": int(max(primary_support_sizes)) if primary_support_sizes else None,
        "top_p_support_size_by_threshold": {
            threshold_key: {
                "threshold": _parse_threshold_key(threshold_key),
                "avg": float(sum(values) / len(values)) if values else 0.0,
                "median": float(statistics.median(values)) if values else 0.0,
                "min": int(min(values)) if values else 0,
                "max": int(max(values)) if values else 0,
            }
            for threshold_key, values in support_sizes_by_threshold.items()
        },
        "avg_num_unique_outputs": float(unique_count_tensor.mean().item()) if unique_counts else None,
        "avg_unique_fraction": float(unique_fraction_tensor.mean().item()) if unique_fractions else None,
        "median_unique_fraction": float(statistics.median(unique_fractions)) if unique_fractions else None,
        "total_empty_outputs": int(sum(empty_output_counts)) if empty_output_counts else None,
        "prompts_with_empty_outputs": int(sum(1 for count in empty_output_counts if count > 0)) if empty_output_counts else None,
        "total_outputs_with_thinking_tags": int(sum(thinking_tag_counts)) if thinking_tag_counts else None,
        "prompts_with_thinking_tags": int(sum(1 for count in thinking_tag_counts if count > 0)) if thinking_tag_counts else None,
    }
    return per_prompt_rows, summary


def write_open_random_gen_artifacts(
    *,
    run_dir: str | Path,
    full_cfg: Any,
    eval_target_name: str,
    per_prompt_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    run_path = ensure_dir(run_dir)
    metrics_dir = ensure_dir(run_path / "metrics")
    samples_dir = ensure_dir(run_path / "samples")
    write_json(run_path / "config_resolved.json", OmegaConf.to_container(full_cfg, resolve=True))
    write_json(
        run_path / "summary.json",
        {
            "eval_target": eval_target_name,
            "benchmark": "open_random_gen",
            "metrics": summary,
        },
    )
    write_json(metrics_dir / "summary.json", summary)
    write_csv(
        metrics_dir / "per_prompt.csv",
        [
            {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "top_p_support_size": row.get("top_p_support_size"),
                "num_unique_outputs": row.get("num_unique_outputs"),
                "unique_fraction": row.get("unique_fraction"),
                **row.get("top_p_support_sizes", {}),
            }
            for row in per_prompt_rows
        ],
    )
    write_jsonl(samples_dir / "per_prompt.jsonl", per_prompt_rows)


def run_open_random_gen(
    *,
    generation_backend: GenerationBackend | None,
    hf_model: Any | None,
    tokenizer: Any,
    model_cfg: Any,
    open_random_gen_cfg: Any,
    eval_target_cfg: Any,
    output_dir: str | Path,
    full_cfg: Any,
    prompts: list[str],
    base_seed: int,
    model: Any | None = None,
) -> dict[str, Any]:
    if hf_model is None and model is not None:
        hf_model = model
    compute_support_metrics = bool(getattr(open_random_gen_cfg, "compute_support_metrics", True))
    compute_sampling_metrics = bool(getattr(open_random_gen_cfg, "compute_sampling_metrics", True))
    if not compute_support_metrics and not compute_sampling_metrics:
        raise ValueError("Open random generation requires at least one of compute_support_metrics or compute_sampling_metrics")

    existing_per_prompt_rows = _load_existing_per_prompt_rows(output_dir)
    support_sizes_by_prompt: list[dict[float, int]] | None = None
    if compute_support_metrics:
        if hf_model is None:
            raise ValueError("Open random generation support metrics require an HF model.")
        support_sizes_by_prompt, _ = compute_top_p_support_metrics(
            model=hf_model,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            prompts=prompts,
            open_random_gen_cfg=open_random_gen_cfg,
        )

    per_prompt_samples: list[dict[str, Any]] | None = None
    if compute_sampling_metrics:
        if generation_backend is None and model is None:
            raise ValueError("Open random generation sampling metrics require a generation backend.")
        per_prompt_samples = sample_prompt_outputs(
            generation_backend=generation_backend,
            prompts=prompts,
            open_random_gen_cfg=open_random_gen_cfg,
            base_seed=base_seed,
            model=model,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
        )

    per_prompt_rows, summary = _aggregate_summary(
        prompts=prompts,
        support_sizes_by_prompt=support_sizes_by_prompt,
        per_prompt_samples=per_prompt_samples,
        top_p_threshold=float(getattr(open_random_gen_cfg, "top_p_threshold", 0.9)),
        top_p_thresholds=_support_thresholds(open_random_gen_cfg),
        existing_per_prompt_rows=existing_per_prompt_rows,
    )
    write_open_random_gen_artifacts(
        run_dir=output_dir,
        full_cfg=full_cfg,
        eval_target_name=str(eval_target_cfg.name),
        per_prompt_rows=per_prompt_rows,
        summary=summary,
    )
    return {
        "per_prompt_rows": per_prompt_rows,
        "summary": summary,
    }

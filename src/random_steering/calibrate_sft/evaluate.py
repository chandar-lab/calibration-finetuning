from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from random_steering.calibrate_sft.data import PreparedPrompt, sample_training_example
from random_steering.calibrate_sft.losses import sequence_calibration_loss
from random_steering.eval.report import sample_record_to_dict, to_metric_rows
from random_steering.eval.protocol_runner import run_protocol
from random_steering.inference.base import GenerationBackend
from random_steering.steering.none import NoneSteeringPolicy
from random_steering.utils.io import ensure_dir, write_csv, write_json, write_jsonl


def _build_family_balanced_summary(
    rows: list[dict[str, Any]],
    *,
    value_keys: list[str],
) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "splits": {}}

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[str(row["split"])].append(row)

    summary: dict[str, Any] = {"num_rows": len(rows), "splits": {}}
    for split_name, split_metric_rows in split_rows.items():
        family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in split_metric_rows:
            family_rows[str(row["family_id"])].append(row)

        split_summary: dict[str, Any] = {
            "num_rows": len(split_metric_rows),
            "num_families": len(family_rows),
            "aggregation": "equal_family_weight",
        }
        for key in value_keys:
            family_means: list[float] = []
            for grouped_rows in family_rows.values():
                values = [float(row[key]) for row in grouped_rows if isinstance(row.get(key), (float, int))]
                if values:
                    family_means.append(float(sum(values) / len(values)))
            if family_means:
                split_summary[f"avg_{key}"] = float(sum(family_means) / len(family_means))

        summary["splits"][split_name] = split_summary

    return summary


def _select_prompts(
    prompts: list[PreparedPrompt],
    max_prompts: int | None,
    *,
    selection_seed: int = 0,
) -> list[PreparedPrompt]:
    if max_prompts is None or len(prompts) <= max_prompts:
        return prompts

    grouped: dict[str, list[PreparedPrompt]] = defaultdict(list)
    for prompt in prompts:
        grouped[prompt.prompt_spec.family_id].append(prompt)

    family_ids = sorted(grouped)
    if not family_ids:
        return []
    offset = int(selection_seed) % len(family_ids)
    ordered_family_ids = family_ids[offset:] + family_ids[:offset]

    selected: list[PreparedPrompt] = []
    family_positions = {family_id: 0 for family_id in ordered_family_ids}
    while len(selected) < int(max_prompts):
        added_any = False
        for family_id in ordered_family_ids:
            position = family_positions[family_id]
            family_prompts = grouped[family_id]
            if position >= len(family_prompts):
                continue
            selected.append(family_prompts[position])
            family_positions[family_id] += 1
            added_any = True
            if len(selected) >= int(max_prompts):
                break
        if not added_any:
            break
    return selected


def _iter_eval_work(
    prepared_splits: dict[str, list[PreparedPrompt]],
    max_prompts: int | None,
    *,
    selection_seed: int = 0,
) -> list[tuple[str, PreparedPrompt]]:
    work_items: list[tuple[str, PreparedPrompt]] = []
    for split_index, (split_name, prompts) in enumerate(prepared_splits.items()):
        split_seed = int(selection_seed) + split_index
        for prompt in _select_prompts(prompts, max_prompts, selection_seed=split_seed):
            work_items.append((split_name, prompt))
    return work_items


def monte_carlo_logit_kl(
    model: Any,
    prompt: PreparedPrompt,
    *,
    tau: float,
    epsilon: float,
    num_samples: int,
    base_seed: int,
    batch_size: int = 1,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []
    examples = [sample_training_example(prompt, base_seed + sample_index) for sample_index in range(num_samples)]
    for start in range(0, len(examples), max(int(batch_size), 1)):
        example_batch = examples[start : start + max(int(batch_size), 1)]
        max_length = max(len(example.prepared_prompt.prompt_token_ids) + len(example.candidate_token_ids) for example in example_batch)
        input_ids = torch.zeros((len(example_batch), max_length), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        prompt_lengths: list[int] = []
        for batch_index, example in enumerate(example_batch):
            full_tokens = list(example.prepared_prompt.prompt_token_ids + example.candidate_token_ids)
            input_ids[batch_index, : len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
            attention_mask[batch_index, : len(full_tokens)] = 1
            prompt_lengths.append(len(example.prepared_prompt.prompt_token_ids))
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for batch_index, example in enumerate(example_batch):
            start_position = prompt_lengths[batch_index] - 1
            positions = [start_position + step_index for step_index in range(len(example.prefix_targets))]
            step_logits = [logits[batch_index, pos] for pos in positions]
            loss = sequence_calibration_loss(step_logits, example.prefix_targets, tau=tau, epsilon=epsilon)
            losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def run_logit_evaluation(
    *,
    model: Any,
    prepared_splits: dict[str, list[PreparedPrompt]],
    experiment_cfg: Any,
    train_cfg: Any,
    output_dir: str | Path,
    base_seed: int,
) -> list[dict[str, Any]]:
    logit_cfg = experiment_cfg.logit_eval
    if not bool(logit_cfg.enabled):
        return []

    rows: list[dict[str, Any]] = []
    max_prompts = getattr(logit_cfg, "max_prompts_per_split", None)
    work_items = _iter_eval_work(prepared_splits, max_prompts, selection_seed=base_seed)
    for split_name, prompt in tqdm(work_items, desc="Logit eval", leave=False):
        rows.append(
            {
                "split": split_name,
                "family_id": prompt.prompt_spec.family_id,
                "tier": prompt.prompt_spec.tier,
                "display_name": prompt.prompt_spec.display_name,
                "distribution_id": prompt.prompt_spec.distribution_id,
                "avg_logit_kl": monte_carlo_logit_kl(
                    model,
                    prompt,
                    tau=float(train_cfg.tau),
                    epsilon=float(train_cfg.epsilon_smoothing),
                    num_samples=int(logit_cfg.num_mc_samples),
                    base_seed=base_seed,
                    batch_size=int(getattr(logit_cfg, "batch_size", 1)),
                ),
            }
        )
    write_csv(Path(output_dir) / "metrics" / "logit_metrics.csv", rows)
    return rows


def run_sample_evaluation(
    *,
    generation_backend: GenerationBackend,
    prepared_splits: dict[str, list[PreparedPrompt]],
    experiment_cfg: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    sample_cfg = experiment_cfg.sample_eval
    if not bool(sample_cfg.enabled):
        return {"summary": {}, "per_distribution_rows": []}

    steering_policy = NoneSteeringPolicy()
    samples_dir = ensure_dir(Path(output_dir) / "samples")
    metrics_dir = ensure_dir(Path(output_dir) / "metrics")
    all_summary_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []

    max_prompts = getattr(sample_cfg, "max_prompts_per_split", None)
    seeds = [int(seed) for seed in sample_cfg.seeds]
    selection_seed = int(seeds[0]) if seeds else 0
    prompt_work_items = _iter_eval_work(prepared_splits, max_prompts, selection_seed=selection_seed)
    total_runs = len(prompt_work_items) * len(seeds)
    with tqdm(total=total_runs, desc="Sample eval", leave=False) as progress:
        for split_name, prompt in prompt_work_items:
            for seed in seeds:
                records = run_protocol(
                    engine=generation_backend,
                    steering_policy=steering_policy,
                    steering_name="calibrate_sft",
                    spec=prompt.distribution_spec,
                    protocol=str(sample_cfg.protocol),
                    num_samples=int(sample_cfg.num_samples),
                    base_seed=seed,
                )
                filename = f"{split_name}_{prompt.prompt_spec.distribution_id}_seed{seed}.jsonl"
                write_jsonl(samples_dir / filename, [sample_record_to_dict(record) for record in records])
                summary_rows, metric_records = to_metric_rows(
                    records=records,
                    spec=prompt.distribution_spec,
                    steering_name="calibrate_sft",
                    seed=seed,
                )
                for row in summary_rows:
                    row["split"] = split_name
                    row["family_id"] = prompt.prompt_spec.family_id
                    row["tier"] = prompt.prompt_spec.tier
                    row["display_name"] = prompt.prompt_spec.display_name
                all_summary_rows.extend(summary_rows)
                all_metric_rows.extend(
                    {
                        "split": split_name,
                        "distribution_id": metric.distribution_id,
                        "protocol": metric.protocol,
                        "steering_name": metric.steering_name,
                        "seed": metric.seed,
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                    }
                    for metric in metric_records
                )
                progress.update(1)

    write_csv(metrics_dir / "per_distribution.csv", all_summary_rows)
    write_csv(metrics_dir / "flat_metrics.csv", all_metric_rows)
    summary = _build_family_balanced_summary(
        all_summary_rows,
        value_keys=[
            "valid_rate",
            "support_violation_rate",
            "mean_error",
            "variance_error",
            "wasserstein_1",
        ],
    )
    write_json(metrics_dir / "summary.json", summary)
    return {"summary": summary, "per_distribution_rows": all_summary_rows}

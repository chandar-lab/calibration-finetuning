from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from random_steering.inference.hf_backend import HFGenerationBackend
from random_steering.retention.tinybenchmarks_data import (
    DATASET_SIZE,
    load_tiny_gsm8k,
    load_tiny_hellaswag,
    load_tiny_mmlu,
    load_tiny_truthfulqa,
    load_tiny_winogrande,
)
from random_steering.retention.tinybenchmarks_metrics import evaluate_tinybenchmarks
from random_steering.retention.tinybenchmarks_tasks import TaskRunResult, run_gsm8k_tasks, run_multiple_choice_task
from random_steering.utils.io import ensure_dir, write_csv, write_json, write_jsonl


def _limit_examples(examples: list[Any], max_examples_per_task: int | None) -> list[Any]:
    if max_examples_per_task is None:
        return examples
    return list(examples[: int(max_examples_per_task)])


def load_enabled_tasks(retention_cfg: Any) -> list[tuple[str, str, list[Any]]]:
    enabled_tasks = {str(task_name) for task_name in retention_cfg.enabled_tasks}
    task_specs = []
    if "tiny_mmlu" in enabled_tasks:
        task_specs.append(("tiny_mmlu", "mmlu", load_tiny_mmlu()))
    if "tiny_hellaswag" in enabled_tasks:
        task_specs.append(("tiny_hellaswag", "hellaswag", load_tiny_hellaswag()))
    if "tiny_truthfulqa" in enabled_tasks:
        task_specs.append(("tiny_truthfulqa", "truthfulqa", load_tiny_truthfulqa()))
    if "tiny_winogrande" in enabled_tasks:
        task_specs.append(("tiny_winogrande", "winogrande", load_tiny_winogrande()))
    if "tiny_gsm8k_strict" in enabled_tasks or "tiny_gsm8k_flexible" in enabled_tasks:
        task_specs.append(("tiny_gsm8k", "gsm8k", load_tiny_gsm8k()))
    return task_specs


def _augment_summary(task_result: TaskRunResult) -> dict[str, Any]:
    summary = dict(task_result.summary_row)
    if len(task_result.example_scores) == DATASET_SIZE:
        summary.update(evaluate_tinybenchmarks(task_result.example_scores, task_result.benchmark_name))
    else:
        summary.update({"irt": None, "pirt": None, "gpirt": None})
    return summary


def _aggregate_summary(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not task_rows:
        return {
            "num_tasks": 0,
            "mean_accuracy": 0.0,
            "mean_gpirt": None,
            "mean_mc_accuracy": 0.0,
            "gsm8k_strict_accuracy": None,
            "gsm8k_flexible_accuracy": None,
        }

    mean_accuracy = sum(float(row["accuracy"]) for row in task_rows) / len(task_rows)
    gpirt_rows = [float(row["gpirt"]) for row in task_rows if row.get("gpirt") is not None]
    mc_rows = [row for row in task_rows if row["benchmark_name"] != "gsm8k"]
    gsm8k_strict = next((row for row in task_rows if row["task_name"] == "tiny_gsm8k_strict"), None)
    gsm8k_flexible = next((row for row in task_rows if row["task_name"] == "tiny_gsm8k_flexible"), None)
    return {
        "num_tasks": len(task_rows),
        "mean_accuracy": mean_accuracy,
        "mean_gpirt": (sum(gpirt_rows) / len(gpirt_rows)) if gpirt_rows else None,
        "mean_mc_accuracy": (
            sum(float(row["accuracy"]) for row in mc_rows) / len(mc_rows) if mc_rows else 0.0
        ),
        "gsm8k_strict_accuracy": None if gsm8k_strict is None else float(gsm8k_strict["accuracy"]),
        "gsm8k_flexible_accuracy": None if gsm8k_flexible is None else float(gsm8k_flexible["accuracy"]),
    }


def run_retention_eval(
    *,
    generation_backend: Any | None = None,
    hf_model: Any | None = None,
    tokenizer: Any,
    model_cfg: Any,
    retention_cfg: Any,
    eval_target_cfg: Any,
    output_dir: str | Path,
    full_cfg: Any,
    model: Any | None = None,
) -> dict[str, Any]:
    if hf_model is None and model is not None:
        hf_model = model
    run_dir = ensure_dir(output_dir)
    per_example_dir = ensure_dir(run_dir / "per_example")

    max_examples_per_task = getattr(retention_cfg, "max_examples_per_task", None)
    write_json(run_dir / "config_resolved.json", OmegaConf.to_container(full_cfg, resolve=True))

    task_results: list[TaskRunResult] = []
    mc_backend = None if hf_model is None else HFGenerationBackend(model=hf_model, tokenizer=tokenizer, model_cfg=model_cfg)
    for task_name, benchmark_name, examples in load_enabled_tasks(retention_cfg):
        selected_examples = _limit_examples(examples, max_examples_per_task)
        if task_name == "tiny_gsm8k":
            if generation_backend is None:
                raise ValueError("GSM8K retention tasks require a generation backend.")
            gsm8k_results = run_gsm8k_tasks(
                generation_backend=generation_backend,
                retention_cfg=retention_cfg,
                examples=selected_examples,
                base_seed=int(getattr(full_cfg, "seed", 0)),
            )
            task_results.extend(
                result for result in gsm8k_results if result.task_name in {str(name) for name in retention_cfg.enabled_tasks}
            )
            continue

        if mc_backend is None:
            raise ValueError("Multiple-choice retention tasks require an HF model.")
        task_results.append(
            run_multiple_choice_task(
                backend=mc_backend,
                retention_cfg=retention_cfg,
                task_name=task_name,
                benchmark_name=benchmark_name,
                examples=selected_examples,
            )
        )

    task_summary_rows: list[dict[str, Any]] = []
    for task_result in task_results:
        task_summary = _augment_summary(task_result)
        task_summary_rows.append(task_summary)
        write_jsonl(per_example_dir / f"{task_result.task_name}.jsonl", task_result.records)

    write_csv(run_dir / "task_summary.csv", task_summary_rows)
    aggregates = _aggregate_summary(task_summary_rows)
    summary = {
        "eval_target": str(eval_target_cfg.name),
        **aggregates,
        "tasks": task_summary_rows,
    }
    write_json(run_dir / "summary.json", summary)
    return summary

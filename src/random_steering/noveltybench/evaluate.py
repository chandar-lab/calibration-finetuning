from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from random_steering.inference.base import GenerationBackend
from random_steering.noveltybench.data import (
    NoveltyBenchPrompt,
    artifact_is_complete,
    enabled_splits,
    load_split_prompts,
    read_jsonl,
    resolve_eval_target_name,
    split_output_dir,
)
from random_steering.noveltybench.partitioning import GenerationSimilarityClassifier, partition_responses
from random_steering.noveltybench.scoring import RewardModelScorer, score_partition, summarize_scores
from random_steering.utils.io import append_jsonl, ensure_dir, write_csv, write_json, write_jsonl


def _stage_model_cfg(model_cfg: Any, noveltybench_cfg: Any) -> Any:
    return OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True)),
        noveltybench_cfg.generation,
        {"use_chat_template": True},
    )


def _seed_for(base_seed: int, prompt_index: int, generation_index: int) -> int:
    return int(base_seed) + (prompt_index * 1009) + generation_index


def _load_prompts_by_split(noveltybench_cfg: Any) -> dict[str, list[NoveltyBenchPrompt]]:
    max_prompts = getattr(noveltybench_cfg, "max_prompts_per_split", None)
    max_prompts = None if max_prompts is None else int(max_prompts)
    return {
        split_name: load_split_prompts(
            noveltybench_cfg.assets_root,
            split_name,
            max_prompts=max_prompts,
        )
        for split_name in enabled_splits(noveltybench_cfg)
    }


def _write_resolved_config(run_dir: Path, full_cfg: Any) -> None:
    write_json(run_dir / "config_resolved.json", OmegaConf.to_container(full_cfg, resolve=True))


def _release_generation_backend(generation_backend: GenerationBackend | None) -> None:
    if generation_backend is None:
        return
    close_fn = getattr(generation_backend, "close", None)
    if callable(close_fn):
        close_fn()


def _generation_manifest(
    *,
    model_cfg: Any,
    noveltybench_cfg: Any,
    eval_target_cfg: Any,
    base_seed: int,
) -> dict[str, Any]:
    generation_cfg = noveltybench_cfg.generation
    return {
        "model_checkpoint": getattr(model_cfg, "checkpoint", None),
        "eval_target_base_checkpoint": getattr(eval_target_cfg, "base_checkpoint", None),
        "eval_target_adapter_checkpoint": getattr(eval_target_cfg, "adapter_checkpoint", None),
        "eval_target_tokenizer_checkpoint": getattr(eval_target_cfg, "tokenizer_checkpoint", None),
        "seed": int(base_seed),
        "enabled_splits": list(enabled_splits(noveltybench_cfg)),
        "max_prompts_per_split": getattr(noveltybench_cfg, "max_prompts_per_split", None),
        "num_generations": int(generation_cfg.num_generations),
        "temperature": float(generation_cfg.temperature),
        "top_p": float(generation_cfg.top_p),
        "do_sample": bool(generation_cfg.do_sample),
        "max_new_tokens": int(generation_cfg.max_new_tokens),
    }


def _ensure_generation_manifest(
    *,
    run_dir: Path,
    model_cfg: Any,
    noveltybench_cfg: Any,
    eval_target_cfg: Any,
    base_seed: int,
) -> None:
    manifest_path = run_dir / "generation_manifest.json"
    expected_manifest = _generation_manifest(
        model_cfg=model_cfg,
        noveltybench_cfg=noveltybench_cfg,
        eval_target_cfg=eval_target_cfg,
        base_seed=base_seed,
    )
    if not manifest_path.exists():
        write_json(manifest_path, expected_manifest)
        return
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError(
            "NoveltyBench generation resume config mismatch for "
            f"{manifest_path}. Existing manifest={actual_manifest}, current={expected_manifest}"
        )


def _load_existing_generation_rows(generations_path: Path) -> dict[str, dict[str, Any]]:
    try:
        existing_rows = read_jsonl(generations_path)
    except json.JSONDecodeError:
        existing_rows = read_jsonl(generations_path, allow_trailing_invalid=True)
        write_jsonl(generations_path, existing_rows)

    existing_by_id: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        prompt_id = str(row.get("id", "")).strip()
        if not prompt_id:
            raise ValueError(f"NoveltyBench generation artifact contains a row without an id: {generations_path}")
        if prompt_id in existing_by_id:
            raise ValueError(f"NoveltyBench generation artifact contains duplicate id {prompt_id}: {generations_path}")
        existing_by_id[prompt_id] = row
    return existing_by_id


def generation_stage_is_complete(
    *,
    noveltybench_cfg: Any,
    run_dir: str | Path,
    model_cfg: Any,
    eval_target_cfg: Any,
    base_seed: int,
    prompts_by_split: dict[str, list[NoveltyBenchPrompt]] | None = None,
) -> bool:
    run_path = ensure_dir(run_dir)
    prompts_by_split = prompts_by_split or _load_prompts_by_split(noveltybench_cfg)
    _ensure_generation_manifest(
        run_dir=run_path,
        model_cfg=model_cfg,
        noveltybench_cfg=noveltybench_cfg,
        eval_target_cfg=eval_target_cfg,
        base_seed=base_seed,
    )
    for split_name, prompt_rows in prompts_by_split.items():
        generations_path = split_output_dir(run_path, split_name) / "generations.jsonl"
        expected_ids = [row.prompt_id for row in prompt_rows]
        if not artifact_is_complete(generations_path, expected_ids):
            return False
    return True


def _write_run_summary(run_dir: Path) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    total_prompts = 0
    weighted_distinct = 0.0
    weighted_utility = 0.0
    for split_name in ("curated", "wildchat"):
        summary_path = split_output_dir(run_dir, split_name) / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = dict(payload.get("metrics", {}))
        split_summaries[split_name] = metrics
        num_prompts = int(metrics.get("num_prompts", 0))
        total_prompts += num_prompts
        weighted_distinct += float(metrics.get("mean_distinct", 0.0)) * num_prompts
        weighted_utility += float(metrics.get("mean_utility", 0.0)) * num_prompts

    aggregate = {
        "num_prompts": total_prompts,
        "num_splits_completed": len(split_summaries),
        "mean_distinct": (weighted_distinct / total_prompts) if total_prompts else 0.0,
        "mean_utility": (weighted_utility / total_prompts) if total_prompts else 0.0,
    }
    summary_payload = {
        "benchmark": "noveltybench",
        "eval_target": resolve_eval_target_name(run_dir),
        "metrics": aggregate,
        "splits": split_summaries,
    }
    metrics_dir = ensure_dir(run_dir / "metrics")
    write_json(run_dir / "summary.json", summary_payload)
    write_json(metrics_dir / "summary.json", aggregate)
    write_csv(
        metrics_dir / "per_split.csv",
        [
            {
                "split": split_name,
                **metrics,
            }
            for split_name, metrics in split_summaries.items()
        ],
    )
    return summary_payload


def run_generation_stage(
    *,
    generation_backend: GenerationBackend,
    noveltybench_cfg: Any,
    output_dir: str | Path,
    full_cfg: Any,
    eval_target_cfg: Any,
    base_seed: int,
    prompts_by_split: dict[str, list[NoveltyBenchPrompt]] | None = None,
) -> dict[str, Any]:
    run_dir = ensure_dir(output_dir)
    _write_resolved_config(run_dir, full_cfg)
    prompts_by_split = prompts_by_split or _load_prompts_by_split(noveltybench_cfg)
    num_generations = int(noveltybench_cfg.generation.num_generations)
    resume = bool(getattr(noveltybench_cfg, "resume", True))
    model_cfg = getattr(full_cfg, "model", None)
    _ensure_generation_manifest(
        run_dir=run_dir,
        model_cfg=model_cfg,
        noveltybench_cfg=noveltybench_cfg,
        eval_target_cfg=eval_target_cfg,
        base_seed=base_seed,
    )
    generation_summary: dict[str, Any] = {}

    for split_name, prompt_rows in prompts_by_split.items():
        split_dir = ensure_dir(split_output_dir(run_dir, split_name))
        generations_path = split_dir / "generations.jsonl"
        expected_ids = [row.prompt_id for row in prompt_rows]
        if resume and artifact_is_complete(generations_path, expected_ids):
            generation_summary[split_name] = {"num_prompts": len(prompt_rows), "skipped": True}
            continue

        existing_by_id: dict[str, dict[str, Any]] = {}
        if resume and generations_path.exists():
            existing_by_id = _load_existing_generation_rows(generations_path)
        else:
            write_jsonl(generations_path, [])

        pending_prompt_rows = [
            (prompt_index, row)
            for prompt_index, row in enumerate(prompt_rows)
            if row.prompt_id not in existing_by_id
        ]

        progress = tqdm(
            total=len(pending_prompt_rows) * num_generations,
            desc=f"NoveltyBench generate [{split_name}]",
            leave=False,
        )
        try:
            batch_size = max(int(getattr(noveltybench_cfg.generation, "batch_size", 1)), 1)
            for batch_start in range(0, len(pending_prompt_rows), batch_size):
                prompt_batch = pending_prompt_rows[batch_start : batch_start + batch_size]
                batch_payloads = [
                    {
                        "id": row.prompt_id,
                        "split": split_name,
                        "prompt": row.prompt,
                        "metadata": row.metadata,
                        "generations": [],
                        "raw_generations": [],
                        "had_thinking_tags": [],
                        "seeds": [],
                    }
                    for _, row in prompt_batch
                ]
                for generation_index in range(num_generations):
                    seed_batch = [
                        _seed_for(base_seed, prompt_index, generation_index)
                        for prompt_index, _ in prompt_batch
                    ]
                    raw_outputs = generation_backend.generate_text_batch(
                        [row.prompt for _, row in prompt_batch],
                        seed_batch,
                    )
                    for row_offset, ((_, prompt_row), raw_output, seed_value) in enumerate(
                        zip(prompt_batch, raw_outputs, seed_batch, strict=True)
                    ):
                        payload = batch_payloads[row_offset]
                        cleaned_output = generation_backend.strip_response(str(raw_output)).strip()
                        payload["raw_generations"].append(str(raw_output))
                        payload["generations"].append(cleaned_output)
                        payload["had_thinking_tags"].append(cleaned_output != str(raw_output).strip())
                        payload["seeds"].append(int(seed_value))
                    progress.update(len(batch_payloads))
                append_jsonl(generations_path, batch_payloads)
        finally:
            progress.close()
        generation_summary[split_name] = {"num_prompts": len(prompt_rows), "skipped": False}

    write_json(
        run_dir / "generation_summary.json",
        {
            "eval_target": str(eval_target_cfg.name),
            "benchmark": "noveltybench",
            "splits": generation_summary,
        },
    )
    return generation_summary


def run_partition_stage(
    *,
    noveltybench_cfg: Any,
    run_dir: str | Path,
    classifier: GenerationSimilarityClassifier | None = None,
) -> dict[str, Any]:
    run_path = ensure_dir(run_dir)
    prompts_by_split = _load_prompts_by_split(noveltybench_cfg)
    classifier = classifier or GenerationSimilarityClassifier.from_config(noveltybench_cfg.classifier)
    resume = bool(getattr(noveltybench_cfg, "resume", True))
    stage_summary: dict[str, Any] = {}

    for split_name, prompt_rows in prompts_by_split.items():
        split_dir = ensure_dir(split_output_dir(run_path, split_name))
        generations_path = split_dir / "generations.jsonl"
        if not generations_path.exists():
            raise FileNotFoundError(f"NoveltyBench generation artifact missing: {generations_path}")
        partition_path = split_dir / "partitions.jsonl"
        expected_ids = [row.prompt_id for row in prompt_rows]
        if resume and artifact_is_complete(partition_path, expected_ids):
            stage_summary[split_name] = {"num_prompts": len(prompt_rows), "skipped": True}
            continue

        generation_rows = read_jsonl(generations_path)
        partition_rows: list[dict[str, Any]] = []
        for row in tqdm(generation_rows, desc=f"NoveltyBench partition [{split_name}]", leave=False):
            partition = partition_responses(
                prompt=str(row["prompt"]),
                responses=[str(generation) for generation in row["generations"]],
                classifier=classifier,
            )
            partition_rows.append(
                {
                    **row,
                    "partition": partition,
                    "num_partitions": (max(partition) + 1) if partition else 0,
                    "distinct": (max(partition) + 1) if partition else 0,
                }
            )
        write_jsonl(partition_path, partition_rows)
        stage_summary[split_name] = {"num_prompts": len(prompt_rows), "skipped": False}

    write_json(run_path / "partition_summary.json", {"benchmark": "noveltybench", "splits": stage_summary})
    return stage_summary


def run_score_stage(
    *,
    noveltybench_cfg: Any,
    run_dir: str | Path,
    scorer: RewardModelScorer | None = None,
) -> dict[str, Any]:
    run_path = ensure_dir(run_dir)
    prompts_by_split = _load_prompts_by_split(noveltybench_cfg)
    scorer = scorer or RewardModelScorer.from_config(noveltybench_cfg.reward_model)
    resume = bool(getattr(noveltybench_cfg, "resume", True))
    patience = float(getattr(noveltybench_cfg, "patience", 0.8))
    split_summaries: dict[str, Any] = {}

    for split_name, prompt_rows in prompts_by_split.items():
        split_dir = ensure_dir(split_output_dir(run_path, split_name))
        partition_path = split_dir / "partitions.jsonl"
        if not partition_path.exists():
            raise FileNotFoundError(f"NoveltyBench partition artifact missing: {partition_path}")
        scores_path = split_dir / "scores.jsonl"
        expected_ids = [row.prompt_id for row in prompt_rows]
        if resume and artifact_is_complete(scores_path, expected_ids):
            summary_path = split_dir / "summary.json"
            if summary_path.exists():
                split_summaries[split_name] = json.loads(summary_path.read_text(encoding="utf-8")).get("metrics", {})
            continue

        partition_rows = read_jsonl(partition_path)
        scored_rows: list[dict[str, Any]] = []
        for row in tqdm(partition_rows, desc=f"NoveltyBench score [{split_name}]", leave=False):
            generation_scores, partition_scores, raw_rewards, utility = score_partition(
                prompt=str(row["prompt"]),
                generations=[str(generation) for generation in row["generations"]],
                partition=[int(partition_id) for partition_id in row["partition"]],
                patience=patience,
                scorer=scorer,
            )
            scored_rows.append(
                {
                    **row,
                    "raw_rewards": raw_rewards,
                    "generation_scores": generation_scores,
                    "partition_scores": partition_scores,
                    "utility": utility,
                }
            )
        write_jsonl(scores_path, scored_rows)
        split_summary = summarize_scores(scored_rows)
        split_summary["split"] = split_name
        split_summary["patience"] = patience
        split_summaries[split_name] = split_summary
        write_json(
            split_dir / "summary.json",
            {
                "benchmark": "noveltybench",
                "split": split_name,
                "metrics": split_summary,
            },
        )

    summary_payload = _write_run_summary(run_path)
    return summary_payload


def run_noveltybench_eval(
    *,
    generation_backend: GenerationBackend,
    noveltybench_cfg: Any,
    output_dir: str | Path,
    full_cfg: Any,
    eval_target_cfg: Any,
    base_seed: int,
    classifier: GenerationSimilarityClassifier | None = None,
    scorer: RewardModelScorer | None = None,
) -> dict[str, Any]:
    run_generation_stage(
        generation_backend=generation_backend,
        noveltybench_cfg=noveltybench_cfg,
        output_dir=output_dir,
        full_cfg=full_cfg,
        eval_target_cfg=eval_target_cfg,
        base_seed=base_seed,
    )
    _release_generation_backend(generation_backend)
    run_partition_stage(
        noveltybench_cfg=noveltybench_cfg,
        run_dir=output_dir,
        classifier=classifier,
    )
    return run_score_stage(
        noveltybench_cfg=noveltybench_cfg,
        run_dir=output_dir,
        scorer=scorer,
    )

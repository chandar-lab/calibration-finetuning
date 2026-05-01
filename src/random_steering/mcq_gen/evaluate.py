from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from random_steering.inference.base import GenerationBackend
from random_steering.inference.hf_backend import HFGenerationBackend
from random_steering.mcq_gen.metrics import compute_mcq_metrics
from random_steering.mcq_gen.parser import parse_mcq_generation
from random_steering.utils.io import ensure_dir, write_csv, write_json, write_jsonl


def _compat_backend_model_cfg(model_cfg: Any, generation_cfg: Any, *, batch_size: int, use_chat_template: bool) -> Any:
    return SimpleNamespace(
        enable_thinking=getattr(model_cfg, "enable_thinking", None),
        reasoning_effort=getattr(model_cfg, "reasoning_effort", None),
        generation_prefix=getattr(model_cfg, "generation_prefix", None),
        max_new_tokens=int(getattr(generation_cfg, "max_new_tokens", 256)),
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

def generate_mcq_rows(
    *,
    generation_backend: GenerationBackend | None = None,
    mcq_gen_cfg: Any,
    prompt: str,
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
                mcq_gen_cfg.generation,
                batch_size=int(getattr(mcq_gen_cfg, "batch_size", 1)),
                use_chat_template=True,
            ),
        )
    batch_size = int(getattr(mcq_gen_cfg, "batch_size", 1))
    num_samples = int(getattr(mcq_gen_cfg, "num_samples", 1000))
    request_indices = list(range(num_samples))
    rows: list[dict[str, Any]] = []

    progress = tqdm(total=num_samples, desc="MCQ generation", leave=False)
    try:
        for batch_request_indices in _iter_batches(request_indices, batch_size):
            prompt_batch = [prompt] * len(batch_request_indices)
            seed_batch = [int(base_seed) + int(request_index) for request_index in batch_request_indices]
            raw_outputs = generation_backend.generate_text_batch(prompt_batch, seed_batch)
            for request_index, raw_output in zip(batch_request_indices, raw_outputs):
                parsed = parse_mcq_generation(raw_output)
                row = {
                    "request_index": int(request_index),
                    "seed": int(base_seed) + int(request_index),
                    "prompt": prompt,
                    "raw_response": raw_output,
                }
                row.update(parsed.to_dict())
                rows.append(row)
            progress.update(len(batch_request_indices))
    finally:
        progress.close()

    return rows


def write_mcq_gen_artifacts(
    *,
    run_dir: str | Path,
    full_cfg: Any,
    eval_target_name: str,
    sample_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    answer_frequency_rows: list[dict[str, Any]],
) -> None:
    run_path = ensure_dir(run_dir)
    metrics_dir = ensure_dir(run_path / "metrics")
    samples_dir = ensure_dir(run_path / "samples")
    write_json(run_path / "config_resolved.json", OmegaConf.to_container(full_cfg, resolve=True))
    write_json(
        run_path / "summary.json",
        {
            "eval_target": eval_target_name,
            "benchmark": "mcq_gen",
            "metrics": summary,
        },
    )
    write_json(metrics_dir / "summary.json", summary)
    write_csv(metrics_dir / "answer_frequencies.csv", answer_frequency_rows)
    write_jsonl(samples_dir / "generated_mcqs.jsonl", sample_rows)


def run_mcq_gen(
    *,
    generation_backend: GenerationBackend | None = None,
    mcq_gen_cfg: Any,
    eval_target_cfg: Any,
    output_dir: str | Path,
    full_cfg: Any,
    prompt: str,
    base_seed: int,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_cfg: Any | None = None,
) -> dict[str, Any]:
    sample_rows = generate_mcq_rows(
        generation_backend=generation_backend,
        mcq_gen_cfg=mcq_gen_cfg,
        prompt=prompt,
        base_seed=base_seed,
        model=model,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
    )
    summary, answer_frequency_rows = compute_mcq_metrics(sample_rows)
    write_mcq_gen_artifacts(
        run_dir=output_dir,
        full_cfg=full_cfg,
        eval_target_name=str(eval_target_cfg.name),
        sample_rows=sample_rows,
        summary=summary,
        answer_frequency_rows=answer_frequency_rows,
    )
    return {
        "sample_rows": sample_rows,
        "summary": summary,
        "answer_frequency_rows": answer_frequency_rows,
    }

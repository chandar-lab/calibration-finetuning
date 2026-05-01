from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.retention.tinybenchmarks_data import DATASET_SIZE
from random_steering.retention.tinybenchmarks_eval import run_retention_eval
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / str(cfg.eval_target.name))


def _configure_external_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    try:
        from datasets.utils import logging as datasets_logging
        from huggingface_hub.utils import logging as hub_logging
        from transformers.utils import logging as transformers_logging

        datasets_logging.disable_progress_bar()
        datasets_logging.set_verbosity_error()
        hub_logging.set_verbosity_error()
        transformers_logging.set_verbosity_warning()
    except Exception:
        pass


def _print_run_summary(cfg: Any, run_dir: Path) -> None:
    max_examples_per_task = getattr(cfg.retention, "max_examples_per_task", None)
    num_examples = DATASET_SIZE if max_examples_per_task is None else min(DATASET_SIZE, int(max_examples_per_task))
    generation_cfg = cfg.retention.generation
    temperature = float(getattr(generation_cfg, "temperature", 0.0))
    do_sample = temperature > 0.0

    print("=== TinyBenchmarks Retention Eval ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"seed: {int(cfg.seed)}", flush=True)
    print(f"eval_target: {cfg.eval_target.name}", flush=True)
    print(f"base_checkpoint: {cfg.eval_target.base_checkpoint}", flush=True)
    print(f"adapter_checkpoint: {getattr(cfg.eval_target, 'adapter_checkpoint', None)}", flush=True)
    print(f"tokenizer_checkpoint: {cfg.eval_target.tokenizer_checkpoint}", flush=True)
    print(f"inference_backend: {cfg.inference.backend}", flush=True)
    print(f"device: {cfg.model.device}", flush=True)
    print(f"dtype: {cfg.model.dtype}", flush=True)
    print(f"enable_thinking: {getattr(cfg.model, 'enable_thinking', None)}", flush=True)
    print(f"use_chat_template: {bool(getattr(cfg.retention, 'use_chat_template', True))}", flush=True)
    print(f"batch_size: {int(getattr(cfg.retention, 'batch_size', 1))}", flush=True)
    print(f"do_sample: {do_sample}", flush=True)
    print(f"temperature: {temperature}", flush=True)
    print(f"top_p: {float(getattr(generation_cfg, 'top_p', 1.0))}", flush=True)
    print(f"max_new_tokens: {int(getattr(generation_cfg, 'max_new_tokens', 256))}", flush=True)
    print("benchmarks:", flush=True)
    for task_name in cfg.retention.enabled_tasks:
        print(f"  - {task_name}: {num_examples} examples", flush=True)
    print("", flush=True)


def main_impl(cfg: Any) -> Path:
    _configure_external_logging()
    set_global_seed(int(cfg.seed))
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    _print_run_summary(cfg, run_dir)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    generation_model_cfg = OmegaConf.merge(
        model_cfg,
        cfg.retention.generation,
        {
            "batch_size": int(getattr(cfg.retention, "batch_size", 1)),
            "use_chat_template": bool(getattr(cfg.retention, "use_chat_template", True)),
            "do_sample": float(getattr(cfg.retention.generation, "temperature", 0.0)) > 0.0,
        },
    )
    enabled_tasks = {str(task_name) for task_name in cfg.retention.enabled_tasks}
    assets = load_evaluation_assets(
        generation_model_cfg,
        cfg.eval_target,
        cfg.inference,
        require_generation_backend=bool({"tiny_gsm8k_strict", "tiny_gsm8k_flexible"} & enabled_tasks),
        require_hf_model=bool(enabled_tasks - {"tiny_gsm8k_strict", "tiny_gsm8k_flexible"}),
    )
    run_retention_eval(
        generation_backend=assets.generation_backend,
        hf_model=assets.hf_model,
        tokenizer=assets.tokenizer,
        model_cfg=generation_model_cfg,
        retention_cfg=cfg.retention,
        eval_target_cfg=cfg.eval_target,
        output_dir=run_dir,
        full_cfg=cfg,
    )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="retention_eval_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

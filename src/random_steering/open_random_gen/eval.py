from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from random_steering.calibrate_sft.modeling import load_evaluation_assets, load_evaluation_bundle
from random_steering.inference.ssot import SSOTMode, maybe_wrap_generation_backend
from random_steering.open_random_gen.data import load_open_random_gen_prompts
from random_steering.open_random_gen.evaluate import run_open_random_gen
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / f"{cfg.eval_target.name}_open_random_gen")


def _print_run_summary(cfg: Any, run_dir: Path, prompt_count: int, use_chat_template: bool) -> None:
    generation_cfg = cfg.open_random_gen.generation
    inference_backend = getattr(getattr(cfg, "inference", None), "backend", "hf")
    print("=== Open Random Generation Eval ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"seed: {int(cfg.seed)}", flush=True)
    print(f"eval_target: {cfg.eval_target.name}", flush=True)
    print(f"base_checkpoint: {cfg.eval_target.base_checkpoint}", flush=True)
    print(f"adapter_checkpoint: {getattr(cfg.eval_target, 'adapter_checkpoint', None)}", flush=True)
    print(f"tokenizer_checkpoint: {cfg.eval_target.tokenizer_checkpoint}", flush=True)
    print(f"inference_backend: {inference_backend}", flush=True)
    print(f"device: {cfg.model.device}", flush=True)
    print(f"dtype: {cfg.model.dtype}", flush=True)
    print(f"enable_thinking: {getattr(cfg.model, 'enable_thinking', None)}", flush=True)
    print(f"use_chat_template: {use_chat_template}", flush=True)
    print(f"compute_support_metrics: {bool(getattr(cfg.open_random_gen, 'compute_support_metrics', True))}", flush=True)
    print(f"compute_sampling_metrics: {bool(getattr(cfg.open_random_gen, 'compute_sampling_metrics', True))}", flush=True)
    print(f"temperature: {float(getattr(generation_cfg, 'temperature', 1.0))}", flush=True)
    print(f"top_p: {float(getattr(generation_cfg, 'top_p', 1.0))}", flush=True)
    print(f"max_new_tokens: {int(getattr(generation_cfg, 'max_new_tokens', 16))}", flush=True)
    print(f"batch_size: {int(getattr(cfg.open_random_gen, 'batch_size', 1))}", flush=True)
    print(f"prompt_count: {prompt_count}", flush=True)
    print("", flush=True)


def _ssot_enabled(cfg: Any) -> bool:
    inference_method_cfg = getattr(cfg, "inference_method", None)
    return bool(getattr(inference_method_cfg, "enabled", False)) and str(getattr(inference_method_cfg, "name", "")) == "ssot"


def main_impl(cfg: Any) -> Path:
    set_global_seed(int(cfg.seed))
    if _ssot_enabled(cfg) and bool(getattr(cfg.open_random_gen, "compute_support_metrics", True)):
        raise ValueError("inference_method=ssot is only supported for sampling metrics; set open_random_gen.compute_support_metrics=false.")
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    prompts = load_open_random_gen_prompts(cfg.open_random_gen.prompts_path)
    if hasattr(cfg, "inference"):
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        generation_model_cfg = OmegaConf.merge(
            model_cfg,
            cfg.open_random_gen.generation,
            {
                "batch_size": int(getattr(cfg.open_random_gen, "batch_size", 1)),
                "use_chat_template": True,
            },
        )
        assets = load_evaluation_assets(
            generation_model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=bool(getattr(cfg.open_random_gen, "compute_sampling_metrics", True)),
            require_hf_model=bool(getattr(cfg.open_random_gen, "compute_support_metrics", True)),
        )
        assets.generation_backend = maybe_wrap_generation_backend(
            assets.generation_backend,
            getattr(cfg, "inference_method", None),
            mode=SSOTMode.DAG_OPEN_RANDOM,
        )
        _print_run_summary(cfg, run_dir, len(prompts), bool(getattr(assets.tokenizer, "chat_template", None)))
        run_open_random_gen(
            generation_backend=assets.generation_backend,
            hf_model=assets.hf_model,
            tokenizer=assets.tokenizer,
            model_cfg=cfg.model,
            open_random_gen_cfg=cfg.open_random_gen,
            eval_target_cfg=cfg.eval_target,
            output_dir=run_dir,
            full_cfg=cfg,
            prompts=prompts,
            base_seed=int(cfg.seed),
        )
    else:
        bundle = load_evaluation_bundle(cfg.model, cfg.eval_target)
        _print_run_summary(cfg, run_dir, len(prompts), bool(getattr(bundle.tokenizer, "chat_template", None)))
        run_open_random_gen(
            generation_backend=None,
            hf_model=bundle.model,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            model_cfg=cfg.model,
            open_random_gen_cfg=cfg.open_random_gen,
            eval_target_cfg=cfg.eval_target,
            output_dir=run_dir,
            full_cfg=cfg,
            prompts=prompts,
            base_seed=int(cfg.seed),
        )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="open_random_gen_eval_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

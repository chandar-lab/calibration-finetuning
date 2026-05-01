from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from random_steering.calibrate_sft.modeling import load_evaluation_assets, load_evaluation_bundle
from random_steering.mcq_gen.evaluate import run_mcq_gen
from random_steering.mcq_gen.prompt import get_medical_mcq_prompt
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / f"{cfg.eval_target.name}_mcq_gen")


def _print_run_summary(cfg: Any, run_dir: Path) -> None:
    generation_cfg = cfg.mcq_gen.generation
    inference_backend = getattr(getattr(cfg, "inference", None), "backend", "hf")
    print("=== MCQ Generation Eval ===", flush=True)
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
    print(f"batch_size: {int(getattr(cfg.mcq_gen, 'batch_size', 1))}", flush=True)
    print(f"num_samples: {int(getattr(cfg.mcq_gen, 'num_samples', 1000))}", flush=True)
    print(f"temperature: {float(getattr(generation_cfg, 'temperature', 1.0))}", flush=True)
    print(f"top_p: {float(getattr(generation_cfg, 'top_p', 1.0))}", flush=True)
    print(f"max_new_tokens: {int(getattr(generation_cfg, 'max_new_tokens', 256))}", flush=True)
    print("", flush=True)


def main_impl(cfg: Any) -> Path:
    set_global_seed(int(cfg.seed))
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    _print_run_summary(cfg, run_dir)
    if hasattr(cfg, "inference"):
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        generation_model_cfg = OmegaConf.merge(
            model_cfg,
            cfg.mcq_gen.generation,
            {
                "batch_size": int(getattr(cfg.mcq_gen, "batch_size", 1)),
                "use_chat_template": True,
            },
        )
        assets = load_evaluation_assets(
            generation_model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=True,
            require_hf_model=False,
        )
        run_mcq_gen(
            generation_backend=assets.generation_backend,
            mcq_gen_cfg=cfg.mcq_gen,
            eval_target_cfg=cfg.eval_target,
            output_dir=run_dir,
            full_cfg=cfg,
            prompt=get_medical_mcq_prompt(),
            base_seed=int(cfg.seed),
        )
    else:
        bundle = load_evaluation_bundle(cfg.model, cfg.eval_target)
        run_mcq_gen(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            model_cfg=cfg.model,
            mcq_gen_cfg=cfg.mcq_gen,
            eval_target_cfg=cfg.eval_target,
            output_dir=run_dir,
            full_cfg=cfg,
            prompt=get_medical_mcq_prompt(),
            base_seed=int(cfg.seed),
        )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="mcq_gen_eval_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

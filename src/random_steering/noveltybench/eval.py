from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.noveltybench.evaluate import _stage_model_cfg, run_noveltybench_eval
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / f"{cfg.eval_target.name}_noveltybench")


def _print_run_summary(cfg: Any, run_dir: Path) -> None:
    generation_cfg = cfg.noveltybench.generation
    print("=== NoveltyBench Eval ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"seed: {int(cfg.seed)}", flush=True)
    print(f"eval_target: {cfg.eval_target.name}", flush=True)
    print(f"base_checkpoint: {cfg.eval_target.base_checkpoint}", flush=True)
    print(f"adapter_checkpoint: {getattr(cfg.eval_target, 'adapter_checkpoint', None)}", flush=True)
    print(f"tokenizer_checkpoint: {cfg.eval_target.tokenizer_checkpoint}", flush=True)
    print(f"inference_backend: {cfg.inference.backend}", flush=True)
    print(f"enabled_splits: {list(cfg.noveltybench.enabled_splits)}", flush=True)
    print(f"num_generations: {int(generation_cfg.num_generations)}", flush=True)
    print(f"batch_size: {int(generation_cfg.batch_size)}", flush=True)
    print(f"max_new_tokens: {int(generation_cfg.max_new_tokens)}", flush=True)
    print(f"temperature: {float(generation_cfg.temperature)}", flush=True)
    print(f"top_p: {float(generation_cfg.top_p)}", flush=True)
    print(f"enable_thinking: {getattr(cfg.model, 'enable_thinking', None)}", flush=True)
    print("", flush=True)


def main_impl(cfg: Any) -> Path:
    set_global_seed(int(cfg.seed))
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    _print_run_summary(cfg, run_dir)
    model_cfg = _stage_model_cfg(cfg.model, cfg.noveltybench)
    assets = load_evaluation_assets(
        model_cfg,
        cfg.eval_target,
        cfg.inference,
        require_generation_backend=True,
        require_hf_model=False,
    )
    run_noveltybench_eval(
        generation_backend=assets.generation_backend,
        noveltybench_cfg=cfg.noveltybench,
        output_dir=run_dir,
        full_cfg=cfg,
        eval_target_cfg=cfg.eval_target,
        base_seed=int(cfg.seed),
    )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="noveltybench_eval_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

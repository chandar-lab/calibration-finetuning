from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.noveltybench.evaluate import (
    _stage_model_cfg,
    generation_stage_is_complete,
    run_generation_stage,
)
from random_steering.noveltybench.data import load_split_prompts
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    configured = getattr(cfg, "run_dir", None)
    if configured:
        return ensure_dir(Path(configured))
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / f"{cfg.eval_target.name}_noveltybench_generate")


def main_impl(cfg: Any) -> Path:
    set_global_seed(int(cfg.seed))
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    model_cfg = _stage_model_cfg(cfg.model, cfg.noveltybench)
    max_prompts = getattr(cfg.noveltybench, "max_prompts_per_split", None)
    max_prompts = None if max_prompts is None else int(max_prompts)
    prompts_by_split = {
        split_name: load_split_prompts(
            cfg.noveltybench.assets_root,
            split_name,
            max_prompts=max_prompts,
        )
        for split_name in cfg.noveltybench.enabled_splits
    }
    if bool(getattr(cfg.noveltybench, "resume", True)) and generation_stage_is_complete(
        noveltybench_cfg=cfg.noveltybench,
        run_dir=run_dir,
        model_cfg=cfg.model,
        eval_target_cfg=cfg.eval_target,
        base_seed=int(cfg.seed),
        prompts_by_split=prompts_by_split,
    ):
        print(run_dir, flush=True)
        return run_dir
    assets = load_evaluation_assets(
        model_cfg,
        cfg.eval_target,
        cfg.inference,
        require_generation_backend=True,
        require_hf_model=False,
    )
    run_generation_stage(
        generation_backend=assets.generation_backend,
        noveltybench_cfg=cfg.noveltybench,
        output_dir=run_dir,
        full_cfg=cfg,
        eval_target_cfg=cfg.eval_target,
        base_seed=int(cfg.seed),
        prompts_by_split=prompts_by_split,
    )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="noveltybench_generate_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

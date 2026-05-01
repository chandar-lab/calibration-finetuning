from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.perplexity.evaluate import run_perplexity_eval
from random_steering.utils.io import ensure_dir
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / f"{cfg.eval_target.name}_perplexity")


def _print_run_summary(cfg: Any, run_dir: Path) -> None:
    print("=== Perplexity Eval ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"seed: {int(cfg.seed)}", flush=True)
    print(f"eval_target: {cfg.eval_target.name}", flush=True)
    print(f"base_checkpoint: {cfg.eval_target.base_checkpoint}", flush=True)
    print(f"adapter_checkpoint: {getattr(cfg.eval_target, 'adapter_checkpoint', None)}", flush=True)
    print(f"tokenizer_checkpoint: {cfg.eval_target.tokenizer_checkpoint}", flush=True)
    print(f"inference_backend: {cfg.inference.backend}", flush=True)
    print(f"device: {cfg.model.device}", flush=True)
    print(f"dtype: {cfg.model.dtype}", flush=True)
    print(f"slices: {list(cfg.perplexity.slices)}", flush=True)
    print(f"split: {cfg.perplexity.split}", flush=True)
    print(f"context_length: {int(cfg.perplexity.context_length)}", flush=True)
    print(f"stride: {int(cfg.perplexity.stride)}", flush=True)
    print(f"batch_size: {int(cfg.perplexity.batch_size)}", flush=True)
    print(f"max_documents: {getattr(cfg.perplexity, 'max_documents', None)}", flush=True)
    print(f"max_tokens: {getattr(cfg.perplexity, 'max_tokens', None)}", flush=True)
    print("", flush=True)


def main_impl(cfg: Any) -> Path:
    if str(cfg.inference.backend) != "hf":
        raise ValueError("Perplexity eval is HF-only.")
    set_global_seed(int(cfg.seed))
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    _print_run_summary(cfg, run_dir)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    assets = load_evaluation_assets(
        model_cfg,
        cfg.eval_target,
        cfg.inference,
        require_generation_backend=False,
        require_hf_model=True,
    )
    run_perplexity_eval(
        model=assets.hf_model,
        tokenizer=assets.tokenizer,
        perplexity_cfg=cfg.perplexity,
        eval_target_cfg=cfg.eval_target,
        output_dir=run_dir,
        full_cfg=cfg,
    )
    print(run_dir, flush=True)
    return run_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="perplexity_eval_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

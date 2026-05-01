from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from random_steering.calibrate_sft.data import build_prepared_selected_splits
from random_steering.calibrate_sft.evaluate import run_logit_evaluation, run_sample_evaluation
from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.inference.ssot import SSOTMode, maybe_wrap_generation_backend
from random_steering.utils.io import ensure_dir, write_json
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / str(cfg.eval_target.name))


def _summarize_logit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "splits": {}}

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row["split"])
        split_rows.setdefault(split, []).append(row)

    summary: dict[str, Any] = {"num_rows": len(rows), "splits": {}}
    for split_name, split_metric_rows in split_rows.items():
        family_rows: dict[str, list[dict[str, Any]]] = {}
        for row in split_metric_rows:
            family_id = str(row["family_id"])
            family_rows.setdefault(family_id, []).append(row)

        family_means = [
            float(sum(float(item["avg_logit_kl"]) for item in grouped_rows) / len(grouped_rows))
            for grouped_rows in family_rows.values()
        ]
        split_summary: dict[str, Any] = {
            "num_rows": len(split_metric_rows),
            "num_families": len(family_rows),
            "aggregation": "equal_family_weight",
        }
        if family_means:
            split_summary["avg_logit_kl"] = float(sum(family_means) / len(family_means))
        summary["splits"][split_name] = split_summary

    return summary


def _cleanup_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _ssot_enabled(cfg: Any) -> bool:
    inference_method_cfg = getattr(cfg, "inference_method", None)
    return bool(getattr(inference_method_cfg, "enabled", False)) and str(getattr(inference_method_cfg, "name", "")) == "ssot"


def run_calibrate_eval(cfg: Any) -> None:
    set_global_seed(int(cfg.seed))
    if _ssot_enabled(cfg) and bool(getattr(cfg.experiment.logit_eval, "enabled", True)):
        raise ValueError("inference_method=ssot is only supported for sample evaluation; set experiment.logit_eval.enabled=false.")
    run_dir = ensure_dir(_resolve_run_dir(cfg))
    metrics_dir = ensure_dir(run_dir / "metrics")

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    generation_model_cfg = OmegaConf.merge(
        model_cfg,
        {
            "batch_size": int(getattr(cfg.experiment.sample_eval, "batch_size", 1)),
            "use_chat_template": True,
        },
    )
    write_json(run_dir / "config_resolved.json", OmegaConf.to_container(cfg, resolve=True))
    split_names = [str(split_name) for split_name in cfg.eval_target.splits]

    sample_result = {"summary": {"num_rows": 0, "splits": {}}, "per_distribution_rows": []}
    logit_rows: list[dict[str, Any]] = []

    if bool(getattr(cfg.experiment.logit_eval, "enabled", True)):
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting logit_eval.load_assets", flush=True)
        logit_assets = load_evaluation_assets(
            model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=False,
            require_hf_model=True,
        )
        print(
            f"[calibrate-eval] finished logit_eval.load_assets in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting logit_eval.build_prepared_splits", flush=True)
        logit_selected_splits = build_prepared_selected_splits(
            logit_assets.tokenizer,
            cfg.model,
            cfg.data,
            split_names,
        )
        print(
            f"[calibrate-eval] finished logit_eval.build_prepared_splits in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting logit_eval.run", flush=True)
        logit_rows = run_logit_evaluation(
            model=logit_assets.hf_model,
            prepared_splits=logit_selected_splits,
            experiment_cfg=cfg.experiment,
            train_cfg=cfg.train,
            output_dir=run_dir,
            base_seed=int(cfg.seed),
        )
        print(
            f"[calibrate-eval] finished logit_eval.run in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        del logit_assets
        del logit_selected_splits
        _cleanup_cuda_state()

    if bool(getattr(cfg.experiment.sample_eval, "enabled", True)):
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting sample_eval.load_assets", flush=True)
        sample_assets = load_evaluation_assets(
            generation_model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=True,
            require_hf_model=False,
        )
        print(
            f"[calibrate-eval] finished sample_eval.load_assets in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        sample_assets.generation_backend = maybe_wrap_generation_backend(
            sample_assets.generation_backend,
            getattr(cfg, "inference_method", None),
            mode=SSOTMode.PIF_NUMERIC,
        )
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting sample_eval.build_prepared_splits", flush=True)
        sample_selected_splits = build_prepared_selected_splits(
            sample_assets.tokenizer,
            cfg.model,
            cfg.data,
            split_names,
        )
        print(
            f"[calibrate-eval] finished sample_eval.build_prepared_splits in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        stage_start = time.perf_counter()
        print("[calibrate-eval] starting sample_eval.run", flush=True)
        sample_result = run_sample_evaluation(
            generation_backend=sample_assets.generation_backend,
            prepared_splits=sample_selected_splits,
            experiment_cfg=cfg.experiment,
            output_dir=run_dir,
        )
        print(
            f"[calibrate-eval] finished sample_eval.run in {time.perf_counter() - stage_start:.2f}s",
            flush=True,
        )
        del sample_assets
        del sample_selected_splits
        _cleanup_cuda_state()

    write_json(
        metrics_dir / "eval_summary.json",
        {
            "eval_target": str(cfg.eval_target.name),
            "splits": split_names,
            "sample_eval": sample_result["summary"],
            "logit_eval": _summarize_logit_rows(logit_rows),
        },
    )
    print(run_dir, flush=True)


@hydra.main(version_base=None, config_path="../../../conf", config_name="calibrate_eval_config")
def main(cfg: DictConfig) -> None:
    run_calibrate_eval(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from random_steering.noveltybench.evaluate import run_score_stage


def main_impl(cfg: Any) -> Path:
    run_dir = getattr(cfg, "run_dir", None)
    if not run_dir:
        raise ValueError("NoveltyBench score stage requires run_dir to point to a partitioned run directory.")
    run_path = Path(run_dir)
    run_score_stage(
        noveltybench_cfg=cfg.noveltybench,
        run_dir=run_path,
    )
    print(run_path, flush=True)
    return run_path


@hydra.main(version_base=None, config_path="../../../conf", config_name="noveltybench_score_config")
def main(cfg: DictConfig) -> None:
    main_impl(cfg)


if __name__ == "__main__":
    main()

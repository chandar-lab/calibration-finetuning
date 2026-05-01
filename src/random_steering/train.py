from __future__ import annotations

import hydra
from omegaconf import DictConfig

from random_steering.calibrate_sft.train_loop import run_calibrate_sft
from random_steering.hard_label_sft.train_loop import run_hard_label_sft


def run_training(cfg: DictConfig) -> None:
    train_name = str(getattr(cfg.train, "name", ""))
    if train_name in {"calibrate_sft", "calibrate_sft_fsdp"}:
        run_calibrate_sft(cfg)
        return
    if train_name in {"hard_label_sft", "hard_label_sft_fsdp"}:
        run_hard_label_sft(cfg)
        return
    raise ValueError(f"Unsupported train config: {train_name}")


@hydra.main(version_base=None, config_path="../../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()

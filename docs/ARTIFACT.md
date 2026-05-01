# Artifact Guide

This artifact is organized around the paper's training and evaluation workflows.

## Main Workflows

1. Train soft-target Calibration Fine-Tuning with `python -m random_steering.train`.
2. Train hard-target Calibration Fine-Tuning with the same entry point and `train=hard_label_sft_final`.
3. Evaluate structured sampling with `python -m random_steering.calibrate_sft.eval`.
4. Evaluate transfer with the open-generation, MCQ, NoveltyBench, PALOMA, and TinyBenchmarks entry points.

## Config Conventions

All workflows use Hydra configs under `conf/`.

- `conf/model/` selects the base model and generation defaults.
- `conf/data/` controls synthetic distribution construction and discretization.
- `conf/train/` controls optimization, LoRA, and logging.
- `conf/eval_target/` controls baseline or fine-tuned checkpoint loading.
- `conf/*_eval_config.yaml` files define evaluation defaults.

The eval-target configs included here are intentionally generic templates. They avoid private absolute paths and should be edited to point to locally produced checkpoints before evaluating fine-tuned models.

## Included Assets

- `benchmarks/open_random_gen/prompts.json`: prompts for open-ended random generation.
- `benchmarks/noveltybench/`: prompt assets used by the NoveltyBench pipeline.
- `src/random_steering/retention/assets/tinyBenchmarks.pkl`: TinyBenchmarks metadata used for GP-IRT aggregation.

PALOMA corpora are not included because they are large. The perplexity pipeline expects the configured PALOMA-style directory layout under `datasets/paloma`.

## Expected Outputs

By default, runs write under:

- `outputs/train/` for training.
- `outputs/runs/` for structured, open-generation, MCQ, and NoveltyBench evaluations.
- `outputs/perplexity/` for PALOMA.
- `outputs/retention/` for TinyBenchmarks.

These directories are ignored by git.

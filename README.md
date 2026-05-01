# Calibration Fine-Tuning

This repository contains the code for training and evaluating language models as calibrated stochastic generators.

The main paper workflow compares two variants of Calibration Fine-Tuning:

- **Soft-target fine-tuning**: discretize the target output distribution, build a prefix trie over tokenized canonical outputs, and train the model to match trie-induced next-token targets.
- **Hard-target fine-tuning**: sample canonical outputs from the same discretized target distribution and train on the sampled completions with standard next-token cross-entropy.

The code also includes evaluation pipelines for structured numeric sampling, open-ended random generation, MCQ answer-position balance, NoveltyBench, PALOMA perplexity, and TinyBenchmarks retention.

## Repository Layout

- `src/random_steering/calibrate_sft/`: soft-target data construction, loss, training, and structured-sampling evaluation.
- `src/random_steering/hard_label_sft/`: hard-target data construction, loss, and training.
- `src/random_steering/inference/`: shared Hugging Face/vLLM generation backends, chat formatting, and String Seed of Thought wrappers.
- `src/random_steering/open_random_gen/`: open-ended random-generation evaluation.
- `src/random_steering/mcq_gen/`: MCQ answer-position balance evaluation.
- `src/random_steering/noveltybench/`: NoveltyBench generation, partitioning, and scoring.
- `src/random_steering/perplexity/`: PALOMA-style teacher-forced perplexity evaluation.
- `src/random_steering/retention/`: TinyBenchmarks retention evaluation.
- `conf/`: Hydra configs for training and evaluation.
- `benchmarks/`: small benchmark assets used by the open-generation and NoveltyBench evaluations.
- `tests/`: lightweight unit and integration tests with mocked models where possible.

The internal Python package is still named `random_steering` for compatibility with the original experiment code. The artifact-level project name is Calibration Fine-Tuning.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For Hugging Face models and datasets, set cache locations as appropriate for your machine:

```bash
export HF_HOME=/path/to/hf_cache
```

If you use gated Hugging Face models or datasets, authenticate first:

```bash
huggingface-cli login
```

The code uses Hydra for configuration. Training logs are written under `outputs/` by default. W&B logging is disabled in the artifact configs; enable it explicitly with `train.wandb.enabled=true train.wandb.mode=online`.

## Evaluation Assets

Most evaluation axes are either synthetic or ship with the repository. The only large external corpus used by the paper pipeline is PALOMA.

### Structured Numeric Sampling

No external dataset is required. Prompts, target distributions, train/test splits, discretized output spaces, and logit/sample metrics are generated from the Hydra data configs under `conf/data/`.

The main configs are:

- `conf/data/calibrate_sft_final.yaml` for soft-target fine-tuning.
- `conf/data/calibrate_sft_hard_label_final.yaml` for hard-target fine-tuning.
- `conf/data/calibrate_sft_selected.yaml` for structured-sampling evaluation.

### Open-Ended Random Generation

The prompt set is included:

- `benchmarks/open_random_gen/prompts.json`

The loader reads this path from `conf/open_random_gen/open_random_gen.yaml`. No download is needed.

### MCQ Answer-Position Balance

No external dataset is required. The benchmark uses a fixed medical-MCQ prompt defined in:

- `src/random_steering/mcq_gen/prompt.py`

The model generates new MCQs, and the evaluator parses the declared `Correct Answer: A/B/C/D` field.

### NoveltyBench

The prompt assets are included:

- `benchmarks/noveltybench/curated.jsonl`
- `benchmarks/noveltybench/wildchat_1k.jsonl`

The evaluator also uses the same scoring models as the NoveltyBench pipeline:

- Similarity classifier tokenizer: `microsoft/deberta-v3-large`
- Similarity classifier: `yimingzhang/deberta-v3-large-generation-similarity`
- Reward model: `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`

These are downloaded automatically by Transformers unless already present in `HF_HOME`. To pre-download them:

```bash
huggingface-cli download microsoft/deberta-v3-large
huggingface-cli download yimingzhang/deberta-v3-large-generation-similarity
huggingface-cli download Skywork/Skywork-Reward-Gemma-2-27B-v0.2
```

The full NoveltyBench run is expensive because it generates responses, partitions them with the DeBERTa classifier, and scores representative outputs with the reward model. The staged entry points are also available:

```bash
PYTHONPATH=src python -m random_steering.noveltybench.generate
PYTHONPATH=src python -m random_steering.noveltybench.partition run_dir=/path/to/generated_run
PYTHONPATH=src python -m random_steering.noveltybench.score run_dir=/path/to/partitioned_run
```

### TinyBenchmarks Retention

The GP-IRT metadata artifact is included:

- `src/random_steering/retention/assets/tinyBenchmarks.pkl`

The 100-example task datasets are loaded from Hugging Face through `datasets.load_dataset`:

- `tinyBenchmarks/tinyMMLU`
- `tinyBenchmarks/tinyHellaswag`
- `tinyBenchmarks/tinyTruthfulQA`
- `tinyBenchmarks/tinyWinogrande`
- `tinyBenchmarks/tinyGSM8k`

They download automatically into the Hugging Face datasets cache. To pre-cache them:

```bash
python - <<'PY'
from datasets import load_dataset

load_dataset("tinyBenchmarks/tinyMMLU", split="test")
load_dataset("tinyBenchmarks/tinyHellaswag", split="validation")
load_dataset("tinyBenchmarks/tinyTruthfulQA", "multiple_choice", split="validation")
load_dataset("tinyBenchmarks/tinyWinogrande", "winogrande_xl", split="validation")
load_dataset("tinyBenchmarks/tinyGSM8k", "main", split="test")
PY
```

### PALOMA Perplexity

PALOMA is not vendored because it is large and gated by the AI2 ImpACT license. The perplexity loader expects local gzip JSONL files with this layout:

```text
datasets/paloma/<slice_name>/<split>/*.jsonl.gz
```

For the paper-style multi-slice run, use `conf/perplexity/paloma_full_stride_1024.yaml`, which expects:

- `wikitext_103`
- `c4_en`
- `dolma-v1_5`
- `mc4`
- `ptb`
- `redpajama`
- `falcon-refinedweb`

After accepting the dataset terms on Hugging Face, the expected layout can be obtained with:

```bash
huggingface-cli download allenai/paloma \
  --repo-type dataset \
  --local-dir datasets/paloma
```

For a lightweight smoke run, the default `conf/perplexity/standard_lm.yaml` only evaluates `wikitext_103`.

### Model Checkpoints

Base models are selected by configs under `conf/model/`, for example `Qwen/Qwen3-1.7B`. Some model families used in the paper, such as Llama and Gemma, may require accepting model terms on Hugging Face before download.

Fine-tuned adapter checkpoints are not included. Train them with the commands below, or edit the template files under `conf/eval_target/` to point to your local checkpoints.

## Quick Validation

Run the lightweight tests:

```bash
pytest tests
```

Some tests that require gated Hugging Face models, local tokens, or large-model inference are skipped automatically when the required environment is unavailable.

## Training

Soft-target fine-tuning:

```bash
PYTHONPATH=src python -m random_steering.train \
  model=qwen3_1p7b \
  data=calibrate_sft_final \
  train=calibrate_sft_final \
  experiment.name=qwen3_1p7b_soft_target
```

Hard-target fine-tuning:

```bash
PYTHONPATH=src python -m random_steering.train \
  model=qwen3_1p7b \
  data=calibrate_sft_hard_label_final \
  train=hard_label_sft_final \
  experiment.name=qwen3_1p7b_hard_target
```

For multi-GPU FSDP runs, use the corresponding `*_fsdp` train configs with `torchrun`.

## Structured-Sampling Evaluation

Baseline evaluation:

```bash
PYTHONPATH=src python -m random_steering.calibrate_sft.eval \
  model=qwen3_1p7b \
  eval_target=calibrate_sft_qwen3_1p7b_baseline_paper
```

Fine-tuned checkpoint evaluation uses the provided template configs. Replace the placeholder checkpoint paths in `conf/eval_target/calibrate_sft_qwen3_1p7b_soft_template.yaml` or `conf/eval_target/calibrate_sft_qwen3_1p7b_hard_template.yaml`, then run:

```bash
PYTHONPATH=src python -m random_steering.calibrate_sft.eval \
  model=qwen3_1p7b \
  eval_target=calibrate_sft_qwen3_1p7b_soft_template
```

## Transfer and Retention Evaluations

Open-ended random generation:

```bash
PYTHONPATH=src python -m random_steering.open_random_gen.eval \
  model=qwen3_1p7b \
  eval_target=open_random_gen_baseline
```

MCQ answer-position balance:

```bash
PYTHONPATH=src python -m random_steering.mcq_gen.eval \
  model=qwen3_1p7b \
  eval_target=mcq_gen_baseline
```

TinyBenchmarks retention:

```bash
PYTHONPATH=src python -m random_steering.retention.eval \
  model=qwen3_1p7b \
  eval_target=tinybenchmarks_baseline
```

PALOMA perplexity:

```bash
PYTHONPATH=src python -m random_steering.perplexity.eval \
  model=qwen3_1p7b \
  eval_target=calibrate_sft_qwen3_1p7b_baseline_paper
```

NoveltyBench:

```bash
PYTHONPATH=src python -m random_steering.noveltybench.eval \
  model=qwen3_1p7b \
  eval_target=noveltybench_baseline
```

## Notes on Reproducibility

- Final soft-target configs use five-decimal canonical outputs and `max_bins=1001`.
- Final hard-target configs use five-decimal canonical outputs and `max_bins=16384`.
- The default training prompt is:

```text
Generate exactly ONE random number from a [distribution] distribution with parameters [params]. Output ONLY the number.
```

- Evaluation target configs in this artifact are intentionally templates. They avoid machine-specific absolute checkpoint paths and should be edited to point to locally trained checkpoints.

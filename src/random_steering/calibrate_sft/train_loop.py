from __future__ import annotations

from collections import defaultdict
import inspect
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from random_steering.calibrate_sft.data import (
    PreparedPrompt,
    TrainingExample,
    build_prepared_splits,
    output_space_to_dict,
    sample_training_example,
)
from random_steering.calibrate_sft.losses import sequence_calibration_loss
from random_steering.calibrate_sft.modeling import TrainingBundle, load_training_bundle, save_training_bundle
from random_steering.utils.distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
    is_main_process,
    reduce_mean,
)
from random_steering.utils.io import ensure_dir, write_csv, write_json, write_jsonl
from random_steering.utils.seed import set_global_seed


def _resolve_run_dir(cfg: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().run.dir)
    except Exception:
        return ensure_dir(Path(cfg.output_root) / "manual_calibrate_sft_run")


def _init_wandb(cfg: Any):
    wandb_cfg = cfg.train.wandb
    if not bool(wandb_cfg.enabled):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Weights & Biases logging is enabled but `wandb` is not installed.") from exc
    return wandb.init(
        project=str(wandb_cfg.project),
        entity=wandb_cfg.entity,
        name=str(wandb_cfg.name),
        tags=list(wandb_cfg.tags),
        mode=str(wandb_cfg.mode),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _prepared_prompt_to_row(prompt: PreparedPrompt) -> dict[str, Any]:
    return {
        "split": prompt.prompt_spec.split,
        "distribution_id": prompt.prompt_spec.distribution_id,
        "family_id": prompt.prompt_spec.family_id,
        "tier": prompt.prompt_spec.tier,
        "parameters": dict(prompt.prompt_spec.parameters),
        "prompt_text": prompt.prompt_text,
        "output_space": output_space_to_dict(prompt.output_space),
    }


def write_dataset_artifacts(dataset_dir: str | Path, prepared_splits: dict[str, list[PreparedPrompt]]) -> None:
    root = ensure_dir(dataset_dir)
    manifest = {
        "splits": {split_name: len(prompts) for split_name, prompts in prepared_splits.items()},
    }
    write_json(root / "manifest.json", manifest)
    for split_name, prompts in prepared_splits.items():
        write_jsonl(root / f"{split_name}.jsonl", [_prepared_prompt_to_row(prompt) for prompt in prompts])


def collate_training_examples(
    examples: list[TrainingExample],
    pad_token_id: int,
    device: torch.device,
    model_cfg: Any | None = None,
) -> dict[str, Any]:
    max_length = max(len(example.prepared_prompt.prompt_token_ids) + len(example.candidate_token_ids) for example in examples)
    input_ids = torch.full((len(examples), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    token_type_ids = torch.zeros((len(examples), max_length), dtype=torch.long) if _requires_token_type_ids(model_cfg) else None
    step_positions: list[list[int]] = []

    for batch_index, example in enumerate(examples):
        full_tokens = list(example.prepared_prompt.prompt_token_ids + example.candidate_token_ids)
        input_ids[batch_index, : len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask[batch_index, : len(full_tokens)] = 1
        start = len(example.prepared_prompt.prompt_token_ids) - 1
        step_positions.append([start + offset for offset in range(len(example.prefix_targets))])

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device) if token_type_ids is not None else None,
        "prefix_targets": [example.prefix_targets for example in examples],
        "step_positions": step_positions,
    }


def _requires_token_type_ids(model_cfg: Any | None) -> bool:
    if model_cfg is None:
        return False
    model_name = str(getattr(model_cfg, "name", "")).lower()
    checkpoint = str(getattr(model_cfg, "checkpoint", "")).lower()
    return "gemma" in model_name or "gemma" in checkpoint


def _forward_supports_kwarg(model: Any, kwarg_name: str) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    if kwarg_name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def compute_batch_loss(model: Any, batch: dict[str, Any], train_cfg: Any) -> torch.Tensor:
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None and _forward_supports_kwarg(model, "token_type_ids"):
        model_inputs["token_type_ids"] = token_type_ids
    outputs = model(**model_inputs)
    logits = outputs.logits
    losses: list[torch.Tensor] = []
    for batch_index, prefix_targets in enumerate(batch["prefix_targets"]):
        step_logits = [logits[batch_index, position] for position in batch["step_positions"][batch_index]]
        losses.append(
            sequence_calibration_loss(
                step_logits,
                prefix_targets,
                tau=float(train_cfg.tau),
                epsilon=float(train_cfg.epsilon_smoothing),
            )
        )
    return torch.stack(losses).mean()


def run_train_step(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    train_cfg: Any,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = compute_batch_loss(model, batch, train_cfg)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _iter_batches(items: list[PreparedPrompt], batch_size: int) -> list[list[PreparedPrompt]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _balanced_prompt_order(items: list[PreparedPrompt], *, seed: int) -> list[PreparedPrompt]:
    if not items:
        return []

    grouped: dict[str, list[PreparedPrompt]] = defaultdict(list)
    for item in items:
        grouped[item.prompt_spec.family_id].append(item)

    generator = torch.Generator().manual_seed(int(seed))
    for family_id, family_items in grouped.items():
        permutation = torch.randperm(len(family_items), generator=generator).tolist()
        grouped[family_id] = [family_items[idx] for idx in permutation]

    ordered_family_ids = sorted(grouped)
    family_positions = {family_id: 0 for family_id in ordered_family_ids}
    ordered: list[PreparedPrompt] = []

    while True:
        active_family_ids = [family_id for family_id in ordered_family_ids if family_positions[family_id] < len(grouped[family_id])]
        if not active_family_ids:
            break

        family_order = active_family_ids
        if len(active_family_ids) > 1:
            permutation = torch.randperm(len(active_family_ids), generator=generator).tolist()
            family_order = [active_family_ids[idx] for idx in permutation]

        for family_id in family_order:
            position = family_positions[family_id]
            family_items = grouped[family_id]
            if position >= len(family_items):
                continue
            ordered.append(family_items[position])
            family_positions[family_id] += 1

    return ordered


def _iter_balanced_batches(items: list[PreparedPrompt], batch_size: int, *, seed: int) -> list[list[PreparedPrompt]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return _iter_batches(_balanced_prompt_order(items, seed=seed), batch_size)


def _shard_training_prompts(items: list[PreparedPrompt], *, rank: int, world_size: int) -> list[PreparedPrompt]:
    if world_size <= 1 or not items:
        return list(items)

    padded = list(items)
    remainder = len(padded) % world_size
    if remainder:
        padded.extend(padded[: world_size - remainder])
    return padded[rank::world_size]


def _sample_batch_examples(batch_prompts: list[PreparedPrompt], base_seed: int) -> list[TrainingExample]:
    return [
        sample_training_example(prompt, sample_seed=base_seed + offset)
        for offset, prompt in enumerate(batch_prompts)
    ]


def _build_optimizer(bundle: TrainingBundle, train_cfg: Any) -> torch.optim.Optimizer:
    params = [param for param in bundle.model.parameters() if param.requires_grad]
    return AdamW(params, lr=float(train_cfg.learning_rate), weight_decay=float(train_cfg.weight_decay))


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def _log_metrics(run, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is not None:
        if step is None:
            run.log(metrics)
        else:
            run.log(metrics, step=step)


def _print_step_log(metrics: dict[str, Any]) -> None:
    epoch = int(metrics["epoch"])
    step = int(metrics["global_step"])
    loss = float(metrics["train_loss"])
    lr = float(metrics["lr"])
    print(f"[train] epoch={epoch} step={step} loss={loss:.6f} lr={lr:.6e}", flush=True)


def run_calibrate_sft(cfg: Any) -> None:
    distributed = initialize_distributed(getattr(cfg, "distributed", None), device_name=str(getattr(cfg.model, "device", "cuda")))
    set_global_seed(int(cfg.seed) + distributed.rank)

    wandb_run = None
    try:
        run_dir = ensure_dir(_resolve_run_dir(cfg))
        dataset_dir = ensure_dir(Path(cfg.dataset_root))
        checkpoint_dir = ensure_dir(run_dir / "checkpoint")
        metrics_dir = ensure_dir(run_dir / "metrics")

        bundle = load_training_bundle(cfg.model, cfg.train, getattr(cfg, "distributed", None))
        prepared_splits = build_prepared_splits(bundle.tokenizer, cfg.model, cfg.data)

        if is_main_process():
            write_dataset_artifacts(dataset_dir, prepared_splits)
            write_json(run_dir / "config_resolved.json", OmegaConf.to_container(cfg, resolve=True))
        barrier()

        train_prompts = list(prepared_splits["train"])
        batch_size = int(cfg.train.per_device_batch_size)
        grad_accum = int(cfg.train.gradient_accumulation_steps)
        num_epochs = int(cfg.train.epochs)
        local_prompt_count = math.ceil(len(train_prompts) / max(distributed.world_size, 1))
        microbatches_per_epoch = max(1, math.ceil(local_prompt_count / max(batch_size, 1)))
        total_microbatches = microbatches_per_epoch * max(num_epochs, 1)
        total_steps = max(1, math.ceil(total_microbatches / max(grad_accum, 1)))

        optimizer = _build_optimizer(bundle, cfg.train)
        scheduler = _build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=float(cfg.train.warmup_ratio))
        if is_main_process():
            wandb_run = _init_wandb(cfg)

        train_rows: list[dict[str, Any]] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch_index in range(num_epochs):
            microbatch_count = 0
            ordered_prompts = _balanced_prompt_order(train_prompts, seed=int(cfg.seed) + epoch_index)
            local_prompts = _shard_training_prompts(
                ordered_prompts,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            epoch_batches = _iter_batches(local_prompts, batch_size)

            for batch_index, batch_prompts in enumerate(epoch_batches):
                sample_seed = int(cfg.seed) + epoch_index * 100_000 + batch_index * batch_size * distributed.world_size + distributed.rank * batch_size
                examples = _sample_batch_examples(batch_prompts, sample_seed)
                batch = collate_training_examples(
                    examples,
                    pad_token_id=int(bundle.tokenizer.pad_token_id),
                    device=bundle.device,
                    model_cfg=cfg.model,
                )
                bundle.model.train()
                loss = compute_batch_loss(bundle.model, batch, cfg.train)
                (loss / grad_accum).backward()
                microbatch_count += 1

                should_step = microbatch_count % grad_accum == 0 or batch_index == len(epoch_batches) - 1
                if not should_step:
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                mean_loss = reduce_mean(loss, device=bundle.device)
                row = {
                    "epoch": epoch_index + 1,
                    "global_step": global_step,
                    "train_loss": mean_loss,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                if is_main_process():
                    train_rows.append(row)
                    _print_step_log(row)
                    _log_metrics(wandb_run, row, step=global_step)

            if is_main_process():
                epoch_rows = [row for row in train_rows if row["epoch"] == epoch_index + 1]
                epoch_summary = {
                    "epoch": epoch_index + 1,
                    "avg_train_loss": float(sum(row["train_loss"] for row in epoch_rows) / max(1, len(epoch_rows))),
                }
                _log_metrics(wandb_run, epoch_summary, step=global_step)

        if is_main_process():
            write_csv(metrics_dir / "train_metrics.csv", train_rows)
        barrier()
        save_training_bundle(bundle, checkpoint_dir, getattr(cfg, "distributed", None))
        barrier()

        if wandb_run is not None:
            wandb_run.finish()
    finally:
        cleanup_distributed()

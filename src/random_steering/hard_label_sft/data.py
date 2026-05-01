from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

from random_steering.calibrate_sft.data import (
    PreparedPrompt,
    build_prepared_splits,
    output_space_to_dict,
    sample_training_example,
)
from random_steering.utils.io import ensure_dir, write_json, write_jsonl


@dataclass(frozen=True)
class HardLabelExample:
    prepared_prompt: PreparedPrompt
    candidate_value: str
    candidate_token_ids: tuple[int, ...]


def build_hard_label_prepared_splits(
    tokenizer: Any,
    model_cfg: Any,
    data_cfg: Any,
) -> dict[str, list[PreparedPrompt]]:
    return build_prepared_splits(tokenizer, model_cfg, data_cfg)


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


def expand_train_prompts_for_epoch(
    prepared_train_prompts: list[PreparedPrompt],
    *,
    samples_per_prompt_per_epoch: int,
    seed: int,
) -> list[PreparedPrompt]:
    if samples_per_prompt_per_epoch <= 0:
        raise ValueError("samples_per_prompt_per_epoch must be positive.")
    repeated = [
        prepared_prompt
        for prepared_prompt in prepared_train_prompts
        for _ in range(int(samples_per_prompt_per_epoch))
    ]
    return _balanced_prompt_order(repeated, seed=seed)


def sample_hard_label_example(prepared_prompt: PreparedPrompt, sample_seed: int, *, eos_token_id: int) -> HardLabelExample:
    sampled = sample_training_example(prepared_prompt, sample_seed=sample_seed)
    candidate_token_ids = sampled.candidate_token_ids + (int(eos_token_id),)
    return HardLabelExample(
        prepared_prompt=prepared_prompt,
        candidate_value=sampled.candidate_value,
        candidate_token_ids=candidate_token_ids,
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


def write_hard_label_dataset_artifacts(
    dataset_dir: str | Path,
    prepared_splits: dict[str, list[PreparedPrompt]],
    *,
    samples_per_prompt_per_epoch: int,
    data_cfg: Any | None = None,
) -> None:
    root = ensure_dir(dataset_dir)
    manifest = {
        "training_mode": "hard_label_sft",
        "samples_per_prompt_per_epoch": int(samples_per_prompt_per_epoch),
        "splits": {split_name: len(prompts) for split_name, prompts in prepared_splits.items()},
    }
    if data_cfg is not None:
        manifest["data_config"] = OmegaConf.to_container(data_cfg, resolve=True)
    write_json(root / "manifest.json", manifest)
    for split_name, prompts in prepared_splits.items():
        write_jsonl(root / f"{split_name}.jsonl", [_prepared_prompt_to_row(prompt) for prompt in prompts])

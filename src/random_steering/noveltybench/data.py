from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_FILE_NAMES = {
    "curated": "curated.jsonl",
    "wildchat": "wildchat_1k.jsonl",
}

SPLIT_EXPECTED_COUNTS = {
    "curated": 100,
    "wildchat": 1000,
}

SPLIT_OUTPUT_NAMES = {
    "curated": "nb-curated",
    "wildchat": "nb-wildchat",
}


@dataclass(frozen=True)
class NoveltyBenchPrompt:
    split: str
    prompt_id: str
    prompt: str
    metadata: dict[str, Any]


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip()
    if normalized not in SPLIT_FILE_NAMES:
        raise ValueError(f"Unsupported NoveltyBench split: {split_name}")
    return normalized


def split_asset_path(assets_root: str | Path, split_name: str) -> Path:
    split = validate_split_name(split_name)
    return Path(assets_root) / SPLIT_FILE_NAMES[split]


def split_output_dir(run_dir: str | Path, split_name: str) -> Path:
    split = validate_split_name(split_name)
    return Path(run_dir) / SPLIT_OUTPUT_NAMES[split]


def read_jsonl(path: str | Path, *, allow_trailing_invalid: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_trailing_invalid and idx == len(lines) - 1:
                break
            raise
    return rows


def load_prompt_rows_from_path(
    path: str | Path,
    *,
    split_name: str,
    expected_count: int | None = None,
    max_prompts: int | None = None,
) -> list[NoveltyBenchPrompt]:
    rows = read_jsonl(path)
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(
            f"NoveltyBench split {split_name} expected {expected_count} prompts, found {len(rows)} in {path}."
        )
    prompts: list[NoveltyBenchPrompt] = []
    for row in rows:
        prompt_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not prompt_id:
            raise ValueError(f"NoveltyBench split {split_name} contains a row without a valid id.")
        if not prompt:
            raise ValueError(f"NoveltyBench split {split_name} contains prompt {prompt_id} without prompt text.")
        metadata = {key: value for key, value in row.items() if key not in {"prompt"}}
        prompts.append(NoveltyBenchPrompt(split=split_name, prompt_id=prompt_id, prompt=prompt, metadata=metadata))
    if max_prompts is not None:
        prompts = prompts[: max(int(max_prompts), 0)]
    return prompts


def load_split_prompts(
    assets_root: str | Path,
    split_name: str,
    *,
    max_prompts: int | None = None,
) -> list[NoveltyBenchPrompt]:
    split = validate_split_name(split_name)
    return load_prompt_rows_from_path(
        split_asset_path(assets_root, split),
        split_name=split,
        expected_count=SPLIT_EXPECTED_COUNTS[split],
        max_prompts=max_prompts,
    )


def enabled_splits(noveltybench_cfg: Any) -> list[str]:
    configured = list(getattr(noveltybench_cfg, "enabled_splits", []))
    if not configured:
        raise ValueError("NoveltyBench requires at least one enabled split.")
    return [validate_split_name(split_name) for split_name in configured]


def artifact_is_complete(path: str | Path, expected_ids: list[str]) -> bool:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return False
    rows = read_jsonl(artifact_path, allow_trailing_invalid=True)
    actual_ids = {str(row.get("id", "")) for row in rows}
    return actual_ids == set(expected_ids)


def resolve_eval_target_name(run_dir: str | Path) -> str | None:
    config_path = Path(run_dir) / "config_resolved.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    eval_target_cfg = payload.get("eval_target")
    if isinstance(eval_target_cfg, dict):
        name = eval_target_cfg.get("name")
        if name is not None:
            return str(name)
    return None

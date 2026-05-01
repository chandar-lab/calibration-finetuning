from __future__ import annotations

import json
from pathlib import Path


def load_open_random_gen_prompts(path: str | Path) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Open random generation prompts file not found: {prompt_path}")

    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in open random generation prompts file: {prompt_path}") from exc

    if not isinstance(payload, list):
        raise ValueError("Open random generation prompts must be a JSON array")
    if not payload:
        raise ValueError("Open random generation prompts list must not be empty")

    prompts: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(f"Open random generation prompt at index {index} must be a string")
        prompt = item.strip()
        if not prompt:
            raise ValueError(f"Open random generation prompt at index {index} must be non-empty")
        prompts.append(item)
    return prompts

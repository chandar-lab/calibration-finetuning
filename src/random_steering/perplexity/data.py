from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class PalomaTextRecord:
    text: str | None
    has_text_field: bool
    is_empty: bool


def resolve_paloma_files(dataset_root: Path, slice_name: str, split: str) -> list[Path]:
    split_dir = Path(dataset_root) / slice_name / split
    if not split_dir.exists():
        raise FileNotFoundError(f"PALOMA slice split directory does not exist: {split_dir}")
    files = sorted(split_dir.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"No .jsonl.gz files found under {split_dir}")
    return files


def resolve_configured_slices(dataset_root: Path, slice_names: list[str], split: str) -> dict[str, list[Path]]:
    return {
        str(slice_name): resolve_paloma_files(Path(dataset_root), str(slice_name), str(split))
        for slice_name in slice_names
    }


def iter_paloma_file_records(file_path: Path, text_field: str = "text") -> Iterator[PalomaTextRecord]:
    saw_valid_row = False
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            has_text_field = text_field in payload
            value = payload.get(text_field) if has_text_field else None
            is_empty = not isinstance(value, str) or not value.strip()
            if not saw_valid_row:
                if not has_text_field:
                    raise KeyError(f"Missing text field {text_field!r} in first valid row of {file_path}")
                saw_valid_row = True
            yield PalomaTextRecord(
                text=value if isinstance(value, str) else None,
                has_text_field=has_text_field,
                is_empty=is_empty,
            )


def iter_paloma_texts(
    dataset_root: Path,
    slice_name: str,
    split: str,
    text_field: str = "text",
) -> Iterator[str]:
    for file_path in resolve_paloma_files(Path(dataset_root), slice_name, split):
        for record in iter_paloma_file_records(file_path, text_field=text_field):
            if record.has_text_field and not record.is_empty and record.text is not None:
                yield record.text

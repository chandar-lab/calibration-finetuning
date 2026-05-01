from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from random_steering.perplexity.data import iter_paloma_file_records, resolve_configured_slices
from random_steering.utils.io import ensure_dir, write_csv, write_json


def _empty_metrics() -> dict[str, int | float]:
    return {
        "num_documents": 0,
        "num_skipped_rows": 0,
        "num_tokens": 0,
        "num_bytes": 0,
        "total_nll": 0.0,
    }


def _finalize_metrics(metrics: dict[str, int | float]) -> dict[str, int | float | None]:
    num_tokens = int(metrics["num_tokens"])
    num_bytes = int(metrics["num_bytes"])
    total_nll = float(metrics["total_nll"])
    avg_nll_per_token = (total_nll / num_tokens) if num_tokens > 0 else None
    perplexity = math.exp(avg_nll_per_token) if avg_nll_per_token is not None else None
    bits_per_byte = (total_nll / (math.log(2.0) * num_bytes)) if num_bytes > 0 else None
    finalized = dict(metrics)
    finalized["avg_nll_per_token"] = avg_nll_per_token
    finalized["perplexity"] = perplexity
    finalized["bits_per_byte"] = bits_per_byte
    return finalized


def _chunk_document(
    token_ids: list[int],
    *,
    context_length: int,
    stride: int,
    max_scored_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if len(token_ids) < 2:
        return [], 0

    windows: list[dict[str, Any]] = []
    prev_end = 0
    total_scored_tokens = 0
    for begin in range(0, len(token_ids), stride):
        end = min(begin + context_length, len(token_ids))
        if end - begin < 2:
            break
        window_ids = token_ids[begin:end]
        absolute_positions = list(range(begin, end))
        score_from = max(prev_end, begin + 1)
        score_mask = [position >= score_from for position in absolute_positions]
        score_count = sum(1 for value in score_mask if value)
        if score_count <= 0:
            prev_end = end
            if end >= len(token_ids):
                break
            continue
        if max_scored_tokens is not None:
            remaining = max_scored_tokens - total_scored_tokens
            if remaining <= 0:
                break
            if score_count > remaining:
                seen = 0
                truncated_mask: list[bool] = []
                for flag in score_mask:
                    if flag and seen < remaining:
                        truncated_mask.append(True)
                        seen += 1
                    else:
                        truncated_mask.append(False)
                score_mask = truncated_mask
                score_count = remaining
        windows.append({"input_ids": window_ids, "score_mask": score_mask})
        total_scored_tokens += score_count
        prev_end = end
        if end >= len(token_ids):
            break
        if max_scored_tokens is not None and total_scored_tokens >= max_scored_tokens:
            break
    return windows, total_scored_tokens


def _batched(iterable: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [iterable[index : index + batch_size] for index in range(0, len(iterable), batch_size)]


def _score_windows(
    model: Any,
    device: torch.device,
    windows: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_token_id: int,
) -> float:
    if not windows:
        return 0.0

    total_nll = 0.0
    for window_batch in _batched(windows, batch_size):
        max_len = max(len(window["input_ids"]) for window in window_batch)
        batch_input_ids = torch.full(
            (len(window_batch), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        batch_attention_mask = torch.zeros((len(window_batch), max_len), dtype=torch.long, device=device)
        batch_labels = torch.full((len(window_batch), max_len), fill_value=-100, dtype=torch.long, device=device)

        for row_index, window in enumerate(window_batch):
            input_ids = torch.tensor(window["input_ids"], dtype=torch.long, device=device)
            score_mask = torch.tensor(window["score_mask"], dtype=torch.bool, device=device)
            seq_len = int(input_ids.shape[0])
            batch_input_ids[row_index, :seq_len] = input_ids
            batch_attention_mask[row_index, :seq_len] = 1
            batch_labels[row_index, :seq_len] = torch.where(score_mask, input_ids, torch.full_like(input_ids, -100))

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch_labels[:, 1:].contiguous()
            valid_mask = labels.ne(-100)
            if valid_mask.any():
                per_token_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(labels)
                total_nll += float(per_token_loss.masked_select(valid_mask).sum().item())
    return total_nll


def _device_for_model(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def run_perplexity_eval(
    *,
    model: Any,
    tokenizer: Any,
    perplexity_cfg: Any,
    eval_target_cfg: Any,
    output_dir: Path,
    full_cfg: Any,
) -> dict[str, Any]:
    run_dir = ensure_dir(output_dir)
    metrics_dir = ensure_dir(run_dir / "metrics")
    OmegaConf.save(full_cfg, run_dir / "config_resolved.yaml")

    device = _device_for_model(model)
    model.eval()

    dataset_root = Path(perplexity_cfg.dataset_root)
    slice_files = resolve_configured_slices(dataset_root, list(perplexity_cfg.slices), str(perplexity_cfg.split))
    context_length = int(perplexity_cfg.context_length)
    stride = int(perplexity_cfg.stride)
    batch_size = int(perplexity_cfg.batch_size)
    text_field = str(getattr(perplexity_cfg, "text_field", "text"))
    max_documents = getattr(perplexity_cfg, "max_documents", None)
    max_documents = None if max_documents is None else int(max_documents)
    max_tokens = getattr(perplexity_cfg, "max_tokens", None)
    max_tokens = None if max_tokens is None else int(max_tokens)
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    per_slice_rows: list[dict[str, Any]] = []
    per_file_rows: list[dict[str, Any]] = []
    total_metrics = _empty_metrics()
    total_files = 0

    for slice_name, files in slice_files.items():
        slice_metrics = _empty_metrics()
        for file_path in files:
            if max_documents is not None and int(slice_metrics["num_documents"]) >= max_documents:
                break
            if max_tokens is not None and int(slice_metrics["num_tokens"]) >= max_tokens:
                break

            file_metrics = _empty_metrics()
            for record in iter_paloma_file_records(file_path, text_field=text_field):
                if max_documents is not None and int(slice_metrics["num_documents"]) >= max_documents:
                    break
                if max_tokens is not None and int(slice_metrics["num_tokens"]) >= max_tokens:
                    break

                if not record.has_text_field or record.is_empty or record.text is None:
                    slice_metrics["num_skipped_rows"] += 1
                    file_metrics["num_skipped_rows"] += 1
                    continue

                text = record.text
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                remaining_tokens = None
                if max_tokens is not None:
                    remaining_tokens = max_tokens - int(slice_metrics["num_tokens"])
                    if remaining_tokens <= 0:
                        break
                windows, scored_tokens = _chunk_document(
                    token_ids,
                    context_length=context_length,
                    stride=stride,
                    max_scored_tokens=remaining_tokens,
                )
                if not windows:
                    file_metrics["num_documents"] += 1
                    slice_metrics["num_documents"] += 1
                    file_metrics["num_bytes"] += len(text.encode("utf-8"))
                    slice_metrics["num_bytes"] += len(text.encode("utf-8"))
                    continue

                total_nll = _score_windows(
                    model,
                    device,
                    windows,
                    batch_size=batch_size,
                    pad_token_id=pad_token_id,
                )
                num_bytes = len(text.encode("utf-8"))
                file_metrics["num_documents"] += 1
                file_metrics["num_tokens"] += scored_tokens
                file_metrics["num_bytes"] += num_bytes
                file_metrics["total_nll"] += total_nll
                slice_metrics["num_documents"] += 1
                slice_metrics["num_tokens"] += scored_tokens
                slice_metrics["num_bytes"] += num_bytes
                slice_metrics["total_nll"] += total_nll

            finalized_file = _finalize_metrics(file_metrics)
            finalized_file["slice_name"] = slice_name
            finalized_file["split"] = str(perplexity_cfg.split)
            finalized_file["file_path"] = str(file_path)
            per_file_rows.append(finalized_file)
            total_files += 1

        finalized_slice = _finalize_metrics(slice_metrics)
        finalized_slice["slice_name"] = slice_name
        finalized_slice["split"] = str(perplexity_cfg.split)
        finalized_slice["num_files"] = len([row for row in per_file_rows if row["slice_name"] == slice_name])
        per_slice_rows.append(finalized_slice)
        for key in total_metrics:
            total_metrics[key] += slice_metrics[key]

    summary = _finalize_metrics(total_metrics)
    summary.update(
        {
            "slices": [str(slice_name) for slice_name in perplexity_cfg.slices],
            "split": str(perplexity_cfg.split),
            "num_slices": len(per_slice_rows),
            "num_files": total_files,
            "context_length": context_length,
            "stride": stride,
            "batch_size": batch_size,
            "eval_target_name": str(eval_target_cfg.name),
        }
    )

    write_json(run_dir / "summary.json", summary)
    write_json(metrics_dir / "summary.json", summary)
    write_csv(metrics_dir / "per_slice_summary.csv", per_slice_rows)
    write_csv(metrics_dir / "per_file_metrics.csv", per_file_rows)
    return summary

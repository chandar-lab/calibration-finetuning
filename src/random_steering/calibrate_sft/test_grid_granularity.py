from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from random_steering.calibrate_sft.data import (
    PreparedPrompt,
    PromptSpec,
    build_distribution_spec,
    build_output_space,
    build_tokenized_output_space,
    get_family_spec,
    sample_training_example,
)
from random_steering.eval.metrics import frozen_distribution


class CharacterNumericTokenizer:
    """Minimal tokenizer fallback for structural trie-cost analysis."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self._vocab = {"<eos>": self.eos_token_id}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        for char in text:
            if char not in self._vocab:
                self._vocab[char] = len(self._vocab)
            token_ids.append(self._vocab[char])
        return token_ids


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _load_tokenizer(tokenizer_checkpoint: str | None, *, local_files_only: bool) -> tuple[Any, str]:
    if not tokenizer_checkpoint:
        return CharacterNumericTokenizer(), "character_numeric"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, local_files_only=local_files_only)
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError(f"Tokenizer {tokenizer_checkpoint} does not define eos_token_id.")
        return tokenizer, str(tokenizer_checkpoint)
    except Exception as exc:
        print(
            f"[warn] failed to load tokenizer '{tokenizer_checkpoint}' ({exc}); "
            "falling back to CharacterNumericTokenizer.",
            file=sys.stderr,
            flush=True,
        )
        return CharacterNumericTokenizer(), "character_numeric"


def _make_prompt_spec(family_id: str, parameters: dict[str, float], *, variant: str) -> PromptSpec:
    family = get_family_spec(family_id)
    canonical_parameters = family.transform_grid_params(dict(parameters))
    return PromptSpec(
        split="granularity_probe",
        family_id=family_id,
        display_name=family.display_name,
        tier=family.tier,
        distribution_id=f"{family_id}_{variant}",
        parameters=canonical_parameters,
        kind=family.kind,
        integer_only=family.integer_only,
    )


def _representative_prompt_specs() -> list[PromptSpec]:
    specs: list[PromptSpec] = []
    suites: dict[str, list[tuple[str, dict[str, float]]]] = {
        "uniform": [
            ("narrow", {"a": -5.0, "width": 1.0}),
            ("mid", {"a": -1.0, "width": 4.0}),
            ("wide", {"a": 2.5, "width": 8.0}),
        ],
        "gaussian": [
            ("narrow", {"mu": 0.0, "sigma": 0.5}),
            ("mid", {"mu": 0.0, "sigma": 2.0}),
            ("wide", {"mu": 2.0, "sigma": 4.0}),
        ],
        "beta": [
            ("u_shaped", {"alpha": 0.5, "beta": 0.5}),
            ("skewed", {"alpha": 2.0, "beta": 5.0}),
            ("concentrated", {"alpha": 7.0, "beta": 7.0}),
        ],
        "gamma": [
            ("light", {"alpha": 1.0, "beta": 1.0}),
            ("mid", {"alpha": 3.0, "beta": 1.5}),
            ("wide", {"alpha": 8.0, "beta": 2.0}),
        ],
        "lognormal": [
            ("light", {"mu": 0.0, "sigma": 0.5}),
            ("mid", {"mu": 0.5, "sigma": 1.0}),
            ("heavy", {"mu": 1.0, "sigma": 1.5}),
        ],
        "cauchy": [
            ("light", {"x0": 0.0, "gamma": 0.5}),
            ("mid", {"x0": 0.0, "gamma": 2.0}),
            ("heavy", {"x0": 2.0, "gamma": 5.0}),
        ],
        "weibull": [
            ("low", {"k": 0.5, "lambda": 0.5}),
            ("mid", {"k": 1.5, "lambda": 1.5}),
            ("high", {"k": 3.0, "lambda": 3.0}),
        ],
        "poisson": [
            ("low", {"lambda": 1.0}),
            ("mid", {"lambda": 4.0}),
            ("high", {"lambda": 10.0}),
        ],
    }
    for family_id, variants in suites.items():
        for variant, parameters in variants:
            specs.append(_make_prompt_spec(family_id, parameters, variant=variant))
    return specs


def _timed_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float, int]:
    tracemalloc.start()
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_s = time.perf_counter() - start_time
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, float(elapsed_s), int(peak_bytes)


def _atomic_support(output_space: Any) -> np.ndarray:
    return np.asarray([float(value) for value in output_space.values], dtype=float)


def _raw_grid_points(output_space: Any) -> int:
    if output_space.integer_only:
        return int(round(output_space.upper_bound) - round(output_space.lower_bound) + 1)
    step = 10.0 ** (-int(output_space.num_decimals))
    start = math.ceil(float(output_space.lower_bound) / step)
    end = math.floor(float(output_space.upper_bound) / step)
    return max(int(end - start + 1), 1)


def _atomic_moment_errors(support: np.ndarray, probabilities: np.ndarray, dist: Any) -> dict[str, float]:
    approx_mean = float(np.sum(probabilities * support))
    approx_var = float(np.sum(probabilities * (support - approx_mean) ** 2))
    target_mean, target_var = dist.stats(moments="mv")
    target_mean_f = float(target_mean) if np.isfinite(target_mean) else math.nan
    target_var_f = float(target_var) if np.isfinite(target_var) else math.nan
    return {
        "approx_mean": approx_mean,
        "approx_variance": approx_var,
        "mean_error": abs(approx_mean - target_mean_f) if np.isfinite(target_mean_f) else math.nan,
        "variance_error": abs(approx_var - target_var_f) if np.isfinite(target_var_f) else math.nan,
    }


def _atomic_wasserstein_1(support: np.ndarray, probabilities: np.ndarray, dist: Any, *, num_quantiles: int) -> float:
    quantiles = (np.arange(num_quantiles, dtype=float) + 0.5) / float(num_quantiles)
    target = np.asarray(dist.ppf(quantiles), dtype=float)
    cdf = np.cumsum(probabilities, dtype=float)
    approx = support[np.searchsorted(cdf, quantiles, side="left")]
    finite_mask = np.isfinite(target) & np.isfinite(approx)
    if not np.any(finite_mask):
        return math.inf
    return float(np.mean(np.abs(target[finite_mask] - approx[finite_mask])))


def _atomic_cdf_sup_error(support: np.ndarray, probabilities: np.ndarray, dist: Any) -> float:
    cumulative = np.cumsum(probabilities, dtype=float)
    previous_mass = 0.0
    sup_error = 0.0
    for index, x_value in enumerate(support):
        true_cdf = float(dist.cdf(float(x_value)))
        sup_error = max(sup_error, abs(true_cdf - previous_mass), abs(true_cdf - float(cumulative[index])))
        previous_mass = float(cumulative[index])
    return float(sup_error)


def _make_prepared_prompt(prompt_spec: PromptSpec, output_space: Any, tokenized_output_space: Any) -> PreparedPrompt:
    return PreparedPrompt(
        prompt_spec=prompt_spec,
        distribution_spec=build_distribution_spec(prompt_spec),
        prompt_text="",
        formatted_prompt="",
        prompt_token_ids=(),
        output_space=output_space,
        tokenized_output_space=tokenized_output_space,
    )


def _sample_path_stats(prepared_prompt: PreparedPrompt, *, num_samples: int) -> dict[str, float]:
    start_time = time.perf_counter()
    path_lengths: list[int] = []
    step_branch_sizes: list[int] = []
    for sample_seed in range(num_samples):
        example = sample_training_example(prepared_prompt, sample_seed=sample_seed)
        path_lengths.append(len(example.prefix_targets))
        step_branch_sizes.extend(len(prefix_target.next_token_ids) for prefix_target in example.prefix_targets)
    elapsed_s = time.perf_counter() - start_time
    return {
        "sample_example_time_s": float(elapsed_s),
        "sample_path_len_avg": float(np.mean(path_lengths)) if path_lengths else math.nan,
        "sample_path_len_max": float(max(path_lengths)) if path_lengths else math.nan,
        "sample_branch_avg": float(np.mean(step_branch_sizes)) if step_branch_sizes else math.nan,
        "sample_branch_max": float(max(step_branch_sizes)) if step_branch_sizes else math.nan,
    }


def _trie_stats(tokenized_output_space: Any) -> dict[str, float]:
    candidate_token_ids = tokenized_output_space.candidate_token_ids
    prefix_targets = tokenized_output_space.prefix_targets
    candidate_lengths = [len(tokens) for tokens in candidate_token_ids]
    branch_sizes = [len(prefix_target.next_token_ids) for prefix_target in prefix_targets.values()]
    return {
        "num_candidates": float(len(candidate_token_ids)),
        "candidate_token_len_avg": float(np.mean(candidate_lengths)) if candidate_lengths else math.nan,
        "candidate_token_len_max": float(max(candidate_lengths)) if candidate_lengths else math.nan,
        "candidate_token_total": float(sum(candidate_lengths)),
        "num_prefixes": float(len(prefix_targets)),
        "prefix_branch_avg": float(np.mean(branch_sizes)) if branch_sizes else math.nan,
        "prefix_branch_max": float(max(branch_sizes)) if branch_sizes else math.nan,
    }


def _base_data_cfg(num_decimals: int, max_bins: int) -> dict[str, Any]:
    return {
        "quantile_low": 0.001,
        "quantile_high": 0.999,
        "num_decimals": int(num_decimals),
        "max_bins": int(max_bins),
        "tail_policy": "clip_to_edge_bin",
    }


def _evaluate_case(
    prompt_spec: PromptSpec,
    *,
    tokenizer: Any,
    tokenizer_name: str,
    num_decimals: int,
    max_bins: int,
    num_quantiles: int,
    num_sample_paths: int,
) -> dict[str, Any]:
    cfg = _base_data_cfg(num_decimals=num_decimals, max_bins=max_bins)
    output_space, output_space_time_s, output_space_peak_bytes = _timed_call(build_output_space, prompt_spec, cfg)
    tokenized_output_space, trie_time_s, trie_peak_bytes = _timed_call(build_tokenized_output_space, output_space, tokenizer)

    support = _atomic_support(output_space)
    probabilities = np.asarray(output_space.probabilities, dtype=float)
    distribution_spec = build_distribution_spec(prompt_spec)
    dist = frozen_distribution(distribution_spec)

    prepared_prompt = _make_prepared_prompt(prompt_spec, output_space, tokenized_output_space)
    path_stats = _sample_path_stats(prepared_prompt, num_samples=num_sample_paths)

    row: dict[str, Any] = {
        "family_id": prompt_spec.family_id,
        "distribution_id": prompt_spec.distribution_id,
        "display_name": prompt_spec.display_name,
        "tier": prompt_spec.tier,
        "integer_only": bool(prompt_spec.integer_only),
        "tokenizer": tokenizer_name,
        "num_decimals": int(num_decimals),
        "max_bins": int(max_bins),
        "raw_grid_points": int(_raw_grid_points(output_space)),
        "final_bins": int(len(output_space.values)),
        "lower_bound": float(output_space.lower_bound),
        "upper_bound": float(output_space.upper_bound),
        "lower_tail_mass": float(dist.cdf(float(output_space.lower_bound))),
        "upper_tail_mass": float(1.0 - dist.cdf(float(output_space.upper_bound))),
        "atomic_w1": _atomic_wasserstein_1(support, probabilities, dist, num_quantiles=num_quantiles),
        "cdf_sup_error": _atomic_cdf_sup_error(support, probabilities, dist),
        "output_space_time_s": output_space_time_s,
        "output_space_peak_tracemalloc_bytes": int(output_space_peak_bytes),
        "trie_time_s": trie_time_s,
        "trie_peak_tracemalloc_bytes": int(trie_peak_bytes),
    }
    row.update(_atomic_moment_errors(support, probabilities, dist))
    row.update(_trie_stats(tokenized_output_space))
    row.update(path_stats)
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["num_decimals"]), int(row["max_bins"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (num_decimals, max_bins), grouped_rows in sorted(grouped.items()):
        atomic_w1_values = [float(row["atomic_w1"]) for row in grouped_rows]
        cdf_errors = [float(row["cdf_sup_error"]) for row in grouped_rows]
        summary.append(
            {
                "num_decimals": num_decimals,
                "max_bins": max_bins,
                "num_prompts": len(grouped_rows),
                "mean_atomic_w1": float(np.mean(atomic_w1_values)),
                "max_atomic_w1": float(np.max(atomic_w1_values)),
                "mean_cdf_sup_error": float(np.mean(cdf_errors)),
                "max_cdf_sup_error": float(np.max(cdf_errors)),
                "mean_output_space_time_s": float(np.mean([float(row["output_space_time_s"]) for row in grouped_rows])),
                "mean_trie_time_s": float(np.mean([float(row["trie_time_s"]) for row in grouped_rows])),
                "mean_num_prefixes": float(np.mean([float(row["num_prefixes"]) for row in grouped_rows])),
                "max_num_prefixes": float(np.max([float(row["num_prefixes"]) for row in grouped_rows])),
                "mean_sample_path_len": float(np.mean([float(row["sample_path_len_avg"]) for row in grouped_rows])),
                "mean_lower_tail_mass": float(np.mean([float(row["lower_tail_mass"]) for row in grouped_rows])),
                "mean_upper_tail_mass": float(np.mean([float(row["upper_tail_mass"]) for row in grouped_rows])),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe approximation quality and cost of trie-grid granularity.")
    parser.add_argument(
        "--num-decimals",
        type=str,
        default="2,3,4,5,6",
        help="Comma-separated list of num_decimals settings to evaluate.",
    )
    parser.add_argument(
        "--max-bins",
        type=str,
        default="1001,2048,4096,8192,16384",
        help="Comma-separated list of max_bins settings to evaluate.",
    )
    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=16384,
        help="Number of quantiles used to approximate W1 between the true and atomic distributions.",
    )
    parser.add_argument(
        "--num-sample-paths",
        type=int,
        default=64,
        help="Number of sampled trie paths used to measure path-length and branching statistics.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default=None,
        help="Optional HF tokenizer checkpoint for realistic trie tokenization cost. Defaults to a character-level numeric tokenizer.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local files when loading a HF tokenizer checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "grid_granularity",
        help="Directory where CSV/JSON summaries will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_decimals_values = _parse_int_list(args.num_decimals)
    max_bins_values = _parse_int_list(args.max_bins)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, tokenizer_name = _load_tokenizer(args.tokenizer_checkpoint, local_files_only=bool(args.local_files_only))
    prompt_specs = _representative_prompt_specs()

    work_items = [
        (prompt_spec, num_decimals, max_bins)
        for prompt_spec in prompt_specs
        for num_decimals in num_decimals_values
        for max_bins in max_bins_values
    ]

    rows: list[dict[str, Any]] = []
    for prompt_spec, num_decimals, max_bins in tqdm(work_items, desc="Grid granularity sweep"):
        rows.append(
            _evaluate_case(
                prompt_spec,
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                num_decimals=num_decimals,
                max_bins=max_bins,
                num_quantiles=int(args.num_quantiles),
                num_sample_paths=int(args.num_sample_paths),
            )
        )

    summary_rows = _summary_rows(rows)
    _write_csv(output_dir / "per_prompt_metrics.csv", rows)
    _write_csv(output_dir / "summary_by_config.csv", summary_rows)
    with (output_dir / "prompt_suite.json").open("w") as handle:
        json.dump([asdict(prompt_spec) for prompt_spec in prompt_specs], handle, indent=2)
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "num_decimals": num_decimals_values,
                "max_bins": max_bins_values,
                "num_quantiles": int(args.num_quantiles),
                "num_sample_paths": int(args.num_sample_paths),
                "tokenizer": tokenizer_name,
            },
            handle,
            indent=2,
        )

    print(f"Wrote per-prompt metrics to {output_dir / 'per_prompt_metrics.csv'}", flush=True)
    print(f"Wrote summary metrics to {output_dir / 'summary_by_config.csv'}", flush=True)
    print("Best configurations by mean atomic W1:", flush=True)
    for row in sorted(summary_rows, key=lambda item: (float(item["mean_atomic_w1"]), float(item["mean_trie_time_s"])))[:10]:
        print(
            "  "
            f"d={row['num_decimals']:>2} | "
            f"max_bins={row['max_bins']:>5} | "
            f"mean_atomic_w1={row['mean_atomic_w1']:.6f} | "
            f"mean_cdf_sup_error={row['mean_cdf_sup_error']:.6f} | "
            f"mean_trie_time_s={row['mean_trie_time_s']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

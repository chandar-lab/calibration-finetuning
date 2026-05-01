from __future__ import annotations

from dataclasses import fields
from typing import Any

try:
    from scipy import stats
except ImportError:
    stats = None

from random_steering.mcq_gen.parser import ParsedMcqGeneration


_OPTION_LABELS = ("A", "B", "C", "D")


def _as_record(record: ParsedMcqGeneration | dict[str, Any]) -> ParsedMcqGeneration:
    if isinstance(record, ParsedMcqGeneration):
        return record
    allowed_keys = {field.name for field in fields(ParsedMcqGeneration)}
    payload = {key: value for key, value in record.items() if key in allowed_keys}
    return ParsedMcqGeneration(**payload)


def compute_mcq_metrics(records: list[ParsedMcqGeneration | dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed_records = [_as_record(record) for record in records]
    num_samples = len(parsed_records)
    num_parseable = sum(1 for record in parsed_records if record.is_parseable)
    num_valid_correct_answer = sum(1 for record in parsed_records if record.has_valid_correct_answer)
    unique_options_count = sum(1 for record in parsed_records if record.unique_options)

    counts = {label: 0 for label in _OPTION_LABELS}
    for record in parsed_records:
        if record.is_parseable and record.correct_answer in counts:
            counts[record.correct_answer] += 1

    if num_parseable:
        frequencies = {label: counts[label] / num_parseable for label in _OPTION_LABELS}
        if stats is None:
            raise ImportError("scipy is required to compute MCQ generation chi-square statistics")
        expected = [num_parseable / len(_OPTION_LABELS)] * len(_OPTION_LABELS)
        result = stats.chisquare(
            f_obs=[counts[label] for label in _OPTION_LABELS],
            f_exp=expected,
        )
        chi_square_statistic = float(result.statistic)
        chi_square_pvalue = float(result.pvalue)
        tv_distance = 0.5 * sum(abs(frequencies[label] - 0.25) for label in _OPTION_LABELS)
        max_abs_deviation = max(abs(frequencies[label] - 0.25) for label in _OPTION_LABELS)
    else:
        frequencies = {label: 0.0 for label in _OPTION_LABELS}
        chi_square_statistic = 0.0
        chi_square_pvalue = 1.0
        tv_distance = 0.0
        max_abs_deviation = 0.0

    answer_frequency_rows = [
        {
            "answer_label": label,
            "count": int(counts[label]),
            "frequency": float(frequencies[label]),
        }
        for label in _OPTION_LABELS
    ]

    summary = {
        "num_samples": int(num_samples),
        "num_parseable": int(num_parseable),
        "parse_rate": float(num_parseable / num_samples) if num_samples else 0.0,
        "num_valid_correct_answer": int(num_valid_correct_answer),
        "valid_correct_answer_rate": float(num_valid_correct_answer / num_samples) if num_samples else 0.0,
        "count_a": int(counts["A"]),
        "count_b": int(counts["B"]),
        "count_c": int(counts["C"]),
        "count_d": int(counts["D"]),
        "freq_a": float(frequencies["A"]),
        "freq_b": float(frequencies["B"]),
        "freq_c": float(frequencies["C"]),
        "freq_d": float(frequencies["D"]),
        "chi_square_statistic": chi_square_statistic,
        "chi_square_pvalue": chi_square_pvalue,
        "tv_distance_from_uniform": float(tv_distance),
        "max_abs_deviation_from_uniform": float(max_abs_deviation),
        "unique_options_rate": float(unique_options_count / num_samples) if num_samples else 0.0,
    }
    return summary, answer_frequency_rows

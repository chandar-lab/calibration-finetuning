from __future__ import annotations

from dataclasses import asdict

from random_steering.eval.metrics import moment_errors
from random_steering.eval.tests import run_gof_test
from random_steering.types import DistributionSpec, MetricRecord, SampleRecord


def sample_record_to_dict(record: SampleRecord) -> dict:
    return asdict(record)


def to_metric_rows(
    *,
    records: list[SampleRecord],
    spec: DistributionSpec,
    steering_name: str,
    seed: int,
) -> tuple[list[dict], list[MetricRecord]]:
    if not records:
        return [], []

    protocol = records[0].protocol
    distribution_id = records[0].distribution_id
    total = len(records)
    valid_values = [r.parsed_value for r in records if r.is_valid and r.parsed_value is not None]
    valid_count = len(valid_values)
    support_violations = sum(1 for r in records if r.parsed_value is not None and not r.support_ok)

    moment_metrics = moment_errors(valid_values, spec)
    gof_metrics = run_gof_test(valid_values, spec)

    summary_row = {
        "distribution_id": distribution_id,
        "protocol": protocol,
        "steering_name": steering_name,
        "seed": seed,
        "num_requested": total,
        "num_valid": valid_count,
        "valid_rate": valid_count / total if total else 0.0,
        "support_violation_rate": support_violations / total if total else 0.0,
        **moment_metrics,
        **gof_metrics,
    }

    metric_records = [
        MetricRecord(
            distribution_id=distribution_id,
            protocol=protocol,
            steering_name=steering_name,
            seed=seed,
            metric_name=k,
            value=float(v),
        )
        for k, v in summary_row.items()
        if isinstance(v, (float, int))
    ]

    return [summary_row], metric_records


def build_summary(metric_rows: list[dict]) -> dict:
    if not metric_rows:
        return {"num_rows": 0}

    numeric_keys = [
        "valid_rate",
        "support_violation_rate",
        "mean_error",
        "variance_error",
        "wasserstein_1",
    ]

    aggregate = {"num_rows": len(metric_rows)}
    for key in numeric_keys:
        values = [row[key] for row in metric_rows if isinstance(row.get(key), (float, int))]
        if values:
            aggregate[f"avg_{key}"] = float(sum(values) / len(values))

    return aggregate

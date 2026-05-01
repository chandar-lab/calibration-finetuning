from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


"""
Vendored and adapted from:
https://github.com/felipemaiapolo/tinyBenchmarks
commit: 9c7e20302301ad531bfdfd9a7288e6e916bf22e9
license: MIT

This copy removes runtime download behavior and loads the packaged metadata
artifact from this repo instead.
"""


_ASSET_PATH = Path(__file__).with_name("assets") / "tinyBenchmarks.pkl"
_LB_SCENARIOS = {"truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"}
_BENCHES = {"lb", "mmlu", "helm_lite", "alpaca"}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _item_curve(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return _sigmoid(z)


def _fit_theta(
    responses_test: np.ndarray,
    seen_items: list[int],
    a: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1e-10,
    optimizer: str = "BFGS",
) -> np.ndarray:
    dimension = a.shape[1]

    def neg_log_like(x: np.ndarray) -> float:
        curve = _item_curve(x.reshape(1, dimension, 1), a[:, :, seen_items], b[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(
            responses_test[seen_items] * np.log(curve + eps) + (1 - responses_test[seen_items]) * np.log(1 - curve + eps)
        )
        return float(-log_likelihood)

    optimum = minimize(neg_log_like, np.zeros(dimension), method=optimizer).x
    return optimum[None, :, None]


def _load_artifact(asset_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(asset_path) if asset_path is not None else _ASSET_PATH
    with path.open("rb") as handle:
        return pickle.load(handle)


def evaluate_tinybenchmarks(
    y: list[int] | list[bool] | np.ndarray,
    benchmark_name: str,
    *,
    asset_path: str | Path | None = None,
) -> dict[str, float]:
    y_input = np.asarray(y, dtype=float)
    if y_input.ndim != 1:
        raise ValueError("y must be a unidimensional vector")
    if y_input.shape[0] != 100:
        raise ValueError(f"y must contain exactly 100 entries, found {y_input.shape[0]}")
    if benchmark_name not in _BENCHES.union(_LB_SCENARIOS):
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

    tiny_benchmarks = _load_artifact(asset_path)
    bench_key = "lb" if benchmark_name in _LB_SCENARIOS else benchmark_name
    bundle = tiny_benchmarks[bench_key]

    seen_examples = list(bundle["seen_examples"])
    examples_weights = bundle["examples_weights"]
    irt_parameters = bundle["irt_parameters"]
    optimal_lambdas = bundle["optimal_lambdas"]
    scenarios_position = bundle["scenarios_position"]
    subscenarios_position = bundle["subscenarios_position"]
    a = irt_parameters["A"]
    b = irt_parameters["B"]

    num_positions = max(max(indices) for indices in scenarios_position.values()) + 1
    balance_weights = np.ones(num_positions)
    for scenario_name, positions in scenarios_position.items():
        num_positions_in_scenario = len(positions)
        num_subscenarios = len(subscenarios_position[scenario_name])
        for subscenario_name, sub_positions in subscenarios_position[scenario_name].items():
            sub_count = len(sub_positions)
            balance_weights[sub_positions] = num_positions_in_scenario / (num_subscenarios * sub_count)

    if benchmark_name not in _BENCHES:
        scenario_names = [benchmark_name]
        scenario_offset = 100 * [index for index, name in enumerate(scenarios_position.keys()) if name == benchmark_name][0]
        seen_examples = seen_examples[scenario_offset : scenario_offset + 100]
    else:
        scenario_names = list(scenarios_position.keys())

    response_vector = np.zeros(num_positions)
    for index, seen_example in enumerate(seen_examples):
        response_vector[seen_example] = y_input[index]

    theta = _fit_theta(response_vector, seen_examples, a, b)
    unseen_examples = [index for index in range(num_positions) if index not in seen_examples]

    estimates: dict[str, dict[str, float]] = {}
    for scenario_name in scenario_names:
        scenario_positions = scenarios_position[scenario_name]
        num_scenario_examples = len(scenario_positions)
        seen_examples_scenario = [index for index in seen_examples if index in scenario_positions]
        unseen_examples_scenario = [index for index in unseen_examples if index in scenario_positions]

        data_part_irtp = ((balance_weights * response_vector)[seen_examples_scenario]).mean()
        irt_part = (balance_weights * _item_curve(theta.reshape(1, a.shape[1], 1), a, b))[0, [unseen_examples_scenario]].mean()
        irtp_lambda = 100 / num_scenario_examples
        irt = float((examples_weights[scenario_name] * response_vector[seen_examples_scenario]).sum())
        pirt = float(irtp_lambda * data_part_irtp + (1 - irtp_lambda) * irt_part)
        gpirt = float(optimal_lambdas[scenario_name] * irt + (1 - optimal_lambdas[scenario_name]) * pirt)

        estimates[scenario_name] = {"irt": irt, "pirt": pirt, "gpirt": gpirt}

    if benchmark_name in estimates:
        return estimates[benchmark_name]
    if len(estimates) == 1:
        return next(iter(estimates.values()))
    raise ValueError(f"Benchmark {benchmark_name} returned multiple scenarios: {sorted(estimates)}")


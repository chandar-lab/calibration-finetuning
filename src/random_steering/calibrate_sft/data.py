from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig, OmegaConf

from random_steering.eval.metrics import frozen_distribution
from random_steering.inference.chat_format import format_prompt
from random_steering.types import DistributionSpec


@dataclass(frozen=True)
class CalibrateFamilySpec:
    family_id: str
    display_name: str
    tier: str
    kind: str
    grid_parameter_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    scipy_name: str
    integer_only: bool
    transform_grid_params: Callable[[dict[str, float]], dict[str, float]]
    build_scipy_params: Callable[[dict[str, float]], dict[str, float]]
    support_bounds: Callable[[dict[str, float]], tuple[float | None, float | None]]


@dataclass(frozen=True)
class PromptSpec:
    split: str
    family_id: str
    display_name: str
    tier: str
    distribution_id: str
    parameters: dict[str, float]
    kind: str
    integer_only: bool


@dataclass(frozen=True)
class OutputSpace:
    values: tuple[str, ...]
    probabilities: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    integer_only: bool
    num_decimals: int


@dataclass(frozen=True)
class PrefixTarget:
    prefix_token_ids: tuple[int, ...]
    next_token_ids: tuple[int, ...]
    next_token_probs: tuple[float, ...]


@dataclass(frozen=True)
class TokenizedOutputSpace:
    output_space: OutputSpace
    candidate_token_ids: tuple[tuple[int, ...], ...]
    prefix_targets: dict[tuple[int, ...], PrefixTarget]


@dataclass(frozen=True)
class PreparedPrompt:
    prompt_spec: PromptSpec
    distribution_spec: DistributionSpec
    prompt_text: str
    formatted_prompt: str
    prompt_token_ids: tuple[int, ...]
    output_space: OutputSpace
    tokenized_output_space: TokenizedOutputSpace


@dataclass(frozen=True)
class TrainingExample:
    prepared_prompt: PreparedPrompt
    candidate_value: str
    candidate_token_ids: tuple[int, ...]
    prefix_targets: tuple[PrefixTarget, ...]


def _identity_params(parameters: dict[str, float]) -> dict[str, float]:
    return dict(parameters)


def _uniform_params(parameters: dict[str, float]) -> dict[str, float]:
    a = float(parameters["a"])
    width = float(parameters["width"])
    return {"a": a, "b": a + width}


def _uniform_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["a"], "scale": parameters["b"] - parameters["a"]}


def _uniform_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return parameters["a"], parameters["b"]


def _gaussian_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["mu"], "scale": parameters["sigma"]}


def _beta_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"a": parameters["alpha"], "b": parameters["beta"]}


def _binomial_params(parameters: dict[str, float]) -> dict[str, float]:
    return {"n": float(round(parameters["n"])), "p": float(parameters["p"])}


def _binomial_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"n": int(round(parameters["n"])), "p": parameters["p"]}


def _finite_integer_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return 0.0, float(round(parameters["n"]))


def _positive_integer_support(_parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return 1.0, None


def _bernoulli_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"p": parameters["p"]}


def _bernoulli_support(_parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return 0.0, 1.0


def _exponential_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"scale": 1.0 / parameters["lambda"]}


def _positive_support(_parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return 0.0, None


def _cauchy_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["x0"], "scale": parameters["gamma"]}


def _student_t_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"df": parameters["nu"]}


def _chi_square_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"df": parameters["nu"]}


def _f_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"dfn": parameters["d1"], "dfd": parameters["d2"]}


def _gamma_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"a": parameters["alpha"], "scale": parameters["beta"]}


def _weibull_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"c": parameters["k"], "scale": parameters["lambda"]}


def _laplace_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["mu"], "scale": parameters["b"]}


def _logistic_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["mu"], "scale": parameters["s"]}


def _negative_binomial_params(parameters: dict[str, float]) -> dict[str, float]:
    return {"r": float(round(parameters["r"])), "p": float(parameters["p"])}


def _negative_binomial_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"n": int(round(parameters["r"])), "p": parameters["p"]}


def _lognormal_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"s": parameters["sigma"], "scale": math.exp(parameters["mu"])}


def _triangular_params(parameters: dict[str, float]) -> dict[str, float]:
    a = float(parameters["a"])
    width = float(parameters["width"])
    mode_fraction = float(parameters["mode_fraction"])
    b = a + width
    c = a + width * mode_fraction
    return {"a": a, "c": c, "b": b}


def _triangular_scipy(parameters: dict[str, float]) -> dict[str, float]:
    width = parameters["b"] - parameters["a"]
    return {"c": (parameters["c"] - parameters["a"]) / width, "loc": parameters["a"], "scale": width}


def _triangular_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return parameters["a"], parameters["b"]


def _rayleigh_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"scale": parameters["sigma"]}


def _pareto_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"b": parameters["alpha"], "scale": parameters["xm"]}


def _pareto_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return parameters["xm"], None


def _hypergeometric_params(parameters: dict[str, float]) -> dict[str, float]:
    population = int(round(parameters["M"]))
    draws = int(round(parameters["N"]))
    success_fraction = float(parameters["success_fraction"])
    successes = min(population, max(0, int(round(population * success_fraction))))
    draws = min(draws, population)
    return {"M": float(population), "K": float(successes), "N": float(draws)}


def _hypergeometric_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {
        "M": int(round(parameters["M"])),
        "n": int(round(parameters["K"])),
        "N": int(round(parameters["N"])),
    }


def _hypergeometric_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    population = int(round(parameters["M"]))
    successes = int(round(parameters["K"]))
    draws = int(round(parameters["N"]))
    lo = max(0, draws - (population - successes))
    hi = min(draws, successes)
    return float(lo), float(hi)


def _gumbel_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"loc": parameters["mu"], "scale": parameters["beta"]}


def _skellam_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"mu1": parameters["mu1"], "mu2": parameters["mu2"]}


def _betabinomial_params(parameters: dict[str, float]) -> dict[str, float]:
    return {
        "n": float(round(parameters["n"])),
        "alpha": float(parameters["alpha"]),
        "beta": float(parameters["beta"]),
    }


def _betabinomial_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {
        "n": int(round(parameters["n"])),
        "a": parameters["alpha"],
        "b": parameters["beta"],
    }


def _lomax_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"c": parameters["alpha"], "scale": parameters["lambda"]}


def _inverse_gaussian_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"mu": parameters["mu"] / parameters["lambda"], "scale": parameters["lambda"]}


def _maxwell_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"scale": parameters["scale"]}


def _chi_scipy(parameters: dict[str, float]) -> dict[str, float]:
    return {"df": parameters["df"]}


def _truncnorm_scipy(parameters: dict[str, float]) -> dict[str, float]:
    mu = parameters["mu"]
    sigma = parameters["sigma"]
    lower = parameters["lower"]
    upper = parameters["upper"]
    return {
        "a": (lower - mu) / sigma,
        "b": (upper - mu) / sigma,
        "loc": mu,
        "scale": sigma,
    }


def _truncnorm_support(parameters: dict[str, float]) -> tuple[float | None, float | None]:
    return parameters["lower"], parameters["upper"]


_FAMILY_SPECS: dict[str, CalibrateFamilySpec] = {
    "uniform": CalibrateFamilySpec(
        family_id="uniform",
        display_name="Uniform",
        tier="I",
        kind="continuous",
        grid_parameter_names=("a", "width"),
        parameter_names=("a", "b"),
        scipy_name="uniform",
        integer_only=False,
        transform_grid_params=_uniform_params,
        build_scipy_params=_uniform_scipy,
        support_bounds=_uniform_support,
    ),
    "gaussian": CalibrateFamilySpec(
        family_id="gaussian",
        display_name="Gaussian",
        tier="I",
        kind="continuous",
        grid_parameter_names=("mu", "sigma"),
        parameter_names=("mu", "sigma"),
        scipy_name="norm",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_gaussian_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "bernoulli": CalibrateFamilySpec(
        family_id="bernoulli",
        display_name="Bernoulli",
        tier="I",
        kind="discrete",
        grid_parameter_names=("p",),
        parameter_names=("p",),
        scipy_name="bernoulli",
        integer_only=True,
        transform_grid_params=_identity_params,
        build_scipy_params=_bernoulli_scipy,
        support_bounds=_bernoulli_support,
    ),
    "beta": CalibrateFamilySpec(
        family_id="beta",
        display_name="Beta",
        tier="II",
        kind="continuous",
        grid_parameter_names=("alpha", "beta"),
        parameter_names=("alpha", "beta"),
        scipy_name="beta",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_beta_scipy,
        support_bounds=lambda _parameters: (0.0, 1.0),
    ),
    "binomial": CalibrateFamilySpec(
        family_id="binomial",
        display_name="Binomial",
        tier="II",
        kind="discrete",
        grid_parameter_names=("n", "p"),
        parameter_names=("n", "p"),
        scipy_name="binom",
        integer_only=True,
        transform_grid_params=_binomial_params,
        build_scipy_params=_binomial_scipy,
        support_bounds=_finite_integer_support,
    ),
    "poisson": CalibrateFamilySpec(
        family_id="poisson",
        display_name="Poisson",
        tier="II",
        kind="discrete",
        grid_parameter_names=("lambda",),
        parameter_names=("lambda",),
        scipy_name="poisson",
        integer_only=True,
        transform_grid_params=_identity_params,
        build_scipy_params=lambda parameters: {"mu": parameters["lambda"]},
        support_bounds=lambda _parameters: (0.0, None),
    ),
    "exponential": CalibrateFamilySpec(
        family_id="exponential",
        display_name="Exponential",
        tier="II",
        kind="continuous",
        grid_parameter_names=("lambda",),
        parameter_names=("lambda",),
        scipy_name="expon",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_exponential_scipy,
        support_bounds=_positive_support,
    ),
    "geometric": CalibrateFamilySpec(
        family_id="geometric",
        display_name="Geometric",
        tier="II",
        kind="discrete",
        grid_parameter_names=("p",),
        parameter_names=("p",),
        scipy_name="geom",
        integer_only=True,
        transform_grid_params=_identity_params,
        build_scipy_params=lambda parameters: {"p": parameters["p"]},
        support_bounds=_positive_integer_support,
    ),
    "negative_binomial": CalibrateFamilySpec(
        family_id="negative_binomial",
        display_name="Negative Binomial",
        tier="II",
        kind="discrete",
        grid_parameter_names=("r", "p"),
        parameter_names=("r", "p"),
        scipy_name="nbinom",
        integer_only=True,
        transform_grid_params=_negative_binomial_params,
        build_scipy_params=_negative_binomial_scipy,
        support_bounds=lambda _parameters: (0.0, None),
    ),
    "lognormal": CalibrateFamilySpec(
        family_id="lognormal",
        display_name="LogNormal",
        tier="II",
        kind="continuous",
        grid_parameter_names=("mu", "sigma"),
        parameter_names=("mu", "sigma"),
        scipy_name="lognorm",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_lognormal_scipy,
        support_bounds=_positive_support,
    ),
    "triangular": CalibrateFamilySpec(
        family_id="triangular",
        display_name="Triangular",
        tier="II",
        kind="continuous",
        grid_parameter_names=("a", "width", "mode_fraction"),
        parameter_names=("a", "c", "b"),
        scipy_name="triang",
        integer_only=False,
        transform_grid_params=_triangular_params,
        build_scipy_params=_triangular_scipy,
        support_bounds=_triangular_support,
    ),
    "rayleigh": CalibrateFamilySpec(
        family_id="rayleigh",
        display_name="Rayleigh",
        tier="II",
        kind="continuous",
        grid_parameter_names=("sigma",),
        parameter_names=("sigma",),
        scipy_name="rayleigh",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_rayleigh_scipy,
        support_bounds=_positive_support,
    ),
    "cauchy": CalibrateFamilySpec(
        family_id="cauchy",
        display_name="Cauchy",
        tier="III",
        kind="continuous",
        grid_parameter_names=("x0", "gamma"),
        parameter_names=("x0", "gamma"),
        scipy_name="cauchy",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_cauchy_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "student_t": CalibrateFamilySpec(
        family_id="student_t",
        display_name="Student's t",
        tier="III",
        kind="continuous",
        grid_parameter_names=("nu",),
        parameter_names=("nu",),
        scipy_name="t",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_student_t_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "chi_square": CalibrateFamilySpec(
        family_id="chi_square",
        display_name="Chi-Square",
        tier="III",
        kind="continuous",
        grid_parameter_names=("nu",),
        parameter_names=("nu",),
        scipy_name="chi2",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_chi_square_scipy,
        support_bounds=_positive_support,
    ),
    "f_distribution": CalibrateFamilySpec(
        family_id="f_distribution",
        display_name="F-Distribution",
        tier="III",
        kind="continuous",
        grid_parameter_names=("d1", "d2"),
        parameter_names=("d1", "d2"),
        scipy_name="f",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_f_scipy,
        support_bounds=_positive_support,
    ),
    "gamma": CalibrateFamilySpec(
        family_id="gamma",
        display_name="Gamma",
        tier="III",
        kind="continuous",
        grid_parameter_names=("alpha", "beta"),
        parameter_names=("alpha", "beta"),
        scipy_name="gamma",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_gamma_scipy,
        support_bounds=_positive_support,
    ),
    "weibull": CalibrateFamilySpec(
        family_id="weibull",
        display_name="Weibull",
        tier="III",
        kind="continuous",
        grid_parameter_names=("k", "lambda"),
        parameter_names=("k", "lambda"),
        scipy_name="weibull_min",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_weibull_scipy,
        support_bounds=_positive_support,
    ),
    "laplace": CalibrateFamilySpec(
        family_id="laplace",
        display_name="Laplace",
        tier="III",
        kind="continuous",
        grid_parameter_names=("mu", "b"),
        parameter_names=("mu", "b"),
        scipy_name="laplace",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_laplace_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "logistic": CalibrateFamilySpec(
        family_id="logistic",
        display_name="Logistic",
        tier="III",
        kind="continuous",
        grid_parameter_names=("mu", "s"),
        parameter_names=("mu", "s"),
        scipy_name="logistic",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_logistic_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "pareto": CalibrateFamilySpec(
        family_id="pareto",
        display_name="Pareto",
        tier="III",
        kind="continuous",
        grid_parameter_names=("alpha", "xm"),
        parameter_names=("alpha", "xm"),
        scipy_name="pareto",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_pareto_scipy,
        support_bounds=_pareto_support,
    ),
    "hypergeometric": CalibrateFamilySpec(
        family_id="hypergeometric",
        display_name="Hypergeometric",
        tier="III",
        kind="discrete",
        grid_parameter_names=("M", "N", "success_fraction"),
        parameter_names=("M", "K", "N"),
        scipy_name="hypergeom",
        integer_only=True,
        transform_grid_params=_hypergeometric_params,
        build_scipy_params=_hypergeometric_scipy,
        support_bounds=_hypergeometric_support,
    ),
    "gumbel": CalibrateFamilySpec(
        family_id="gumbel",
        display_name="Gumbel",
        tier="III",
        kind="continuous",
        grid_parameter_names=("mu", "beta"),
        parameter_names=("mu", "beta"),
        scipy_name="gumbel_r",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_gumbel_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "skellam": CalibrateFamilySpec(
        family_id="skellam",
        display_name="Skellam",
        tier="III",
        kind="discrete",
        grid_parameter_names=("mu1", "mu2"),
        parameter_names=("mu1", "mu2"),
        scipy_name="skellam",
        integer_only=True,
        transform_grid_params=_identity_params,
        build_scipy_params=_skellam_scipy,
        support_bounds=lambda _parameters: (None, None),
    ),
    "beta_binomial": CalibrateFamilySpec(
        family_id="beta_binomial",
        display_name="Beta-Binomial",
        tier="III",
        kind="discrete",
        grid_parameter_names=("n", "alpha", "beta"),
        parameter_names=("n", "alpha", "beta"),
        scipy_name="betabinom",
        integer_only=True,
        transform_grid_params=_betabinomial_params,
        build_scipy_params=_betabinomial_scipy,
        support_bounds=_finite_integer_support,
    ),
    "lomax": CalibrateFamilySpec(
        family_id="lomax",
        display_name="Lomax",
        tier="III",
        kind="continuous",
        grid_parameter_names=("alpha", "lambda"),
        parameter_names=("alpha", "lambda"),
        scipy_name="lomax",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_lomax_scipy,
        support_bounds=_positive_support,
    ),
    "inverse_gaussian": CalibrateFamilySpec(
        family_id="inverse_gaussian",
        display_name="Inverse Gaussian",
        tier="III",
        kind="continuous",
        grid_parameter_names=("mu", "lambda"),
        parameter_names=("mu", "lambda"),
        scipy_name="invgauss",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_inverse_gaussian_scipy,
        support_bounds=_positive_support,
    ),
    "maxwell": CalibrateFamilySpec(
        family_id="maxwell",
        display_name="Maxwell",
        tier="II",
        kind="continuous",
        grid_parameter_names=("scale",),
        parameter_names=("scale",),
        scipy_name="maxwell",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_maxwell_scipy,
        support_bounds=_positive_support,
    ),
    "chi": CalibrateFamilySpec(
        family_id="chi",
        display_name="Chi",
        tier="III",
        kind="continuous",
        grid_parameter_names=("df",),
        parameter_names=("df",),
        scipy_name="chi",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_chi_scipy,
        support_bounds=_positive_support,
    ),
    "truncnorm": CalibrateFamilySpec(
        family_id="truncnorm",
        display_name="TruncNorm",
        tier="II",
        kind="continuous",
        grid_parameter_names=("mu", "sigma", "lower", "upper"),
        parameter_names=("mu", "sigma", "lower", "upper"),
        scipy_name="truncnorm",
        integer_only=False,
        transform_grid_params=_identity_params,
        build_scipy_params=_truncnorm_scipy,
        support_bounds=_truncnorm_support,
    ),
}


def get_family_spec(family_id: str) -> CalibrateFamilySpec:
    if family_id not in _FAMILY_SPECS:
        raise KeyError(f"Unknown Calibrate-SFT family: {family_id}")
    return _FAMILY_SPECS[family_id]


def _to_plain_data(cfg: dict[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _slugify_number(value: float) -> str:
    if float(value).is_integer():
        text = str(int(round(value)))
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _make_distribution_id(family: CalibrateFamilySpec, parameters: dict[str, float]) -> str:
    parts = [family.family_id]
    for key in family.parameter_names:
        parts.append(key)
        parts.append(_slugify_number(parameters[key]))
    return "_".join(parts)


def _parameter_values(parameter_cfg: dict[str, Any], default_num_points: int) -> list[float]:
    if "values" in parameter_cfg:
        return [float(v) for v in parameter_cfg["values"]]
    lo = float(parameter_cfg["min"])
    hi = float(parameter_cfg["max"])
    num_points = int(parameter_cfg.get("num_points", default_num_points))
    if num_points <= 1 or math.isclose(lo, hi):
        return [lo]
    spacing = str(parameter_cfg.get("spacing", "linspace"))
    if spacing == "linspace":
        return np.linspace(lo, hi, num=num_points).astype(float).tolist()
    if spacing == "logspace":
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("logspace parameter sampling requires strictly positive min/max.")
        return np.geomspace(lo, hi, num=num_points).astype(float).tolist()
    raise ValueError(f"Unsupported parameter spacing: {spacing}")


def _build_parameter_grid(
    family: CalibrateFamilySpec,
    split_cfg: dict[str, Any],
    points_1d: int,
    points_2d: int,
) -> list[dict[str, float]]:
    per_param: list[list[float]] = []
    num_points = points_1d if len(family.grid_parameter_names) == 1 else points_2d
    for key in family.grid_parameter_names:
        if key not in split_cfg:
            raise KeyError(f"Missing grid parameter '{key}' for family '{family.family_id}'")
        per_param.append(_parameter_values(split_cfg[key], num_points))

    rows: list[dict[str, float]] = []
    for combo in itertools.product(*per_param):
        raw = {key: float(value) for key, value in zip(family.grid_parameter_names, combo, strict=True)}
        materialized = family.transform_grid_params(raw)
        rows.append(materialized)
    return rows


def _build_selected_rows(
    family: CalibrateFamilySpec,
    selected_cfg: list[dict[str, Any]],
) -> list[dict[str, float]]:
    expected_keys = set(family.grid_parameter_names)
    rows: list[dict[str, float]] = []
    for row_index, raw_row in enumerate(selected_cfg):
        actual_keys = set(raw_row)
        if actual_keys != expected_keys:
            raise ValueError(
                f"Selected config #{row_index} for family '{family.family_id}' must use exactly "
                f"{family.grid_parameter_names}, got {tuple(sorted(actual_keys))}."
            )
        materialized = family.transform_grid_params(
            {key: float(raw_row[key]) for key in family.grid_parameter_names}
        )
        rows.append(materialized)
    return rows


def _evenly_subsample(rows: list[dict[str, float]], max_items: int) -> list[dict[str, float]]:
    if max_items <= 0 or len(rows) <= max_items:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=max_items, dtype=int)
    return [rows[int(idx)] for idx in indices]


def _make_prompt_spec(split: str, family: CalibrateFamilySpec, parameters: dict[str, float]) -> PromptSpec:
    return PromptSpec(
        split=split,
        family_id=family.family_id,
        display_name=family.display_name,
        tier=family.tier,
        distribution_id=_make_distribution_id(family, parameters),
        parameters={key: float(parameters[key]) for key in family.parameter_names},
        kind=family.kind,
        integer_only=family.integer_only,
    )


def build_dataset_splits(cfg: dict[str, Any] | DictConfig) -> dict[str, list[PromptSpec]]:
    plain_cfg = _to_plain_data(cfg)
    splits: dict[str, list[PromptSpec]] = {
        "train": [],
        "test_unseen_params": [],
        "test_ood_family": [],
    }
    points_1d = int(plain_cfg["linspace_points_1d"])
    points_2d = int(plain_cfg["linspace_points_2d_per_axis"])
    eval_prompt_selection = str(plain_cfg.get("eval_prompt_selection", "full_grid"))
    family_cfgs = plain_cfg["families"]

    for family_id in plain_cfg["train_families"]:
        family = get_family_spec(str(family_id))
        config_entry = family_cfgs[family_id]
        train_rows = _build_parameter_grid(family, config_entry["train"], points_1d, points_2d)
        train_max_specs = int(config_entry["train"].get("max_specs", plain_cfg["train_max_specs_per_family"]))
        for parameters in _evenly_subsample(train_rows, train_max_specs):
            splits["train"].append(_make_prompt_spec("train", family, parameters))

        unseen_cfg = config_entry.get("test_unseen")
        if unseen_cfg is None:
            continue
        if eval_prompt_selection == "paper_selected" and "test_selected" in config_entry:
            unseen_rows = _build_selected_rows(family, list(config_entry["test_selected"]))
        else:
            unseen_rows = _build_parameter_grid(family, unseen_cfg, points_1d, points_2d)
            unseen_max_specs = int(unseen_cfg.get("max_specs", plain_cfg["test_unseen_max_specs_per_family"]))
            unseen_rows = _evenly_subsample(unseen_rows, unseen_max_specs)
        for parameters in unseen_rows:
            splits["test_unseen_params"].append(_make_prompt_spec("test_unseen_params", family, parameters))

    for family_id in plain_cfg["heldout_families"]:
        family = get_family_spec(str(family_id))
        config_entry = family_cfgs[family_id]
        if eval_prompt_selection == "paper_selected" and "ood_selected" in config_entry:
            ood_rows = _build_selected_rows(family, list(config_entry["ood_selected"]))
        else:
            ood_rows = _build_parameter_grid(family, config_entry["ood"], points_1d, points_2d)
            ood_max_specs = int(config_entry["ood"].get("max_specs", plain_cfg["ood_max_specs_per_family"]))
            ood_rows = _evenly_subsample(ood_rows, ood_max_specs)
        for parameters in ood_rows:
            splits["test_ood_family"].append(_make_prompt_spec("test_ood_family", family, parameters))

    return splits


def build_distribution_spec(prompt_spec: PromptSpec) -> DistributionSpec:
    family = get_family_spec(prompt_spec.family_id)
    support_min, support_max = family.support_bounds(prompt_spec.parameters)
    return DistributionSpec(
        distribution_id=prompt_spec.distribution_id,
        display_name=prompt_spec.display_name,
        parameters=dict(prompt_spec.parameters),
        family=prompt_spec.kind,
        support_min=support_min,
        support_max=support_max,
        integer_only=prompt_spec.integer_only,
        scipy_name=family.scipy_name,
        scipy_params=family.build_scipy_params(prompt_spec.parameters),
    )


def build_protocol_b_prompt(prompt_spec: PromptSpec) -> str:
    family = get_family_spec(prompt_spec.family_id)
    params_text = ", ".join(f"{key}={prompt_spec.parameters[key]}" for key in family.parameter_names)
    return (
        "Generate exactly ONE random number from a "
        f"{prompt_spec.display_name} distribution with parameters {params_text}. "
        "Output ONLY the number."
    )


def _format_canonical_value(value: float, integer_only: bool, num_decimals: int) -> str:
    if integer_only:
        return str(int(round(value)))
    return f"{value:.{num_decimals}f}"


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), a_min=0.0, a_max=None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("Target probabilities collapsed to zero mass.")
    return clipped / total


def _merge_duplicate_values(values: list[str], probabilities: list[float]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    merged_values: list[str] = []
    merged_probs: list[float] = []
    for value, probability in zip(values, probabilities, strict=True):
        if merged_values and merged_values[-1] == value:
            merged_probs[-1] += probability
            continue
        merged_values.append(value)
        merged_probs.append(probability)
    merged = _normalize_probabilities(np.asarray(merged_probs, dtype=float))
    return tuple(merged_values), tuple(float(v) for v in merged.tolist())


def _continuous_grid(lower: float, upper: float, num_decimals: int, max_bins: int) -> np.ndarray:
    step = 10.0 ** (-num_decimals)
    start = math.ceil(lower / step)
    end = math.floor(upper / step)
    if start > end:
        return np.asarray([round((lower + upper) / 2.0, num_decimals)], dtype=float)
    values = np.arange(start, end + 1, dtype=float) * step
    if values.size > max_bins:
        indices = np.linspace(0, values.size - 1, num=max_bins, dtype=int)
        values = values[indices]
    return values


def build_output_space(prompt_spec: PromptSpec, cfg: dict[str, Any] | DictConfig) -> OutputSpace:
    plain_cfg = _to_plain_data(cfg)
    dist_spec = build_distribution_spec(prompt_spec)
    dist = frozen_distribution(dist_spec)
    q_low = float(plain_cfg["quantile_low"])
    q_high = float(plain_cfg["quantile_high"])
    num_decimals = int(plain_cfg["num_decimals"])
    max_bins = int(plain_cfg["max_bins"])
    tail_policy = str(plain_cfg["tail_policy"])

    if prompt_spec.integer_only:
        if dist_spec.support_min is None:
            raw_support_min = float(dist.ppf(q_low))
            support_min = int(math.floor(raw_support_min)) if np.isfinite(raw_support_min) else -max_bins // 2
        else:
            support_min = int(math.floor(dist_spec.support_min))
        if dist_spec.support_max is not None:
            support_max = int(math.ceil(dist_spec.support_max))
        else:
            raw_support_max = float(dist.ppf(q_high))
            support_max = int(math.ceil(raw_support_max)) if np.isfinite(raw_support_max) else max_bins // 2
        if support_min > support_max:
            support_min, support_max = support_max, support_min
        support_values = np.arange(support_min, support_max + 1, dtype=int)
        if support_values.size > max_bins:
            support_values = support_values[:max_bins]
        probabilities = np.asarray(dist.pmf(support_values), dtype=float)
        if tail_policy == "clip_to_edge_bin" and dist_spec.support_max is None:
            upper_tail = max(0.0, 1.0 - float(dist.cdf(float(support_values[-1]))))
            probabilities[-1] += upper_tail
        probabilities = _normalize_probabilities(probabilities)
        values = tuple(str(int(v)) for v in support_values.tolist())
        return OutputSpace(
            values=values,
            probabilities=tuple(float(v) for v in probabilities.tolist()),
            lower_bound=float(support_values[0]),
            upper_bound=float(support_values[-1]),
            integer_only=True,
            num_decimals=0,
        )

    lower = float(dist.ppf(q_low))
    upper = float(dist.ppf(q_high))
    if dist_spec.support_min is not None:
        lower = max(lower, float(dist_spec.support_min))
    if dist_spec.support_max is not None:
        upper = min(upper, float(dist_spec.support_max))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower, upper = -10.0, 10.0

    centers = _continuous_grid(lower, upper, num_decimals, max_bins)
    if centers.size == 1:
        boundaries = np.asarray([lower, upper], dtype=float)
    else:
        midpoints = (centers[:-1] + centers[1:]) / 2.0
        boundaries = np.concatenate(([lower], midpoints, [upper]))
    probabilities = np.asarray(dist.cdf(boundaries[1:]) - dist.cdf(boundaries[:-1]), dtype=float)
    if tail_policy == "clip_to_edge_bin":
        probabilities[0] += max(0.0, float(dist.cdf(lower)))
        probabilities[-1] += max(0.0, 1.0 - float(dist.cdf(upper)))
    probabilities = _normalize_probabilities(probabilities)
    values, merged_probs = _merge_duplicate_values(
        [_format_canonical_value(float(center), False, num_decimals) for center in centers.tolist()],
        probabilities.tolist(),
    )
    return OutputSpace(
        values=values,
        probabilities=merged_probs,
        lower_bound=float(lower),
        upper_bound=float(upper),
        integer_only=False,
        num_decimals=num_decimals,
    )


def _tokenize_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    if hasattr(tokenizer, "encode"):
        return tuple(int(tok) for tok in tokenizer.encode(text, add_special_tokens=False))
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    return tuple(int(tok) for tok in input_ids)


def tokenize_prompt(tokenizer: Any, model_cfg: Any, prompt_text: str) -> tuple[str, tuple[int, ...]]:
    formatted_prompt = format_prompt(tokenizer, prompt_text, enable_thinking=getattr(model_cfg, "enable_thinking", None), reasoning_effort=getattr(model_cfg, "reasoning_effort", None), generation_prefix=getattr(model_cfg, "generation_prefix", None))
    tokenized = tokenizer(formatted_prompt, add_special_tokens=True)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    return formatted_prompt, tuple(int(tok) for tok in input_ids)


def build_tokenized_output_space(output_space: OutputSpace, tokenizer: Any) -> TokenizedOutputSpace:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for trie construction.")

    candidate_token_ids = tuple(_tokenize_text(tokenizer, value) for value in output_space.values)
    next_mass: dict[tuple[int, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    prefix_mass: dict[tuple[int, ...], float] = defaultdict(float)

    for tokens, probability in zip(candidate_token_ids, output_space.probabilities, strict=True):
        sequence = tokens + (int(eos_token_id),)
        prefix: tuple[int, ...] = ()
        for token_id in sequence:
            prefix_mass[prefix] += float(probability)
            next_mass[prefix][int(token_id)] += float(probability)
            prefix = prefix + (int(token_id),)

    prefix_targets: dict[tuple[int, ...], PrefixTarget] = {}
    for prefix, token_masses in next_mass.items():
        total_mass = float(prefix_mass[prefix])
        token_ids = tuple(int(token_id) for token_id in token_masses.keys())
        probabilities = tuple(float(token_masses[token_id] / total_mass) for token_id in token_ids)
        prefix_targets[prefix] = PrefixTarget(
            prefix_token_ids=prefix,
            next_token_ids=token_ids,
            next_token_probs=probabilities,
        )

    return TokenizedOutputSpace(
        output_space=output_space,
        candidate_token_ids=candidate_token_ids,
        prefix_targets=prefix_targets,
    )


def prepare_prompt_spec(prompt_spec: PromptSpec, tokenizer: Any, model_cfg: Any, data_cfg: dict[str, Any] | DictConfig) -> PreparedPrompt:
    distribution_spec = build_distribution_spec(prompt_spec)
    prompt_text = build_protocol_b_prompt(prompt_spec)
    formatted_prompt, prompt_token_ids = tokenize_prompt(tokenizer, model_cfg, prompt_text)
    output_space = build_output_space(prompt_spec, data_cfg)
    tokenized_output_space = build_tokenized_output_space(output_space, tokenizer)
    return PreparedPrompt(
        prompt_spec=prompt_spec,
        distribution_spec=distribution_spec,
        prompt_text=prompt_text,
        formatted_prompt=formatted_prompt,
        prompt_token_ids=prompt_token_ids,
        output_space=output_space,
        tokenized_output_space=tokenized_output_space,
    )


def build_prepared_splits(
    tokenizer: Any,
    model_cfg: Any,
    data_cfg: dict[str, Any] | DictConfig,
) -> dict[str, list[PreparedPrompt]]:
    splits = build_dataset_splits(data_cfg)
    prepared: dict[str, list[PreparedPrompt]] = {}
    for split_name, prompt_specs in splits.items():
        prepared[split_name] = [prepare_prompt_spec(prompt_spec, tokenizer, model_cfg, data_cfg) for prompt_spec in prompt_specs]
    return prepared


def build_prepared_selected_splits(
    tokenizer: Any,
    model_cfg: Any,
    data_cfg: dict[str, Any] | DictConfig,
    split_names: list[str],
) -> dict[str, list[PreparedPrompt]]:
    splits = build_dataset_splits(data_cfg)
    prepared: dict[str, list[PreparedPrompt]] = {}
    for split_name in split_names:
        if split_name not in splits:
            raise KeyError(f"Unknown Calibrate-SFT split for evaluation: {split_name}")
        prepared[split_name] = [
            prepare_prompt_spec(prompt_spec, tokenizer, model_cfg, data_cfg)
            for prompt_spec in splits[split_name]
        ]
    return prepared


def sample_training_example(prepared_prompt: PreparedPrompt, sample_seed: int) -> TrainingExample:
    rng = np.random.default_rng(int(sample_seed))
    probabilities = np.asarray(prepared_prompt.output_space.probabilities, dtype=float)
    sample_index = int(rng.choice(len(prepared_prompt.output_space.values), p=probabilities))
    candidate_value = prepared_prompt.output_space.values[sample_index]
    candidate_tokens = prepared_prompt.tokenized_output_space.candidate_token_ids[sample_index]
    prefix_targets: list[PrefixTarget] = []
    for step_index in range(len(candidate_tokens) + 1):
        prefix = candidate_tokens[:step_index]
        prefix_targets.append(prepared_prompt.tokenized_output_space.prefix_targets[prefix])
    return TrainingExample(
        prepared_prompt=prepared_prompt,
        candidate_value=candidate_value,
        candidate_token_ids=candidate_tokens,
        prefix_targets=tuple(prefix_targets),
    )


def prompt_spec_to_dict(prompt_spec: PromptSpec) -> dict[str, Any]:
    return asdict(prompt_spec)


def output_space_to_dict(output_space: OutputSpace) -> dict[str, Any]:
    return asdict(output_space)

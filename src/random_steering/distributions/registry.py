from __future__ import annotations

from random_steering.types import DistributionSpec


_DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "uniform_0_1": DistributionSpec(
        distribution_id="uniform_0_1",
        display_name="Uniform",
        parameters={"a": 0.0, "b": 1.0},
        family="continuous",
        support_min=0.0,
        support_max=1.0,
        integer_only=False,
        scipy_name="uniform",
        scipy_params={"loc": 0.0, "scale": 1.0},
    ),
    "gaussian_0_1": DistributionSpec(
        distribution_id="gaussian_0_1",
        display_name="Gaussian",
        parameters={"mu": 0.0, "sigma": 1.0},
        family="continuous",
        support_min=None,
        support_max=None,
        integer_only=False,
        scipy_name="norm",
        scipy_params={"loc": 0.0, "scale": 1.0},
    ),
    "bernoulli_0_7": DistributionSpec(
        distribution_id="bernoulli_0_7",
        display_name="Bernoulli",
        parameters={"p": 0.7},
        family="discrete",
        support_min=0.0,
        support_max=1.0,
        integer_only=True,
        scipy_name="bernoulli",
        scipy_params={"p": 0.7},
    ),
    "beta_2_2": DistributionSpec(
        distribution_id="beta_2_2",
        display_name="Beta",
        parameters={"alpha": 2.0, "beta": 2.0},
        family="continuous",
        support_min=0.0,
        support_max=1.0,
        integer_only=False,
        scipy_name="beta",
        scipy_params={"a": 2.0, "b": 2.0},
    ),
    "binomial_10_0_5": DistributionSpec(
        distribution_id="binomial_10_0_5",
        display_name="Binomial",
        parameters={"n": 10.0, "p": 0.5},
        family="discrete",
        support_min=0.0,
        support_max=10.0,
        integer_only=True,
        scipy_name="binom",
        scipy_params={"n": 10, "p": 0.5},
    ),
    "poisson_5": DistributionSpec(
        distribution_id="poisson_5",
        display_name="Poisson",
        parameters={"lambda": 5.0},
        family="discrete",
        support_min=0.0,
        support_max=None,
        integer_only=True,
        scipy_name="poisson",
        scipy_params={"mu": 5.0},
    ),
    "exponential_1": DistributionSpec(
        distribution_id="exponential_1",
        display_name="Exponential",
        parameters={"lambda": 1.0},
        family="continuous",
        support_min=0.0,
        support_max=None,
        integer_only=False,
        scipy_name="expon",
        scipy_params={"scale": 1.0},
    ),
    "cauchy_0_1": DistributionSpec(
        distribution_id="cauchy_0_1",
        display_name="Cauchy",
        parameters={"x0": 0.0, "gamma": 1.0},
        family="continuous",
        support_min=None,
        support_max=None,
        integer_only=False,
        scipy_name="cauchy",
        scipy_params={"loc": 0.0, "scale": 1.0},
    ),
    "student_t_3": DistributionSpec(
        distribution_id="student_t_3",
        display_name="Student's t",
        parameters={"nu": 3.0},
        family="continuous",
        support_min=None,
        support_max=None,
        integer_only=False,
        scipy_name="t",
        scipy_params={"df": 3.0},
    ),
    "chi_square_5": DistributionSpec(
        distribution_id="chi_square_5",
        display_name="Chi-Square",
        parameters={"nu": 5.0},
        family="continuous",
        support_min=0.0,
        support_max=None,
        integer_only=False,
        scipy_name="chi2",
        scipy_params={"df": 5.0},
    ),
    "f_5_10": DistributionSpec(
        distribution_id="f_5_10",
        display_name="F-Distribution",
        parameters={"d1": 5.0, "d2": 10.0},
        family="continuous",
        support_min=0.0,
        support_max=None,
        integer_only=False,
        scipy_name="f",
        scipy_params={"dfn": 5.0, "dfd": 10.0},
    ),
    "gamma_2_2": DistributionSpec(
        distribution_id="gamma_2_2",
        display_name="Gamma",
        parameters={"alpha": 2.0, "beta": 2.0},
        family="continuous",
        support_min=0.0,
        support_max=None,
        integer_only=False,
        scipy_name="gamma",
        scipy_params={"a": 2.0, "scale": 2.0},
    ),
    "weibull_1_5_1": DistributionSpec(
        distribution_id="weibull_1_5_1",
        display_name="Weibull",
        parameters={"k": 1.5, "lambda": 1.0},
        family="continuous",
        support_min=0.0,
        support_max=None,
        integer_only=False,
        scipy_name="weibull_min",
        scipy_params={"c": 1.5, "scale": 1.0},
    ),
    "laplace_0_1": DistributionSpec(
        distribution_id="laplace_0_1",
        display_name="Laplace",
        parameters={"mu": 0.0, "b": 1.0},
        family="continuous",
        support_min=None,
        support_max=None,
        integer_only=False,
        scipy_name="laplace",
        scipy_params={"loc": 0.0, "scale": 1.0},
    ),
    "logistic_0_1": DistributionSpec(
        distribution_id="logistic_0_1",
        display_name="Logistic",
        parameters={"mu": 0.0, "s": 1.0},
        family="continuous",
        support_min=None,
        support_max=None,
        integer_only=False,
        scipy_name="logistic",
        scipy_params={"loc": 0.0, "scale": 1.0},
    ),
}


def all_distribution_ids() -> list[str]:
    return list(_DISTRIBUTIONS.keys())


def get_distribution(distribution_id: str) -> DistributionSpec:
    if distribution_id not in _DISTRIBUTIONS:
        raise KeyError(f"Unknown distribution: {distribution_id}")
    return _DISTRIBUTIONS[distribution_id]


def resolve_distributions(distribution_ids: list[str] | str) -> list[DistributionSpec]:
    if distribution_ids == "all":
        return [get_distribution(did) for did in all_distribution_ids()]
    if isinstance(distribution_ids, list) and len(distribution_ids) == 1 and distribution_ids[0] == "all":
        return [get_distribution(did) for did in all_distribution_ids()]
    if not isinstance(distribution_ids, list):
        raise TypeError("distribution_ids must be a list or 'all'")
    return [get_distribution(did) for did in distribution_ids]

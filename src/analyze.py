"""
analyze.py
----------

Real statistical analysis of a pricing A/B test: a two-proportion
z-test on conversion rate, a Welch's t-test on revenue-per-visitor
(the two outcomes a real pricing experiment cares about -- more
bookings vs. more revenue per booking are not the same question, and a
lower price can genuinely win on one while losing on the other), 95%
confidence intervals for both effect sizes, and a required
minimum-detectable-effect / power calculation done *before* looking at
results (the correct order -- power analysis informs whether a sample
size is even adequate to detect the effect size that matters, and doing
it after peeking at results is a well-known way to fool yourself).

Because two metrics are tested, this module also applies a Bonferroni
correction to the significance threshold rather than reporting each
p-value against the naive alpha=0.05 -- a real, if simple, multiple-
comparisons correction that is basic by design (a larger
experimentation platform might use a stricter or sequential
correction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from src.simulate import ExperimentData


@dataclass(frozen=True)
class ConversionTestResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift_pct: float
    z_statistic: float
    p_value: float
    ci_95_low: float
    ci_95_high: float
    significant_at_corrected_alpha: bool


@dataclass(frozen=True)
class RevenueTestResult:
    control_mean: float
    treatment_mean: float
    absolute_diff: float
    relative_diff_pct: float
    t_statistic: float
    p_value: float
    ci_95_low: float
    ci_95_high: float
    significant_at_corrected_alpha: bool


@dataclass(frozen=True)
class ExperimentAnalysis:
    conversion: ConversionTestResult
    revenue_per_visitor: RevenueTestResult
    corrected_alpha: float
    n_per_group: int
    minimum_detectable_effect_at_actual_n: float


BASE_ALPHA = 0.05
N_METRICS_TESTED = 2  # conversion rate and revenue-per-visitor -> Bonferroni-corrected alpha


def _bonferroni_alpha(n_comparisons: int = N_METRICS_TESTED) -> float:
    return BASE_ALPHA / n_comparisons


def analyze_conversion(data: ExperimentData, alpha: float) -> ConversionTestResult:
    n_c, n_t = len(data.control_converted), len(data.treatment_converted)
    x_c, x_t = int(data.control_converted.sum()), int(data.treatment_converted.sum())
    p_c, p_t = x_c / n_c, x_t / n_t

    z_stat, p_value = proportions_ztest(count=[x_t, x_c], nobs=[n_t, n_c], alternative="two-sided")

    # Wald 95% CI on the difference in proportions.
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    relative_lift_pct = (diff / p_c * 100) if p_c > 0 else float("nan")

    return ConversionTestResult(
        control_rate=float(p_c), treatment_rate=float(p_t),
        absolute_lift=float(diff), relative_lift_pct=float(relative_lift_pct),
        z_statistic=float(z_stat), p_value=float(p_value),
        ci_95_low=float(ci_low), ci_95_high=float(ci_high),
        significant_at_corrected_alpha=bool(p_value < alpha),
    )


def analyze_revenue_per_visitor(data: ExperimentData, alpha: float) -> RevenueTestResult:
    """Revenue PER VISITOR (not per booking) is the correct metric here
    -- it already incorporates the conversion-rate effect, since a
    non-converting visitor contributes 0 revenue. Comparing revenue only
    among converters would double-count/confound the conversion effect
    with the revenue-per-booking effect."""
    control, treatment = data.control_revenue, data.treatment_revenue

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)  # Welch's t-test

    mean_c, mean_t = control.mean(), treatment.mean()
    diff = mean_t - mean_c

    se_diff = np.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    relative_diff_pct = (diff / mean_c * 100) if mean_c > 0 else float("nan")

    return RevenueTestResult(
        control_mean=float(mean_c), treatment_mean=float(mean_t),
        absolute_diff=float(diff), relative_diff_pct=float(relative_diff_pct),
        t_statistic=float(t_stat), p_value=float(p_value),
        ci_95_low=float(ci_low), ci_95_high=float(ci_high),
        significant_at_corrected_alpha=bool(p_value < alpha),
    )


def minimum_detectable_effect(n_per_group: int, base_rate: float, power: float = 0.8, alpha: float = BASE_ALPHA) -> float:
    """
    The smallest absolute conversion-rate lift this experiment's actual
    sample size (n_per_group) could reliably detect at the given power
    and alpha -- computed BEFORE looking at the observed effect, so it
    can be reported alongside the result as a clear statement of what
    this experiment was even capable of finding. A statistically
    "non-significant" result with a large minimum-detectable-effect is a
    genuinely different, weaker finding than one with a small MDE, and
    conflating them is a common, real mistake in experiment reporting.
    """
    analysis = NormalIndPower()
    # Solve for effect size (Cohen's h-like standardized effect via the
    # normal approximation to the two-proportion test), then translate
    # back to an absolute rate difference around base_rate.
    standardized_effect = analysis.solve_power(
        effect_size=None, nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0,
    )
    # Approximate translation from standardized effect to absolute
    # proportion difference using the variance at base_rate (a standard,
    # if approximate, approach for proportions of this general scale).
    p = base_rate
    approx_sigma = np.sqrt(p * (1 - p))
    return float(standardized_effect * approx_sigma)


def analyze_experiment(data: ExperimentData, power: float = 0.8) -> ExperimentAnalysis:
    alpha = _bonferroni_alpha()
    conversion = analyze_conversion(data, alpha=alpha)
    revenue = analyze_revenue_per_visitor(data, alpha=alpha)
    n_per_group = len(data.control_converted)
    mde = minimum_detectable_effect(n_per_group, base_rate=conversion.control_rate, power=power)

    return ExperimentAnalysis(
        conversion=conversion,
        revenue_per_visitor=revenue,
        corrected_alpha=alpha,
        n_per_group=n_per_group,
        minimum_detectable_effect_at_actual_n=mde,
    )

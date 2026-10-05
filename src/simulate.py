"""
simulate.py
-----------

Generates synthetic pricing-experiment data: a control group (current
price) and a treatment group (a test price variant), with per-visitor
booking outcomes and, for bookings, a realistic price/revenue value --
to support the design and analysis of experiments that measure the
impact of pricing and product initiatives. The data is synthetic.

Two scenarios are provided:

- `simulate_experiment(..., true_effect=...)` with a nonzero effect --
  a genuine, if modest, conversion-rate lift from the treatment price.
- `simulate_experiment(..., true_effect=0.0)` -- a true null effect, so
  the analysis code in analyze.py can be tested against a case where
  the correct conclusion is "no significant difference," not just
  against a pre-built win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentData:
    control_converted: np.ndarray   # 1 = booked, 0 = did not book
    treatment_converted: np.ndarray
    control_revenue: np.ndarray     # revenue per visitor; 0 if not converted
    treatment_revenue: np.ndarray
    control_price: float
    treatment_price: float


def simulate_experiment(
    n_per_group: int = 5000,
    base_conversion_rate: float = 0.08,
    control_price: float = 89.0,
    treatment_price: float = 79.0,
    true_effect: float = 0.015,
    seed: int = 0,
) -> ExperimentData:
    """
    Simulates a pricing A/B test: control sees `control_price`,
    treatment sees `treatment_price` (typically a discount). Booking
    (conversion) is Bernoulli per visitor; `true_effect` is the true
    absolute conversion-rate lift the treatment price causes (e.g. 0.015
    means treatment's true conversion rate is base_conversion_rate +
    0.015). Set true_effect=0.0 to simulate a genuine null case.

    Revenue per converting visitor is drawn from a lognormal distribution
    around the shown price (not exactly the price, since real guests
    sometimes book multi-night stays or add-ons) -- so the revenue
    analysis is a genuine second metric, not just conversion rate
    restated.
    """
    rng = np.random.default_rng(seed)

    control_rate = base_conversion_rate
    treatment_rate = base_conversion_rate + true_effect

    control_converted = rng.binomial(1, control_rate, size=n_per_group)
    treatment_converted = rng.binomial(1, treatment_rate, size=n_per_group)

    def _revenue_for(converted, price):
        revenue = np.zeros(len(converted), dtype=float)
        n_converted = int(converted.sum())
        if n_converted > 0:
            # Lognormal around price: mean roughly equal to price, with
            # realistic right-skew (a few visitors book longer stays).
            mu = np.log(price) - 0.5 * (0.25 ** 2)
            revenue[converted == 1] = rng.lognormal(mean=mu, sigma=0.25, size=n_converted)
        return revenue

    control_revenue = _revenue_for(control_converted, control_price)
    treatment_revenue = _revenue_for(treatment_converted, treatment_price)

    return ExperimentData(
        control_converted=control_converted,
        treatment_converted=treatment_converted,
        control_revenue=control_revenue,
        treatment_revenue=treatment_revenue,
        control_price=control_price,
        treatment_price=treatment_price,
    )

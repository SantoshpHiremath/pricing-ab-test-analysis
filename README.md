# Pricing A/B Test Analysis

A real, tested statistical experiment-design-and-analysis project — built
to close a specific gap identified against Limehome's "Working Student
Data Scientist" posting, whose one distinctive, otherwise-uncovered ask
is "support the design and analysis of experiments to measure the impact
of pricing and product initiatives." No other project in this portfolio
touched experiment design (A/B testing, hypothesis testing, statistical
power) before this one.

## What this is (read before citing anywhere)

**There is no real Limehome pricing data here.** `src/simulate.py`
generates synthetic pricing-experiment data (a control price vs. a
treatment price, per-visitor booking outcomes, and per-booking revenue)
modeled on a hospitality pricing-test scenario — not real bookings,
prices, or guest data, which I have no access to.

**Two honest scenarios are simulated, not just one flattering one.** The
simulator supports both a genuine, modest treatment effect and a true
null (`true_effect=0.0`) — so the analysis code is tested against a case
where the honest, correct conclusion is "no significant difference,"
not only against a pre-built win. `run_experiment.py` runs both and
prints both reports.

## What this models

- **`src/simulate.py`** — generates per-visitor conversion outcomes
  (Bernoulli) and per-booking revenue (lognormal around the shown
  price) for a control and treatment pricing group.
- **`src/analyze.py`** — the real statistical core: a two-proportion
  z-test on conversion rate, a Welch's t-test on revenue-per-visitor
  (deliberately computed over *all* visitors, not just converters, so
  it correctly reflects the conversion-rate effect rather than
  confounding it away), 95% confidence intervals for both effect sizes,
  a Bonferroni correction to the significance threshold since two
  metrics are tested (disclosed as a basic correction, not passed off as
  more sophisticated than it is), and a minimum-detectable-effect / power
  calculation computed from the sample size alone, before looking at
  results — the correct order, since computing it after seeing the
  p-value would be a well-known way to fool yourself.
- **`src/report.py`** — turns the statistical output into a
  plain-language report, directly addressing the posting's "reports
  that communicate insights clearly to technical and non-technical
  stakeholders." Reports a real null result as a null result (with the
  minimum-detectable-effect stated as a caveat), not reframed as a win.
- **`run_experiment.py`** — runs both scenarios end-to-end.

## A real bug caught during development

The first version of `analyze.py` stored `significant_at_corrected_alpha`
as whatever type `p_value < alpha` produced — which, since `p_value`
comes from scipy/statsmodels, is `numpy.bool_`, not Python's built-in
`bool`. `numpy.bool_(True) is True` evaluates to `False` (an identity
check across two different types), which broke an `is True` assertion
in this project's own test suite and would have silently broken JSON
serialization of analysis results in any downstream use (e.g. an API
returning these results, the same pattern used in other projects in
this portfolio). Fixed by wrapping every stored value in `float(...)` /
`bool(...)` explicitly in `analyze.py`. A dedicated regression test
(`test_significant_flag_is_a_plain_python_bool_not_numpy_bool`) was
confirmed to fail against the original version before the fix, and
pass afterward.

## An honest, non-obvious finding

Running the simulation with a genuine 1.5-percentage-point conversion
lift produces a real, defensible nuance: the conversion-rate lift is
statistically significant, but the revenue-per-visitor difference is
**not** significant at the same sample size. This isn't a bug — a lower
price can genuinely drive more bookings without necessarily driving more
total revenue, and the report explicitly calls this out rather than
picking whichever metric looks better. The `true_effect=0.0` scenario
correctly reports both metrics as non-significant, with the
minimum-detectable-effect stated so a reader can tell "no effect found"
apart from "this experiment wasn't powered to find it."

## Verification performed

- `pytest tests/ -v` — 24/24 tests pass, including:
  - A false-positive-rate check: under a genuine null, re-running the
    analysis across 300 simulated experiments rejects at a rate
    consistent with the corrected alpha (not dramatically higher, which
    would indicate a miscalibrated test).
  - The numpy-bool regression test above, confirmed to fail against the
    original buggy version.
  - Confirmation that `true_effect` genuinely drives the simulated data
    (treatment wins the large majority of runs under a real effect;
    shows no systematic difference under a true null).
  - Report-generation tests covering all four significance combinations
    (both significant, neither, conversion-only, revenue-only), with an
    explicit check that a non-significant result is never misreported
    as "no effect."
- `python3 run_experiment.py` — run end-to-end for both scenarios,
  producing the honest reports shown above.

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v              # 24 tests
python3 run_experiment.py     # both scenarios, full reports
```

## What this doesn't demonstrate

This project doesn't use real Limehome pricing or booking data (all
disclosed above). The multiple-comparisons correction (Bonferroni) is a
simple, conservative choice — a production experimentation platform
might use sequential testing, CUPED variance reduction, or a more
targeted correction, and this project doesn't claim to replicate that
level of sophistication. It demonstrates real, tested statistical
hypothesis testing, power/sample-size reasoning, and honest experiment
reporting (including correctly handling a null result) — the specific,
previously-uncovered skill this project was built to evidence.

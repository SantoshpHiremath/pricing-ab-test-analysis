# Pricing A/B Test Analysis

A tested statistical experiment-design-and-analysis project for measuring the
impact of pricing and product initiatives: simulating a pricing A/B test,
analyzing it with proper hypothesis tests, and reporting the result in plain
language for technical and non-technical readers.

## What it does

- **`src/simulate.py`** — generates per-visitor conversion outcomes
  (Bernoulli) and per-booking revenue (lognormal around the shown
  price) for a control and treatment pricing group.
- **`src/analyze.py`** — the statistical core: a two-proportion
  z-test on conversion rate, a Welch's t-test on revenue-per-visitor
  (computed over *all* visitors, not just converters, so it correctly
  reflects the conversion-rate effect rather than confounding it away),
  95% confidence intervals for both effect sizes, a Bonferroni
  correction to the significance threshold since two metrics are tested,
  and a minimum-detectable-effect / power calculation computed from the
  sample size alone, before looking at results, which is the correct order
  since computing it after seeing the p-value would bias the
  interpretation.
- **`src/report.py`** — turns the statistical output into a
  plain-language report that communicates insights clearly to technical and
  non-technical stakeholders. A null result is reported as a null result,
  with the minimum-detectable-effect stated as context.
- **`run_experiment.py`** — runs both scenarios end-to-end and prints both
  reports.

## Data

The data is synthetic; the analysis is built so real experiment data can
replace it. `src/simulate.py` generates a control price vs. a treatment price,
per-visitor booking outcomes, and per-booking revenue, modeled on a
hospitality pricing-test scenario.

The simulator supports both a modest treatment effect and a true null
(`true_effect=0.0`), so the analysis code is tested against a case where the
correct conclusion is "no significant difference" as well as against a
pre-built win.

## Results

Running the simulation with a 1.5-percentage-point conversion lift shows a
useful nuance: the conversion-rate lift is statistically significant, but the
revenue-per-visitor difference is **not** significant at the same sample
size. A lower price can drive more bookings without necessarily driving more
total revenue, and the report calls this out explicitly rather than picking
whichever metric looks better. The `true_effect=0.0` scenario correctly
reports both metrics as non-significant, with the minimum-detectable-effect
stated so a reader can tell "no effect found" apart from "this experiment
wasn't powered to find it."

## Tests

`pytest tests/ -v` — 24/24 tests pass, including:

- A false-positive-rate check: under a true null, re-running the analysis
  across 300 simulated experiments rejects at a rate consistent with the
  corrected alpha (not dramatically higher, which would indicate a
  miscalibrated test).
- A regression test for a bug caught during development (below).
- Confirmation that `true_effect` drives the simulated data (treatment wins
  the large majority of runs under a real effect and shows no systematic
  difference under a true null).
- Report-generation tests covering all four significance combinations (both
  significant, neither, conversion-only, revenue-only), with an explicit
  check that a non-significant result is never misreported as "no effect."

### A bug caught during development

The first version of `analyze.py` stored `significant_at_corrected_alpha` as
whatever type `p_value < alpha` produced. Since `p_value` comes from
scipy/statsmodels, that is `numpy.bool_`, not Python's built-in `bool`.
`numpy.bool_(True) is True` evaluates to `False` (an identity check across two
different types), which broke an `is True` assertion in the test suite and
would have broken JSON serialization of analysis results in downstream use
(for example an API returning them). I fixed it by wrapping every stored value
in `float(...)` / `bool(...)` explicitly in `analyze.py`. A dedicated
regression test
(`test_significant_flag_is_a_plain_python_bool_not_numpy_bool`) fails against
the original version and passes after the fix.

## Project structure

```
src/        simulate.py, analyze.py, report.py
tests/      test_simulate.py, test_analyze.py, test_report.py
run_experiment.py
requirements.txt, pytest.ini
.github/workflows/ci.yml
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v              # 24 tests
python3 run_experiment.py     # both scenarios, full reports
```

## Notes

The Bonferroni correction is a simple, conservative choice for the two tested
metrics.

## Possible extensions

- Sequential testing for early stopping.
- CUPED variance reduction to improve power.
- A more targeted multiple-comparisons correction as the number of metrics
  grows.

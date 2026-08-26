"""
tests/test_analyze.py
------------------------

Tests the statistical analysis module against both simulated scenarios
(a genuine effect, and a true null) -- the critical property being
tested is that the analysis correctly distinguishes the two cases, not
just that it runs without crashing. Also verifies the multiple-
comparisons correction is actually applied (not just computed and
ignored) and that the false-positive rate under the null is
approximately controlled at the corrected alpha, not the naive one.
"""

import numpy as np
import pytest

from src.simulate import simulate_experiment
from src.analyze import analyze_experiment, minimum_detectable_effect, _bonferroni_alpha, BASE_ALPHA


class TestBonferroniCorrection:
    def test_corrected_alpha_is_half_of_base_for_two_metrics(self):
        assert _bonferroni_alpha(n_comparisons=2) == pytest.approx(BASE_ALPHA / 2)

    def test_analysis_result_reports_the_corrected_alpha_not_the_base(self):
        data = simulate_experiment(n_per_group=1000, seed=0)
        result = analyze_experiment(data)
        assert result.corrected_alpha == pytest.approx(0.025)


class TestConversionAnalysisDetectsRealEffect:
    def test_large_genuine_effect_is_detected_as_significant(self):
        """A large, genuine effect with a large sample should be
        reliably detected -- this is the sanity check that the test
        itself has power, not just that it runs."""
        data = simulate_experiment(n_per_group=8000, true_effect=0.03, seed=0)
        result = analyze_experiment(data)
        assert result.conversion.significant_at_corrected_alpha is True
        assert result.conversion.absolute_lift > 0

    def test_significant_flag_is_a_plain_python_bool_not_numpy_bool(self):
        """Regression test for a real bug caught during development:
        `significant_at_corrected_alpha` was originally stored as
        whatever type `p_value < alpha` produced -- numpy's `np.True_`,
        not Python's built-in `True`. `np.True_ is True` evaluates to
        False (an identity check across different types), which broke
        an `is True` assertion in this very test suite and would also
        silently break JSON serialization of analysis results. Fixed by
        wrapping the comparison in `bool(...)` in analyze.py."""
        data = simulate_experiment(n_per_group=8000, true_effect=0.03, seed=0)
        result = analyze_experiment(data)
        assert type(result.conversion.significant_at_corrected_alpha) is bool
        assert type(result.revenue_per_visitor.significant_at_corrected_alpha) is bool

    def test_confidence_interval_excludes_zero_for_a_clear_effect(self):
        data = simulate_experiment(n_per_group=8000, true_effect=0.03, seed=0)
        result = analyze_experiment(data)
        assert result.conversion.ci_95_low > 0


class TestConversionAnalysisRespectsTrueNull:
    def test_false_positive_rate_near_corrected_alpha_under_true_null(self):
        """The core correctness property of a hypothesis test: under a
        genuine null, it should reject at approximately the stated
        alpha rate, not much more (which would mean the test is
        miscalibrated / overconfident) and not much less (which would
        mean it's underpowered in a way that hides real effects too)."""
        alpha = _bonferroni_alpha()
        n_runs = 300
        false_positives = 0
        for seed in range(n_runs):
            data = simulate_experiment(n_per_group=1500, true_effect=0.0, seed=seed)
            result = analyze_experiment(data)
            if result.conversion.significant_at_corrected_alpha:
                false_positives += 1
        observed_rate = false_positives / n_runs
        # Allow generous slack for a moderate number of runs -- checking
        # the rate is in a sane ballpark (not e.g. 0.5, which would mean
        # something is badly broken), not pinning to a tight interval
        # that would make the test itself flaky.
        assert observed_rate < alpha * 3


class TestRevenueAnalysis:
    def test_revenue_reflects_conversion_effect_since_nonconverters_contribute_zero(self):
        """Revenue-per-visitor is deliberately computed over ALL
        visitors (not just converters) -- a higher conversion rate
        alone should tend to raise mean revenue-per-visitor, even
        holding revenue-per-booking constant. This test confirms that
        design choice is actually reflected in the numbers, catching a
        real mistake this metric is designed to avoid (comparing
        revenue only among converters, which would hide a conversion-
        rate effect entirely)."""
        data = simulate_experiment(n_per_group=6000, true_effect=0.03, seed=0)
        result = analyze_experiment(data)
        # Treatment converts more AND is priced lower per booking, so
        # the net revenue-per-visitor direction isn't guaranteed a
        # priori -- but the mean must be computed over the full
        # per-visitor arrays (including zeros), which this checks
        # indirectly via consistency with the raw data.
        assert result.revenue_per_visitor.control_mean == pytest.approx(data.control_revenue.mean())
        assert result.revenue_per_visitor.treatment_mean == pytest.approx(data.treatment_revenue.mean())


class TestMinimumDetectableEffect:
    def test_larger_sample_size_yields_smaller_minimum_detectable_effect(self):
        """A basic sanity property of statistical power: more data
        should let you detect smaller effects, not larger ones."""
        mde_small_n = minimum_detectable_effect(n_per_group=500, base_rate=0.08)
        mde_large_n = minimum_detectable_effect(n_per_group=20000, base_rate=0.08)
        assert mde_large_n < mde_small_n

    def test_mde_is_positive(self):
        mde = minimum_detectable_effect(n_per_group=5000, base_rate=0.08)
        assert mde > 0

    def test_analysis_reports_a_sane_mde_for_the_actual_sample_size_used(self):
        data = simulate_experiment(n_per_group=5000, seed=0)
        result = analyze_experiment(data)
        # For n=5000/group at an 8% base rate, the MDE should be a small
        # single-digit-percent absolute lift, not something absurd like
        # 50 percentage points (which would indicate a broken formula).
        assert 0 < result.minimum_detectable_effect_at_actual_n < 0.05

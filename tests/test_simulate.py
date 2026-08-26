"""
tests/test_simulate.py
------------------------

Tests the experiment-data simulator: correct shapes, deterministic given
a seed, and that the true_effect parameter actually moves the simulated
conversion rate in the expected direction (not just present but inert).
"""

import numpy as np

from src.simulate import simulate_experiment


class TestSimulateExperimentShape:
    def test_returns_correct_group_sizes(self):
        data = simulate_experiment(n_per_group=1000, seed=0)
        assert len(data.control_converted) == 1000
        assert len(data.treatment_converted) == 1000
        assert len(data.control_revenue) == 1000
        assert len(data.treatment_revenue) == 1000

    def test_conversion_arrays_are_binary(self):
        data = simulate_experiment(n_per_group=1000, seed=0)
        assert set(np.unique(data.control_converted)).issubset({0, 1})
        assert set(np.unique(data.treatment_converted)).issubset({0, 1})

    def test_non_converters_have_zero_revenue(self):
        data = simulate_experiment(n_per_group=1000, seed=0)
        assert (data.control_revenue[data.control_converted == 0] == 0).all()
        assert (data.treatment_revenue[data.treatment_converted == 0] == 0).all()

    def test_converters_have_positive_revenue(self):
        data = simulate_experiment(n_per_group=1000, seed=0)
        converted_revenue = data.control_revenue[data.control_converted == 1]
        assert len(converted_revenue) > 0
        assert (converted_revenue > 0).all()


class TestDeterminism:
    def test_same_seed_produces_identical_data(self):
        data1 = simulate_experiment(n_per_group=500, seed=42)
        data2 = simulate_experiment(n_per_group=500, seed=42)
        assert np.array_equal(data1.control_converted, data2.control_converted)
        assert np.array_equal(data1.treatment_revenue, data2.treatment_revenue)

    def test_different_seeds_produce_different_data(self):
        data1 = simulate_experiment(n_per_group=500, seed=1)
        data2 = simulate_experiment(n_per_group=500, seed=2)
        assert not np.array_equal(data1.control_converted, data2.control_converted)


class TestTrueEffectParameter:
    def test_positive_true_effect_produces_higher_treatment_conversion_on_average(self):
        """With a real effect and a large enough sample, treatment's
        realized conversion rate should exceed control's on average
        across repeated simulations -- confirms true_effect actually
        drives the simulation, not just a documented but unused
        parameter."""
        treatment_wins = 0
        n_runs = 50
        for seed in range(n_runs):
            data = simulate_experiment(n_per_group=3000, true_effect=0.02, seed=seed)
            if data.treatment_converted.mean() > data.control_converted.mean():
                treatment_wins += 1
        # With a real, reasonably large effect, treatment should win
        # the large majority of runs, not merely half.
        assert treatment_wins >= n_runs * 0.8

    def test_zero_true_effect_produces_no_systematic_difference(self):
        """Under a genuine null (true_effect=0.0), treatment should NOT
        systematically beat control -- averaged across many seeds, the
        mean conversion-rate difference should be close to zero."""
        diffs = []
        for seed in range(100):
            data = simulate_experiment(n_per_group=1000, true_effect=0.0, seed=seed)
            diffs.append(data.treatment_converted.mean() - data.control_converted.mean())
        mean_diff = sum(diffs) / len(diffs)
        assert abs(mean_diff) < 0.01  # small relative to the ~0.08 base rate

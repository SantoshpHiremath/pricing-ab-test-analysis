"""
run_experiment.py
------------------

Entry point: runs a full pricing A/B test simulation and analysis end
to end, printing a plain-language report. Run twice with
different true_effect values to see both a genuine-effect case and a
true-null case handled correctly.
"""

from src.simulate import simulate_experiment
from src.analyze import analyze_experiment
from src.report import format_report


def main():
    print("### Scenario 1: genuine (modest) treatment effect ###\n")
    data = simulate_experiment(
        n_per_group=5000, control_price=89.0, treatment_price=79.0,
        true_effect=0.015, seed=0,
    )
    analysis = analyze_experiment(data)
    print(format_report(analysis, control_price=89.0, treatment_price=79.0))

    print("\n\n### Scenario 2: true null (no real effect) ###\n")
    data_null = simulate_experiment(
        n_per_group=5000, control_price=89.0, treatment_price=79.0,
        true_effect=0.0, seed=1,
    )
    analysis_null = analyze_experiment(data_null)
    print(format_report(analysis_null, control_price=89.0, treatment_price=79.0))


if __name__ == "__main__":
    main()

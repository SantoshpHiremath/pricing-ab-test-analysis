"""
tests/test_report.py
----------------------

Tests that the report correctly reflects each of the four possible
significance combinations (both significant, neither, conversion-only,
revenue-only) with the right plain-language conclusion -- the actual
"communicate insights clearly" behavior this module exists for, not
just that it produces some string.
"""

from src.analyze import ConversionTestResult, RevenueTestResult, ExperimentAnalysis
from src.report import format_report


def _make_analysis(conv_significant: bool, rev_significant: bool) -> ExperimentAnalysis:
    conv = ConversionTestResult(
        control_rate=0.08, treatment_rate=0.09, absolute_lift=0.01, relative_lift_pct=12.5,
        z_statistic=2.0, p_value=0.02 if conv_significant else 0.5,
        ci_95_low=0.001, ci_95_high=0.019, significant_at_corrected_alpha=conv_significant,
    )
    rev = RevenueTestResult(
        control_mean=7.0, treatment_mean=7.3, absolute_diff=0.3, relative_diff_pct=4.3,
        t_statistic=1.5, p_value=0.02 if rev_significant else 0.5,
        ci_95_low=-0.1, ci_95_high=0.7, significant_at_corrected_alpha=rev_significant,
    )
    return ExperimentAnalysis(
        conversion=conv, revenue_per_visitor=rev,
        corrected_alpha=0.025, n_per_group=5000, minimum_detectable_effect_at_actual_n=0.015,
    )


class TestReportReflectsSignificanceCombinations:
    def test_both_significant_reports_clear_win(self):
        analysis = _make_analysis(conv_significant=True, rev_significant=True)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "significant improvement in both" in report

    def test_conversion_only_significant_reports_nuanced_finding(self):
        analysis = _make_analysis(conv_significant=True, rev_significant=False)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "more bookings" in report
        assert "genuinely different questions" in report

    def test_neither_significant_reports_null_with_mde_caveat(self):
        analysis = _make_analysis(conv_significant=False, rev_significant=False)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "Neither conversion rate nor revenue" in report
        assert "minimum-detectable-effect" in report or "1.50%" in report

    def test_neither_significant_does_not_falsely_claim_no_effect(self):
        """The report must not overclaim 'no effect' from a
        non-significant result -- it should hedge with the MDE."""
        analysis = _make_analysis(conv_significant=False, rev_significant=False)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "before concluding the treatment price has no effect" in report

    def test_revenue_only_significant_reports_investigate_note(self):
        analysis = _make_analysis(conv_significant=False, rev_significant=True)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "worth investigating" in report

    def test_report_includes_prices_and_sample_size(self):
        analysis = _make_analysis(conv_significant=True, rev_significant=False)
        report = format_report(analysis, control_price=89.0, treatment_price=79.0)
        assert "89.00" in report
        assert "79.00" in report
        assert "5,000" in report

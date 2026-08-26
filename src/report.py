"""
report.py
---------

Turns an ExperimentAnalysis into a plain-language summary report --
directly addressing the posting's "build visualizations and reports
that communicate insights clearly to technical and non-technical
stakeholders." Reports the honest result either way: a real, disclosed
null/non-significant outcome is reported as such, not reframed as a
win, and the minimum-detectable-effect is always stated so a reader can
judge whether "not significant" means "no effect" or "this experiment
wasn't big enough to tell."
"""

from __future__ import annotations

from src.analyze import ExperimentAnalysis


def format_report(analysis: ExperimentAnalysis, control_price: float, treatment_price: float) -> str:
    conv = analysis.conversion
    rev = analysis.revenue_per_visitor

    lines = []
    lines.append("PRICING A/B TEST -- RESULTS SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Control price: EUR {control_price:.2f}   |   Treatment price: EUR {treatment_price:.2f}")
    lines.append(f"Sample size: {analysis.n_per_group:,} visitors per group")
    lines.append(f"Significance threshold: {analysis.corrected_alpha:.3f} "
                 f"(Bonferroni-corrected from 0.05 for 2 metrics tested)")
    lines.append("")

    lines.append("1. CONVERSION RATE")
    lines.append(f"   Control:   {conv.control_rate:.2%}")
    lines.append(f"   Treatment: {conv.treatment_rate:.2%}")
    lines.append(f"   Absolute lift: {conv.absolute_lift:+.2%}  (95% CI: [{conv.ci_95_low:+.2%}, {conv.ci_95_high:+.2%}])")
    lines.append(f"   Relative lift: {conv.relative_lift_pct:+.1f}%")
    lines.append(f"   p-value: {conv.p_value:.4g}")
    if conv.significant_at_corrected_alpha:
        lines.append("   -> STATISTICALLY SIGNIFICANT at the corrected threshold.")
    else:
        lines.append("   -> NOT statistically significant at the corrected threshold.")
        lines.append(f"   -> This experiment could reliably detect an absolute lift of "
                     f"{analysis.minimum_detectable_effect_at_actual_n:.2%} or larger (80% power). "
                     f"A non-significant result here means either there is no effect, or any real "
                     f"effect is smaller than that -- not necessarily 'no effect at all.'")
    lines.append("")

    lines.append("2. REVENUE PER VISITOR")
    lines.append(f"   Control:   EUR {rev.control_mean:.2f}")
    lines.append(f"   Treatment: EUR {rev.treatment_mean:.2f}")
    lines.append(f"   Absolute difference: EUR {rev.absolute_diff:+.2f}  "
                 f"(95% CI: [EUR {rev.ci_95_low:+.2f}, EUR {rev.ci_95_high:+.2f}])")
    lines.append(f"   Relative difference: {rev.relative_diff_pct:+.1f}%")
    lines.append(f"   p-value: {rev.p_value:.4g}")
    if rev.significant_at_corrected_alpha:
        lines.append("   -> STATISTICALLY SIGNIFICANT at the corrected threshold.")
    else:
        lines.append("   -> NOT statistically significant at the corrected threshold.")
    lines.append("")

    lines.append("3. HONEST BOTTOM LINE")
    if conv.significant_at_corrected_alpha and not rev.significant_at_corrected_alpha:
        lines.append("   The treatment price drove significantly more bookings, but the effect on")
        lines.append("   revenue per visitor was not statistically significant at this sample size.")
        lines.append("   These are genuinely different questions -- 'more bookings' and 'more revenue'")
        lines.append("   don't have to move together, especially when the treatment is a lower price.")
    elif conv.significant_at_corrected_alpha and rev.significant_at_corrected_alpha:
        lines.append("   The treatment price drove a statistically significant improvement in both")
        lines.append("   conversion rate and revenue per visitor.")
    elif not conv.significant_at_corrected_alpha and not rev.significant_at_corrected_alpha:
        lines.append("   Neither conversion rate nor revenue per visitor showed a statistically")
        lines.append("   significant difference at this sample size. See the minimum-detectable-effect")
        lines.append("   note above before concluding the treatment price has no effect.")
    else:
        lines.append("   Revenue per visitor moved significantly while conversion rate did not --")
        lines.append("   worth investigating whether this is driven by a shift in booking mix (e.g.")
        lines.append("   longer stays or add-ons) rather than the price itself.")

    return "\n".join(lines)

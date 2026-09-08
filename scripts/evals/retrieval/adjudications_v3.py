"""Explicit source decisions, not rules inferred from retrieval scores.

Keep this specification separate from ranking and metric implementation. Each
addition names a source document already inspected in the frozen full corpus.
Unlisted documents remain unjudged, not implicitly source-adjudicated negatives.
"""

# Two query cases, each with two missed equivalent tables: four evidence-ID
# defects. Original HTML reports the same current-year consolidated totals.
MICROSOFT_TOTALS = {
    2024: {"tables": [8, 72], "revenue": "245122", "operating_income": "109433"},
    2025: {"tables": [8, 65], "revenue": "281724", "operating_income": "128528"},
}

SUBSCRIPTION_ALTERNATIVE = {
    2023: "AMZN_10-K_2023::text::26::split::1",
    2024: "AMZN_10-K_2024::text::26::split::2",
}
SUBSCRIPTION_QUOTE = (
    "Subscriptions are paid for at the time of or in advance of delivering the services. "
    "Revenue from such arrangements is recognized over the subscription period."
)


def decisions():
    result = []
    for year, spec in MICROSOFT_TOTALS.items():
        for index in spec["tables"]:
            document = f"MSFT_10-K_{year}::table::{index}"
            for suffix, grade in [("01", 2), ("11", 1)]:
                result.append({
                    "parent_case_id": f"KBV2_MSFT_{year}_{suffix}", "evidence_id": document,
                    "grade": grade, "groups": ["revenue"],
                    "reason_code": "B1_MISSED_EQUIVALENT_TOTALS" if grade == 2 else "B1_PARTIAL_NUMERIC_CONTEXT",
                    "reason": (
                        "Original filing segment table explicitly reports consolidated revenue and operating-income totals "
                        "for the requested year; capitalized Income does not change the measure."
                        if grade == 2 else
                        "Same source totals/segment changes provide partial numeric context but do not explain the "
                        "business drivers of overall revenue growth. No full explanation credit."
                    ),
                    "source_cell_values": [spec["revenue"], spec["operating_income"]],
                    "unit": "USD_millions", "fiscal_year": year,
                })
    for year, document in SUBSCRIPTION_ALTERNATIVE.items():
        for suffix, grade in [("20", 2), ("05", 1)]:
            result.append({
                "parent_case_id": f"KBV2_AMZN_{year}_{suffix}", "evidence_id": document,
                "grade": grade, "groups": ["unearned"],
                "reason_code": "B1_EQUIVALENT_SUBSCRIPTION_POLICY" if grade == 2 else "B1_PARTIAL_POLICY_FACET",
                "reason": (
                    "Prime is explicitly within the adjacent subscription-services policy. Advance payment followed "
                    "by recognition over the subscription period answers the prepaid-membership accounting facet. "
                    "It does not replace the separate required customer-benefits facet."
                    if grade == 2 else
                    "Supports service-period recognition of prepaid subscriptions, but not the full general "
                    "unearned-revenue rule including payments received OR due. Partial, not equivalent full gold."
                ),
                "quote": SUBSCRIPTION_QUOTE,
                "context_anchor": "Prime memberships provide our customers with access to an evolving suite of benefits that represent a single stand-ready obligation.",
            })
    return result


# Concept-level inspection notes apply to every retained corresponding case,
# including both fiscal years. They explain when text/table alternatives would
# and would not be genuinely equivalent, independent of document type.
REVIEW_NOTES = {
    "direct_fact": "Require the requested measure/year, not a neighboring measure. An explicitly requested table/statement scope remains binding; otherwise a source-explicit equivalent text fact may qualify. No textual equivalent of the audited numeric totals was found in the frozen text corpus.",
    "narrative": "Require the actual policy rule, not a balance or generic discussion. Microsoft 2025 goodwill questions preserve their documented source-year override; they are not silently treated as inventory questions.",
    "risk_factors": "Require the requested risk mechanism or consequence. Section-restricted questions retain that scope; a similar business-section discussion is at most partial when risk-factor location is explicitly requested.",
    "business_growth": "Require the requested business description, channels, principles, offerings or economies of scale. Related risk warnings are not equivalent business descriptions.",
    "mda": "Require causal/explanatory prose for grade 2; matching numeric tables are partial context only. Preserve direction and offsets (for example, AWS pricing is an offset, not the positive growth driver).",
    "paraphrase": "Preserve the underlying source intent despite wording differences; lookups of policy, seasonality, funding purpose, distribution and multi-tenancy are not numeric balance lookups.",
    "section_specific": "Use source-verified section locations, not the chunker's inherited Item 6 or forward-looking label. Require the requested committee/role/frequency/disclosure content.",
    "hard_negative": "A similar-but-wrong policy or measure earns zero only when explicitly adjudicated. Document unknowns stay unjudged. A chunk containing both the right and wrong topic is not negative merely for containing the wrong term.",
    "multi_evidence": "Each named facet is independently necessary. Any grade-2 alternative can satisfy its own facet, but alternatives do not substitute for another required facet. Prime subscription accounting has an additional source-equivalent alternative; numeric tables do not substitute for management explanations.",
}

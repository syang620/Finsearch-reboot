"""Source-adjudicated facets. No system outputs or retrieval rankings as input.

One entry maps each old narrative requirement to atomic v2 requirements. Qualifier
omission is partial completeness; a contradictory qualifier is not a paraphrase.
"""
FACETS = {
    ('AAPL', 'policy'): [('definition', 'Highly liquid investments with maturities of three months or less at purchase qualify as cash equivalents.')],
    ('AAPL', 'growth_drivers'): [
        ('advertising', 'Higher advertising sales primarily contributed to Services net-sales growth.'),
        ('app_store', 'Higher App Store sales primarily contributed to Services net-sales growth.'),
        ('cloud', 'Higher cloud-services sales primarily contributed to Services net-sales growth.')],
    ('AAPL', 'business_or_governance'): [('sourcing', 'Apple uses some custom components available from a single or limited source; this is not a statement that all components are single-sourced.')],
    ('AAPL', 'risk'): [('supply', 'Single or limited component sourcing exposes Apple to significant supply risk.'),
                       ('pricing', 'Single or limited component sourcing exposes Apple to significant pricing risk.')],
    ('AAPL', 'attribution'): [('mix', 'The Services gross-margin percentage increase was attributed to a different services mix, not higher sales as the explanation for that percentage.')],
    ('AMZN', 'policy'): [
        ('received', 'Payments received before service obligations are performed trigger recording of unearned revenue.'),
        ('due', 'Payments due before service obligations are performed also trigger recording of unearned revenue; receipt is not the only trigger.'),
        ('recognition', 'Unearned revenue is recognized as revenue over the service period, not all immediately on receipt.')],
    ('AMZN', 'growth_drivers'): [
        ('usage', 'Increased customer usage was the primary AWS sales-growth driver.'),
        ('pricing_offset', 'Pricing changes partially offset AWS sales growth, rather than being a positive growth driver.'),
        ('contracts', 'The pricing changes were primarily driven by long-term customer contracts.')],
    ('AMZN', 'business_or_governance'): [
        ('networks', 'Order fulfillment uses Amazon-operated North America and International fulfillment networks.'),
        ('partners', 'Order fulfillment also uses co-sourced or outsourced arrangements in certain countries.'),
        ('digital', 'Order fulfillment includes digital delivery.'),
        ('stores', 'Order fulfillment includes physical stores.')],
    ('AMZN', 'risk'): [('carrier_risk', 'Reliance on limited shipping companies means unacceptable terms or carrier difficulties can harm operating results and customer experience.')],
    ('AMZN', 'attribution'): [
        ('usage', 'The primary stated AWS sales-growth driver was increased customer usage, not higher pricing.'),
        ('offset', 'Pricing changes were a partial offset to AWS sales growth.')],
    ('MSFT', 'policy'): [('definition', 'Highly liquid interest-earning investments with maturities of three months or less at purchase qualify as cash equivalents.')],
    ('MSFT', 'growth_drivers'): [
        ('intelligent_cloud', 'Azure drove Intelligent Cloud revenue growth.'),
        ('productivity', 'Office 365 Commercial drove Productivity and Business Processes revenue growth.'),
        ('personal_computing', 'Gaming drove More Personal Computing revenue growth.')],
    ('MSFT', 'business_or_governance'): [
        ('board', 'The Board of Directors oversees cybersecurity risk.'),
        ('cadence', 'Board cybersecurity reviews are scheduled at least quarterly.')],
    ('MSFT', 'risk'): [('evolution', 'Constantly evolving, increasingly sophisticated and complex cyberthreats make detection and successful defense harder.')],
    ('MSFT', 'attribution'): [
        ('gaming', 'Gaming contributed to the R&D expense increase; the increase was not attributed solely to AI.'),
        ('cloud', 'Investments in cloud engineering contributed to the R&D expense increase.'),
        ('acquisition', 'The Gaming driver includes the impact of the Activision Blizzard acquisition.')],
}


def facets(ticker, year, old_id):
    rows = list(FACETS[(ticker, old_id)])
    if ticker == 'AAPL' and year == 2025 and old_id == 'attribution':
        rows += [('cost_offset', 'Higher costs partially offset the favorable services-mix effect on Services gross-margin percentage.')]
    if ticker == 'MSFT' and year == 2025:
        if old_id == 'growth_drivers':
            rows = [('intelligent_cloud', 'Azure drove Intelligent Cloud revenue growth.'),
                    ('productivity', 'Microsoft 365 Commercial cloud drove Productivity and Business Processes revenue growth.'),
                    ('personal_computing', 'Gaming and Search and news advertising drove More Personal Computing revenue growth.')]
        if old_id == 'attribution':
            rows = [(k, 'Investments in cloud and AI engineering contributed to the R&D expense increase.' if k == 'cloud' else text) for k, text in rows]
    return rows


def optional_detail(ticker, year, old_id):
    if ticker == 'AAPL' and old_id == 'business_or_governance':
        return ['New products often use custom components available from only one source. The question asks sourcing and exposure, not a new-product-specific qualifier.']
    if ticker == 'MSFT' and old_id == 'business_or_governance':
        return ['Reviews may occur more often when necessary or advisable; at least quarterly already preserves the minimum cadence.']
    if ticker == 'MSFT' and year == 2024 and old_id == 'attribution':
        return ['The acquisition contributed seven points of R&D expense growth; the question requests acquisition context, not its quantified contribution.']
    if ticker == 'AAPL' and year == 2024 and old_id == 'attribution':
        return ['No offset is stated for the Services percentage in the cited FY2024 discussion. Do not transfer Products offsets to Services.']
    return []


EQUIVALENCE = {
    'phrasing': 'Synonyms, reordered clauses, issuer names/tickers, equivalent explicit units and non-contradictory concise paraphrases are acceptable. Gold wording is not a required string.',
    'scope': 'Use the named issuer and fiscal period. Explicit named-filing questions, comparisons and calculations remain bound to that filing. Generic numeric questions without an explicit filing-year restriction may use independently verified equivalent evidence for the same original amount, entity, metric and fact period; a later filing is not wrong merely because it repeats that fact. Different restated values, segments, periods or metrics are not interchangeable. Narrative/attribution facets retain their named fiscal-year disclosure scope.',
    'evidence': 'Listed source passages are independently verified examples, not an exhaustive whitelist for semantic adjudication. Inspect any unlisted passage against its original filing and the requirement scope before accepting it. Generic numeric evidence may come from another filing only when financial equivalence is established. Log its identity/span and rationale without rewriting frozen gold.',
    'numeric_evidence': 'Financial truth is independent of route: an original-filing table/text with explicit correct metric, issuer, period, unit and amount may support a numeric claim. Structured numeric-to-structured evidence policy is a separate compatibility metric. No credit from context metadata without matching answer semantics.',
    'partial': 'Omitting a qualifying facet may be incomplete while all emitted claims remain supported. Contradicting a material qualifier, issuer, period, amount or attribution is unsupported, not merely incomplete. Mixed true and unsupported separable assertions in one emitted claim are partially supported; a wrong central numeric fact is unsupported.',
    'abstention': 'For future-actual questions, explicitly identify that the named filing cannot establish audited actuals for the later year; no guessed value or later-filing substitution. For answerable questions a refusal is incomplete/incorrect answerability, even if the system failed to retrieve evidence. Execution failures are a different layer.',
}

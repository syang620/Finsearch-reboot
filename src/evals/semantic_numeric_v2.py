"""Conservative numeric assertion checks, independent of production validation.

This is deliberately a bounded assertion grammar, not general entailment. Prose
outside the grammar is unknown and must not acquire numeric correctness credit
from a matching substring or from structured citation metadata alone.
"""
from decimal import Decimal, InvalidOperation
import re

ENTITIES = {"AAPL": r"(?:Apple(?: Inc\.?)?|AAPL)",
            "AMZN": r"(?:Amazon(?:\.com)?(?: Inc\.?)?|AMZN)",
            "MSFT": r"(?:Microsoft(?: Corporation)?|MSFT)"}
METRICS = {
    "revenue": r"(?:(?:total|consolidated)\s+)?(?:revenue|revenues|net sales)",
    "operating_income": r"operating income",
    "net_income": r"net income",
    "total_assets": r"total assets",
    "cash_and_cash_equivalents": r"cash and cash equivalents",
    "research_and_development": r"(?:research and development|R&D) (?:expense|expenses)",
    "services_gross_margin_percent": r"Services gross[- ]margin (?:percentage|percent)",
    "aws_leased_area": r"AWS(?:'s)? leased (?:facility area|square footage)",
    "aws_owned_area": r"AWS(?:'s)? owned (?:facility area|square footage)",
    "revenue_growth_percent": r"revenue (?:growth|percentage change|growth rate)",
}
SCALES = {"": 1, "thousand": 1000, "million": 10**6, "billion": 10**9,
          "trillion": 10**12, "bn": 10**9, "mn": 10**6}
CURRENCIES = {"$": "USD", "US$": "USD", "USD": "USD", "EUR": "EUR", "€": "EUR",
              "GBP": "GBP", "£": "GBP", "JPY": "JPY", "¥": "JPY"}
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
YEAR = r"(?:FY\s*|fiscal(?: year)?\s+)(?P<{name}>20\d{{2}})"


def result(status, reason, **detail):
    return {"status": status, "reason": reason, **detail}


def quantity(text):
    """Parse a complete explicit quantity into a base-unit Decimal, or unknown."""
    text = text.strip().replace("−", "-")
    currency = r"(?:US\$|USD|EUR|GBP|JPY|\$|€|£|¥)"
    pattern = rf"(?P<prefix>{currency})?\s*(?P<open>\()?\s*(?P<number>{NUMBER})\s*(?P<scale>trillion|billion|million|thousand|bn|mn)?\s*(?P<unit>{currency}|%|percent|square feet|sq\.?\s*ft\.?)?\s*(?P<close>\))?"
    match = re.fullmatch(pattern, text, re.I)
    if not match or bool(match['open']) != bool(match['close']):
        return result("unknown", "quantity_not_in_explicit_grammar")
    try:
        value = Decimal(match['number'].replace(',', ''))
    except InvalidOperation:
        return result("unknown", "invalid_decimal")
    if match['open']:
        if match['number'][0] in '+-': return result("unknown", "ambiguous_parenthesized_sign")
        value = -value
    scale = (match['scale'] or '').lower()
    prefix = CURRENCIES.get((match['prefix'] or '').upper())
    unit_text = match['unit'] or ''
    suffix = CURRENCIES.get(unit_text.upper())
    if prefix and suffix and prefix != suffix:
        return result("unknown", "conflicting_currency_markers")
    if prefix or suffix:
        if unit_text and not suffix: return result("unknown", "conflicting_unit_markers")
        unit = prefix or suffix
    elif unit_text.lower() in {'%', 'percent'}:
        if scale: return result("unknown", "scaled_percentage_ambiguous")
        unit = 'percent'
    elif re.fullmatch(r'square feet|sq\.?\s*ft\.?', unit_text, re.I):
        unit = 'square_feet'
    else:
        return result("unknown", "explicit_unit_or_currency_missing")
    return result("parsed", "explicit_quantity", value=value * SCALES[scale], unit=unit)


def assertion(text):
    """Extract one simple affirmed assertion without borrowing gold metadata.

    Entity, metric and fiscal year must be stated, not guessed from the expected
    number or a cited context. This conservative coverage limit is reported.
    """
    text = text.strip().replace('’', "'").replace('−', '-')
    # Markdown emphasis is typography, not a source of semantic field values.
    text = text.replace('**', '').replace('__', '')
    if re.search(r"\b(?:not|never|no|isn't|wasn't|didn't|cannot|can't)\b", text, re.I):
        return result("unknown", "negation_requires_semantic_adjudication")
    if text.endswith('.'):
        text = text[:-1]
    y1, y2 = YEAR.format(name='year_before'), YEAR.format(name='year_after')
    for entity, entity_pattern in ENTITIES.items():
        for metric, metric_pattern in METRICS.items():
            pattern = (rf"{entity_pattern}(?:'s)?\s+(?:reported\s+)?(?:{y1}\s+)?"
                       rf"{metric_pattern}(?:\s+(?:for|in|at the end of)\s+{y2})?"
                       rf"\s*(?:was|were|is|totaled|amounted to|of|:)\s*(?P<quantity>.+)")
            match = re.fullmatch(pattern, text, re.I)
            if not match: continue
            years = {int(match[k]) for k in ('year_before', 'year_after') if match[k]}
            if len(years) != 1:
                return result("unknown", "missing_or_conflicting_explicit_fiscal_year")
            parsed = quantity(match['quantity'])
            if parsed['status'] != 'parsed': return parsed
            return result("parsed", "explicit_entity_metric_period_quantity", entity=entity,
                          metric_id=metric, fiscal_year=years.pop(), value=parsed['value'], unit=parsed['unit'])
    return result("unknown", "assertion_outside_bounded_grammar")


def assess_statement(text, expected):
    """Check parsed financial meaning separately from status/evidence/calculator.

    Callers must still enforce answer eligibility and evidence support. This
    function alone never confers accepted-answer or grounding credit.
    """
    parsed = assertion(text)
    if parsed['status'] != 'parsed': return parsed
    mismatch = [key for key, actual in [('ticker', parsed['entity']), ('metric_id', parsed['metric_id']),
                                      ('fiscal_year', parsed['fiscal_year'])] if actual != expected[key]]
    unit = expected['unit']
    expected_value = Decimal(str(expected['value']))
    tolerance = Decimal(str(expected['absolute_tolerance']))
    if not expected_value.is_finite() or not tolerance.is_finite() or tolerance < 0:
        raise ValueError('Invalid numeric expectation/tolerance')
    if unit == 'thousand_square_feet':
        unit = 'square_feet'; expected_value *= 1000; tolerance *= 1000
    if parsed['unit'] != unit: mismatch.append('unit_or_currency')
    if abs(parsed['value'] - expected_value) > tolerance: mismatch.append('value_or_scale_or_sign')
    detail = {**parsed, 'value': str(parsed['value']), 'mismatches': mismatch}
    return {**detail, 'status': 'incorrect' if mismatch else 'correct',
            'reason': 'explicit_numeric_semantics_mismatch' if mismatch else 'explicit_numeric_semantics_match'}

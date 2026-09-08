from decimal import Decimal

import pytest

from evals.semantic_numeric_v2 import assess_statement, assertion, quantity


GOLD = {'ticker':'AAPL', 'metric_id':'revenue', 'fiscal_year':2024,
        'value':391035000000, 'unit':'USD', 'absolute_tolerance':500000}


@pytest.mark.parametrize('text', [
    "Apple's FY2024 revenue was 391,035 million USD.",
    'Apple FY2024 total revenue was $391.035 billion.',
    'AAPL reported fiscal year 2024 net sales of USD 391035000000.',
    '**Apple** revenue for FY2024 was US$391,035 million.',
    'Apple Inc. FY2024 consolidated revenue amounted to 391035000000 USD.',
])
def test_equivalent_explicit_assertions(text):
    assert assess_statement(text, GOLD)['status'] == 'correct'


@pytest.mark.parametrize('text', [
    'Microsoft FY2024 revenue was 391035 million USD.',
    'Apple FY2024 revenue was 391035 million EUR.',
    'Apple FY2024 revenue was 391035 billion USD.',
    'Apple FY2024 revenue was -391035 million USD.',
    'Apple FY2024 revenue was (391035 million USD).',
    'Apple FY2023 revenue was 391035 million USD.',
    'Apple FY2024 cash and cash equivalents were 391035 million USD.',
])
def test_explicit_wrong_semantics_never_pass(text):
    assert assess_statement(text, GOLD)['status'] == 'incorrect'


@pytest.mark.parametrize('text', [
    'Apple FY2024 revenue was not 391035 million USD; it was 1 million USD.',
    'Apple FY2024 operating income was 391035 million USD; revenue was 1 million USD.',
    "Apple said Tesla's FY2024 revenue was 391035 million USD.",
    'Revenue was 391035 million USD.',
    'Apple revenue was 391035 million USD.',
    'Apple FY2024 revenue was 391035 million.',
    'Apple FY2024 revenue was 391035.',
    'Apple FY2024 revenue was $391035 million EUR.',
    'Apple FY2024 revenue was 391035 million USD and operating income was 1 million USD.',
    'Apple FY2024 revenue for FY2023 was 391035 million USD.',
])
def test_unsupported_grammar_or_negation_is_explicit_unknown(text):
    assert assess_statement(text, GOLD)['status'] == 'unknown'


def test_wrong_metric_does_not_borrow_declared_claim_metadata():
    expected = {**GOLD, 'metric_id':'cash_and_cash_equivalents'}
    result = assess_statement('Apple FY2024 revenue was 391035 million USD.', expected)
    assert result['status'] == 'incorrect' and 'metric_id' in result['mismatches']


@pytest.mark.parametrize('delta,expected', [(500000,'correct'), (500001,'incorrect'), (-500001,'incorrect')])
def test_absolute_display_tolerance_boundary(delta, expected):
    assert assess_statement(f'Apple FY2024 revenue was {GOLD["value"]+delta} USD.', GOLD)['status'] == expected


@pytest.mark.parametrize('text', ['73.9%', '73.9 percent'])
def test_percent_formatting(text):
    gold = {**GOLD, 'metric_id':'services_gross_margin_percent', 'value':73.9, 'unit':'percent', 'absolute_tolerance':0.000001}
    assert assess_statement(f"Apple's FY2024 Services gross-margin percentage was {text}.", gold)['status'] == 'correct'


@pytest.mark.parametrize('text', ['20,434 thousand square feet','20.434 million square feet','20434000 sq. ft.'])
def test_equivalent_area_scale(text):
    gold = {**GOLD, 'ticker':'AMZN','metric_id':'aws_leased_area','fiscal_year':2023,
            'value':20434,'unit':'thousand_square_feet','absolute_tolerance':0.000001}
    assert assess_statement(f'Amazon FY2023 AWS leased facility area was {text}', gold)['status'] == 'correct'


@pytest.mark.parametrize('text', ['39,10 million USD','73.9 million percent','$3 percent','(-3 USD)','1e12 USD','NaN USD'])
def test_malformed_or_conflicting_quantity(text):
    assert quantity(text)['status'] == 'unknown'


def test_negative_sign_normalization_and_no_gold_values_in_parser():
    parsed = assertion('Apple FY2024 revenue was −2 million USD.')
    assert parsed['value'] == Decimal('-2000000')
    assert quantity('$1.2 billion')['value'] == Decimal('1200000000')

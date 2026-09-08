from copy import deepcopy
import pytest

from evals.semantic_judge_v2 import packets, validate_support, validate_completeness, validation_metrics, whole_answer


def setup():
    case = {'user_query': 'What increased?', 'expected_answerability': 'answerable',
            'required_claims': [{'claim_id': 'g1', 'requirement': 'Sales increased.'}, {'claim_id': 'g2', 'requirement': 'Costs increased.'}]}
    out = {'ok': True, 'status': 'completed', 'analyst': {'ok': True, 'status': 'ok', 'answer': 'Sales increased.',
        'claims': [{'claim_id': 'c1', 'text': 'Sales increased.', 'context_ids': ['x']}],
        'trace': {'analyst_visible_context_ids': ['x', 'uncited']}},
        'evaluation_trace': {'analyst_packet': {'context_items': [
            {'context_id': 'x', 'kind': 'text', 'payload': {'content': 'Sales increased. Costs increased.'}},
            {'context_id': 'uncited', 'kind': 'text', 'payload': {'content': 'Gold rescue text.'}}]}}}
    p = packets(case, out)
    s = {'claims': [{'claim_id': 'c1', 'support': 'fully_supported', 'reason': 'Direct.',
                    'evidence_quotes': [{'context_id': 'x', 'quote': 'Sales increased.'}]}], 'unbound_factual_prose': False, 'reason': 'One grounded claim.'}
    c = {'requirements': [{'claim_id': 'g1', 'fulfillment': 'complete', 'reason': 'In answer.', 'answer_quotes': ['Sales increased.']},
                          {'claim_id': 'g2', 'fulfillment': 'missing', 'reason': 'Only in source.', 'answer_quotes': []}],
         'answer_relevant': True, 'answerability_correct': True, 'reason': 'Incomplete but relevant.'}
    return p, s, c


def test_phase_boundaries_and_grounded_not_complete():
    p,s,c = setup()
    assert 'requirements' not in p['support']
    assert len(p['support']['cited_contexts']) == 1
    assert 'cited_contexts' not in p['completeness']
    validate_support(p['support'],s); validate_completeness(p['completeness'],c)
    assert whole_answer(s,c,'ok') == {'fully_grounded': True, 'complete': False}


@pytest.mark.parametrize('mutation', ['extra_key', 'missing_id', 'duplicate_id', 'invented_quote', 'uncited_quote', 'wrong_bool', 'empty_reason'])
def test_strict_support_schema(mutation):
    p,s,_ = setup()
    if mutation == 'extra_key': s['score'] = 1
    if mutation == 'missing_id': s['claims'] = []
    if mutation == 'duplicate_id': s['claims'] *= 2
    if mutation == 'invented_quote': s['claims'][0]['evidence_quotes'][0]['quote'] = 'Invented.'
    if mutation == 'uncited_quote': s['claims'][0]['evidence_quotes'][0]['context_id'] = 'uncited'
    if mutation == 'wrong_bool': s['unbound_factual_prose'] = 'false'
    if mutation == 'empty_reason': s['reason'] = ' '
    with pytest.raises(ValueError): validate_support(p['support'],s)


def test_evidence_cannot_rescue_omitted_answer_fact():
    p,_,c = setup(); c['requirements'][1].update(fulfillment='complete', answer_quotes=['Costs increased.'])
    with pytest.raises(ValueError, match='evidence-only'): validate_completeness(p['completeness'], c)


def fixtures():
    rows = []
    for i in range(36):
        label = ['fully_supported','partially_supported','unsupported'][i%3]
        rows.append({'id': str(i), 'repeat_selected': i<12, 'labels': {
            'claims': {'c': label}, 'requirements': {'g': 'complete'}, 'fully_grounded': label=='fully_supported',
            'complete': True, 'answerability_correct': True, 'answer_relevant': True, 'unbound_factual_prose': False}})
    return rows


def test_perfect_candidate_passes_all_gates():
    f=fixtures(); predictions={r['id']:deepcopy(r['labels']) for r in f}
    assert validation_metrics(f, predictions, predictions)['full_benchmark_judge_enabled']


def test_supported_only_judge_fails_unsupported_sensitivity():
    f=fixtures(); predictions={r['id']:deepcopy(r['labels']) for r in f}
    for p in predictions.values(): p['claims']['c']='fully_supported'
    r=validation_metrics(f,predictions,predictions)
    assert r['metrics']['unsupported_recall']['rate']==0
    assert not r['full_benchmark_judge_enabled']


def test_parse_failures_and_invalid_repeats_are_not_dropped():
    f=fixtures(); predictions={r['id']:deepcopy(r['labels']) for r in f}
    del predictions['2']
    r=validation_metrics(f,predictions,{})
    assert r['metrics']['unsupported_recall']['denominator']==12
    assert r['metrics']['unsupported_recall']['numerator']==11
    assert r['metrics']['claim_agreement']['denominator']==36
    assert r['metrics']['repeat_agreement']['rate']==0
    assert not r['full_benchmark_judge_enabled']

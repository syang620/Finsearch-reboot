"""Strict secondary-judge contracts and prospectively fixed validation gates.

Schema validation establishes well-formedness, never semantic correctness.
The two packets intentionally have different information boundaries.
"""
from collections import Counter

from evals.semantic_answer_v1 import evidence_text, rate, visible_contexts
from evals.semantic_outcomes_v2 import accepted_answer

SUPPORT = {'fully_supported', 'partially_supported', 'unsupported'}
FULFILLMENT = {'complete', 'partial', 'missing'}
GATES = {'parse_success': .95, 'claim_agreement': .85, 'supported_precision': .90,
         'supported_recall': .85, 'unsupported_recall': .90, 'partial_agreement': .80,
         'grounded_answer_agreement': .90, 'completeness_agreement': .85, 'repeat_agreement': .90,
         'requirement_agreement': .85, 'partial_fulfillment_recall': .80,
         'off_topic_recall': .90, 'unbound_prose_recall': .90,
         'cross_filing_generic_supported':1.0,'cross_filing_named_rejected':1.0}


def packets(case, output):
    answer = accepted_answer(output)
    if answer is None: raise ValueError('No eligible answer to judge')
    contexts = visible_contexts(output)
    claims = answer.get('claims') or []
    cited = {cid for c in claims for cid in c.get('context_ids', [])}
    support = {'question': case['user_query'], 'answer': answer,
               'cited_contexts': [{'context_id': cid, 'kind': c.get('kind'), 'source': c.get('source'),
                                  'evidence': evidence_text(c)} for cid, c in sorted(contexts.items()) if cid in cited]}
    completeness = {'question': case['user_query'], 'answer': answer,
                    'expected_answerability': case['expected_answerability'],
                    'answerability_reason': case.get('answerability_reason'),
                    'equivalence_rules': case.get('equivalence_rules'), 'optional_detail': case.get('optional_detail', []),
                    'requirements': [{k: g[k] for k in ('claim_id', 'requirement', 'numeric') if k in g} for g in case['required_claims']]}
    return {'support': support, 'completeness': completeness}


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys): raise ValueError('Unexpected/missing schema keys')


def nonempty(value):
    if not isinstance(value, str) or not value.strip(): raise ValueError('Nonempty string required')


def ids(rows, key, expected):
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows): raise ValueError('Expected object list')
    values = [r.get(key) for r in rows]
    if any(not isinstance(v, str) for v in values) or len(values) != len(set(values)) or set(values) != set(expected):
        raise ValueError('Exact unique ID coverage required')


def validate_support(packet, judgment):
    exact_keys(judgment, {'claims', 'unbound_factual_prose', 'reason'})
    nonempty(judgment['reason'])
    if type(judgment['unbound_factual_prose']) is not bool: raise ValueError('Boolean required')
    claims = {c['claim_id']: c for c in packet['answer'].get('claims') or []}
    contexts = {c['context_id']: c for c in packet['cited_contexts']}
    ids(judgment['claims'], 'claim_id', claims)
    for row in judgment['claims']:
        exact_keys(row, {'claim_id', 'support', 'reason', 'evidence_quotes'})
        if row['support'] not in SUPPORT: raise ValueError('Unknown support label')
        nonempty(row['reason'])
        quotes = row['evidence_quotes']
        if not isinstance(quotes, list) or (row['support'] != 'unsupported' and not quotes): raise ValueError('Support needs evidence quotes')
        for quote in quotes:
            exact_keys(quote, {'context_id', 'quote'}); nonempty(quote['quote'])
            cid = quote['context_id']
            if not isinstance(cid, str) or cid not in claims[row['claim_id']].get('context_ids', []) or cid not in contexts or quote['quote'] not in contexts[cid]['evidence']:
                raise ValueError('Invented/non-cited evidence quotation')
    return judgment


def validate_completeness(packet, judgment):
    exact_keys(judgment, {'requirements', 'answer_relevant', 'answerability_correct', 'reason'})
    nonempty(judgment['reason'])
    for key in ('answer_relevant', 'answerability_correct'):
        if type(judgment[key]) is not bool: raise ValueError('Boolean required')
    ids(judgment['requirements'], 'claim_id', [g['claim_id'] for g in packet['requirements']])
    # Answer prose only: facts existing solely in claim metadata, comparison
    # artifacts or cited evidence cannot rescue an omitted final-answer fact.
    text = packet['answer']['answer']
    for row in judgment['requirements']:
        exact_keys(row, {'claim_id', 'fulfillment', 'reason', 'answer_quotes'})
        if row['fulfillment'] not in FULFILLMENT: raise ValueError('Unknown fulfillment label')
        nonempty(row['reason'])
        quotes = row['answer_quotes']
        if not isinstance(quotes, list) or (row['fulfillment'] != 'missing' and not quotes): raise ValueError('Fulfillment needs answer quotes')
        for quote in quotes:
            nonempty(quote)
            if quote not in text: raise ValueError('Invented answer quotation or evidence-only rescue')
    return judgment


def whole_answer(support, completeness, answer_status):
    """Groundedness and completeness deliberately do not imply one another."""
    claims = support['claims']
    nonvacuous = bool(claims) or (answer_status == 'insufficient_data' and completeness['answerability_correct'])
    grounded = (nonvacuous and all(c['support'] == 'fully_supported' for c in claims)
                and not support['unbound_factual_prose'] and completeness['answerability_correct'])
    complete = (completeness['answer_relevant'] and completeness['answerability_correct']
                and all(c['fulfillment'] == 'complete' for c in completeness['requirements']))
    return {'fully_grounded': bool(grounded), 'complete': bool(complete)}


def validation_metrics(fixtures, assessments, repeats):
    """Inputs are parser-validated assessments or explicit None/errors.

    Missing and invalid predictions never disappear from gold denominators.
    Repeat success requires every label/whole-answer decision to agree per case;
    two invalid responses are not agreement.
    """
    if len(fixtures) != 36 or len({f['id'] for f in fixtures}) != 36: raise ValueError('Expected frozen 36 fixtures')
    gold_counts = Counter(); pred_counts = Counter(); true_positive = Counter()
    gold_fulfillment=Counter(); correct_fulfillment=Counter()
    off_topic=unbound=off_topic_correct=unbound_correct=0
    scope_total=Counter(); scope_correct=Counter()
    valid = agreement = grounded = complete = repeat_agreement = 0
    repeat_ids = [f['id'] for f in fixtures if f['repeat_selected']]
    if len(repeat_ids) != 12: raise ValueError('Expected 12 frozen repeat fixtures')
    for f in fixtures:
        expected = f['labels']; predicted = assessments.get(f['id'])
        scope_class=f.get('validation_class')
        if scope_class in {'cross_filing_generic_supported','cross_filing_named_rejected'}:
            scope_total[scope_class]+=1
            scope_correct[scope_class]+=(predicted is not None and predicted['claims']==expected['claims']
                                         and predicted['fully_grounded']==expected['fully_grounded'])
        gold_counts.update(expected['claims'].values())
        gold_fulfillment.update(expected['requirements'].values())
        off_topic+=not expected['answer_relevant']; unbound+=expected['unbound_factual_prose']
        if predicted is not None:
            if set(predicted['claims']) != set(expected['claims']): raise ValueError('Unvalidated prediction IDs')
            if set(predicted['requirements']) != set(expected['requirements']): raise ValueError('Unvalidated requirement IDs')
            valid += 1; pred_counts.update(predicted['claims'].values())
            for cid, label in expected['claims'].items():
                if predicted['claims'][cid] == label:
                    agreement += 1; true_positive[label] += 1
            grounded += predicted['fully_grounded'] == expected['fully_grounded']
            complete += predicted['complete'] == expected['complete']
            for gid,label in expected['requirements'].items():
                if predicted['requirements'][gid]==label: correct_fulfillment[label]+=1
            off_topic_correct+=(not expected['answer_relevant'] and predicted['answer_relevant'] is False)
            unbound_correct+=(expected['unbound_factual_prose'] and predicted['unbound_factual_prose'] is True)
        repeated = repeats.get(f['id'])
        if f['id'] in repeat_ids and predicted is not None and repeated is not None:
            repeat_agreement += all(predicted[k] == repeated[k] for k in ('claims', 'requirements', 'fully_grounded', 'complete', 'answerability_correct', 'unbound_factual_prose', 'answer_relevant'))
    metrics = {'parse_success': rate(valid, len(fixtures)), 'claim_agreement': rate(agreement, sum(gold_counts.values())),
               'supported_precision': rate(true_positive['fully_supported'], pred_counts['fully_supported']),
               'supported_recall': rate(true_positive['fully_supported'], gold_counts['fully_supported']),
               'unsupported_recall': rate(true_positive['unsupported'], gold_counts['unsupported']),
               'partial_agreement': rate(true_positive['partially_supported'], gold_counts['partially_supported']),
               'grounded_answer_agreement': rate(grounded, len(fixtures)), 'completeness_agreement': rate(complete, len(fixtures)),
               'repeat_agreement': rate(repeat_agreement, len(repeat_ids)),
               'requirement_agreement':rate(sum(correct_fulfillment.values()),sum(gold_fulfillment.values())),
               'partial_fulfillment_recall':rate(correct_fulfillment['partial'],gold_fulfillment['partial']),
               'off_topic_recall':rate(off_topic_correct,off_topic),'unbound_prose_recall':rate(unbound_correct,unbound)}
    metrics.update({k:rate(scope_correct[k],scope_total[k]) for k in ('cross_filing_generic_supported','cross_filing_named_rejected')})
    passed = {k: metrics[k]['rate'] is not None and metrics[k]['rate'] >= threshold for k, threshold in GATES.items()}
    return {'metrics': metrics, 'thresholds': GATES, 'gates_passed': passed, 'full_benchmark_judge_enabled': all(passed.values()),
            'gold_label_distribution': dict(gold_counts), 'predicted_label_distribution': dict(pred_counts),
            'gold_requirement_label_distribution':dict(gold_fulfillment),
            'invalid_or_missing_cases': len(fixtures)-valid, 'repeat_valid_pairs': sum(assessments.get(i) is not None and repeats.get(i) is not None for i in repeat_ids)}

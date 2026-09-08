"""Explicit population/assessment denominators; never extrapolate a subset."""
from collections import Counter
import math

from evals.semantic_answer_v1 import rate
from evals.semantic_judge_v2 import packets, validate_support, validate_completeness, whole_answer
from evals.semantic_outcomes_v2 import summarize_numeric, summarize_outcomes


def deterministic_summary(rows):
    summary={'execution':summarize_outcomes(rows),'numeric':summarize_numeric(rows)}
    for key in ('claim_citation_coverage','valid_context_id_rate','structured_evidence_compatibility','kb_evidence_compatibility'):
        summary[key]=rate(sum(r[key]['numerator'] for r in rows),sum(r[key]['denominator'] for r in rows))
    checks=[c for r in rows for c in r['numeric_checks']]
    eligible_checks=[c for r in rows if r['execution']['eligible'] for c in r['numeric_checks']]
    summary['numeric']['truth_evidence_calculator_verified_over_all_gold']=rate(sum(c['credit'] for c in checks),len(checks))
    summary['numeric']['truth_evidence_calculator_verified_given_eligible']=rate(sum(c['credit'] for c in eligible_checks),len(eligible_checks))
    numeric_wrong=[]
    for row in rows:
        if row['execution']['eligible']:
            valid_citations=(row['emitted_eligible_claims']>0 and row['claim_citation_coverage']['rate']==1
                             and row['valid_context_id_rate']['rate']==1)
            numeric_wrong.append(valid_citations and any(c['truth']=='incorrect' for c in row['numeric_checks']))
    summary['detected_numeric_wrong_answer_with_valid_citations_over_eligible_answers']=rate(sum(numeric_wrong),len(numeric_wrong))
    summary['detected_numeric_wrong_answer_with_valid_citations_over_all_cases']=rate(sum(numeric_wrong),len(rows))
    summary['wrong_answer_detection_limitation']='Bounded deterministic numeric mismatches only, not an exhaustive semantic wrong-answer rate. Unsupported does not necessarily mean factually false.'
    times=sorted(r['latency_ms'] for r in rows if type(r.get('latency_ms')) in (int,float) and math.isfinite(r['latency_ms']) and r['latency_ms']>=0)
    summary['orchestrator_latency_ms']={'n':len(times),'p50':times[math.ceil(len(times)*.5)-1] if times else None,
                                      'p95':times[math.ceil(len(times)*.95)-1] if times else None}
    return summary


def semantic_summary(cases, outputs, deterministic, assessments, *, scope_ids, channel):
    """Validated support/completeness assessments for a declared fixed scope.

    `scope_ids` must be the preregistered audit subset or all cases for an enabled
    judge. Missing assessments remain unknown; ineligible cases remain execution
    losses. All semantic rates below describe ONLY this declared scope.
    """
    if channel not in {'source_adjudicated_subset','validated_secondary_judge'}: raise ValueError('Unknown assessment channel')
    case_map={c['id']:c for c in cases}; row_map={r['case_id']:r for r in deterministic}
    if len(case_map)!=len(cases) or len(row_map)!=len(deterministic) or set(row_map)!=set(case_map) or set(outputs)!=set(case_map):
        raise ValueError('Complete unique benchmark execution membership required')
    scope=set(scope_ids)
    if len(scope)!=len(scope_ids) or not scope <= set(case_map): raise ValueError('Invalid assessment population')
    expected=({c['id'] for c in cases if c['baseline_audit_selected']} if channel=='source_adjudicated_subset' else set(case_map))
    if scope!=expected: raise ValueError('Assessment subset changed after preregistration')
    if not set(assessments)<=scope: raise ValueError('Out-of-scope assessments')
    eligible={cid for cid in scope if row_map[cid]['execution']['eligible']}
    emitted=sum(row_map[cid]['emitted_eligible_claims'] for cid in eligible)
    gold_all=sum(len(case_map[cid]['required_claims']) for cid in scope)
    gold_eligible=sum(len(case_map[cid]['required_claims']) for cid in eligible)
    counts=Counter(); fulfillment=Counter(); assessed=grounded=complete=abstentions=unsupported_valid=0
    for cid,judgment in assessments.items():
        if judgment is None: continue
        if cid not in eligible: raise ValueError('Ineligible error output cannot acquire semantic credit')
        p=packets(case_map[cid],outputs[cid])
        support=validate_support(p['support'],judgment['support'])
        completeness=validate_completeness(p['completeness'],judgment['completeness'])
        whole=whole_answer(support,completeness,outputs[cid]['analyst']['status'])
        assessed+=1; grounded+=whole['fully_grounded']; complete+=whole['complete']
        counts.update(r['support'] for r in support['claims'])
        fulfillment.update(r['fulfillment'] for r in completeness['requirements'])
        abstentions+=(case_map[cid]['expected_answerability']=='insufficient_data' and outputs[cid]['analyst']['status']=='insufficient_data' and completeness['answerability_correct'] and not support['unbound_factual_prose'] and not support['claims'])
        row=row_map[cid]
        unsupported_valid+=(any(c['support']=='unsupported' for c in support['claims']) and row['claim_citation_coverage']['rate']==1 and row['valid_context_id_rate']['rate']==1)
    insufficient_scope={cid for cid in scope if case_map[cid]['expected_answerability']=='insufficient_data'}
    judged_insufficient=sum(cid in insufficient_scope and assessments.get(cid) is not None for cid in eligible)
    return {'assessment_channel':channel,'population_case_ids':sorted(scope),'population_size':len(scope),
            'benchmark_size':len(cases),'population_note':'All semantic rates are restricted to the declared assessment population; subset results are not full-benchmark accuracy.',
            'execution_in_population':summarize_outcomes([row_map[cid] for cid in sorted(scope)]),
            'assessed_eligible_answers':rate(assessed,len(eligible)),
            'unassessed_eligible_answers':len(eligible)-assessed,'unassessed_execution_cases':len(scope)-len(eligible),
            'emitted_claim_assessment_coverage':rate(sum(counts.values()),emitted),
            'support_labels':dict(counts),
            'claim_support_rates_assessed_only':{label:rate(counts[label],sum(counts.values())) for label in ('fully_supported','partially_supported','unsupported')},
            'fully_grounded_answer_rate_assessed_only':rate(grounded,assessed),
            'observed_grounded_answers_over_eligible_population':rate(grounded,len(eligible)),
            'observed_grounded_answers_over_all_population_cases':rate(grounded,len(scope)),
            'complete_answer_rate_assessed_only':rate(complete,assessed),
            'observed_complete_answers_over_all_population_cases':rate(complete,len(scope)),
            'required_facet_labels':dict(fulfillment),
            'required_facet_completeness_assessed_only':rate(fulfillment['complete'],sum(fulfillment.values())),
            'observed_complete_facets_over_eligible_gold':rate(fulfillment['complete'],gold_eligible),
            'observed_complete_facets_over_all_population_gold':rate(fulfillment['complete'],gold_all),
            'unassessed_gold_due_to_execution':gold_all-gold_eligible,
            'unassessed_gold_with_eligible_answer':gold_eligible-sum(fulfillment.values()),
            'correct_abstention_rate_assessed_only':rate(abstentions,judged_insufficient),
            'observed_correct_abstentions_over_expected_insufficient_population':rate(abstentions,len(insufficient_scope)),
            'unsupported_answer_with_valid_citations_assessed_only':rate(unsupported_valid,assessed),
            'limitations':['Observed-positive rates are descriptive, not mathematical lower bounds when assessments are fallible.',
                           'Unsupported is evidence failure, not proof of factual falsity. Do not relabel it wrong-answer accuracy.']}

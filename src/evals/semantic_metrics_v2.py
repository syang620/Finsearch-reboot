"""Explicit population/assessment denominators; never extrapolate a subset."""
from collections import Counter
import json
import math
from pathlib import Path

from evals.semantic_answer_v1 import rate
from evals.semantic_dataset_v2 import sha, verify_files
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


def enabled_judge_policy(data_root):
    """Read the frozen decision, never trust a caller's channel label alone."""
    root=Path(data_root)
    manifest_path=root/'optimization_manifest.json'
    if not manifest_path.is_file(): raise ValueError('Frozen enabled judge decision required')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='OPTIMIZATION_FROZEN' or manifest.get('judge_enabled') is not True:
        raise ValueError('Full-benchmark secondary judge is disabled')
    files=manifest['files_sha256']
    if not {'judge_decision.json','judge_config.json','validation_manifest.json'}<=set(files):
        raise ValueError('Judge policy missing from optimization freeze')
    verify_files(root,files)
    decision_path=root/'judge_decision.json'; decision=json.loads(decision_path.read_text())
    if decision.get('full_benchmark_judge_enabled') is not True or decision.get('metrics',{}).get('full_benchmark_judge_enabled') is not True:
        raise ValueError('Full-benchmark secondary judge is disabled')
    return {'optimization_manifest_sha256':sha(manifest_path),'judge_decision_sha256':sha(decision_path)}


def deterministic_breakdowns(cases,rows):
    """Expose correlated question and requirement families, without pooling them."""
    case_map={c['id']:c for c in cases}; row_map={r['case_id']:r for r in rows}
    if len(case_map)!=len(cases) or len(row_map)!=len(rows) or set(case_map)!=set(row_map):
        raise ValueError('Complete unique case membership required for breakdowns')
    result={'overall':deterministic_summary(rows)}
    for key,name in [('stratum','by_stratum'),('ticker','by_issuer'),('question_family','by_question_family')]:
        result[name]={value:deterministic_summary([row_map[c['id']] for c in cases if c[key]==value])
                      for value in sorted({c[key] for c in cases})}
    families=sorted({g['requirement_family'] for c in cases for g in c['required_claims']})
    result['by_requirement_family']={}
    for family in families:
        members={c['id']:[g['claim_id'] for g in c['required_claims'] if g['requirement_family']==family]
                 for c in cases if any(g['requirement_family']==family for g in c['required_claims'])}
        selected=[{**row_map[cid],'numeric_checks':[n for n in row_map[cid]['numeric_checks'] if n['claim_id'] in ids]}
                  for cid,ids in members.items()]
        result['by_requirement_family'][family]={'gold_requirements':sum(map(len,members.values())),
            'case_requirement_ids':members,'execution_of_containing_cases':summarize_outcomes(selected),
            'numeric_requirements':summarize_numeric(selected),
            'note':'Execution is case-level; numeric checks are restricted to this requirement family. Non-numeric fulfillment requires the separate semantic assessment channel.'}
    result['family_note']='Families overlap and share evidence; do not sum their case counts or treat paired-year cases as independent trials.'
    return result


def semantic_summary(cases, outputs, deterministic, assessments, *, scope_ids, channel,
                     data_root='data/evals/semantic_answer/v2'):
    """Validated support/completeness assessments for a declared fixed scope.

    `scope_ids` must be the preregistered audit subset or all cases for an enabled
    judge. Missing assessments remain unknown; ineligible cases remain execution
    losses. All semantic rates below describe ONLY this declared scope.
    """
    if channel not in {'source_adjudicated_subset','validated_secondary_judge'}: raise ValueError('Unknown assessment channel')
    policy=enabled_judge_policy(data_root) if channel=='validated_secondary_judge' else None
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
    return {'assessment_channel':channel,'frozen_judge_policy':policy,'population_case_ids':sorted(scope),'population_size':len(scope),
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

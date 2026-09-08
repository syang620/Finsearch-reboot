from copy import deepcopy
import json
import pytest

from evals.semantic_answer_v2 import deterministic_case
from evals.semantic_metrics_v2 import deterministic_summary, semantic_summary
from evals.semantic_dataset_v2 import sha


def sample():
    cases=[{'id':str(i),'stratum':'narrative','ticker':'AAPL','baseline_audit_selected':i<3,
            'user_query':'What happened?','expected_answerability':'answerable',
            'required_claims':[{'claim_id':'g','requirement':'Sales increased.'}]} for i in range(4)]
    out={'ok':True,'status':'completed','analyst':{'ok':True,'status':'ok','answer':'Sales increased.',
         'claims':[{'claim_id':'c','text':'Sales increased.','claim_type':'narrative','context_ids':['x']}],
         'trace':{'analyst_visible_context_ids':['x']}},
         'evaluation_trace':{'analyst_packet':{'context_items':[{'context_id':'x','kind':'text','payload':{'content':'Sales increased.'}}]}}}
    outputs={c['id']:deepcopy(out) for c in cases}
    outputs['2']={'status':'failed','ok':False,'analyst':{'status':'error','ok':False,'error':'ANALYST_MODEL_TIMEOUT'}}
    rows=[deterministic_case(c,outputs[c['id']]) for c in cases]
    verdict={'support':{'claims':[{'claim_id':'c','support':'fully_supported','reason':'Direct source support.',
        'evidence_quotes':[{'context_id':'x','quote':'Sales increased.'}]}],'unbound_factual_prose':False,'reason':'Bound answer.'},
        'completeness':{'requirements':[{'claim_id':'g','fulfillment':'complete','reason':'Explicit answer.',
        'answer_quotes':['Sales increased.']}],'answer_relevant':True,'answerability_correct':True,'reason':'Relevant and answerable.'}}
    return cases,outputs,rows,{'0':verdict}


def test_manual_subset_never_extrapolates_or_hides_failures():
    args=sample(); s=semantic_summary(*args,scope_ids=['0','1','2'],channel='source_adjudicated_subset')
    assert s['benchmark_size']==4 and s['population_size']==3
    assert s['fully_grounded_answer_rate_assessed_only']['rate']==1
    assert s['observed_grounded_answers_over_eligible_population']['rate']==.5
    assert s['observed_grounded_answers_over_all_population_cases']['rate']==1/3
    assert s['unassessed_execution_cases']==1 and s['unassessed_eligible_answers']==1
    assert s['required_facet_completeness_assessed_only']['denominator']==1
    assert s['observed_complete_facets_over_all_population_gold']['denominator']==3
    assert s['unassessed_gold_due_to_execution']==1 and s['unassessed_gold_with_eligible_answer']==1


def test_cannot_cherry_pick_audit_scope():
    with pytest.raises(ValueError,match='preregistration'):
        semantic_summary(*sample(),scope_ids=['0'],channel='source_adjudicated_subset')


def test_ineligible_output_never_receives_support_credit():
    c,o,r,a=sample(); a['2']=a['0']
    with pytest.raises(ValueError,match='Ineligible'):
        semantic_summary(c,o,r,a,scope_ids=['0','1','2'],channel='source_adjudicated_subset')


def test_deterministic_empty_and_no_numeric_denominators():
    summary=deterministic_summary(sample()[2])
    assert summary['numeric']['numeric_correctness_resolved_only']['rate'] is None
    assert summary['execution']['eligible_produced_answers']['denominator']==4
    assert summary['claim_citation_coverage']['denominator']==3
    assert deterministic_summary([])['orchestrator_latency_ms']['p50'] is None


def frozen_policy(root,enabled):
    decision={'full_benchmark_judge_enabled':enabled,'metrics':{'full_benchmark_judge_enabled':enabled}}
    for name,record in [('judge_decision.json',decision),('judge_config.json',{}),('validation_manifest.json',{})]:
        (root/name).write_text(json.dumps(record))
    manifest={'status':'OPTIMIZATION_FROZEN','judge_enabled':enabled,
              'files_sha256':{p.name:sha(p) for p in root.iterdir()}}
    (root/'optimization_manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('state',['missing','disabled','changed'])
def test_channel_string_cannot_enable_unfrozen_or_disabled_judge(tmp_path,state):
    if state!='missing': frozen_policy(tmp_path,state=='changed')
    if state=='changed': (tmp_path/'judge_decision.json').write_text('{}')
    with pytest.raises(ValueError):
        semantic_summary(*sample(),scope_ids=['0','1','2','3'],channel='validated_secondary_judge',data_root=tmp_path)


def test_enabled_judge_reports_exact_frozen_policy_hashes(tmp_path):
    frozen_policy(tmp_path,True)
    result=semantic_summary(*sample(),scope_ids=['0','1','2','3'],channel='validated_secondary_judge',data_root=tmp_path)
    assert result['frozen_judge_policy']['judge_decision_sha256']==sha(tmp_path/'judge_decision.json')
    assert result['population_size']==4

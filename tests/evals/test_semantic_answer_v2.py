from copy import deepcopy
import hashlib
import pytest

from evals.semantic_answer_v2 import deterministic_case as evaluate, evidence_support, calculator_provenance


NUMBER={'ticker':'AAPL','metric_id':'revenue','fiscal_year':2024,'value':391035000000,'unit':'USD','absolute_tolerance':500000}
SOURCE={'kind':'inline_xbrl','fact_id':'f1','ticker':'AAPL','metric_id':'revenue','fact_fiscal_year':2024,
        'value':391035000000,'unit':'USD','form_type':'10-K','start_date':'2023-10-01','report_date':'2024-09-28','source_sha256':'filing-hash'}
GOLD={'claim_id':'revenue','claim_type':'structured_numeric','numeric':NUMBER,'sources':[SOURCE]}
CASE={'id':'test','stratum':'structured_numeric','ticker':'AAPL','required_claims':[GOLD]}
TEXT='Apple FY2024 revenue was $391.035 billion.'
FILING={'ticker':'AAPL','form_type':'10-K','accession_number':'0000320193-24-000123',
        'report_date':'2024-09-28','filed_date':'2024-11-01',
        'source_url':'https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm'}
CATALOG={'filings':{'filing-hash':FILING}}


def deterministic_case(case,out,catalog=CATALOG): return evaluate(case,out,catalog)


def structured(source):
    return {**{k:source[k] for k in ('ticker','metric_id','value','unit','start_date')},
            **FILING,'fiscal_year':source['fact_fiscal_year'],'metric_label':source['metric_id'],'status':'ok'}


def output(text=TEXT):
    return {'ok':True,'status':'completed','failure_stage':'none','analyst':{
        'ok':True,'status':'ok','answer':text,'claims':[{'claim_id':'c1','claim_type':'structured_numeric','metric_id':'revenue','text':text,'context_ids':['x']}],
        'trace':{'analyst_visible_context_ids':['x']}},
        'evaluation_trace':{'analyst_packet':{'context_items':[{'context_id':'x','kind':'structured_fact',
            'structured_fact':structured(SOURCE)}]}}}


def test_correct_number_needs_separate_truth_evidence_and_status():
    out=output(); row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='correct' and row['evidence_support']=='supported' and row['credit']
    out['evaluation_trace']['analyst_packet']['context_items'][0]['structured_fact']['accession_number']='other-filing'
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='correct' and row['evidence_support']=='unknown' and not row['credit']


@pytest.mark.parametrize('text',[
    'Microsoft FY2024 revenue was $391.035 billion.', 'Apple FY2024 revenue was 391.035 billion EUR.',
    'Apple FY2024 revenue was $391035 billion.', 'Apple FY2024 revenue was -391035 million USD.',
    'Apple FY2024 revenue was not $391.035 billion.', 'Apple FY2023 revenue was $391.035 billion.',
    'Apple FY2024 cash and cash equivalents were $391.035 billion.',
    'Apple FY2024 operating income was $391.035 billion; revenue was $1 million.',
])
def test_audit_adversaries_do_not_gain_credit(text):
    row=deterministic_case(CASE,output(text))['numeric_checks'][0]
    assert row['truth']!='correct' and not row['credit']


@pytest.mark.parametrize('status',['grounding_error','tool_error','error','rejected'])
def test_retained_failed_claims_do_not_score(status):
    out=output(); out['analyst']['status']=status
    row=deterministic_case(CASE,out)
    assert row['emitted_eligible_claims']==0
    assert row['numeric_checks'][0]['truth']=='unassessed'
    assert not row['numeric_checks'][0]['credit']
    assert row['claim_citation_coverage']['denominator']==0


@pytest.mark.parametrize('answer',['No answer.','It is false that '+TEXT,TEXT.replace('was','was not')])
def test_answer_channel_must_contain_actual_affirmed_claim(answer):
    out=output(); out['analyst']['answer']=answer
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='unknown' and not row['credit']


def test_conflicting_assertions_do_not_choose_favorable_one():
    out=output(); wrong=deepcopy(out['analyst']['claims'][0]); wrong.update(claim_id='c2',text='Apple FY2024 revenue was $1 million.')
    out['analyst']['claims'].append(wrong); out['analyst']['answer']+='\n'+wrong['text']
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='incorrect' and not row['credit']


def test_same_number_kb_table_truth_not_confused_with_route_policy():
    out=output(); text='| Revenue | FY2024 | USD millions |\n| Total | 391035 | |'
    context={'context_id':'x','kind':'table','source':{'doc_id':'d1'},'payload':{'table_markdown':text}}
    catalog=[{'fact_id':'f1','evidence_id':'d1','content_sha256':hashlib.sha256(text.encode()).hexdigest()}]
    out['evaluation_trace']['analyst_packet']['context_items']=[context]
    row=deterministic_case(CASE,out,catalog)
    assert row['numeric_checks'][0]['credit']
    assert row['structured_evidence_compatibility']['numerator']==0
    context['payload']['table_markdown']=text.replace('391035','1')
    assert not deterministic_case(CASE,out,catalog)['numeric_checks'][0]['credit']


def test_missing_citation_is_not_grounded_numeric_credit():
    out=output(); out['analyst']['claims'][0]['context_ids']=[]
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='correct' and not row['credit']


def test_answer_display_tolerance_does_not_relax_source_fact_identity():
    out=output()
    out['evaluation_trace']['analyst_packet']['context_items'][0]['structured_fact']['value']+=1
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='correct' and not row['credit']


def test_one_valid_reference_does_not_rescue_an_invented_reference():
    out=output(); out['analyst']['claims'][0]['context_ids'].append('invented')
    row=deterministic_case(CASE,out)['numeric_checks'][0]
    assert row['truth']=='correct' and row['evidence_support']=='unsupported' and not row['credit']


def test_missing_metric_ids_are_not_a_compatible_pair():
    out=output(); out['analyst']['claims'][0].pop('metric_id')
    out['evaluation_trace']['analyst_packet']['context_items'][0]['structured_fact'].pop('metric_id')
    assert deterministic_case(CASE,out)['structured_evidence_compatibility']['numerator']==0


def calc():
    previous={**SOURCE,'fact_id':'f0','fact_fiscal_year':2023,'value':383285000000,'start_date':'2022-09-25','report_date':'2023-09-30'}
    growth=(NUMBER['value']-previous['value'])/previous['value']*100
    gold={'claim_id':'growth','claim_type':'calculation','sources':[previous,SOURCE],
          'numeric':{**NUMBER,'metric_id':'revenue_growth_percent','unit':'percent','value':growth,'absolute_tolerance':.005001}}
    contexts={str(i):{'kind':'structured_fact','structured_fact':structured(s)} for i,s in enumerate([previous,SOURCE])}
    computation={'expression':'(current - previous) / previous * 100','variables':{'current':'391035','previous':'383285'},'result':growth}
    analyst={'computation':computation,'trace':{'used_financial_evaluator':True,'tool_calls':[{'name':'financial_evaluator','args':deepcopy(computation)}]}}
    analyst['trace']['tool_calls'][0]['args'].pop('result')
    return gold,{'context_ids':['0','1']},contexts,analyst


def test_calculator_requires_bound_call_operands_result_not_matching_digits():
    args=calc(); assert calculator_provenance(*args,CATALOG)['status']=='supported'
    args[3]['trace']['used_financial_evaluator']=False
    assert calculator_provenance(*args,CATALOG)['status']=='missing'


@pytest.mark.parametrize('expression',['((current - previous) / previous) * 100','100 * ((current - previous) / previous)'])
def test_calculator_equivalent_parentheses_and_argument_types(expression):
    args=calc(); analyst=args[3]
    analyst['computation']['expression']=expression
    analyst['trace']['tool_calls'][0]['args']['expression']=expression
    analyst['trace']['tool_calls'][0]['args']['variables']={'current':391035,'previous':383285}
    assert calculator_provenance(*args,CATALOG)['status']=='supported'


@pytest.mark.parametrize('mutation',['wrong_operand','wrong_result','no_call','wrong_period','bare_percent_match'])
def test_calculator_provenance_adversaries(mutation):
    gold,claim,contexts,analyst=calc()
    if mutation=='wrong_operand':
        analyst['computation']['variables']['previous']='10'; analyst['trace']['tool_calls'][0]['args']['variables']['previous']='10'
    if mutation=='wrong_result': analyst['computation']['result']=100
    if mutation=='no_call': analyst['trace']['tool_calls']=[]
    if mutation=='wrong_period': contexts['0']['structured_fact']['fiscal_year']=2022
    if mutation=='bare_percent_match':
        analyst['computation']['expression']='2.02'; analyst['trace']['tool_calls'][0]['args']['expression']='2.02'
    assert calculator_provenance(gold,claim,contexts,analyst,CATALOG)['status']!='supported'


def test_supported_structured_context_conforms_to_real_runtime_contract():
    from agents.contracts import StructuredFactEvidence
    fact=StructuredFactEvidence.model_validate(structured(SOURCE)).model_dump(mode='json')
    assert 'source_sha256' not in fact
    assert evidence_support({'kind':'structured_fact','structured_fact':fact},GOLD,CATALOG)=='supported'


@pytest.mark.parametrize('field',['accession_number','filed_date','report_date','source_url','start_date'])
def test_missing_or_wrong_filing_provenance_never_gets_credit(field):
    for value in (None,'different'):
        fact=structured(SOURCE); fact[field]=value
        assert evidence_support({'kind':'structured_fact','structured_fact':fact},GOLD,CATALOG)!='supported'


def test_fabricated_source_hash_does_not_replace_runtime_filing_identity():
    fact=structured(SOURCE); fact['source_sha256']='filing-hash'; fact.pop('accession_number')
    assert evidence_support({'kind':'structured_fact','structured_fact':fact},GOLD,CATALOG)=='unknown'

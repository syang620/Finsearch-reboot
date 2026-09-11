from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from evals.semantic_dataset_v2 import load_dataset, read, sha, validate_lineage, verify_files
from evals.semantic_judge_v2 import GATES, packets

ROOT=Path('data/evals/semantic_answer/v2')


def test_dataset_determinism_source_compatibility_history_and_counts():
    first=load_dataset(ROOT); second=load_dataset(ROOT)
    assert first==second
    cases,counts=first
    assert len(cases)==60 and counts['v1_required_claims']==86 and counts['v2_required_claims']==119
    assert counts['normalized_question_families']==30
    assert sum(counts['numeric_metric_distribution'].values())==56


@pytest.mark.parametrize('mutation',['duplicate','missing_lineage','changed_value','changed_query'])
def test_lineage_rejects_silent_relabel_or_membership_change(mutation):
    cases=read(ROOT/'queries.jsonl'); old=read('data/evals/semantic_answer/v1/queries.jsonl'); lineage=read(ROOT/'claim_lineage.jsonl')
    if mutation=='duplicate': cases.append(deepcopy(cases[0]))
    if mutation=='missing_lineage': lineage.pop()
    if mutation=='changed_value': cases[0]['required_claims'][0]['numeric']['value']=1
    if mutation=='changed_query': cases[0]['user_query']='Easier query'
    with pytest.raises(ValueError): validate_lineage(cases,old,lineage)


def test_hash_guard_detects_missing_or_edited_file(tmp_path):
    target=tmp_path/'fixture.json'; target.write_text('{}')
    frozen={'fixture.json':sha(target)}
    target.write_text('{"easier":true}')
    with pytest.raises(ValueError): verify_files(tmp_path,frozen)


def test_validation_composition_and_packet_isolation():
    fixtures=read(ROOT/'validation_fixtures.jsonl')
    cases={c['id']:c for c in read(ROOT/'queries.jsonl')}
    assert len(fixtures)==36 and sum(f['repeat_selected'] for f in fixtures)==12
    counts=Counter(label for f in fixtures for label in f['labels']['claims'].values())
    assert counts==Counter({'fully_supported':19,'partially_supported':6,'unsupported':14})
    assert sum(f['labels']['fully_grounded'] and not f['labels']['complete'] for f in fixtures)==2
    assert sum(label=='partial' for f in fixtures for label in f['labels']['requirements'].values())>=3
    assert sum(not f['labels']['answer_relevant'] for f in fixtures)==2
    assert sum(f['labels']['unbound_factual_prose'] for f in fixtures)==2
    classes={f['validation_class']:f for f in fixtures if f.get('validation_class')}
    assert set(classes)=={'cross_filing_generic_supported','cross_filing_named_rejected'}
    generic=classes['cross_filing_generic_supported']; named=classes['cross_filing_named_rejected']
    assert generic['labels']['fully_grounded'] and not named['labels']['fully_grounded']
    assert named['labels']['requirements']=={'current_revenue':'complete','previous_revenue':'missing'}
    for f in (generic,named):
        context=f['output']['evaluation_trace']['analyst_packet']['context_items'][0]['structured_fact']
        assert context['report_date']=='2025-09-27' and context['fiscal_year']==2024
    for f in fixtures:
        assert f['case']==cases[f['case']['id']]
        p=packets(f['case'],f['output'])
        assert 'labels' not in p['support'] and 'labels' not in p['completeness']
        assert 'requirements' not in p['support'] and 'cited_contexts' not in p['completeness']
        assert set(f['labels']['claims'])=={c['claim_id'] for c in f['output']['analyst']['claims']}
        assert set(f['labels']['requirements'])=={g['claim_id'] for g in f['case']['required_claims']}


def test_judge_thresholds_match_preregistered_config():
    assert json.loads((ROOT/'judge_config.json').read_text())['acceptance_thresholds']==GATES


def test_all_synthetic_structured_contexts_are_runtime_shaped():
    from agents.contracts import StructuredFactEvidence
    from evals.semantic_dataset_v2 import load_numeric_catalog
    from evals.semantic_answer_v2 import deterministic_case
    catalog=load_numeric_catalog(ROOT)
    for fixture in read(ROOT/'validation_fixtures.jsonl'):
        for context in fixture['output']['evaluation_trace']['analyst_packet']['context_items']:
            if context['kind']=='structured_fact':
                fact=context['structured_fact']
                assert 'source_sha256' not in fact
                StructuredFactEvidence.model_validate(fact)
        if fixture['id'] in {'SEM2_VALID_01','SEM2_VALID_09','SEM2_VALID_10','SEM2_VALID_11'}:
            row=deterministic_case(fixture['case'],fixture['output'],catalog)
            assert all(check['credit'] for check in row['numeric_checks'])


def test_generic_fact_alternative_does_not_relax_named_filing_comparison():
    from evals.semantic_dataset_v2 import load_numeric_catalog
    from evals.semantic_answer_v2 import evidence_support
    cases={c['id']:c for c in read(ROOT/'queries.jsonl')}; catalog=load_numeric_catalog(ROOT)
    generic=cases['SEM2_AAPL_2024_01']['required_claims'][0]
    alternative=generic['acceptable_source_alternatives'][0]
    filing=catalog['filings'][alternative['source_sha256']]
    fact={**{k:alternative[k] for k in ('ticker','metric_id','value','unit','start_date')},
          **{k:filing[k] for k in ('form_type','accession_number','report_date','filed_date','source_url')},
          'fiscal_year':2024,'status':'ok'}
    context={'kind':'structured_fact','structured_fact':fact}
    assert evidence_support(context,generic,catalog)=='supported'
    named=next(g for g in cases['SEM2_AAPL_2024_05']['required_claims'] if g['numeric']['fiscal_year']==2024)
    assert named['source_scope']['mode']=='named_filing' and not named['acceptable_source_alternatives']
    assert evidence_support(context,named,catalog)=='unknown'


@pytest.mark.parametrize('year,productivity,personal',[(2024,'Office 365 Commercial','Gaming'),
    (2025,'Microsoft 365 Commercial cloud','Gaming and Search and news advertising')])
def test_all_microsoft_segment_requirements_have_adjudicated_support(year,productivity,personal):
    case=next(c for c in read(ROOT/'queries.jsonl') if c['id']==f'SEM2_MSFT_{year}_04')
    expected={'growth_drivers_intelligent_cloud':'Intelligent Cloud revenue increased driven by Azure.',
              'growth_drivers_productivity':f'Productivity and Business Processes revenue increased driven by {productivity}.',
              'growth_drivers_personal_computing':f'More Personal Computing revenue increased driven by {personal}.'}
    for gold in case['required_claims']:
        if gold['claim_id'] in expected:
            assert expected[gold['claim_id']] in [p['quote'] for s in gold['sources'] for p in s['adjudicated_spans']]


def test_decimal_truncations_and_bare_word_tails_do_not_become_adjudicated_sentences():
    cases=read(ROOT/'queries.jsonl')
    for case in cases:
        for gold in case['required_claims']:
            for source in gold['sources']:
                for span in source.get('adjudicated_spans',[]):
                    assert span['quote'] not in {'Gaming revenue increased $6.','Research and development expenses increased $2.',
                                                 'Research and development expenses increased $3.','App Store® and cloud services.','cloud services.'}


def test_source_only_rebuild_is_byte_deterministic_after_v2_exists(tmp_path):
    out=tmp_path/'draft'
    subprocess.run([sys.executable,'scripts/evals/agents/build_semantic_dataset_v2.py','--out-dir',str(out)],
                   check=True,capture_output=True,env={**os.environ,'PYTHONPATH':'src:.'})
    for path in out.iterdir():
        assert path.read_bytes()==(ROOT/path.name).read_bytes(),path.name

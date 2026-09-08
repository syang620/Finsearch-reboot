from copy import deepcopy
from pathlib import Path

from agents.analyst.agent import _table_dict_to_markdown
from evals.semantic_answer_v2 import evidence_support, deterministic_case
from evals.semantic_dataset_v2 import read, load_numeric_catalog

ROOT=Path('data/evals/semantic_answer/v2')


def test_source_display_catalog_matches_unchanged_runtime_renderer():
    catalog=load_numeric_catalog(ROOT)
    assert len(catalog['displays'])==68
    sidecars={}
    for row in catalog['displays'].values():
        path=row['sidecar_path']
        if path not in sidecars: sidecars[path]=read(path)
        table=sidecars[path][int(row['evidence_id'].split('::')[-1])]
        actual=_table_dict_to_markdown(table['table_dict'])
        assert any(r['text']==actual for r in row['representations'])


def sample():
    fixtures=read(ROOT/'validation_fixtures.jsonl')
    fixture=next(f for f in fixtures if f['id']=='SEM2_VALID_09')
    return fixture,load_numeric_catalog(ROOT)


def test_hydrated_prefix_and_regenerated_markdown_retain_numeric_support():
    f,catalog=sample(); out=f['output']
    context=out['evaluation_trace']['analyst_packet']['context_items'][0]
    assert context['payload']['content']!=context['payload']['table_markdown']
    assert context['payload']['table_markdown'].startswith('matched_row:')
    assert evidence_support(context,f['case']['required_claims'][0],catalog)=='supported'
    assert deterministic_case(f['case'],out,catalog)['numeric_checks'][0]['credit']


def test_stable_source_payload_cannot_rescue_altered_or_invisible_evidence():
    f,catalog=sample(); context=f['output']['evaluation_trace']['analyst_packet']['context_items'][0]
    gold=f['case']['required_claims'][0]
    for text in ('Nothing relevant visible.',context['payload']['table_markdown'].replace('391035','1')):
        bad=deepcopy(context); bad['payload']['table_markdown']=text
        assert evidence_support(bad,gold,catalog)!='supported'


def test_wrong_filing_metadata_is_not_rescued_by_same_display():
    f,catalog=sample(); context=f['output']['evaluation_trace']['analyst_packet']['context_items'][0]
    context['source']['ticker']='MSFT'
    assert evidence_support(context,f['case']['required_claims'][0],catalog)=='unsupported'

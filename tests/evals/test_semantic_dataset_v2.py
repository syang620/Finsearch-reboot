from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
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
    assert len(fixtures)==36 and sum(f['repeat_selected'] for f in fixtures)==12
    counts=Counter(label for f in fixtures for label in f['labels']['claims'].values())
    assert counts==Counter({'fully_supported':19,'partially_supported':6,'unsupported':14})
    assert sum(f['labels']['fully_grounded'] and not f['labels']['complete'] for f in fixtures)==2
    for f in fixtures:
        p=packets(f['case'],f['output'])
        assert 'labels' not in p['support'] and 'labels' not in p['completeness']
        assert 'requirements' not in p['support'] and 'cited_contexts' not in p['completeness']
        assert set(f['labels']['claims'])=={c['claim_id'] for c in f['output']['analyst']['claims']}
        assert set(f['labels']['requirements'])=={g['claim_id'] for g in f['case']['required_claims']}


def test_judge_thresholds_match_preregistered_config():
    assert json.loads((ROOT/'judge_config.json').read_text())['acceptance_thresholds']==GATES

"""Read-only v2 draft/freeze verification. Historical data is an input only."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): raise ValueError('Blank JSONL record')
        value=json.loads(line)
        if not isinstance(value,dict): raise ValueError('JSON object required')
        rows.append(value)
    return rows


def relative(path):
    path=Path(path)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe artifact reference')
    return path


def verify_files(root, mapping):
    for name,digest in mapping.items():
        path=Path(root)/relative(name)
        if not path.is_file() or sha(path)!=digest: raise ValueError(f'Frozen file changed or absent: {name}')


def unique(rows,key):
    values=[r.get(key) for r in rows]
    if any(not isinstance(v,str) or not v for v in values) or len(values)!=len(set(values)):
        raise ValueError(f'Duplicate/empty {key}')
    return {r[key]:r for r in rows}


def validate_lineage(cases,old,lineage):
    unique(cases,'id'); previous=unique(old,'id')
    if len(cases)!=len(old) or {c['v1_id'] for c in cases}!=set(previous): raise ValueError('Unexpected v1 membership difference')
    mapping={(r['v1_case_id'],r['v1_claim_id']):r for r in lineage}
    if len(mapping)!=len(lineage): raise ValueError('Duplicate lineage')
    expected={(c['id'],g['claim_id']) for c in old for g in c['required_claims']}
    if set(mapping)!=expected: raise ValueError('Missing/extra original claim lineage')
    for case in cases:
        original=previous[case['v1_id']]; new=unique(case['required_claims'],'claim_id')
        for key in ('user_query','ticker','fiscal_year','form_type','expected_answerability'):
            if case[key]!=original[key]: raise ValueError(f'Unexpected semantic membership/question change: {key}')
        mapped=[]
        for gold in original['required_claims']:
            link=mapping[(original['id'],gold['claim_id'])]
            digest=hashlib.sha256(json.dumps(gold,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if link['v1_claim_sha256']!=digest or link['v2_case_id']!=case['id'] or not link['reason']:
                raise ValueError('Invalid original claim lineage')
            for cid in link['v2_claim_ids']:
                mapped.append(cid); target=new[cid]
                if target['v1_claim_id']!=gold['claim_id'] or not target['requirement'] or not target['sources']:
                    raise ValueError('Invalid adjudicated requirement')
                if target.get('numeric')!=gold.get('numeric'): raise ValueError('Numeric target/tolerance changed')
        if len(mapped)!=len(set(mapped)) or set(mapped)!=set(new): raise ValueError('New claim lacks unique lineage')


def load_dataset(root, repository='.'):
    root=Path(root); repository=Path(repository)
    manifest=json.loads((root/'draft_manifest.json').read_text())
    verify_files(root,manifest['files_sha256'])
    refs=json.loads((root/'source_references.json').read_text())
    verify_files(repository,{refs['corpus_path']:refs['corpus_sha256'],refs['numeric_catalog_path']:refs['numeric_catalog_sha256'],
                             refs['source_sections_path']:refs['source_sections_sha256']})
    for source in refs['source_manifest']['sources']:
        verify_files(repository,{source['source_html']:source['source_sha256'],
                                'data/evals/semantic_answer/v1/'+source['table_sidecar']:source['table_sha256']})
    verify_files(repository,json.loads((root/'historical_sha256.json').read_text()))
    cases=read(root/'queries.jsonl')
    validate_lineage(cases,read(repository/'data/evals/semantic_answer/v1/queries.jsonl'),read(root/'claim_lineage.jsonl'))
    docs=unique(read(repository/refs['corpus_path']),'id')
    for case in cases:
        for gold in case['required_claims']:
            for source in gold['sources']:
                if source['kind']!='kb': continue
                doc=docs.get(source['evidence_id'])
                if doc is None or hashlib.sha256(doc['content'].encode()).hexdigest()!=source['content_sha256']:
                    raise ValueError('Missing/changed gold evidence')
                for span in source.get('adjudicated_spans',[]):
                    if doc['content'][span['start']:span['end']]!=span['quote'] or not span['normalized_source_offsets'] or not span['canonical_source_items']:
                        raise ValueError('Invalid adjudicated evidence span')
    if (root/'validation_manifest.json').exists():
        verify_files(root,json.loads((root/'validation_manifest.json').read_text())['files_sha256'])
    composition=json.loads((root/'composition.json').read_text())
    if len(cases)!=composition['cases'] or sum(len(c['required_claims']) for c in cases)!=composition['v2_required_claims']:
        raise ValueError('Composition denominator mismatch')
    selected=[c for c in cases if c['baseline_audit_selected']]
    if len(selected)!=30 or Counter(c['ticker'] for c in selected)!=Counter({'AAPL':10,'AMZN':10,'MSFT':10}):
        raise ValueError('Audit subset balance mismatch')
    return cases,composition

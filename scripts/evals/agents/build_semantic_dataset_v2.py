"""Source-only deterministic v2 draft builder; never overwrites a dataset.

No retrieval, system answers, baseline artifacts or judge predictions are inputs.
This is not the final optimization freeze (which requires judge/audit decisions).
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from bs4 import BeautifulSoup
from build_semantic_dataset_v1 import canonical
from build_semantic_sources_v1 import extract_facts
from semantic_annotations_v2 import EQUIVALENCE, facets, optional_detail

V1 = Path('data/evals/semantic_answer/v1')
CORPUS = Path('data/evals/retrieval/benchmark_v2/corpus.jsonl')
BASE = 'ef847550c80077bf9d785dc2694c1a9d6afb1ed3'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]
def stable(value): return json.dumps(value, sort_keys=True, ensure_ascii=False)


def source_quotes(source, text):
    """Repair v1 sentence/offset defects, preserving the old source identity.

    Full first anchored sentence (decimal points are not sentence boundaries).
    Additional anchors are retained only when full sentences, avoiding v1's
    ambiguous bare 'mix'/'Gaming'/'App Store' matches in unrelated paragraphs.
    """
    anchors = [s['anchor'] for s in source['spans']]
    anchors = [anchors[0]] + [a for a in anchors[1:] if a.endswith('.') and len(a) > 60]
    if source['ticker'] == 'MSFT' and 'Our Board' in anchors[0]:
        anchors += ['Cybersecurity reviews by the Board are scheduled to occur at least quarterly']
    if source['ticker'] == 'AMZN' and anchors[0].startswith('AWS sales increased'):
        anchors += ['The sales growth primarily reflects increased customer usage']
    if source['ticker'] == 'AMZN' and anchors[0].startswith('We rely on a limited number'):
        anchors += ['An inability to negotiate acceptable terms']
    if source['ticker'] == 'AAPL' and anchors[0].startswith('The Company uses some custom'):
        anchors += ['Although most components essential to the Company']
    rows = []
    for anchor in dict.fromkeys(anchors):
        start = text.index(anchor)
        end = start + len(anchor) if anchor.endswith('.') else None
        if end is None:
            match = re.search(r'\.(?=\s|$)', text[start + len(anchor):])
            if match is None: raise ValueError('Source sentence boundary absent')
            end = start + len(anchor) + match.end()
        rows.append({'start': start, 'end': end, 'quote': text[start:end]})
    return rows


def build(out):
    if out.exists(): raise ValueError('Refusing to overwrite a semantic dataset')
    old = read(V1 / 'queries.jsonl'); docs = {d['id']: d for d in read(CORPUS)}
    source_manifest = json.loads((V1 / 'source_manifest.json').read_text())
    source_text = {}; extracted = {}
    for source in source_manifest['sources']:
        path = Path(source['source_html'])
        if digest(path) != source['source_sha256']: raise ValueError('Historical filing changed')
        source_text[str(path)] = canonical(BeautifulSoup(path.read_text(), 'xml').get_text(' ', strip=True))
        doc = next(d for d in docs.values() if d['metadata']['source_html'] == str(path))
        extracted.update({f['fact_id']: f for f in extract_facts(doc['metadata'])})
    for fact in read(V1 / 'numeric_source_facts.jsonl'):
        if extracted[fact['fact_id']] != fact: raise ValueError('Source re-extraction differs from historical numeric catalog')
    section_map = {s['source_html']: s for s in json.loads(Path('data/evals/retrieval/benchmark_v3/source_sections.json').read_text())}
    lineage = []; cases = []; evidence_groups = defaultdict(list)
    for old_case in old:
        case = deepcopy(old_case); case['id'] = old_case['id'].replace('SEM1_', 'SEM2_', 1)
        case['v1_id'] = old_case['id']; case.pop('audit_selected', None)
        case['annotation_method'] = 'Source-adjudicated coding-assistant annotation, independent of judge predictions; not independent human annotation.'
        case['equivalence_rules'] = EQUIVALENCE
        case['optional_detail'] = []; requirements = []
        for old_gold in old_case['required_claims']:
            gold = deepcopy(old_gold)
            gold['v1_claim_id'] = old_gold['claim_id']
            if not gold.get('numeric'):
                for source in gold['sources']:
                    doc = docs[source['evidence_id']]
                    if digest_text(doc['content']) != source['content_sha256']: raise ValueError('Historical corpus changed')
                    source['adjudicated_spans'] = source_quotes(source, doc['content'])
                    original = source_text[source['source_html']]
                    for span in source['adjudicated_spans']:
                        needle = canonical(span['quote'])
                        offsets = [m.start() for m in re.finditer(re.escape(needle), original)]
                        if not offsets: raise ValueError(f"Source quote not corroborated: {case['id']} {span['quote']}")
                        span['normalized_source_offsets'] = offsets
                        span['canonical_source_items'] = sorted({s['item'] for s in section_map[source['source_html']]['sections'] for i in offsets if s['start'] <= i < s['end']})
                    source['section_warning'] = 'Historical chunk section_path retained as provenance, not trusted canonical SEC Item. Use adjudicated source offsets/items.'
                expanded = []
                for suffix, requirement in facets(case['ticker'], case['fiscal_year'], gold['claim_id']):
                    item = deepcopy(gold); item['claim_id'] += '_' + suffix; item['requirement'] = requirement
                    item['requirement_family'] = case['ticker'] + ':' + old_gold['claim_id'] + ':' + suffix
                    expanded.append(item)
                case['optional_detail'] += optional_detail(case['ticker'], case['fiscal_year'], gold['claim_id'])
                reason = 'B4: source-adjudicated atomic facets and material qualifiers; optional detail separated; repaired source quotation/offset precision.'
            else:
                gold['requirement_family'] = gold['numeric']['metric_id']
                gold['source_scope'] = {'ticker': case['ticker'], 'filing_fiscal_year': case['fiscal_year'], 'form_type': '10-K'}
                gold['accepted_financial_evidence'] = 'same-filing equivalent numeric source; adjudicate explicit semantics independently from declared claim/evidence type'
                expanded = [gold]
                reason = 'B2/B4: numeric target and tolerance preserved and reverified from original source; financial evidence equivalence separated from production route compatibility.'
            requirements += expanded
            lineage.append({'v1_case_id': old_case['id'], 'v2_case_id': case['id'], 'v1_claim_id': old_gold['claim_id'],
                            'v1_claim_sha256': digest_text(stable(old_gold)), 'v2_claim_ids': [g['claim_id'] for g in expanded],
                            'reason': reason, 'numeric_target_changed': False})
            for s in gold['sources']:
                evidence_groups[s.get('evidence_id', s.get('fact_id'))].append(case['id'])
        case['required_claims'] = requirements
        case['question_family'] = re.sub(r'20\d{2}', '<YEAR>', case['user_query'])
        # One case per issuer in each ordinary stratum, paired-year rotation;
        # two numeric per issuer = ten/issuer and thirty overall. No answer input.
        order = int(case['id'][-2:]); issuer = ['AAPL','AMZN','MSFT'].index(case['ticker'])
        years = sorted({c['fiscal_year'] for c in old if c['ticker'] == case['ticker']})
        case['baseline_audit_selected'] = case['fiscal_year'] == years[(order + issuer) % 2]
        cases.append(case)
    historical_paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','data/evals','artifacts/evals'],text=True).splitlines()
    historical = {p:digest(p) for prefix in ('data/evals/','artifacts/evals/') for p in sorted(historical_paths,key=Path) if p.startswith(prefix)}
    for path,value in historical.items():
        original=subprocess.check_output(['git','show',f'{BASE}:{path}'])
        if hashlib.sha256(original).hexdigest()!=value: raise ValueError('Pre-v2 historical file changed')
    report = {'status': 'DRAFT_NOT_OPTIMIZATION_FROZEN', 'cases': len(cases), 'v1_required_claims': len(lineage),
              'v2_required_claims': sum(len(c['required_claims']) for c in cases),
              'issuers': dict(Counter(c['ticker'] for c in cases)),
              'filings': dict(Counter(f"{c['ticker']}:{c['fiscal_year']}" for c in cases)),
              'strata': dict(Counter(c['stratum'] for c in cases)),
              'normalized_question_families': len({c['question_family'] for c in cases}),
              'required_claim_families': dict(Counter(g['requirement_family'] for c in cases for g in c['required_claims'])),
              'numeric_metric_distribution': dict(Counter(g['numeric']['metric_id'] for c in cases for g in c['required_claims'] if g.get('numeric'))),
              'shared_evidence_groups': {k: sorted(set(v)) for k,v in sorted(evidence_groups.items())},
              'baseline_audit_ids': [c['id'] for c in cases if c['baseline_audit_selected']],
              'membership_added': [], 'membership_removed': [], 'question_text_changes': [],
              'limitations': ['Three technology/commerce issuers, six filings; exposed correlated question families, not an unseen or cross-sector holdout.', 'Atomic splitting changes denominators; v1 to v2 is measurement correction, not system improvement.', 'No independent human financial annotation. Source-first assistant adjudication requires external review.']}
    out.mkdir(parents=True)
    outputs = {'queries.jsonl': ''.join(stable(c)+'\n' for c in cases),
               'claim_lineage.jsonl': ''.join(stable(c)+'\n' for c in lineage),
               'composition.json': json.dumps(report, indent=2)+'\n',
               'historical_sha256.json': json.dumps(historical, indent=2)+'\n',
               'source_references.json': json.dumps({'v1_manifest_sha256': digest(V1/'manifest.json'), 'v1_queries_sha256': digest(V1/'queries.jsonl'),
                   'corpus_path': str(CORPUS), 'corpus_sha256': digest(CORPUS), 'numeric_catalog_path': str(V1/'numeric_source_facts.jsonl'),
                   'numeric_catalog_sha256': digest(V1/'numeric_source_facts.jsonl'), 'source_manifest': source_manifest,
                   'source_sections_path': 'data/evals/retrieval/benchmark_v3/source_sections.json',
                   'source_sections_sha256': digest('data/evals/retrieval/benchmark_v3/source_sections.json')}, indent=2)+'\n'}
    for name, content in outputs.items(): (out/name).write_text(content)
    (out/'draft_manifest.json').write_text(json.dumps({'version': 'semantic_answer_v2_draft', 'files_sha256': {name: digest(out/name) for name in outputs}}, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'shared_evidence_groups'}, indent=2))


def digest_text(value): return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out-dir', type=Path, required=True)
    build(parser.parse_args().out_dir)

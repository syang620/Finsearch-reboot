"""Link original inline facts to exact source-derived KB tables, not rankings."""
import json
from pathlib import Path
from bs4 import BeautifulSoup
from build_semantic_dataset_v2 import digest, read, stable, V1, CORPUS


def build(out):
    if out.exists(): raise ValueError('Refusing to overwrite evidence catalog')
    facts = read(V1/'numeric_source_facts.jsonl')
    docs = {d['id']:d for d in read(CORPUS)}
    links = []
    for sidecar in sorted((V1/'tables').glob('*.jsonl')):
        prefix = sidecar.name.removesuffix('.tables.jsonl')
        for index, table in enumerate(read(sidecar)):
            doc = docs[f'{prefix}::table::{index}']
            if table['text'] != doc['content']: raise ValueError('Source sidecar/corpus table disagreement')
            soup = BeautifulSoup(table['table_html'], 'xml')
            elements = {e.get('id') for e in soup.find_all() if e.get('id')}
            for fact in facts:
                if fact['source_html'] != doc['metadata']['source_html']: continue
                matched = [e['element_id'] for e in fact['source_elements'] if e['element_id'] in elements]
                if not matched: continue
                links.append({'fact_id': fact['fact_id'], 'evidence_id':doc['id'], 'content_sha256':doc['content_sha256'],
                              'source_html':fact['source_html'], 'source_sha256':fact['source_sha256'],
                              'source_element_ids':matched, 'sidecar_path':str(sidecar), 'sidecar_sha256':digest(sidecar),
                              'method':'Original filing inline-XBRL element identity in source-derived table HTML; no retriever output.'})
    if not links: raise ValueError('No source-provenance links')
    out.write_text(''.join(stable(row)+'\n' for row in sorted(links,key=lambda r:(r['fact_id'],r['evidence_id']))))
    print(json.dumps({'source_table_links':len(links), 'facts_with_kb_alternatives':len({r['fact_id'] for r in links}), 'sha256':digest(out)}))


if __name__ == '__main__':
    build(Path('data/evals/semantic_answer/v2/numeric_evidence_catalog.jsonl'))

"""Source-derived display alternatives, never answer/retriever-derived labels.

The unchanged runtime hydrates split-orientation table sidecars and displays a
pandas rendering with an index. Catalog that presentation independently; do not
mistake differing Markdown whitespace/index columns for differing source facts.
"""
import json
from pathlib import Path
import pandas as pd

from build_semantic_dataset_v2 import read, stable, digest, CORPUS, V1


def renderings(table):
    data=table['table_dict']
    try: frame=pd.DataFrame(data)
    except Exception: frame=pd.DataFrame.from_dict(data,orient='index')
    if len(frame)>40: raise ValueError('Truncated tables need independently annotated visible-cell bindings before display credit')
    frame=frame.head(40)
    variants=[{'format':'canonical_corpus','text':table['text']},
              {'format':'hydrated_csv','text':frame.to_csv(index=True)},
              {'format':'hydrated_json_without_pandas','text':json.dumps(data,ensure_ascii=False,indent=2)}]
    try: variants.append({'format':'hydrated_markdown','text':frame.to_markdown(index=True)})
    except ImportError: pass
    return variants


def build(out):
    if out.exists(): raise ValueError('Refusing to overwrite display catalog')
    docs={d['id']:d for d in read(CORPUS)}; rows=[]
    numeric=read(Path('data/evals/semantic_answer/v2/numeric_evidence_catalog.jsonl'))
    wanted={r['evidence_id'] for r in numeric}
    for case in read(Path('data/evals/semantic_answer/v2/queries.jsonl')):
        wanted.update(s['evidence_id'] for g in case['required_claims'] if g.get('numeric') for s in g['sources'] if s['kind']=='kb')
    for path in sorted((V1/'tables').glob('*.jsonl')):
        prefix=path.name.removesuffix('.tables.jsonl')
        for index,table in enumerate(read(path)):
            identity=f'{prefix}::table::{index}'
            if identity not in wanted: continue
            doc=docs[identity]
            if doc['content']!=table['text']: raise ValueError('Sidecar/corpus source mismatch')
            rows.append({'evidence_id':identity,'content_sha256':doc['content_sha256'],'metadata':doc['metadata'],
                         'sidecar_path':str(path),'sidecar_sha256':digest(path),'representations':renderings(table)})
    if {r['evidence_id'] for r in rows}!=wanted: raise ValueError('Missing numeric source display')
    out.write_text(''.join(stable(r)+'\n' for r in rows))
    print(json.dumps({'source_tables':len(rows),'sha256':digest(out)}))


if __name__=='__main__': build(Path('data/evals/semantic_answer/v2/source_table_displays.jsonl'))

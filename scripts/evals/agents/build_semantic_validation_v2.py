"""Prospective source-authored judge calibration, not production answer samples.

Answers below are authored and labeled before any judge invocation. Changing
these labels after observing judge behavior is prohibited by the validation
freeze. Each fixture carries its original-filing gold/source provenance.
"""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from build_semantic_dataset_v2 import read, digest, stable, CORPUS

ROOT=Path('data/evals/semantic_answer/v2')
REPEATS={1,2,7,11,13,14,15,22,23,24,31,34}

# Six distinct source scopes, not six variants of the same passage. Each has a
# supported, mixed-support, and centrally wrong answer. Mixed-support explicitly
# combines a source-supported assertion with an unsupported separable assertion.
NARRATIVE=[
    ('SEM2_AAPL_2024_03',
     ['Apple treats highly liquid investments maturing within three months of purchase as cash equivalents.'],
     'Some highly liquid investments qualify as Apple cash equivalents, and Apple guarantees these investments can never lose value.',
     'Apple treats all investments maturing within five years of purchase as cash equivalents.'),
    ('SEM2_AAPL_2025_07',
     ['Some custom Apple components are obtained from a single or limited source.',
      'This sourcing exposes Apple to significant supply risk.', 'It also exposes Apple to significant pricing risk.'],
     'Some custom Apple components are obtained from a single or limited source, and Apple guarantees suppliers will never raise prices.',
     'Apple sources all custom components from many interchangeable suppliers and therefore has no supply or pricing risk.'),
    ('SEM2_AMZN_2023_03',
     ['Amazon records unearned revenue when payments are received before the service is performed.',
      'Payments due before the service is performed also trigger unearned revenue.',
      'Amazon recognizes that revenue over the service period.'],
     'Amazon records unearned revenue when payments are received before service, and recognizes all of it immediately on receipt.',
     'Amazon records unearned revenue only after all service obligations have been performed.'),
    ('SEM2_AMZN_2024_08',
     ['Increased customer usage was the primary driver of AWS sales growth.',
      'Pricing changes partially offset that growth; higher pricing was not the stated positive driver.'],
     'Increased customer usage was the primary AWS growth driver, and higher prices were another positive growth driver.',
     'Higher prices, rather than increased customer usage, were the primary driver of AWS sales growth.'),
    ('SEM2_MSFT_2024_07',
     ['The Microsoft Board of Directors oversees cybersecurity risk.',
      'Its cybersecurity reviews are scheduled at least quarterly.',
      'Evolving and increasingly sophisticated, complex cyberthreats make detection and defense harder.'],
     'The Microsoft Board of Directors oversees cybersecurity risk, and is required to conduct those reviews every day.',
     'Microsoft delegates all cybersecurity oversight exclusively to its external auditor and the board has no role.'),
    ('SEM2_MSFT_2025_08',
     ['Gaming contributed to Microsoft’s R&D expense increase, so the increase was not attributed solely to AI.',
      'Investments in cloud and AI engineering also contributed.',
      'The Gaming contribution included the impact of the Activision Blizzard acquisition.'],
     'Gaming contributed to Microsoft’s R&D expense increase, and the filing says the acquisition of Apple was the reason.',
     'Microsoft attributed the entire R&D expense increase solely to AI; Gaming and acquisitions contributed nothing.'),
]


def build(out):
    if out.exists(): raise ValueError('Refusing to overwrite validation fixtures')
    cases={c['id']:c for c in read(ROOT/'queries.jsonl')}; docs={d['id']:d for d in read(CORPUS)}
    links=read(ROOT/'numeric_evidence_catalog.jsonl')
    displays={r['evidence_id']:r for r in read(ROOT/'source_table_displays.jsonl')}
    rows=[]

    def context(source, cid, kb=False):
        if source['kind']=='kb' or kb:
            doc_id=source['evidence_id'] if source['kind']=='kb' else next(r['evidence_id'] for r in links if r['fact_id']==source['fact_id'])
            doc=docs[doc_id]
            body=doc['content']
            if doc_id in displays:
                body=next(r['text'] for r in displays[doc_id]['representations'] if r['format']=='hydrated_markdown')
            return {'context_id':cid,'kind':'table' if doc['metadata']['doc_type']=='table' else 'text',
                    'source':{'doc_id':doc_id,**doc['metadata']},'payload':{'content':doc['content'],
                       'table_markdown':('matched_row: source table evidence\n\n'+body) if doc_id in displays else body}}
        filing=next(f for f in json.loads((ROOT/'filing_identities.json').read_text()) if f['source_sha256']==source['source_sha256'])
        return {'context_id':cid,'kind':'structured_fact','structured_fact':{
            **{k:source.get(k) for k in ('ticker','metric_id','value','unit','start_date')},
            **{k:filing[k] for k in ('form_type','report_date','filed_date','accession_number','source_url')},
            'metric_label':source['metric_id'],
            'fiscal_year':source['fact_fiscal_year'],'status':'ok'}}

    def add(case_id, texts, support, fulfillment, *, answerability=True, status='ok', kb=False, reason, computation=None,
            answer_tail='', relevant=True, unbound=False):
        case=deepcopy(cases[case_id]); contexts=[]; claims=[]
        sources={}
        for g in case['required_claims']:
            for s in g['sources']:
                identity=s.get('fact_id',s.get('evidence_id'))
                sources.setdefault(identity,s)
        for s in sources.values(): contexts.append(context(s,f'e{len(contexts)+1}',kb))
        refs=[c['context_id'] for c in contexts]
        for i,text in enumerate(texts):
            g=case['required_claims'][min(i,len(case['required_claims'])-1)] if case['required_claims'] else {}
            claim={'claim_id':f'c{i+1}','text':text,'claim_type':g.get('claim_type','narrative'),'context_ids':refs}
            if g.get('numeric'): claim['metric_id']=g['numeric']['metric_id']
            claims.append(claim)
        answer='\n'.join(texts)
        if answer_tail: answer+='\n'+answer_tail
        if status=='insufficient_data':
            answer=(f"The specified FY{case['fiscal_year']} filing cannot establish audited actual net income for FY{case['fiscal_year']+1}; that later fiscal year's actuals are outside this filing." if answerability else 'I cannot answer this question from the filing.')
        output={'ok':True,'status':'completed','failure_stage':'none','analyst':{
            'ok':True,'status':status,'answer':answer,'claims':claims,'compare_rows':[],
            'computation':computation, 'trace':{'analyst_visible_context_ids':refs,'used_financial_evaluator':computation is not None,
                'tool_calls':[{'name':'financial_evaluator','args':{k:v for k,v in computation.items() if k!='result'}}] if computation else []}},
            'evaluation_trace':{'analyst_packet':{'context_items':contexts}}}
        number=len(rows)+1
        labels={'claims':{c['claim_id']:s for c,s in zip(claims,support,strict=True)},
                'requirements':{g['claim_id']:f for g,f in zip(case['required_claims'],fulfillment,strict=True)},
                'answerability_correct':answerability,'answer_relevant':relevant,'unbound_factual_prose':unbound,
                'fully_grounded':answerability and not unbound and all(s=='fully_supported' for s in support) and (bool(claims) or status=='insufficient_data'),
                'complete':answerability and relevant and all(f=='complete' for f in fulfillment)}
        rows.append({'id':f'SEM2_VALID_{number:02}','case':case,'output':output,'labels':labels,
                     'repeat_selected':number in REPEATS,'adjudication_reason':reason,
                     'annotation_method':'Prospectively source-authored synthetic answer and coding-assistant source adjudication before judge predictions; not a production sample or independent human label.'})

    revenue='SEM2_AAPL_2024_01'
    texts=[(revenue,'Apple FY2024 revenue was $391.035 billion.'), (revenue,'Microsoft FY2024 revenue was $391.035 billion.'),
           ('SEM2_AMZN_2024_01','Amazon FY2024 revenue was 637959 million EUR.'),
           ('SEM2_MSFT_2025_01','Microsoft FY2025 revenue was $281724 billion.'),
           (revenue,'Apple FY2024 revenue was not $391.035 billion.'),
           ('SEM2_AMZN_2023_01','Amazon FY2023 revenue was -574785 million USD.'),
           ('SEM2_MSFT_2024_01','Microsoft FY2024 cash and cash equivalents were $245.122 billion.'),
           ('SEM2_AMZN_2024_01','Amazon FY2023 revenue was $637.959 billion.'),
           (revenue,'Apple FY2024 revenue was 391,035 million USD.')]
    for i,(case_id,text) in enumerate(texts):
        correct=i in {0,8}
        add(case_id,[text],['fully_supported' if correct else 'unsupported'],['complete' if correct else 'missing'],kb=i==8,
            reason='The case carries the original issuer/year consolidated revenue fact and source element. Issuer, fiscal period, currency, scale, sign, metric and affirmation are material; matching digits alone do not establish truth. KB table alternative is linked by original inline fact element, not retrieval ranking.')
    add('SEM2_AAPL_2024_09',['Apple FY2024 Services gross-margin percentage was 73.9%.'],['fully_supported'],['complete'],
        reason='The original Services margin-percentage table reports 73.9; gross-profit dollars are a different quantity.')
    case=cases['SEM2_AAPL_2024_06']; values=[g['numeric']['value'] for g in case['required_claims']]
    growth=(Decimal(values[1])-Decimal(values[0]))/Decimal(values[0])*100
    add(case['id'],['Apple FY2023 revenue was 383285 million USD.', 'Apple FY2024 revenue was 391035 million USD.',
                   f'Apple FY2024 revenue growth was {growth:.2f}%.'], ['fully_supported']*3,['complete']*3,
        computation={'expression':'(current - previous) / previous * 100','variables':{'previous':'383285','current':'391035'},'result':float(growth)},
        reason='Both original comparative revenue facts and the bound growth calculation support the three assertions. Two-decimal rounding uses the frozen tolerance.')
    add('SEM2_MSFT_2025_01',['Microsoft FY2025 operating income was $281.724 billion.'],['unsupported'],['missing'],
        reason='Revenue evidence does not support a different metric merely because the digits match.')
    for case_id,full,partial,wrong in NARRATIVE:
        n=len(cases[case_id]['required_claims'])
        fulfillment=['complete']*n
        if case_id=='SEM2_MSFT_2024_07':
            full=[full[0],'Board cybersecurity reviews are scheduled at least once per year.']
            fulfillment=['complete','partial','missing']
        if case_id=='SEM2_AMZN_2023_03':
            full=[full[0],'Amazon recognizes that revenue over time.']; fulfillment=['complete','missing','partial']
        tail=('Apple stock will double tomorrow.' if case_id=='SEM2_AAPL_2024_03' else
              'AWS sales will double next quarter.' if case_id=='SEM2_AMZN_2024_08' else '')
        add(case_id,full,['fully_supported']*len(full),fulfillment,
            answer_tail=tail,unbound=bool(tail),
            reason='Each emitted claim is fully source-supported. Amazon-policy omits the due trigger and states recognition over time without the service-period boundary; Microsoft-governance weakens quarterly to the entailed but incomplete annual minimum and omits the threat explanation. Those two answers are grounded but incomplete. Apple-policy and AWS-attribution add an unsupported future prediction outside emitted claims: unbound_factual_prose=true, so their whole answers are not fully grounded.')
        # The first required facet is retained; other facets absent/contradicted.
        partial_labels=['partial' if case_id=='SEM2_AAPL_2024_03' else 'complete']+['missing']*(n-1)
        add(case_id,[partial],['partially_supported'],partial_labels,
            reason='The first source-backed facet is asserted alongside a separable unsupported or contradictory assertion. Mixed claim support is partial, not fully supported; absent/contradicted other requirements receive no completeness credit.')
        off_topic=case_id in {'SEM2_AAPL_2024_03','SEM2_AMZN_2023_03'}
        if off_topic: wrong='Tomorrow the weather in Paris will certainly be sunny.'
        add(case_id,[wrong],['unsupported'],['missing']*n,relevant=not off_topic,
            reason='The central assertion is unsupported by the cited filing. Apple-policy and Amazon-policy wrong variants are deliberately off-topic weather answers (answer_relevant=false); other wrong variants contradict issuer, driver or policy. A valid citation ID cannot make them supported.')
    for case_id in ('SEM2_AAPL_2024_10','SEM2_AMZN_2023_10','SEM2_MSFT_2024_10'):
        add(case_id,[],[],[],status='insufficient_data',reason='Correct scoped abstention: future audited actuals cannot be established by the earlier named filing; no fabricated amount or later filing is used.')
    for case_id in ('SEM2_AAPL_2024_03','SEM2_AMZN_2023_03','SEM2_MSFT_2024_03'):
        add(case_id,[],[],['missing']*len(cases[case_id]['required_claims']),status='insufficient_data',answerability=False,
            reason='Incorrect refusal of an answerable question: the source filing explicitly provides the policy. Retrieval failure would not change gold answerability.')
    assert len(rows)==36
    counts=Counter(s for f in rows for s in f['labels']['claims'].values())
    assert counts['fully_supported']>=12 and counts['unsupported']>=10 and counts['partially_supported']>=6
    out.write_text(''.join(stable(r)+'\n' for r in rows))
    print(json.dumps({'fixtures':len(rows),'labels':dict(counts),'repeats':sum(r['repeat_selected'] for r in rows),'sha256':digest(out)}))


if __name__=='__main__': build(ROOT/'validation_fixtures.jsonl')

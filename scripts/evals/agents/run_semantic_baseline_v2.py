"""One controlled unchanged-system pass; no selective reruns or runtime tuning.

This command stays blocked until the separate optimization freeze and externally
verified quality approval exist. Judge calibration alone does not release it.
"""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import requests
from qdrant_client import QdrantClient, models as qmodels

from evals.semantic_answer_v2 import deterministic_case
from evals.semantic_dataset_v2 import load_dataset, load_numeric_catalog, read, sha, verify_files
from evals.semantic_metrics_v2 import deterministic_summary
from scripts.evals.agents.run_semantic_v1 import append, save, now, runtime_environment
from scripts.evals.retrieval.run_benchmark_v3 import (clean_checkout, committed_approval, controls, git,
    hardware, snapshot, verify_index, verify_remote_review)
from scripts.evals.retrieval.index_provenance_v3 import verify_frozen_index

DATA=Path('data/evals/semantic_answer/v2')
BASE='ef847550c80077bf9d785dc2694c1a9d6afb1ed3'


def verify_freeze(approval_path):
    clean_checkout()
    approval=committed_approval(approval_path)
    if approval.get('status')!='approved_for_narrow_semantic_v2_baseline': raise ValueError('Semantic benchmark-quality approval required')
    manifest_path=DATA/'optimization_manifest.json'
    if not manifest_path.exists() or approval.get('optimization_manifest_sha256')!=sha(manifest_path): raise ValueError('Optimization freeze/approval hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='OPTIMIZATION_FROZEN': raise ValueError('Not frozen for optimization')
    verify_files(DATA,manifest['files_sha256']); verify_files('.',manifest['code_sha256'])
    reviewed=approval['reviewed_commit']
    if not re.fullmatch('[0-9a-f]{40}',reviewed): raise ValueError('Exact reviewed implementation SHA required')
    git('merge-base','--is-ancestor',reviewed,'HEAD')
    # Only the newly written hash manifest/approval may follow the reviewed
    # candidate. Existing dataset files, scorers and runners must be identical.
    if git('diff',reviewed,'--','src','scripts/evals/agents','scripts/evals/retrieval',str(DATA),f':(exclude){DATA}/optimization_manifest.json'):
        raise ValueError('Reviewed evaluation inputs or behavior changed')
    production=[p for p in git('ls-files','src').splitlines() if not p.startswith('src/evals/')]
    if git('diff',BASE,'--',*production): raise ValueError('Production source changed from merged PR30 baseline')
    verify_remote_review(approval)
    return manifest,approval


def check_controls(state):
    if state.get('ac_power') is not True or state.get('low_power_mode')!=0:
        raise ValueError('AC power and Low Power Mode off required')
    if state.get('browser_process_count') or state.get('heavy_non_model_processes'):
        raise ValueError('Close browsers and settle other heavy workloads before baseline')


def finalize_validity(ending, expected_ids, evaluated_ids):
    """Raw capture completion is not approval to publish a controlled baseline."""
    reasons=[]
    captured=ending['captured_cases']
    if len(captured)!=len(set(captured)) or set(captured)!=set(expected_ids): reasons.append('incomplete_or_duplicate_capture')
    if len(evaluated_ids)!=len(set(evaluated_ids)) or set(evaluated_ids)!=set(expected_ids): reasons.append('incomplete_or_duplicate_evaluation')
    if ending.get('evaluation_errors'): reasons.append('evaluation_errors')
    if ending.get('control_violations'): reasons.append('control_violations')
    try: check_controls(ending.get('controls_after',{}))
    except ValueError: reasons.append('final_control_check_failed')
    for field in ('model_identities_unchanged','index_unchanged'):
        if ending.get(field) is not True: reasons.append(field+'_not_verified')
    for field in ('model_verification_error','index_verification_error','runtime_cleanup_error'):
        if ending.get(field): reasons.append(field)
    ending.update(status='complete' if not reasons else 'invalid_diagnostic',
                  official_baseline_eligible=not reasons,invalidity_reasons=reasons,
                  capture_complete=len(captured)==len(expected_ids) and set(captured)==set(expected_ids))
    return not reasons


def service_preflight(config):
    def get(url,headers=None):
        start=time.perf_counter(); r=requests.get(url,headers=headers,timeout=30); r.raise_for_status()
        return r,{'status_code':r.status_code,'wall_ms':(time.perf_counter()-start)*1000}
    response,_=get('http://127.0.0.1:11434/api/tags')
    tags={m['name']:m for m in response.json()['models']}; identities={}
    for name,digest in [(config['analyst_model'].removeprefix('ollama/'),config['model_digest']),
                        (config['embedding_model'],config['embedding_digest'])]:
        if tags.get(name,{}).get('digest')!=digest: raise ValueError('Frozen model identity mismatch')
        identities[name]={k:tags[name].get(k) for k in ('name','digest','size')}
    ollama,_=get('http://127.0.0.1:11434/api/version')
    qdrant,qdrant_health=get('http://127.0.0.1:6333/')
    _,sec_health=get('https://data.sec.gov/submissions/CIK0000320193.json',{'User-Agent':os.environ['SEC_USER_AGENT']})
    from mcp_server.tools import sec_retrieval as runtime
    probes=[qmodels.ScoredPoint(id=i,version=0,score=1.0,payload={'doc_id':f'health-{i}',
              'content':text}) for i,text in enumerate(['A service health check.','An unrelated document.'],start=1)]
    start=time.perf_counter(); result,rerank_meta=runtime._rerank_candidates('service health check',probes)
    if not result or rerank_meta.get('fallback_used') or rerank_meta.get('applied_backend')!='qwen3_api': raise ValueError('Reranker service health failed')
    return {'ollama_version':ollama.json(),'model_identities':identities,'qdrant_service':qdrant.json(),
            'qdrant_health':qdrant_health,'sec_health':sec_health,
            'reranker_health':{'wall_ms':(time.perf_counter()-start)*1000,'metadata':rerank_meta,
                               'requested_model':runtime._current_rerank_model(),'served_model_digest':None},
            'preflight_note':'Non-benchmark health probe only; requested reranker model recorded, provider does not expose a weight digest. This warms the service and is outside case latency.'}


async def run(args):
    freeze,approval=verify_freeze(args.approval); cases,counts=load_dataset(DATA)
    config=json.loads(Path('data/evals/semantic_answer/v1/evaluation_config.json').read_text())
    # Use historical settings verbatim; v1 judge/audit configuration is not used.
    head=git('rev-parse','HEAD'); out=args.out_root/head; cache=Path('.cache/semantic_answer_v2')/head
    if args.out_root.exists() and any(args.out_root.glob('*/started.json')):
        raise ValueError('A semantic v2 baseline was already started; a new documentation SHA is not authorization to rerun')
    if out.exists() or cache.exists(): raise ValueError('SHA-keyed baseline already attempted; do not overwrite or selectively rerun')
    environment=runtime_environment(config,cache)
    state=controls(); check_controls(state)
    services=service_preflight(config)
    client=QdrantClient(host='127.0.0.1',port=6333,timeout=120)
    before,records=snapshot(client,config['collection'])
    historical,_=snapshot(client,config['historical_collection'])
    verify_index(records,read('data/evals/retrieval/benchmark_v2/corpus.jsonl'))
    index=json.loads(args.index_manifest.read_text())
    reference=verify_frozen_index(index,before,sha(args.index_manifest.parent/'embedded.jsonl'))
    old_reference=json.loads(Path('artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f/manifest.json').read_text())
    if historical!=old_reference['historical_index_after']: raise ValueError('Historical collection changed')
    check_controls(controls())
    out.mkdir(parents=True,exist_ok=False); cache.mkdir(parents=True,exist_ok=False)
    schedule=sorted(cases,key=lambda c:c['id']); random.Random(config['order_seed']).shuffle(schedule)
    catalog=load_numeric_catalog(DATA)
    manifest={'implementation_sha':head,'production_identical_to':BASE,'started_at':now(),
              'dataset_sha256':sha(DATA/'queries.jsonl'),'optimization_manifest_sha256':sha(DATA/'optimization_manifest.json'),
              'quality_approval':approval,'controls_before':state,'service_preflight':services,'hardware':hardware(),
              'environment':environment,'schedule':[c['id'] for c in schedule], 'dataset_composition':counts,
              'runtime_config':{k:v for k,v in config.items() if k not in {'judge','audit','runtime_policy','base_runtime_sha'}},
              'historical_runtime_config_sha256':sha('data/evals/semantic_answer/v1/evaluation_config.json'),
              'python':os.sys.version.split()[0],'packages':{p:importlib.metadata.version(p) for p in ('pytest','requests','qdrant-client','langchain-ollama')},
              'index_before':before,'historical_index_before':historical,'index_origin':reference,
              'policy':'One sequential pass, unchanged runtime retries and 120s analyst timeout; no harness retries or clarification follow-ups; preserve all failures.'}
    save(out/'started.json',manifest)
    from agents.planner.interactive_target_resolution import InteractivePlannerAgent
    from agents.orchestrator.agent_orchestrator import run_multi_agent_orchestration, aclose_orchestrator_runtime
    planner=InteractivePlannerAgent(model=config['planner_model'],log_timing=False)
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    rows=[]; captured=[]; evaluation_errors=[]; control_violations=[]
    try:
        for i,case in enumerate(schedule):
            started=now(); timer=time.perf_counter(); before_state=controls()
            print(f'START {i+1}/60 {case["id"]}',flush=True)
            try:
                output=await run_multi_agent_orchestration(case['user_query'],planner=planner,analyst_model=config['analyst_model'],
                    tables_dir='data/evals/semantic_answer/v1/tables',debug=False,include_evidence_trace=True)
            except Exception as exc:
                output={'ok':False,'status':'harness_captured_runtime_error','error':f'{type(exc).__name__}: {exc}'}
            elapsed=(time.perf_counter()-timer)*1000; after_state=controls()
            append(out/'raw_answers.jsonl',{'case_id':case['id'],'started_at':started,'wall_ms':elapsed,
                   'controls_before':before_state,'controls_after':after_state,'output':output})
            captured.append(case['id'])
            for when,check in [('before',before_state),('after',after_state)]:
                try: check_controls(check)
                except ValueError as exc: control_violations.append({'case_id':case['id'],'when':when,'reason':str(exc)})
            try:
                row=deterministic_case(case,output,catalog); rows.append(row); append(out/'deterministic.jsonl',row)
            except Exception as exc:
                record={'case_id':case['id'],'error':f'{type(exc).__name__}: {exc}'}
                evaluation_errors.append(record); append(out/'evaluation_errors.jsonl',record)
            print(f'END {case["id"]} {output.get("status")} {elapsed/1000:.1f}s',flush=True)
    finally:
        ending={'finished_at':now(),'captured_cases':captured,'status':'complete' if len(captured)==len(cases) else 'incomplete',
                'evaluation_errors':evaluation_errors,'control_violations':control_violations,'controls_after':controls()}
        try: await aclose_orchestrator_runtime()
        except Exception as exc: ending['runtime_cleanup_error']=f'{type(exc).__name__}: {exc}'
        awake.terminate(); awake.wait(timeout=10)
        try:
            response=requests.get('http://127.0.0.1:11434/api/tags',timeout=10); response.raise_for_status()
            identities={m['name']:{k:m.get(k) for k in ('name','digest','size')} for m in response.json()['models'] if m['name'] in services['model_identities']}
            ending.update(model_identities_after=identities,model_identities_unchanged=identities==services['model_identities'])
        except Exception as exc: ending['model_verification_error']=f'{type(exc).__name__}: {exc}'
        try:
            after,_=snapshot(client,config['collection']); historical_after,_=snapshot(client,config['historical_collection'])
            ending.update(index_after=after,historical_index_after=historical_after,index_unchanged=before==after and historical==historical_after)
        except Exception as exc: ending['index_verification_error']=f'{type(exc).__name__}: {exc}'
        valid=finalize_validity(ending,[c['id'] for c in cases],[r['case_id'] for r in rows])
        save(out/'completion.json',ending); client.close()
        if valid:
            save(out/'deterministic_summary.json',{'overall':deterministic_summary(rows),
                 'by_stratum':{s:deterministic_summary([r for r in rows if r['stratum']==s]) for s in sorted({c['stratum'] for c in cases})},
                 'by_issuer':{s:deterministic_summary([r for r in rows if r['ticker']==s]) for s in sorted({c['ticker'] for c in cases})}})
        save(out/'files_sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--index-manifest',type=Path,required=True); parser.add_argument('--env-file',type=Path)
    parser.add_argument('--out-root',type=Path,default=Path('artifacts/evals/semantic_answer/v2/baselines'))
    args=parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    asyncio.run(run(args))

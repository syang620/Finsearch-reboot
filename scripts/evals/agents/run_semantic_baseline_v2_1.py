"""Versioned preflight-only correction; the original v2 launcher stays immutable.

Two fixed 30-second setup settling periods precede the unchanged strict controls.
No case-level sleep, process exemption, retry, timeout or scoring change.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import requests
from qdrant_client import QdrantClient

from evals.semantic_dataset_v2 import load_dataset, load_numeric_catalog, read, sha, verify_files
from scripts.evals.agents.run_semantic_baseline_v2 import (
    DATA, BASE, append, save, now, runtime_environment, clean_checkout,
    committed_approval, controls, git, hardware, snapshot, verify_index,
    verify_remote_review, verify_frozen_index, check_controls, finalize_validity,
    service_preflight, deterministic_case, deterministic_breakdowns)
from scripts.evals.retrieval.run_benchmark_v3 import github_json

ORIGINAL_FREEZE='31797803dc0ef262068fa55a8a76745d15ac758f'
ORIGINAL_MANIFEST_SHA='653a65e778a5633c4d1adc2d90ea7a56a578b81695e69d679f9d69915e54bec0'
LAUNCHER='scripts/evals/agents/run_semantic_baseline_v2_1.py'
CONTRACT=Path('data/evals/semantic_answer/launcher_v2_1/manifest.json')
SETTLE_SECONDS=30


def verify_original(approval_path):
    clean_checkout()
    approval=committed_approval(approval_path)
    manifest_path=DATA/'optimization_manifest.json'
    if (sha(manifest_path)!=ORIGINAL_MANIFEST_SHA
        or approval.get('status')!='approved_for_narrow_semantic_v2_baseline'
        or approval.get('optimization_manifest_sha256')!=ORIGINAL_MANIFEST_SHA):
        raise ValueError('Original v2 optimization freeze/approval changed')
    manifest=json.loads(manifest_path.read_text())
    verify_files(DATA,manifest['files_sha256']); verify_files('.',manifest['code_sha256'])
    git('merge-base','--is-ancestor',ORIGINAL_FREEZE,'HEAD')
    changed=set(git('diff','--name-only',ORIGINAL_FREEZE,'--','src','scripts/evals/agents',
                    'scripts/evals/retrieval','data/evals/semantic_answer').splitlines())
    if changed-{LAUNCHER,str(CONTRACT)}:
        raise ValueError('Only the separately versioned launcher/contract may differ from v2 freeze')
    verify_remote_review(approval)
    return manifest,approval


def freeze_launcher(approval_path,pr,comment_id):
    freeze,original=verify_original(approval_path)
    if CONTRACT.exists(): raise ValueError('Launcher contract already frozen; never overwrite')
    if any(Path('artifacts/evals/semantic_answer/v2/baselines').glob('*/started.json')):
        raise ValueError('Baseline already started; preflight correction is not rerun authority')
    head=git('rev-parse','HEAD')
    comment=github_json(f'repos/syang620/Finsearch-reboot/issues/comments/{comment_id}')
    body=comment['body']
    reviewed=re.search(r'\*\*Reviewed commit:\*\*\s*`([0-9a-f]{10,40})`',body)
    if not reviewed or not head.startswith(reviewed.group(1)):
        raise ValueError('Require fresh clean review of the exact launcher candidate')
    record={'status':'approved_preflight_only_launcher_v2_1','pull_request':pr,
            'review_comment_id':comment_id,'review_url':comment['html_url'],
            'review_body_sha256':hashlib.sha256(body.encode()).hexdigest(),
            'reviewed_commit':head,'original_freeze_commit':ORIGINAL_FREEZE,
            'optimization_manifest_sha256':ORIGINAL_MANIFEST_SHA,
            'quality_approval_sha256':sha(approval_path),'launcher_sha256':sha(LAUNCHER),
            'dataset_sha256':freeze['dataset_sha256'],'judge_enabled':freeze['judge_enabled'],
            'settle_seconds':SETTLE_SECONDS,
            'policy':'Preflight only: fixed settling after imports/configuration and after index/planner setup. All per-case controls, runtime, schedule, labels, scorers, retry/timeout policy and original immutable evidence unchanged. No baseline has started; this authorizes the original single pass, not a rerun.'}
    verify_remote_review(record)
    CONTRACT.parent.mkdir(parents=True,exist_ok=True); save(CONTRACT,record)
    print(json.dumps(record,indent=2))


def verify_launcher(approval_path):
    freeze,original=verify_original(approval_path)
    launcher=committed_approval(CONTRACT)
    if (launcher.get('status')!='approved_preflight_only_launcher_v2_1'
        or launcher.get('optimization_manifest_sha256')!=ORIGINAL_MANIFEST_SHA
        or launcher.get('quality_approval_sha256')!=sha(approval_path)
        or launcher.get('launcher_sha256')!=sha(LAUNCHER)
        or launcher.get('dataset_sha256')!=freeze['dataset_sha256']
        or launcher.get('judge_enabled')!=freeze['judge_enabled']
        or launcher.get('settle_seconds')!=SETTLE_SECONDS):
        raise ValueError('Launcher freeze/identity mismatch')
    reviewed=launcher.get('reviewed_commit','')
    if not re.fullmatch('[0-9a-f]{40}',reviewed):
        raise ValueError('Exact launcher review SHA required')
    git('merge-base','--is-ancestor',reviewed,'HEAD')
    if git('diff',reviewed,'--',LAUNCHER):
        raise ValueError('Launcher changed after review')
    verify_remote_review(launcher)
    return freeze,original,launcher


async def settle_preflight(stage):
    before=controls()
    print(f'PREFLIGHT {stage}: fixed {SETTLE_SECONDS}s setup settling',flush=True)
    timer=time.perf_counter()
    await asyncio.sleep(SETTLE_SECONDS)
    after=controls()
    check_controls(after)
    return {'stage':stage,'requested_seconds':SETTLE_SECONDS,
            'wall_seconds':time.perf_counter()-timer,'before':before,'after':after,
            'note':'Setup-only observation before settling; the unchanged strict check applies after settling. No processes are exempted and no case has started.'}


async def run_once(args):
    freeze,approval,launcher=verify_launcher(args.approval); cases,counts=load_dataset(DATA)
    config=json.loads(Path('data/evals/semantic_answer/v1/evaluation_config.json').read_text())
    # Use historical settings verbatim; v1 judge/audit configuration is not used.
    head=git('rev-parse','HEAD'); out=args.out_root/head; cache=Path('.cache/semantic_answer_v2')/head
    if args.out_root.exists() and any(args.out_root.glob('*/started.json')):
        raise ValueError('A semantic v2 baseline was already started; a new documentation SHA is not authorization to rerun')
    if out.exists() or cache.exists(): raise ValueError('SHA-keyed baseline already attempted; do not overwrite or selectively rerun')
    environment=runtime_environment(config,cache)
    initial_settle=await settle_preflight('imports_and_configuration')
    state=initial_settle['after']
    services=service_preflight(config)
    client=QdrantClient(host='127.0.0.1',port=6333,timeout=120)
    before,records=snapshot(client,config['collection'])
    historical,_=snapshot(client,config['historical_collection'])
    verify_index(records,read('data/evals/retrieval/benchmark_v2/corpus.jsonl'))
    index=json.loads(args.index_manifest.read_text())
    reference=verify_frozen_index(index,before,sha(args.index_manifest.parent/'embedded.jsonl'))
    old_reference=json.loads(Path('artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f/manifest.json').read_text())
    if historical!=old_reference['historical_index_after']: raise ValueError('Historical collection changed')
    schedule=sorted(cases,key=lambda c:c['id']); random.Random(config['order_seed']).shuffle(schedule)
    catalog=load_numeric_catalog(DATA)
    from agents.planner.interactive_target_resolution import InteractivePlannerAgent
    from agents.orchestrator.agent_orchestrator import run_multi_agent_orchestration, aclose_orchestrator_runtime
    planner=InteractivePlannerAgent(model=config['planner_model'],log_timing=False)
    final_settle=await settle_preflight('index_verification_and_planner_setup')
    out.mkdir(parents=True,exist_ok=False); cache.mkdir(parents=True,exist_ok=False)
    manifest={'launcher_contract':launcher,'launcher_contract_sha256':sha(CONTRACT),
              'preflight_settling':[initial_settle,final_settle],
              'implementation_sha':head,'production_identical_to':BASE,'started_at':now(),
              'dataset_sha256':sha(DATA/'queries.jsonl'),'optimization_manifest_sha256':sha(DATA/'optimization_manifest.json'),
              'quality_approval':approval,'controls_before':state,'service_preflight':services,'hardware':hardware(),
              'environment':environment,'schedule':[c['id'] for c in schedule], 'dataset_composition':counts,
              'runtime_config':{k:v for k,v in config.items() if k not in {'judge','audit','runtime_policy','base_runtime_sha'}},
              'historical_runtime_config_sha256':sha('data/evals/semantic_answer/v1/evaluation_config.json'),
              'python':os.sys.version.split()[0],'packages':{p:importlib.metadata.version(p) for p in ('pytest','requests','qdrant-client','langchain-ollama')},
              'index_before':before,'historical_index_before':historical,'index_origin':reference,
              'policy':'One sequential pass, unchanged runtime retries and 120s analyst timeout; no harness retries or clarification follow-ups; preserve all failures.'}
    save(out/'started.json',manifest)
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
            save(out/'deterministic_summary.json',deterministic_breakdowns(cases,rows))
        save(out/'files_sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})


async def run(args):
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        await run_once(args)
    finally:
        awake.terminate(); awake.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--approval',type=Path,default=Path('docs/evals/semantic_answer_v2_quality_approval.json'))
    parser.add_argument('--index-manifest',type=Path)
    parser.add_argument('--out-root',type=Path,default=Path('artifacts/evals/semantic_answer/v2/baselines'))
    parser.add_argument('--freeze-review-comment',type=int); parser.add_argument('--pr',type=int,default=31)
    args=parser.parse_args()
    if args.freeze_review_comment:
        freeze_launcher(args.approval,args.pr,args.freeze_review_comment)
    else:
        if args.index_manifest is None: parser.error('--index-manifest required')
        asyncio.run(run(args))

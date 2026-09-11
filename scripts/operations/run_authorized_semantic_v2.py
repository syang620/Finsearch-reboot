"""Consume one external run permission before invoking the unchanged launcher.

Only local registration checks precede consumption. Import/review/network,
preflight and child-process failures all leave a durable consumed marker.
This module neither implements nor patches benchmark/runtime behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

AUTH=Path('docs/evals/semantic_answer_v2_fresh_attempt_20260908.json')
REVIEW=Path('docs/evals/semantic_answer_v2_fresh_attempt_20260908_review.json')
WRAPPER=Path('scripts/operations/run_authorized_semantic_v2.py')
MARKER=Path('.cache/semantic_answer_v2_fresh_authorized_20260908.consumed.json')
OUTCOME=Path('.cache/semantic_answer_v2_fresh_authorized_20260908.launch_outcome.json')
LOG=Path('.cache/semantic_answer_v2_fresh_authorized_20260908.console.log')
STAGING=Path('.cache/semantic_answer_v2_fresh_authorized_20260908')
LAUNCHER=Path('scripts/evals/agents/run_semantic_baseline_v2_1.py')
QUALITY=Path('docs/evals/semantic_answer_v2_quality_approval.json')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def git(*args): return subprocess.check_output(['git',*args],text=True).strip()


def write_once(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump(record,stream,indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def registration():
    # These local checks establish which permission is being consumed. No
    # project/runtime import, service/model call or workload preflight occurs.
    if git('status','--porcelain'): raise ValueError('Clean committed registration required')
    for path in (AUTH,REVIEW):
        if path.read_bytes()!=subprocess.check_output(['git','show',f'HEAD:{path}']):
            raise ValueError('Registration/approval must match committed bytes')
    auth=json.loads(AUTH.read_text()); review=json.loads(REVIEW.read_text())
    if (auth.get('authorization_id')!='SEMANTIC-V2-FRESH-20260908'
        or auth.get('max_new_attempts')!=1
        or auth.get('consumption_marker')!=str(MARKER)
        or auth.get('staging_output_root')!=str(STAGING)
        or auth.get('operation_wrapper')!=str(WRAPPER)
        or auth.get('launcher_path')!=str(LAUNCHER)
        or auth.get('launcher_sha256')!=sha(LAUNCHER)
        or review.get('status')!='approved_for_single_fresh_semantic_v2_attempt'
        or review.get('authorization_sha256')!=sha(AUTH)
        or review.get('operation_wrapper_sha256')!=sha(WRAPPER)):
        raise ValueError('Registered one-attempt identity changed')
    reviewed=review.get('reviewed_commit')
    if not isinstance(reviewed,str) or not re.fullmatch(r'[0-9a-f]{40}',reviewed):
        raise ValueError('Reviewed commit identity changed: full lowercase SHA required')
    git('merge-base','--is-ancestor',reviewed,'HEAD')
    if git('diff',reviewed,'--',str(AUTH),str(WRAPPER)):
        raise ValueError('Authorization/wrapper changed after review')
    return auth,review,git('rev-parse','HEAD')


def verify_review(review):
    # Deliberately after the marker: even an import or GitHub outage consumes
    # this invocation, unlike the old started.json-only preflight boundary.
    from scripts.evals.retrieval.run_benchmark_v3 import verify_remote_review
    verify_remote_review(review)


def run(index_manifest,env_file=None):
    auth,review,head=registration()
    marker={'status':'consumed','authorization_id':auth['authorization_id'],
        'authorization_sha256':sha(AUTH),'review_approval_sha256':sha(REVIEW),
        'operation_wrapper_sha256':sha(WRAPPER),'launcher_sha256':sha(LAUNCHER),
        'implementation_sha':head,'consumed_at':now(),'staging_output_root':str(STAGING),
        'policy':'Exactly one invocation, including preflight failure. Never delete/reset this marker or retry with a new SHA/root.'}
    write_once(MARKER,marker)
    print('Authorization consumed durably; invoking the frozen launcher once.',flush=True)
    result={'authorization_id':auth['authorization_id'],'implementation_sha':head,
            'consumption_marker_sha256':sha(MARKER),'stage':'review_verification',
            'child_started':False,'child_returncode':None,'error_type':None}
    child=None
    try:
        verify_review(review)
        command=[sys.executable,'-u',str(LAUNCHER),'--approval',str(QUALITY),
                 '--out-root',str(STAGING),'--index-manifest',str(index_manifest)]
        if env_file is not None: command+=['--env-file',str(env_file)]
        result['stage']='launcher'
        log_fd=os.open(LOG,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(log_fd,'w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            result['child_started']=True
            result['child_returncode']=child.wait()
        result['stage']='finished'
        return result['child_returncode']
    except BaseException as exc:
        result['error_type']=type(exc).__name__
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            result['child_returncode']=child.wait()
        raise
    finally:
        result['finished_at']=now()
        result['policy']='Invocation ended; permission remains consumed regardless of captured cases or exit status. Frozen completion.json alone determines baseline eligibility.'
        write_once(OUTCOME,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index-manifest',type=Path,required=True)
    parser.add_argument('--env-file',type=Path)
    args=parser.parse_args()
    raise SystemExit(run(args.index_manifest,args.env_file))

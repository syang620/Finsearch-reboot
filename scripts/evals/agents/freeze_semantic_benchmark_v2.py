"""Publish a reproduced judge decision, then freeze only after fresh review.

Both operations create new records; neither overwrites labels or runs a model.
Optimization freeze needs a clean Codex review of the complete current candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from evals.semantic_dataset_v2 import load_dataset, read, sha, verify_files
from evals.semantic_judge_v2 import packets, validate_support, validate_completeness, whole_answer, validation_metrics
from scripts.evals.retrieval.run_benchmark_v3 import git, github_json, verify_remote_review

DATA=Path('data/evals/semantic_answer/v2')


def clean():
    if git('status','--porcelain'): raise ValueError('Commit a clean candidate before recording a freeze/decision')


def reproduce_validation(root):
    root=Path(root); load_dataset(DATA)
    freeze=json.loads((DATA/'validation_manifest.json').read_text())
    verify_files(DATA,freeze['files_sha256']); verify_files('.',freeze['code_sha256'])
    verify_files(root,json.loads((root/'files_sha256.json').read_text()))
    config=json.loads((DATA/'judge_config.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text()); completion=json.loads((root/'completion.json').read_text())
    if (provenance['validation_manifest_sha256']!=sha(DATA/'validation_manifest.json') or provenance['config']!=config
        or completion['validation_manifest_sha256']!=sha(DATA/'validation_manifest.json')): raise ValueError('Calibration identity changed')
    fixtures=read(DATA/'validation_fixtures.jsonl'); primary={}; repeats={}
    for fixture in fixtures:
        for repeat in ([False,True] if fixture['repeat_selected'] else [False]):
            prefix=fixture['id']+('_repeat' if repeat else ''); pair={}; p=packets(fixture['case'],fixture['output'])
            for phase in config['phases']:
                record=json.loads((root/f'{prefix}_{phase}.json').read_text())
                if record['packet']!=p[phase] or record['fixture_id']!=fixture['id'] or record['repeat']!=repeat or record['phase']!=phase:
                    raise ValueError('Calibration input/identity mismatch')
                if record.get('error'): continue
                response=record['raw_response']
                if (response.get('done') is not True or response.get('done_reason')=='length'
                    or response.get('prompt_eval_count',0)>=config['num_ctx']-config['num_predict']): raise ValueError('Unrecorded judge capacity failure')
                verdict=json.loads(response['message']['content'])
                pair[phase]=(validate_support if phase=='support' else validate_completeness)(p[phase],verdict)
                if pair[phase]!=record['validated']: raise ValueError('Stored judgment differs from raw response')
            result=None
            if set(pair)=={'support','completeness'}:
                result={'claims':{c['claim_id']:c['support'] for c in pair['support']['claims']},
                        'requirements':{c['claim_id']:c['fulfillment'] for c in pair['completeness']['requirements']},
                        'unbound_factual_prose':pair['support']['unbound_factual_prose'],
                        **{k:pair['completeness'][k] for k in ('answerability_correct','answer_relevant')},
                        **whole_answer(pair['support'],pair['completeness'],fixture['output']['analyst']['status'])}
            (repeats if repeat else primary)[fixture['id']]=result
    metrics=validation_metrics(fixtures,primary,repeats)
    if metrics!=json.loads((root/'metrics.json').read_text()): raise ValueError('Metrics do not reproduce from raw calibration')
    if {'primary':primary,'repeat':repeats}!=json.loads((root/'assessments.json').read_text()): raise ValueError('Stored assessments do not reproduce')
    return metrics


def decision(root):
    clean(); root=Path(root)
    if root.is_absolute() or '..' in root.parts: raise ValueError('Use a repository-relative calibration artifact path')
    target=DATA/'judge_decision.json'
    if target.exists(): raise ValueError('Judge decision already recorded')
    metrics=reproduce_validation(root)
    enabled=metrics['full_benchmark_judge_enabled']
    record={'full_benchmark_judge_enabled':enabled,'validation_path':str(root),'raw_artifact_manifest_sha256':sha(root/'files_sha256.json'),
            'validation_metrics_sha256':sha(root/'metrics.json'),'metrics':metrics,
            'decision_rule':'All preregistered gates must pass; no exception or post-result tuning.',
            'interpretation':'Validated secondary candidate on small synthetic calibration only.' if enabled else 'Candidate failed validation. Full-benchmark automated judge disabled; publish source-adjudicated subset metrics with coverage, not a fabricated full-population semantic score.'}
    with target.open('x') as stream: stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


def freeze(pr,comment_id):
    clean(); load_dataset(DATA)
    target=DATA/'optimization_manifest.json'; approval_path=Path('docs/evals/semantic_answer_v2_quality_approval.json')
    if target.exists() or approval_path.exists(): raise ValueError('Optimization contract already frozen; corrections require v3')
    decision_record=json.loads((DATA/'judge_decision.json').read_text())
    recomputed=reproduce_validation(decision_record['validation_path'])
    if (recomputed!=decision_record['metrics'] or decision_record['full_benchmark_judge_enabled']!=recomputed['full_benchmark_judge_enabled']):
        raise ValueError('Judge decision evidence mismatch')
    head=git('rev-parse','HEAD')
    comment=github_json(f'repos/syang620/Finsearch-reboot/issues/comments/{comment_id}')
    body=comment['body']; reviewed=re.search(r'\*\*Reviewed commit:\*\*\s*`([0-9a-f]{10,40})`',body)
    if not reviewed or not head.startswith(reviewed.group(1)): raise ValueError('Require clean review of the current exact candidate before freeze')
    approval={'status':'approved_for_narrow_semantic_v2_baseline','pull_request':pr,'review_comment_id':comment_id,
              'review_url':comment['html_url'],'review_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'reviewed_commit':head}
    verify_remote_review(approval)
    paths=git('ls-files','src','scripts/evals/agents','scripts/evals/retrieval').splitlines()
    manifest={'status':'OPTIMIZATION_FROZEN','reviewed_candidate_sha':head,
              'files_sha256':{p.name:sha(p) for p in sorted(DATA.iterdir()) if p.is_file()},
              'code_sha256':{p:sha(p) for p in paths},'dataset_sha256':sha(DATA/'queries.jsonl'),
              'manual_validation_subset_sha256':sha(DATA/'validation_fixtures.jsonl'),
              'judge_enabled':decision_record['full_benchmark_judge_enabled'],
              'policy':'Future optimization cannot change membership, labels, scoring or judge policy. Corrections require semantic v3.'}
    with target.open('x') as stream: stream.write(json.dumps(manifest,indent=2)+'\n')
    approval['optimization_manifest_sha256']=sha(target)
    with approval_path.open('x') as stream: stream.write(json.dumps(approval,indent=2)+'\n')
    print(json.dumps(approval,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['decision','freeze'])
    parser.add_argument('--validation',type=Path); parser.add_argument('--pr',type=int); parser.add_argument('--review-comment',type=int)
    args=parser.parse_args()
    if args.phase=='decision':
        if args.validation is None: parser.error('--validation required')
        decision(args.validation)
    else:
        if args.pr is None or args.review_comment is None: parser.error('--pr and --review-comment required')
        freeze(args.pr,args.review_comment)

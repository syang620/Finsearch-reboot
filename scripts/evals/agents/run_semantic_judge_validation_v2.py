"""One fixed calibration run. No production calls and no silent retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from evals.semantic_dataset_v2 import load_dataset, read, sha, verify_files
from evals.semantic_judge_v2 import packets, validate_support, validate_completeness, whole_answer, validation_metrics

ROOT=Path('data/evals/semantic_answer/v2')


def command(*args): return subprocess.check_output(args,text=True).strip()
def now(): return datetime.now(timezone.utc).isoformat()


def request(path,payload=None,timeout=10):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request('http://127.0.0.1:11434'+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)


def models():
    # Select identities, never persist model installation/home paths from tags.
    return [{k:m.get(k) for k in ('name','digest','size')} for m in request('/api/tags')['models']]


def controls():
    battery=command('pmset','-g','batt'); power=command('pmset','-g','custom')
    tasks=command('ps','-Ao','pcpu,comm').splitlines()[1:]
    loads=[]
    for line in tasks:
        parts=line.strip().split(maxsplit=1)
        if len(parts)==2:
            try: loads.append({'cpu_percent':float(parts[0]),'process':Path(parts[1]).name})
            except ValueError: pass
    return {'battery':battery,'power_settings':power,'highest_cpu':sorted(loads,key=lambda r:r['cpu_percent'],reverse=True)[:10]}


def verify_validation():
    load_dataset(ROOT)
    manifest=json.loads((ROOT/'validation_manifest.json').read_text())
    verify_files(ROOT,manifest['files_sha256']); verify_files('.',manifest['code_sha256'])
    return manifest


def run(out):
    if command('git','status','--porcelain'): raise RuntimeError('Calibration requires a clean committed worktree')
    implementation=command('git','rev-parse','HEAD'); manifest=verify_validation()
    config=json.loads((ROOT/'judge_config.json').read_text()); fixtures=read(ROOT/'validation_fixtures.jsonl')
    before=models(); live=next((m for m in before if m['name']==config['model']),None)
    if live is None or live['digest']!=config['digest']: raise RuntimeError('Judge model digest mismatch')
    control=controls()
    if "'AC Power'" not in control['battery'] or any(line.split()[-1]!='0' for line in control['power_settings'].splitlines() if line.strip().startswith('lowpowermode')):
        raise RuntimeError('AC power and Low Power Mode off required')
    out=out/implementation
    if out.exists(): raise RuntimeError('Refusing to overwrite an existing calibration run')
    out.mkdir(parents=True)
    awake=subprocess.Popen(['caffeinate','-dims'])
    provenance={'kind':'secondary_judge_calibration_not_release_baseline','implementation_sha':implementation,
                'started_at':now(),'validation_manifest_sha256':sha(ROOT/'validation_manifest.json'),
                'fixtures_sha256':sha(ROOT/'validation_fixtures.jsonl'),'config':config,'models_before':before,
                'ollama_version':request('/api/version'),'controls_before':control,'awake_protection':True,
                'attempts_per_phase':1,'production_calls':0}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    assessments={}; repeats={}
    try:
        ordered=[(f,False) for f in fixtures]+[(f,True) for f in fixtures if f['repeat_selected']]
        for index,(fixture,repeat) in enumerate(ordered):
            pair={}; p=packets(fixture['case'],fixture['output'])
            prefix=fixture['id']+('_repeat' if repeat else '')
            for phase in config['phases']:
                record={'fixture_id':fixture['id'],'repeat':repeat,'phase':phase,'started_at':now(), 'packet':p[phase]}
                start=time.perf_counter()
                payload={'model':config['model'],'stream':False,'think':config['think'],'format':config['format'],
                         'options':{k:config[k] for k in ('temperature','num_ctx','num_predict')},
                         'messages':[{'role':'system','content':(ROOT/f'judge_{phase}_rubric.txt').read_text()},
                                     {'role':'user','content':json.dumps(p[phase],sort_keys=True)}]}
                try:
                    raw=request('/api/chat',payload,timeout=config['timeout_seconds']); record['raw_response']=raw
                    parsed=json.loads(raw['message']['content'])
                    validator=validate_support if phase=='support' else validate_completeness
                    pair[phase]=validator(p[phase],parsed); record['validated']=pair[phase]
                except Exception as exc:
                    # Record every error; do not retry or repair model text.
                    record['error']={'type':type(exc).__name__,'message':str(exc).replace(str(Path.home()),'<HOME>')}
                record['wall_ms']=(time.perf_counter()-start)*1000; record['finished_at']=now()
                (out/f'{prefix}_{phase}.json').write_text(json.dumps(record,indent=2)+'\n')
            assessment=None
            if set(pair)=={'support','completeness'}:
                assessment={'claims':{c['claim_id']:c['support'] for c in pair['support']['claims']},
                            'requirements':{c['claim_id']:c['fulfillment'] for c in pair['completeness']['requirements']},
                            'unbound_factual_prose':pair['support']['unbound_factual_prose'],
                            **{k:pair['completeness'][k] for k in ('answerability_correct','answer_relevant')},
                            **whole_answer(pair['support'],pair['completeness'],fixture['output']['analyst']['status'])}
            (repeats if repeat else assessments)[fixture['id']]=assessment
            print(f'{index+1}/48 {prefix}: {"valid" if assessment else "invalid"}',flush=True)
        report=validation_metrics(fixtures,assessments,repeats)
        (out/'assessments.json').write_text(json.dumps({'primary':assessments,'repeat':repeats},indent=2)+'\n')
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        ending={'finished_at':now(),'controls_after':controls(),'models_after':models(),'validation_manifest_sha256':sha(ROOT/'validation_manifest.json')}
        verify_validation()
        if next(m for m in ending['models_after'] if m['name']==config['model'])['digest']!=config['digest']:
            raise RuntimeError('Judge identity changed during validation')
        (out/'completion.json').write_text(json.dumps(ending,indent=2)+'\n')
        hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        (out/'files_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        awake.terminate(); awake.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out-root',type=Path,required=True)
    run(parser.parse_args().out_root)

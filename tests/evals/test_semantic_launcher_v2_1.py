import ast
import asyncio
import copy
import inspect
from pathlib import Path

import pytest

from scripts.evals.agents import run_semantic_baseline_v2 as original
from scripts.evals.agents import run_semantic_baseline_v2_1 as launcher


def quiet():
    return {'ac_power':True,'low_power_mode':0,'browser_process_count':0,
            'heavy_non_model_processes':[]}


def settling(monkeypatch, before, after):
    states=iter([before,after]); sleeps=[]
    monkeypatch.setattr(launcher,'controls',lambda:next(states))
    async def sleep(seconds): sleeps.append(seconds)
    monkeypatch.setattr(launcher.asyncio,'sleep',sleep)
    return sleeps


def test_setup_cpu_settles_without_exempting_python(monkeypatch):
    busy={**quiet(),'heavy_non_model_processes':[{'process':'python','cpu':89.4}]}
    sleeps=settling(monkeypatch,busy,quiet())
    record=asyncio.run(launcher.settle_preflight('index'))
    assert sleeps==[30] and record['before']==busy and record['after']==quiet()
    assert record['stage']=='index' and record['requested_seconds']==30


@pytest.mark.parametrize('change',[{'ac_power':False},{'low_power_mode':1},
    {'browser_process_count':1},
    {'heavy_non_model_processes':[{'process':'python','cpu':89.4}]},
    {'heavy_non_model_processes':[{'process':'spotlightknowledged','cpu':100}]}])
def test_still_uncontrolled_after_bounded_settling_fails(monkeypatch,change):
    sleeps=settling(monkeypatch,quiet(),{**quiet(),**change})
    with pytest.raises(ValueError): asyncio.run(launcher.settle_preflight('setup'))
    assert sleeps==[30]


def test_unchanged_helpers_and_per_case_execution():
    for name in ('controls','check_controls','finalize_validity','service_preflight',
                 'deterministic_case','deterministic_breakdowns','runtime_environment'):
        assert getattr(launcher,name) is getattr(original,name)
    old=ast.parse(inspect.getsource(original.run)).body[0]
    new=ast.parse(inspect.getsource(launcher.run_once)).body[0]
    old_try=copy.deepcopy(next(n for n in old.body if isinstance(n,ast.Try)))
    new_try=next(n for n in new.body if isinstance(n,ast.Try))
    # Awake protection moved around the entire launch, including preflight.
    old_try.finalbody=[n for n in old_try.finalbody if not (
        isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute)
        and isinstance(n.value.func.value,ast.Name)
        and n.value.func.value.id=='awake')]
    assert ast.dump(old_try)==ast.dump(new_try)


def test_original_frozen_inputs_and_code_are_unchanged():
    import json
    manifest=launcher.DATA/'optimization_manifest.json'
    assert launcher.sha(manifest)==launcher.ORIGINAL_MANIFEST_SHA
    frozen=json.loads(manifest.read_text())
    launcher.verify_files(launcher.DATA,frozen['files_sha256'])
    launcher.verify_files('.',frozen['code_sha256'])


def mock_contract(monkeypatch):
    freeze={'dataset_sha256':'dataset','judge_enabled':False}
    record={'status':'approved_preflight_only_launcher_v2_1',
        'optimization_manifest_sha256':launcher.ORIGINAL_MANIFEST_SHA,
        'quality_approval_sha256':'approval','launcher_sha256':'launcher',
        'dataset_sha256':'dataset','judge_enabled':False,'settle_seconds':30,
        'reviewed_commit':'a'*40}
    monkeypatch.setattr(launcher,'verify_original',lambda p:(freeze,{}))
    monkeypatch.setattr(launcher,'committed_approval',lambda p:record)
    monkeypatch.setattr(launcher,'sha',lambda p:'launcher' if str(p)==launcher.LAUNCHER else 'approval')
    monkeypatch.setattr(launcher,'git',lambda *a:'')
    reviewed=[]
    monkeypatch.setattr(launcher,'verify_remote_review',lambda r:reviewed.append(r))
    return record,reviewed


def test_launcher_requires_fresh_review_and_matching_contract(monkeypatch):
    record,reviewed=mock_contract(monkeypatch)
    assert launcher.verify_launcher(Path('approval'))[2]==record
    assert reviewed==[record]


@pytest.mark.parametrize('field,value',[('launcher_sha256','changed'),
    ('optimization_manifest_sha256','changed'),('quality_approval_sha256','changed'),
    ('dataset_sha256','changed'),('judge_enabled',True),('settle_seconds',0),
    ('reviewed_commit','short'),('status','unapproved')])
def test_changed_contract_cannot_run(monkeypatch,field,value):
    record,_=mock_contract(monkeypatch); record[field]=value
    with pytest.raises(ValueError): launcher.verify_launcher(Path('approval'))


def test_code_changes_after_review_cannot_run(monkeypatch):
    mock_contract(monkeypatch)
    monkeypatch.setattr(launcher,'git',lambda *a:'diff' if a[0]=='diff' else '')
    with pytest.raises(ValueError,match='changed after review'):
        launcher.verify_launcher(Path('approval'))


def test_initial_control_failure_cannot_start_baseline(tmp_path,monkeypatch):
    from argparse import Namespace
    monkeypatch.setattr(launcher,'verify_launcher',lambda p:({}, {}, {}))
    monkeypatch.setattr(launcher,'load_dataset',lambda p:([],{}))
    monkeypatch.setattr(launcher,'git',lambda *a:'unused-test-sha')
    monkeypatch.setattr(launcher,'runtime_environment',lambda *a:{})
    async def fail(stage): raise ValueError('uncontrolled')
    monkeypatch.setattr(launcher,'settle_preflight',fail)
    monkeypatch.setattr(launcher,'service_preflight',lambda *a:pytest.fail('No services before controls'))
    with pytest.raises(ValueError,match='uncontrolled'):
        asyncio.run(launcher.run_once(Namespace(approval=Path('approval'),out_root=tmp_path/'baseline')))
    assert not (tmp_path/'baseline').exists()


def test_awake_cleanup_also_runs_when_preflight_fails(monkeypatch):
    calls=[]
    class Awake:
        def terminate(self): calls.append('terminate')
        def wait(self,timeout): calls.append(('wait',timeout))
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a:Awake())
    async def fail(args): raise ValueError('preflight')
    monkeypatch.setattr(launcher,'run_once',fail)
    with pytest.raises(ValueError): asyncio.run(launcher.run(None))
    assert calls==['terminate',('wait',10)]


def test_original_env_file_option_loads_without_overriding_environment(monkeypatch):
    import dotenv
    calls=[]
    monkeypatch.setattr(dotenv,'load_dotenv',lambda path,override:calls.append(('env',path,override)))
    async def run(args): calls.append(('run',args.index_manifest))
    monkeypatch.setattr(launcher,'run',run)
    launcher.main(['--env-file','local.env','--index-manifest','index.json'])
    assert calls==[('env',Path('local.env'),False),('run',Path('index.json'))]


def test_inherited_environment_needs_no_env_file(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:pytest.fail('No file requested'))
    called=[]
    async def run(args): called.append(args.env_file)
    monkeypatch.setattr(launcher,'run',run)
    launcher.main(['--index-manifest','index.json'])
    assert called==[None]

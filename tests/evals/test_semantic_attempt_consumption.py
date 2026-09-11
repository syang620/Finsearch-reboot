import json
from pathlib import Path
import sys

import pytest

from scripts.operations import run_authorized_semantic_v2 as operation


def setup_operation(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('AUTH','REVIEW','WRAPPER','LAUNCHER'):
        path=tmp_path/name.lower(); path.write_text('{}')
        monkeypatch.setattr(operation,name,path)
    for name in ('MARKER','OUTCOME','LOG','STAGING'):
        monkeypatch.setattr(operation,name,tmp_path/name.lower())
    monkeypatch.setattr(operation,'registration',lambda:({'authorization_id':'test'}, {}, 'a'*40))
    return tmp_path


def test_marker_is_exclusive_and_durable(tmp_path,monkeypatch):
    sync=[]; original=operation.os.fsync
    monkeypatch.setattr(operation.os,'fsync',lambda fd:(sync.append(fd),original(fd)))
    path=tmp_path/'marker.json'
    operation.write_once(path,{'status':'consumed'})
    assert len(sync)==2 and json.loads(path.read_text())=={'status':'consumed'}
    before=path.read_bytes()
    with pytest.raises(FileExistsError): operation.write_once(path,{'status':'reset'})
    assert path.read_bytes()==before


def test_review_failure_consumes_permission_before_any_child(tmp_path,monkeypatch):
    setup_operation(tmp_path,monkeypatch)
    def failure(review):
        assert json.loads(operation.MARKER.read_text())['status']=='consumed'
        raise ConnectionError('preflight review unavailable')
    monkeypatch.setattr(operation,'verify_review',failure)
    monkeypatch.setattr(operation.subprocess,'Popen',lambda *a,**k:pytest.fail('No child expected'))
    with pytest.raises(ConnectionError): operation.run(Path('index.json'))
    record=json.loads(operation.OUTCOME.read_text())
    assert record['error_type']=='ConnectionError' and not record['child_started']
    with pytest.raises(FileExistsError): operation.run(Path('index.json'))


@pytest.mark.parametrize('code',[0,1,130])
def test_child_result_never_resets_permission_or_changes_command(tmp_path,monkeypatch,code):
    setup_operation(tmp_path,monkeypatch); calls=[]
    monkeypatch.setattr(operation,'verify_review',lambda r:None)
    class Child:
        def wait(self): return code
    def spawn(command,**kwargs):
        assert operation.MARKER.exists()
        assert kwargs['stderr']==operation.subprocess.STDOUT
        calls.append(command); return Child()
    monkeypatch.setattr(operation.subprocess,'Popen',spawn)
    assert operation.run(Path('index.json'),Path('local.env'))==code
    assert calls==[[sys.executable,'-u',str(operation.LAUNCHER),'--approval',str(operation.QUALITY),
        '--out-root',str(operation.STAGING),'--index-manifest','index.json','--env-file','local.env']]
    record=json.loads(operation.OUTCOME.read_text())
    assert record['child_started'] and record['child_returncode']==code
    marker=operation.MARKER.read_bytes(); outcome=operation.OUTCOME.read_bytes()
    with pytest.raises(FileExistsError): operation.run(Path('index.json'))
    assert operation.MARKER.read_bytes()==marker and operation.OUTCOME.read_bytes()==outcome
    assert len(calls)==1


def test_launch_error_is_preserved_without_reinvocation(tmp_path,monkeypatch):
    setup_operation(tmp_path,monkeypatch)
    monkeypatch.setattr(operation,'verify_review',lambda r:None)
    def failure(*a,**k): raise OSError('cannot spawn')
    monkeypatch.setattr(operation.subprocess,'Popen',failure)
    with pytest.raises(OSError): operation.run(Path('index.json'))
    assert json.loads(operation.OUTCOME.read_text())['error_type']=='OSError'
    with pytest.raises(FileExistsError): operation.run(Path('index.json'))


def test_interrupt_forwards_to_child_and_preserves_outcome(tmp_path,monkeypatch):
    setup_operation(tmp_path,monkeypatch); signals=[]
    monkeypatch.setattr(operation,'verify_review',lambda r:None)
    class Child:
        calls=0
        def wait(self):
            self.calls+=1
            if self.calls==1: raise KeyboardInterrupt()
            return 130
        def poll(self): return None
        def send_signal(self,sig): signals.append(sig)
    monkeypatch.setattr(operation.subprocess,'Popen',lambda *a,**k:Child())
    with pytest.raises(KeyboardInterrupt): operation.run(Path('index.json'))
    assert signals==[operation.signal.SIGINT]
    record=json.loads(operation.OUTCOME.read_text())
    assert record['error_type']=='KeyboardInterrupt' and record['child_returncode']==130


@pytest.mark.parametrize('bad',[None,'attempts','root','wrapper_hash','authorization_hash',
    'short_sha','uppercase_sha','nonhex_sha','missing_sha','nonstring_sha'])
def test_local_registration_requires_committed_review_bound_identity(tmp_path,monkeypatch,bad):
    registration=operation.registration
    setup_operation(tmp_path,monkeypatch)
    auth={'authorization_id':'SEMANTIC-V2-FRESH-20260908','max_new_attempts':1,
        'consumption_marker':str(operation.MARKER),'staging_output_root':str(operation.STAGING),
        'operation_wrapper':str(operation.WRAPPER),'launcher_path':str(operation.LAUNCHER),
        'launcher_sha256':operation.sha(operation.LAUNCHER)}
    if bad=='attempts': auth['max_new_attempts']=2
    if bad=='root': auth['staging_output_root']='another-root'
    operation.AUTH.write_text(json.dumps(auth))
    review={'status':'approved_for_single_fresh_semantic_v2_attempt',
        'authorization_sha256':operation.sha(operation.AUTH),
        'operation_wrapper_sha256':operation.sha(operation.WRAPPER),'reviewed_commit':'a'*40}
    if bad=='wrapper_hash': review['operation_wrapper_sha256']='changed'
    if bad=='authorization_hash': review['authorization_sha256']='changed'
    if bad=='short_sha': review['reviewed_commit']='a'*10
    if bad=='uppercase_sha': review['reviewed_commit']='A'*40
    if bad=='nonhex_sha': review['reviewed_commit']='g'*40
    if bad=='missing_sha': del review['reviewed_commit']
    if bad=='nonstring_sha': review['reviewed_commit']=123
    operation.REVIEW.write_text(json.dumps(review))
    monkeypatch.setattr(operation,'git',lambda *a:'a'*40 if a[0]=='rev-parse' else '')
    monkeypatch.setattr(operation.subprocess,'check_output',lambda command:Path(command[-1][5:]).read_bytes())
    if bad:
        with pytest.raises(ValueError,match='identity changed'): registration()
    else:
        assert registration()==(auth,review,'a'*40)
    assert not operation.MARKER.exists()

"""No model, retrieval, or live service calls: real harmless children and fault injection."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import MappingProxyType

import pytest

from scripts.operations import run_semantic_v7 as controller
from scripts.operations import semantic_v7_snapshot as snapshot
from scripts.operations import semantic_v7_environment as environment


@pytest.fixture
def contract(tmp_path):
    root = tmp_path / 'attempt'
    root.mkdir(mode=0o700)
    (root / 'cache').mkdir(mode=0o700)
    (root / 'source').mkdir()
    metadata = {'prepared': {'head': 'a' * 40, 'image_sha256': 'b' * 64},
                'approval_sha256': 'c' * 64}
    env = MappingProxyType({'PATH': os.environ['PATH'], 'SEC_USER_AGENT': 'private-contact',
                            'DASHSCOPE_API_KEY': 'private-credential'})
    return controller.Contract(root, root / 'source', sys.executable,
                               (sys.executable, '-c', 'print("harmless")'), env,
                               json.dumps(metadata).encode())


def auth_record(contract):
    record = dict(version='2', authorization_id=controller.AUTHORIZATION_ID,
                  status='authorized_for_one_invocation', explicit_user_authorization=True,
                  max_invocations=1, approval_sha256='c' * 64, reviewed_commit='a' * 40,
                  image_sha256='b' * 64, artifact_root=str(contract.root),
                  review_status='completed_clean', pull_request=31, review_comment_id=123,
                  review_url='https://example.invalid/review', review_body_sha256='d' * 64)
    path = contract.root / 'execution_authorization.json'
    snapshot.write_once(path, record)
    return path, record


def test_authorization_hashes_same_bytes_when_path_replaced(contract):
    path, record = auth_record(contract)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    def remote(actual, label):
        assert actual == record
        replacement = path.with_suffix('.replacement')
        replacement.write_text('{"revoked":true}')
        replacement.replace(path)

    result = controller.authorization(contract, {'pull_request': 31}, remote)
    assert result['record_sha256'] == expected
    assert result['record_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('bad', ['symlink', 'mode', 'duplicate', 'oversize', 'array'])
def test_record_rejections(contract, bad):
    path, _ = auth_record(contract)
    if bad == 'symlink':
        target = path.with_suffix('.target')
        path.rename(target)
        path.symlink_to(target)
    elif bad == 'mode':
        path.chmod(0o644)
    elif bad == 'duplicate':
        path.write_text('{"status":1,"status":2}')
    elif bad == 'oversize':
        path.write_bytes(b' ' * 65537)
    else:
        path.write_text('[]')
    with pytest.raises((ValueError, OSError)):
        snapshot.read_record(path)


def test_same_open_descriptor_used_after_lstat_equivalent(contract, monkeypatch):
    path, _ = auth_record(contract)
    original_open = os.open
    expected = path.read_bytes()

    def swap(name, *args):
        fd = original_open(name, *args)
        replacement = path.with_suffix('.replacement')
        replacement.write_text('{"changed":true}')
        replacement.replace(path)
        return fd

    monkeypatch.setattr(snapshot.os, 'open', swap)
    _, actual = snapshot.read_record(path)
    assert actual == hashlib.sha256(expected).hexdigest()


def test_supervisor_real_child_and_captured_contract(contract, monkeypatch):
    captured = []
    original = subprocess.Popen

    def capture(argv, **kwargs):
        captured.append((argv, kwargs))
        return original(argv, **kwargs)

    monkeypatch.setattr(controller.subprocess, 'Popen', capture)
    with controller.attempt_lock(contract.root):
        assert controller.supervise(contract, {}, lambda c: {'ok': True}) == 0
    argv, options = captured[0]
    assert argv == contract.argv
    assert options['env'] is contract.env
    assert options['cwd'] == contract.cwd
    outcome, _ = snapshot.read_record(contract.root / 'outcome.json')
    assert outcome['child_started'] is True
    assert outcome['child_returncode'] == 0
    assert (contract.root / 'consumed.json').exists()
    with pytest.raises(ValueError, match='no retry'):
        controller.absent_artifacts(contract.root)


@pytest.mark.parametrize('bad', ['symlink', 'mode'])
def test_cache_must_be_owned_private_directory(contract, tmp_path, bad):
    cache = contract.root / 'cache'
    if bad == 'symlink':
        cache.rmdir()
        target = tmp_path / 'redirected-cache'
        target.mkdir(mode=0o700)
        cache.symlink_to(target, target_is_directory=True)
    else:
        cache.chmod(0o755)
    with pytest.raises(ValueError, match='mode-0700 cache'):
        controller.absent_artifacts(contract.root)


def test_cache_replacement_after_preflight_does_not_consume(contract, tmp_path, monkeypatch):
    def replace(stage):
        if stage == 'after_preflight':
            cache = contract.root / 'cache'
            cache.rmdir()
            target = tmp_path / 'redirected-cache'
            target.mkdir(mode=0o700)
            cache.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(controller.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('No child expected'))
    assert controller.supervise(contract, {}, lambda c: {}, transition=replace) == 1
    assert not (contract.root / 'consumed.json').exists()
    assert not (contract.root / 'outcome.json').exists()


def test_cache_replacement_before_spawn_consumes_without_child(contract, tmp_path, monkeypatch):
    def replace(stage):
        if stage == 'before_spawn':
            cache = contract.root / 'cache'
            cache.rmdir()
            target = tmp_path / 'redirected-cache'
            target.mkdir(mode=0o700)
            cache.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(controller.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('No child expected'))
    assert controller.supervise(contract, {}, lambda c: {}, transition=replace) == 1
    outcome, _ = snapshot.read_record(contract.root / 'outcome.json')
    assert outcome['child_started'] is False
    assert outcome['failure']['reason'] == 'Owned external mode-0700 cache directory required'


def test_real_launcher_trampoline_matches_argv_builder(contract):
    launcher = contract.cwd / controller.LAUNCHER
    launcher.parent.mkdir(parents=True)
    launcher.write_text('import os,sys,json\nprint(json.dumps({"cwd":os.getcwd(),'
                        '"path0":sys.path[0],"exe":sys.executable}))\n')
    argv = controller.child_argv(contract.root, contract.cwd, sys.executable)
    current = controller.Contract(contract.root, contract.cwd, sys.executable, argv,
                                  MappingProxyType({'PATH': os.environ['PATH']}), contract.metadata)
    assert controller.supervise(current, {}, lambda c: {}) == 0
    observed = json.loads((contract.root / 'console.log').read_text())
    assert observed == {'cwd': str(contract.cwd), 'path0': str(launcher.parent),
                        'exe': sys.executable}


def test_real_launcher_trampoline_clears_inherited_signal_mask(contract):
    launcher = contract.cwd / controller.LAUNCHER
    launcher.parent.mkdir(parents=True)
    launcher.write_text('import signal\nprint(signal.SIGUSR1 in '
                        'signal.pthread_sigmask(signal.SIG_BLOCK, set()))\n')
    current = controller.Contract(
        contract.root, contract.cwd, sys.executable,
        controller.child_argv(contract.root, contract.cwd, sys.executable),
        MappingProxyType({'PATH': os.environ['PATH']}), contract.metadata)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    try:
        assert controller.supervise(current, {}, lambda c: {}) == 0
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    assert (contract.root / 'console.log').read_bytes() == b'False\n'


@pytest.mark.parametrize('stage,consumed', [('after_preflight', False),
                                          ('after_marker', True), ('before_spawn', True)])
@pytest.mark.parametrize('number', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_boundaries_skip_child(contract, stage, consumed, number):
    def interrupt(actual):
        if actual == stage:
            os.kill(os.getpid(), number)
    assert controller.supervise(contract, {}, lambda c: {}, transition=interrupt) == 128 + number
    assert (contract.root / 'consumed.json').exists() is consumed
    assert (contract.root / 'outcome.json').exists() is consumed
    if consumed:
        result, _ = snapshot.read_record(contract.root / 'outcome.json')
        assert result['child_started'] is False
        assert result['wrapper_exit_code'] == 128 + number


def test_signal_during_popen_reaches_unmasked_child(contract, monkeypatch):
    original = subprocess.Popen
    launcher = contract.cwd / controller.LAUNCHER
    launcher.parent.mkdir(parents=True)
    launcher.write_text('import time\ntime.sleep(20)\n')
    current = controller.Contract(contract.root, contract.cwd, sys.executable,
                                  controller.child_argv(contract.root, contract.cwd, sys.executable),
                                  contract.env, contract.metadata)

    def spawn(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)

    monkeypatch.setattr(controller.subprocess, 'Popen', spawn)
    assert controller.supervise(current, {}, lambda c: {}) == 143
    result, _ = snapshot.read_record(contract.root / 'outcome.json')
    assert result['child_started'] is True
    assert result['child_returncode'] != 0


def test_partial_marker_is_consumed_and_gets_outcome(contract, monkeypatch):
    original = snapshot.write_once

    def fail(path, record):
        if path.name == 'consumed.json':
            path.write_bytes(b'{')
            raise OSError('disk failure')
        original(path, record)

    monkeypatch.setattr(snapshot, 'write_once', fail)
    assert controller.supervise(contract, {}, lambda c: {}) == 1
    outcome, _ = snapshot.read_record(contract.root / 'outcome.json')
    assert outcome['child_started'] is False
    assert outcome['error_type'] == 'OSError'
    assert (contract.root / 'consumed.json').read_bytes() == b'{'


def test_preflight_failure_no_artifacts(contract):
    def fail(c):
        raise RuntimeError('private-credential')
    assert controller.supervise(contract, {}, fail) == 1
    for name in ('consumed.json', 'outcome.json', 'console.log', 'staging'):
        assert not (contract.root / name).exists()


def test_concurrent_attempt_lock(contract):
    with controller.attempt_lock(contract.root):
        with pytest.raises(BlockingIOError):
            with controller.attempt_lock(contract.root):
                pytest.fail('Second lock acquired')


def test_stream_redaction_across_chunks():
    redactor = controller.Redactor(['secret-credential', 'private-contact'])
    output = b''.join(redactor.feed(chunk) for chunk in
                      [b'hello sec', b'ret-cred', b'ential pri', b'vate-contact bye'])
    output += redactor.feed(b'', final=True)
    assert output == b'hello [REDACTED] [REDACTED] bye'


def test_supervisor_redacts_sensitive_values_without_corrupting_numbers(contract):
    env = MappingProxyType({
        'PATH': os.environ['PATH'],
        'SEC_USER_AGENT': 'private-contact',
        'DASHSCOPE_API_KEY': 'private-credential',
        'OTHER_TOKEN': 'another-secret',
        'SHLVL': '1',
        'NO_COLOR': '1',
    })
    script = ('print("case 101 latency 1.25"); '
              'print("private-contact private-credential another-secret")')
    current = controller.Contract(contract.root, contract.cwd, sys.executable,
                                  (sys.executable, '-c', script), env, contract.metadata)
    assert controller.supervise(current, {}, lambda c: {}) == 0
    content = (contract.root / 'console.log').read_bytes()
    assert b'case 101 latency 1.25' in content
    assert content.count(b'[REDACTED]') == 3
    assert b'private-contact' not in content
    assert b'private-credential' not in content
    assert b'another-secret' not in content


def test_frozen_mapping_cannot_change(contract):
    with pytest.raises(TypeError):
        contract.env['SEC_USER_AGENT'] = 'changed'


def test_env_file_changes_cannot_affect_real_child(contract, tmp_path, monkeypatch):
    monkeypatch.setenv('SEC_USER_AGENT', 'inherited-private-contact')
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('QWEN3_RERANK_API_KEY', raising=False)
    env_file = tmp_path / 'runtime.env'
    env_file.write_text('SEC_USER_AGENT=file-contact\nDASHSCOPE_API_KEY=file-secret\n')
    effective, provenance = environment.freeze_effective_child_environment(env_file)
    assert provenance['env_file_supplied'] is True
    assert provenance['required_key_sources']['SEC_USER_AGENT'] == 'inherited'
    assert provenance['required_key_sources']['DASHSCOPE_API_KEY'] == 'env_file'
    env_file.write_text('DASHSCOPE_API_KEY=changed\n')
    env_file.unlink()
    script = ('import os; assert os.environ["SEC_USER_AGENT"] == "inherited-private-contact"; '
              'assert os.environ["DASHSCOPE_API_KEY"] == "file-secret"; '
              'print(os.environ["DASHSCOPE_API_KEY"]); print(os.environ["SEC_USER_AGENT"])')
    current = controller.Contract(contract.root, contract.cwd, sys.executable,
                                  (sys.executable, '-c', script), effective, contract.metadata)
    assert controller.supervise(current, {}, lambda c: {}) == 0
    for name in ('consumed.json', 'outcome.json', 'console.log'):
        content = (contract.root / name).read_bytes()
        assert b'file-secret' not in content
        assert b'inherited-private-contact' not in content


def test_inherited_only_source_flag(monkeypatch):
    monkeypatch.setenv('SEC_USER_AGENT', 'private')
    _, contract = environment.freeze_effective_child_environment(None)
    assert contract['env_file_supplied'] is False
    assert contract['required_key_sources']['SEC_USER_AGENT'] == 'inherited'


def test_missing_authorization_never_calls_remote(contract):
    with pytest.raises(FileNotFoundError):
        controller.authorization(contract, {}, lambda *a: pytest.fail('Remote called'))


@pytest.mark.parametrize('key,value', [('explicit_user_authorization', False),
                                     ('max_invocations', True),
                                     ('image_sha256', 'different'),
                                     ('reviewed_commit', 'b' * 40)])
def test_mismatched_authority_fails_before_remote(contract, key, value):
    path, record = auth_record(contract)
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        controller.authorization(contract, {'pull_request': 31},
                                 lambda *a: pytest.fail('Remote called'))


def test_snapshot_builder_rejects_symlink_without_mount(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    snapshot.git(repo, 'init', '--quiet')
    (repo / 'link').symlink_to('/tmp')
    snapshot.git(repo, 'add', 'link')
    snapshot.git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'bad symlink')
    head = snapshot.git(repo, 'rev-parse', 'HEAD').decode().strip()
    with pytest.raises(ValueError, match='regular tracked'):
        snapshot.verify_tree(repo, head, tmp_path / 'cache')


def test_image_mount_must_be_readonly(contract, monkeypatch):
    class Writable:
        f_flag = 0
    monkeypatch.setattr(snapshot.os, 'statvfs', lambda p: Writable())
    with pytest.raises(ValueError, match='not read-only'):
        snapshot.verify_mount(contract.root, {'version': '2'})


def test_external_pythonpath_rejected(contract):
    with pytest.raises(ValueError, match='PYTHONPATH'):
        controller.validate_paths({'PYTHONPATH': '/tmp'}, contract.cwd, Path(sys.prefix))


def test_failed_outcome_write_keeps_consumed_marker(contract, monkeypatch):
    original = snapshot.write_once
    def fail(path, record):
        if path.name == 'outcome.json':
            raise OSError('storage unavailable')
        original(path, record)
    monkeypatch.setattr(snapshot, 'write_once', fail)
    with pytest.raises(OSError):
        controller.supervise(contract, {}, lambda c: {})
    assert (contract.root / 'consumed.json').exists()
    with pytest.raises(ValueError):
        controller.absent_artifacts(contract.root)


def test_signal_after_outcome_cutoff_cannot_rewrite_outcome(contract, monkeypatch):
    original = snapshot.write_once
    def signal_at_write(path, record):
        if path.name == 'outcome.json':
            os.kill(os.getpid(), signal.SIGTERM)
        original(path, record)
    monkeypatch.setattr(snapshot, 'write_once', signal_at_write)
    assert controller.supervise(contract, {}, lambda c: {}) == 0
    outcome, _ = snapshot.read_record(contract.root / 'outcome.json')
    assert outcome['signals'] == []
    assert outcome['wrapper_exit_code'] == 0


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired('probe', 1),
                                 subprocess.CalledProcessError(2, 'probe'), OSError()])
def test_probe_subprocess_failures(contract, monkeypatch, error):
    monkeypatch.setattr(snapshot, 'verify_mount', lambda *a: {})
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(controller.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='preflight failed'):
        controller.preflight(contract)
    assert not (contract.root / 'consumed.json').exists()


@pytest.mark.parametrize('stdout', [b'bad', b'{}', b'{"ok":true,"interpreter":"wrong"}'])
def test_probe_malformed_or_wrong_identity(contract, monkeypatch, stdout):
    monkeypatch.setattr(snapshot, 'verify_mount', lambda *a: {})
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess([], 0, stdout, b''))
    with pytest.raises((RuntimeError, ValueError)):
        controller.preflight(contract)


def test_inactive_registration_stops_without_imports(contract, monkeypatch):
    monkeypatch.setattr(snapshot, 'verify_mount', lambda *a: {})
    monkeypatch.setattr(controller, '__file__', str(contract.cwd / controller.CONTROLLER))
    monkeypatch.setattr(controller, 'INTERPRETER', Path(sys.executable))
    monkeypatch.setattr(controller, 'validate_paths', lambda *a: None)
    monkeypatch.chdir(contract.cwd)
    with pytest.raises(ValueError, match='candidate inactive'):
        controller.registration(contract.root, {})


@pytest.mark.parametrize('reason', ['Required credentials absent', 'Fixture mode forbidden'])
def test_safe_probe_reason_survives_without_secret_output(contract, monkeypatch, reason):
    monkeypatch.setattr(snapshot, 'verify_mount', lambda *a: {})
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, 'probe',
                output=json.dumps({'reason': reason}).encode(), stderr=b'private-credential')
    monkeypatch.setattr(controller.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match=reason):
        controller.preflight(contract)


def test_failure_reporting_does_not_echo_unknown_exception_text():
    assert 'private-credential' not in json.dumps(
        controller.safe_failure(RuntimeError('private-credential')))


def test_native_readonly_image_executes_original_after_checkout_edit(tmp_path):
    if sys.platform != 'darwin':
        pytest.skip('macOS hdiutil integration requires macOS')
    repository = tmp_path / 'repository'
    repository.mkdir()
    snapshot.git(repository, 'init', '--quiet')
    (repository / 'harmless.py').write_text('print("reviewed source")\n')
    snapshot.git(repository, 'add', 'harmless.py')
    snapshot.git(repository, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'harmless fixture')
    head = snapshot.git(repository, 'rev-parse', 'HEAD').decode().strip()
    root = tmp_path / 'sealed'
    try:
        receipt = snapshot.prepare(repository, head, root)
        (repository / 'harmless.py').write_text('raise RuntimeError("changed checkout")\n')
        snapshot.verify_mount(root, receipt)
        with pytest.raises(OSError):
            (root / 'source' / 'harmless.py').write_text('changed sealed source')
        current = controller.Contract(root, root / 'source', sys.executable,
                    (sys.executable, str(root / 'source' / 'harmless.py')),
                    MappingProxyType({'PATH': os.environ['PATH']}),
                    json.dumps({'prepared': receipt}).encode())
        with controller.attempt_lock(root):
            assert controller.supervise(current, {}, lambda c:
                                        snapshot.verify_mount(root, receipt)) == 0
        assert (root / 'console.log').read_bytes() == b'reviewed source\n'
        outcome, _ = snapshot.read_record(root / 'outcome.json')
        assert outcome['child_started'] is True
        assert outcome['contract']['prepared'] == receipt
        assert snapshot.git(root / 'source', 'status', '--porcelain').strip() == b''
    finally:
        # Only fixture-owned paths; never production images or attempt artifacts.
        if (root / 'source').is_mount():
            snapshot.command(['/usr/bin/hdiutil', 'detach', str(root / 'source')])
        if (root / 'source.dmg').exists():
            os.chflags(root / 'source.dmg', 0)

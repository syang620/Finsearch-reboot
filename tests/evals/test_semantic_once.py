"""Controller tests use temporary Git repositories and harmless child processes only."""
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

from scripts.operations import run_semantic_once as controller


def git(root, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(root), *args]).decode().strip()


def commit(root, message='fixture'):
    git(root, 'add', '-A')
    subprocess.run(
        ['/usr/bin/git', '-C', str(root), '-c', 'user.name=Test',
         '-c', 'user.email=test@example.invalid', 'commit', '-qm', message],
        check=True,
    )
    return git(root, 'rev-parse', 'HEAD')


def repository(tmp_path, files=None):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init', '--quiet')
    for name, content in (files or {'.gitignore': '.cache/\n__pycache__/\n',
                                    'harmless.py': 'print("reviewed")\n'}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root, commit(root)


@pytest.fixture
def prepared_contract(tmp_path):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)
    env = MappingProxyType({
        'PATH': os.environ['PATH'],
        'SEC_USER_AGENT': 'private-contact',
        'DASHSCOPE_API_KEY': 'private-credential',
    })
    metadata = {'prepared': receipt, 'approval_sha256': 'c' * 64}
    contract = controller.Contract(
        root, root / 'source', sys.executable,
        (sys.executable, '-c', 'print("harmless")'), env,
        json.dumps(metadata).encode(),
    )
    return contract


def authorization_record(contract):
    prepared = contract.record()['prepared']
    record = {
        'version': controller.VERSION,
        'authorization_id': controller.AUTHORIZATION_ID,
        'status': 'authorized_for_one_invocation',
        'explicit_user_authorization': True,
        'max_invocations': 1,
        'approval_sha256': 'c' * 64,
        'reviewed_commit': prepared['head'],
        'prepared_tree': prepared['tree'],
        'prepared_files_sha256': prepared['files_sha256'],
        'artifact_root': str(contract.root),
        'review_status': 'completed_clean',
        'pull_request': 31,
        'review_comment_id': 123,
        'review_url': 'https://example.invalid/review',
        'review_body_sha256': 'd' * 64,
    }
    controller.write_once(contract.root / 'execution_authorization.json', record)
    return record


def test_prepare_creates_private_independent_checkout(tmp_path):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / 'source').stat().st_mode) == 0o700
    assert receipt == json.loads((root / 'prepared.json').read_text())
    assert receipt['head'] == head
    assert (root / 'source/.git').is_dir()
    assert not (root / 'source/.git/objects/info/alternates').exists()
    assert subprocess.run(
        ['/usr/bin/git', '-C', str(root / 'source'), 'symbolic-ref', '-q', 'HEAD'],
        capture_output=True,
    ).returncode != 0
    assert git(root / 'source', 'config', '--local', '--get', 'core.hooksPath') == '/dev/null'
    assert controller.verify_prepared(root, receipt)['head'] == head

    (repo / 'harmless.py').write_text('raise RuntimeError("changed")\n')
    assert (root / 'source/harmless.py').read_text() == 'print("reviewed")\n'


def test_ignored_checkout_local_cache_is_allowed(tmp_path):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)
    cache = root / 'source/.cache/unrelated'
    cache.mkdir(parents=True)
    (cache / 'entry').write_text('ok')

    assert controller.verify_prepared(root, receipt)['head'] == head


def test_external_cache_symlink_is_rejected(tmp_path):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)
    external = tmp_path / 'external-cache'
    external.mkdir()
    (root / 'source/.cache').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='identity differs'):
        controller.verify_prepared(root, receipt)


@pytest.mark.parametrize('change', ['tracked', 'untracked', 'hooks', 'alternates'])
def test_prepared_checkout_rejects_identity_changes(tmp_path, change):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)
    source = root / 'source'
    if change == 'tracked':
        (source / 'harmless.py').write_text('changed\n')
    elif change == 'untracked':
        (source / 'unexpected').write_text('changed\n')
    elif change == 'hooks':
        git(source, 'config', '--local', 'core.hooksPath', '.git/hooks')
    else:
        path = source / '.git/objects/info/alternates'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('/tmp/objects\n')
    with pytest.raises(ValueError, match='identity differs'):
        controller.verify_prepared(root, receipt)


def test_prepare_rejects_tracked_symlink(tmp_path):
    repo, _ = repository(tmp_path)
    (repo / 'link').symlink_to('/tmp')
    head = commit(repo, 'symlink')
    with pytest.raises(ValueError, match='regular tracked'):
        controller.prepare(repo, head, tmp_path / 'attempt')


def test_prepare_rejects_gitlink(tmp_path):
    repo, existing = repository(tmp_path)
    subprocess.run(
        ['/usr/bin/git', '-C', str(repo), 'update-index', '--add', '--cacheinfo',
         f'160000,{existing},nested'],
        check=True,
    )
    subprocess.run(
        ['/usr/bin/git', '-C', str(repo), '-c', 'user.name=Test',
         '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'gitlink'],
        check=True,
    )
    head = git(repo, 'rev-parse', 'HEAD')
    with pytest.raises(ValueError, match='regular tracked'):
        controller.prepare(repo, head, tmp_path / 'attempt')


def test_prepare_is_write_once(tmp_path):
    repo, head = repository(tmp_path)
    root = tmp_path / 'attempt'
    controller.prepare(repo, head, root)
    with pytest.raises(FileExistsError):
        controller.prepare(repo, head, root)


@pytest.mark.parametrize('bad', ['symlink', 'mode', 'duplicate', 'oversize', 'array'])
def test_record_rejections(tmp_path, bad):
    path = tmp_path / 'record.json'
    controller.write_once(path, {'ok': True})
    if bad == 'symlink':
        target = tmp_path / 'target.json'
        path.rename(target)
        path.symlink_to(target)
    elif bad == 'mode':
        path.chmod(0o644)
    elif bad == 'duplicate':
        path.write_text('{"ok":true,"ok":false}')
    elif bad == 'oversize':
        path.write_bytes(b' ' * 65537)
    else:
        path.write_text('[]')
    with pytest.raises((ValueError, OSError)):
        controller.read_record(path)


@pytest.mark.parametrize('cache_function', [controller.cache_path,
                                            controller.workload_control_cache_path])
def test_attempt_cache_blocks_launch(prepared_contract, cache_function):
    receipt = prepared_contract.record()['prepared']
    cache = cache_function(prepared_contract.root, receipt)
    cache.mkdir(parents=True)
    with pytest.raises(ValueError, match='cache already exists'):
        controller.absent_artifacts(prepared_contract.root, receipt)


@pytest.mark.parametrize('cache_function', [controller.cache_path,
                                            controller.workload_control_cache_path])
@pytest.mark.parametrize('kind', ['file', 'dangling-symlink', 'external-symlink'])
def test_attempt_cache_namespace_must_be_in_checkout_directory(
        prepared_contract, cache_function, kind, tmp_path):
    receipt = prepared_contract.record()['prepared']
    namespace = cache_function(prepared_contract.root, receipt).parent
    namespace.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'file':
        namespace.write_text('invalid')
    elif kind == 'dangling-symlink':
        namespace.symlink_to(tmp_path / 'missing', target_is_directory=True)
    else:
        external = tmp_path / 'external'
        external.mkdir()
        namespace.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='cache namespace is invalid'):
        controller.absent_artifacts(prepared_contract.root, receipt)


def test_attempt_lock_is_exclusive(prepared_contract):
    with controller.attempt_lock(prepared_contract.root):
        with pytest.raises(BlockingIOError):
            with controller.attempt_lock(prepared_contract.root):
                pytest.fail('second lock acquired')


def test_authorization_binds_prepared_checkout(prepared_contract):
    expected = authorization_record(prepared_contract)
    result = controller.authorization(
        prepared_contract, {'pull_request': 31},
        lambda actual, label: (actual == expected and label == 'execution') or
        pytest.fail('unexpected remote record'),
    )
    assert result['record_sha256'] == hashlib.sha256(
        (prepared_contract.root / 'execution_authorization.json').read_bytes()
    ).hexdigest()


@pytest.mark.parametrize('key,value', [
    ('explicit_user_authorization', False), ('max_invocations', True),
    ('prepared_tree', 'different'), ('reviewed_commit', 'b' * 40),
])
def test_authorization_mismatch_fails_before_remote(prepared_contract, key, value):
    record = authorization_record(prepared_contract)
    record[key] = value
    path = prepared_contract.root / 'execution_authorization.json'
    path.write_text(json.dumps(record))
    path.chmod(0o600)
    with pytest.raises(ValueError, match='authorization mismatch'):
        controller.authorization(
            prepared_contract, {'pull_request': 31},
            lambda *args: pytest.fail('remote called'),
        )


def test_supervisor_runs_once_and_uses_checkout_local_cache(prepared_contract):
    receipt = prepared_contract.record()['prepared']
    relative = Path('.cache/semantic_answer_v2') / receipt['head']
    script = ('from pathlib import Path; '
              f'p=Path({str(relative)!r}); p.mkdir(parents=True); '
              'print("case 101 latency 1.25")')
    contract = controller.Contract(
        prepared_contract.root, prepared_contract.cwd, sys.executable,
        (sys.executable, '-c', script), prepared_contract.env,
        prepared_contract.metadata,
    )
    with controller.attempt_lock(contract.root):
        assert controller.supervise(contract, {}, lambda item: {'ok': True}) == 0
    assert controller.cache_path(contract.root, receipt).is_dir()
    assert (contract.root / 'console.log').read_text() == 'case 101 latency 1.25\n'
    outcome, _ = controller.read_record(contract.root / 'outcome.json')
    assert outcome['child_started'] is True
    assert outcome['child_returncode'] == 0
    with pytest.raises(ValueError, match='no retry'):
        controller.absent_artifacts(contract.root, receipt)


def test_failed_preflight_does_not_consume(prepared_contract):
    with controller.attempt_lock(prepared_contract.root):
        assert controller.supervise(
            prepared_contract, {}, lambda item: (_ for _ in ()).throw(RuntimeError('failed'))
        ) == 1
    assert not (prepared_contract.root / 'consumed.json').exists()
    assert not (prepared_contract.root / 'outcome.json').exists()


def test_failure_after_marker_remains_consumed(prepared_contract):
    def transition(stage):
        if stage == 'after_marker':
            raise OSError('storage failed')

    with controller.attempt_lock(prepared_contract.root):
        assert controller.supervise(
            prepared_contract, {}, lambda item: {'ok': True}, transition=transition
        ) == 1
    assert (prepared_contract.root / 'consumed.json').exists()
    outcome, _ = controller.read_record(prepared_contract.root / 'outcome.json')
    assert outcome['child_started'] is False
    assert outcome['stage'] == 'consumption'


def test_console_redacts_only_sensitive_values(prepared_contract):
    script = ('print("case 101 latency 1.25"); '
              'print("private-contact private-credential")')
    contract = controller.Contract(
        prepared_contract.root, prepared_contract.cwd, sys.executable,
        (sys.executable, '-c', script), prepared_contract.env,
        prepared_contract.metadata,
    )
    with controller.attempt_lock(contract.root):
        assert controller.supervise(contract, {}, lambda item: {}) == 0
    content = (contract.root / 'console.log').read_bytes()
    assert b'case 101 latency 1.25' in content
    assert content.count(b'[REDACTED]') == 2
    assert b'private-contact' not in content
    assert b'private-credential' not in content


def test_preflight_subprocess_failure_is_non_consuming(prepared_contract, monkeypatch):
    monkeypatch.setattr(controller, 'verify_prepared', lambda *args: {})
    monkeypatch.setattr(
        controller.subprocess, 'run',
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired('probe', 1)),
    )
    with pytest.raises(RuntimeError, match='preflight failed'):
        controller.preflight(prepared_contract)
    assert not (prepared_contract.root / 'consumed.json').exists()


def test_inactive_registration_stops_before_environment_freeze(tmp_path, monkeypatch):
    repo, head = repository(tmp_path, {
        '.gitignore': '.cache/\n',
        'controller.py': '# candidate\n',
    })
    root = tmp_path / 'attempt'
    receipt = controller.prepare(repo, head, root)
    monkeypatch.setattr(controller, 'CONTROLLER', Path('controller.py'))
    monkeypatch.setattr(controller, '__file__', str(root / 'source/controller.py'))
    monkeypatch.setattr(controller, 'INTERPRETER', Path(sys.executable))
    monkeypatch.chdir(root / 'source')
    with pytest.raises(ValueError, match='Candidate inactive'):
        controller.registration(root, receipt)


def test_registration_accepts_separate_approval_commit(tmp_path, monkeypatch):
    repo, reviewed = repository(tmp_path, {
        '.gitignore': '.cache/\n',
        'controller.py': '# reviewed candidate\n',
    })
    root = tmp_path / 'attempt'
    approval_path = Path('approval.json')
    controller_path = Path('controller.py')
    approval = {
        'version': controller.VERSION,
        'authorization_id': controller.AUTHORIZATION_ID,
        'status': 'approved_candidate_not_execution',
        'artifact_root': str(root),
        'effective_environment_contract_version': '2',
        'review_status': 'completed_clean',
        'pull_request': 31,
        'review_comment_id': 123,
        'review_url': 'https://example.invalid/review',
        'review_body_sha256': 'd' * 64,
        'reviewed_commit': reviewed,
        'interpreter_path': sys.executable,
        'interpreter_sha256': controller.sha(sys.executable),
        'files_sha256': {str(controller_path): controller.sha(repo / controller_path)},
    }
    (repo / approval_path).write_text(json.dumps(approval))
    head = commit(repo, 'approval')
    receipt = controller.prepare(repo, head, root)
    monkeypatch.setattr(controller, 'CONTROLLER', controller_path)
    monkeypatch.setattr(controller, 'BOUND_FILES', (controller_path,))
    monkeypatch.setattr(controller, 'APPROVAL', approval_path)
    monkeypatch.setattr(controller, '__file__', str(root / 'source' / controller_path))
    monkeypatch.setattr(controller, 'INTERPRETER', Path(sys.executable))
    monkeypatch.chdir(root / 'source')
    assert controller.registration(root, receipt) == approval


def test_signal_before_consumption_skips_launch(prepared_contract):
    def transition(stage):
        if stage == 'after_preflight':
            os.kill(os.getpid(), signal.SIGTERM)

    with controller.attempt_lock(prepared_contract.root):
        assert controller.supervise(
            prepared_contract, {}, lambda item: {'ok': True}, transition=transition
        ) == 128 + signal.SIGTERM
    assert not (prepared_contract.root / 'consumed.json').exists()
    assert not (prepared_contract.root / 'outcome.json').exists()


def test_signal_arriving_during_spawn_is_forwarded_once(prepared_contract, monkeypatch):
    original_popen = controller.subprocess.Popen
    original_killpg = controller.os.killpg
    forwarded = []

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if args[0] == contract.argv:
            os.kill(os.getpid(), signal.SIGTERM)
        return child

    def forward(pid, number):
        forwarded.append((pid, number))
        return original_killpg(pid, number)

    contract = controller.Contract(
        prepared_contract.root, prepared_contract.cwd, sys.executable,
        (sys.executable, '-c', 'import signal,time; '
         'signal.pthread_sigmask(signal.SIG_SETMASK,set()); time.sleep(5)'),
        prepared_contract.env,
        prepared_contract.metadata,
    )
    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    monkeypatch.setattr(controller.os, 'killpg', forward)
    with controller.attempt_lock(contract.root):
        assert controller.supervise(contract, {}, lambda item: {'ok': True}) == 143
    outcome, _ = controller.read_record(contract.root / 'outcome.json')
    assert [number for _, number in forwarded].count(signal.SIGTERM) == 1, outcome


def test_signal_during_outcome_persistence_returns_to_caller(prepared_contract, monkeypatch):
    original_write_once = controller.write_once
    received = []
    previous = signal.signal(signal.SIGTERM, lambda number, frame: received.append(number))

    def interrupt_outcome(path, record):
        if path.name == 'outcome.json':
            os.kill(os.getpid(), signal.SIGTERM)
        original_write_once(path, record)

    monkeypatch.setattr(controller, 'write_once', interrupt_outcome)
    try:
        with controller.attempt_lock(prepared_contract.root):
            assert controller.supervise(
                prepared_contract, {}, lambda item: {'ok': True}
            ) == 0
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert received == [signal.SIGTERM]
    outcome, _ = controller.read_record(prepared_contract.root / 'outcome.json')
    assert outcome['signals'] == []
    assert outcome['signal_observation_closed_at']


def test_runtime_overrides_and_external_pythonpath_are_rejected(prepared_contract):
    with pytest.raises(ValueError, match='override'):
        controller.validate_runtime_paths({'GIT_DIR': '/tmp/repo'}, prepared_contract.cwd)
    with pytest.raises(ValueError, match='PYTHONPATH'):
        controller.validate_runtime_paths({'PYTHONPATH': '/tmp'}, prepared_contract.cwd)


@pytest.mark.parametrize('changed', [None, 'QDRANT_HOST', 'QDRANT_PORT',
                                     'QDRANT_COLLECTION_NAME'])
def test_retrieval_target_must_match_canonical_attestation(tmp_path, monkeypatch, changed):
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'collection': 'canonical-collection'}))
    monkeypatch.setattr(controller, 'INDEX', Path('index.json'))
    environment = {
        'QDRANT_HOST': '127.0.0.1',
        'QDRANT_PORT': '6333',
        'QDRANT_COLLECTION_NAME': 'canonical-collection',
    }
    if changed is None:
        environment.pop('QDRANT_HOST')
    else:
        environment[changed] = 'different'
    with pytest.raises(ValueError, match='Retrieval target differs'):
        controller.validate_retrieval_target(environment, tmp_path)


def test_retrieval_target_accepts_exact_canonical_identity(tmp_path, monkeypatch):
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'collection': 'canonical-collection'}))
    monkeypatch.setattr(controller, 'INDEX', Path('index.json'))
    environment = {
        'QDRANT_HOST': '127.0.0.1',
        'QDRANT_PORT': '6333',
        'QDRANT_COLLECTION_NAME': 'canonical-collection',
    }
    assert controller.validate_retrieval_target(environment, tmp_path) == environment


def test_failure_reporting_never_echoes_unknown_exception_text():
    assert 'private-credential' not in json.dumps(
        controller.safe_failure(RuntimeError('private-credential')))


def test_stream_redaction_across_chunks():
    redactor = controller.Redactor(['secret-credential', 'private-contact'])
    output = b''.join(redactor.feed(chunk) for chunk in
                      [b'hello sec', b'ret-cred', b'ential pri', b'vate-contact bye'])
    output += redactor.feed(b'', final=True)
    assert output == b'hello [REDACTED] [REDACTED] bye'

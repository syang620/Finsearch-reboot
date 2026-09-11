"""Trusted-local, one-use controller for the semantic-v2 baseline.

This module grants no execution authority.  Candidate approval and a separate
external execution-authorization record are both required before launch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from types import MappingProxyType


ROOT = Path('/Users/shicheny/.local/share/finsearch/semantic-baseline/local-once-v1')
VERSION = '1'
AUTHORIZATION_ID = 'SEMANTIC-V2-LOCAL-ONCE-V1'
APPROVAL = Path('docs/evals/semantic_answer_v2_local_once_v1_approval.json')
CONTROLLER = Path('scripts/operations/run_semantic_once.py')
ENVIRONMENT = Path('scripts/operations/semantic_v7_environment.py')
LAUNCHER = Path('scripts/evals/agents/run_semantic_baseline_v2_3.py')
INTEGRATION = Path('docs/evals/semantic_answer_v2_control_v2_fresh_v7_approval.json')
QUALITY = Path('docs/evals/semantic_answer_v2_quality_approval.json')
INDEX = Path('docs/evals/semantic_answer_v2_fresh_v7_qdrant_candidate.json')
INTERPRETER = Path('/Users/shicheny/miniforge3/envs/finsearch-arm/bin/python')
BOUND_FILES = (CONTROLLER, ENVIRONMENT, LAUNCHER, INTEGRATION, QUALITY, INDEX)
REQUIRED_MODULES = (
    'requests', 'qdrant_client', 'dotenv', 'evals.semantic_dataset_v2',
    'scripts.evals.agents.run_semantic_baseline_v2_3',
    'scripts.evals.agents.semantic_workload_control_v2',
    'scripts.diagnostics.run_workload_control_v2_calibration',
    'scripts.diagnostics.observe_semantic_workload',
    'agents.planner.interactive_target_resolution',
    'agents.orchestrator.agent_orchestrator',
    'agents.retrieval.mcp_client', 'mcp_server.server',
)
UNBLOCK_EXEC = (
    'import os,signal,sys; '
    'signal.pthread_sigmask(signal.SIG_SETMASK,set()); '
    'os.execv(sys.executable,[sys.executable,"-u",*sys.argv[1:]])'
)
STOP_GRACE_SECONDS = 30
SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r'(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_|$)'
)
FORBIDDEN_GIT_ENVIRONMENT = frozenset({
    'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_CEILING_DIRECTORIES',
    'GIT_COMMON_DIR', 'GIT_CONFIG_COUNT', 'GIT_CONFIG_GLOBAL',
    'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_SYSTEM', 'GIT_DIR', 'GIT_INDEX_FILE',
    'GIT_NAMESPACE', 'GIT_OBJECT_DIRECTORY', 'GIT_REPLACE_REF_BASE', 'GIT_WORK_TREE',
})
SAFE_FAILURE_REASONS = frozenset({
    'Candidate inactive: approval absent', 'Candidate approval mismatch',
    'Execution authorization mismatch', 'Required credentials absent',
    'Fixture mode forbidden', 'Runtime preflight failed',
    'Preflight execution identity differs', 'Pinned interpreter required',
    'Controller must execute from prepared source',
    'Controller cwd must be prepared source', 'Runtime override environment is forbidden',
    'PYTHONPATH must resolve inside prepared source',
    'Repository module escaped prepared source',
    'Attempt artifacts already exist; no retry permitted',
    'Attempt cache already exists; no retry permitted',
    'Prepared source identity differs', 'Retired v6 identity differs',
})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=pairs)
    except (ValueError, UnicodeError):
        raise ValueError('Invalid JSON record') from None
    if not isinstance(result, dict):
        raise ValueError('JSON object required')
    return result


def read_record(path, limit=65536):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid()):
            raise ValueError('Record must be an owned regular mode-0600 file')
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError('Record exceeds size limit')
    return strict_json(data), digest(data)


def write_once(path, record):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(json.dumps(record, sort_keys=True).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def safe_failure(exc):
    reason = str(exc)
    return {'status': 'failed', 'error_type': type(exc).__name__,
            'reason': reason if reason in SAFE_FAILURE_REASONS else 'Validation or operation failed'}


def command(argv, cwd=None, *, timeout=180):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, check=True,
                              env=env, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError('Preparation or provenance command failed') from None


def git(root, *args):
    return command(['/usr/bin/git', '--no-replace-objects', '-c',
                    'core.hooksPath=/dev/null', *args], root)


def private_directory(path):
    path = Path(path).absolute()
    try:
        info = path.lstat()
    except OSError:
        raise ValueError('Prepared source identity differs') from None
    if (path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700 or path.resolve() != path):
        raise ValueError('Prepared source identity differs')


def verify_tree(source, head):
    source = Path(source).absolute()
    private_directory(source)
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Full commit required')
    if git(source, 'rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('Prepared source identity differs')
    git_dir = source / '.git'
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise ValueError('Prepared source identity differs')
    if (git_dir / 'HEAD').read_text().strip() != head:
        raise ValueError('Prepared source identity differs')
    common = Path(git(source, 'rev-parse', '--git-common-dir').decode().strip())
    if common != Path('.git'):
        raise ValueError('Prepared source identity differs')
    for name in ('objects/info/alternates', 'info/grafts', 'shallow'):
        if (git_dir / name).exists():
            raise ValueError('Prepared source identity differs')
    if git(source, 'for-each-ref', 'refs/replace').strip():
        raise ValueError('Prepared source identity differs')
    if git(source, 'config', '--local', '--get', 'core.hooksPath').decode().strip() != '/dev/null':
        raise ValueError('Prepared source identity differs')

    entries = []
    expected = set()
    for entry in git(source, 'ls-tree', '-rz', '-r', head).split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        relative = os.fsdecode(raw_path)
        if mode not in ('100644', '100755') or kind != 'blob':
            raise ValueError('Only regular tracked files are supported')
        path = source / relative
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(source)):
            raise ValueError('Prepared source identity differs')
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if blob != oid or bool(path.stat().st_mode & 0o111) != (mode == '100755'):
            raise ValueError('Prepared source identity differs')
        entries.append((relative, mode, digest(content)))
        expected.add(relative)
    if git(source, 'status', '--porcelain=v1', '-z', '--untracked-files=all').strip():
        raise ValueError('Prepared source identity differs')
    git(source, 'check-ignore', '-q', '--', '.cache/semantic_answer_v2/probe')
    cache = source / '.cache'
    if exists(cache) and (cache.is_symlink() or not cache.is_dir()
                          or not cache.resolve().is_relative_to(source)):
        raise ValueError('Prepared source identity differs')
    tree = git(source, 'rev-parse', 'HEAD^{tree}').decode().strip()
    manifest = digest(json.dumps(sorted(entries), separators=(',', ':')).encode())
    return {'head': head, 'tree': tree, 'files_sha256': manifest, 'file_count': len(expected)}


def verify_prepared(root, receipt):
    root = Path(root).absolute()
    private_directory(root)
    expected_keys = {'version', 'authorization_id', 'artifact_root', 'head', 'tree',
                     'files_sha256', 'file_count', 'prepared_at'}
    if (set(receipt) != expected_keys or receipt.get('version') != VERSION
            or receipt.get('authorization_id') != AUTHORIZATION_ID
            or receipt.get('artifact_root') != str(root)):
        raise ValueError('Prepared source identity differs')
    actual = verify_tree(root / 'source', receipt.get('head', ''))
    if any(receipt.get(key) != value for key, value in actual.items()):
        raise ValueError('Prepared source identity differs')
    return actual


def prepare(repository, head, root=ROOT):
    repository = Path(repository).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('Full commit required')
    if not repository.is_dir() or not (repository / '.git').exists():
        raise ValueError('Git repository required')
    command(['/usr/bin/git', '--no-replace-objects', '-C', str(repository),
             'cat-file', '-e', head + '^{commit}'])
    root = Path(root).absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    root.chmod(0o700)
    source = root / 'source'
    source.mkdir(mode=0o700)
    template = root / 'empty-git-template'
    template.mkdir(mode=0o700)
    git(source, 'init', '--quiet', '--template=' + str(template))
    git(source, 'config', '--local', 'core.hooksPath', '/dev/null')
    git(source, 'fetch', '--no-tags', str(repository), head)
    git(source, 'checkout', '--detach', head)
    source.chmod(0o700)
    identity = verify_tree(source, head)
    receipt = dict(identity, version=VERSION, authorization_id=AUTHORIZATION_ID,
                   artifact_root=str(root), prepared_at=now())
    write_once(root / 'prepared.json', receipt)
    verify_prepared(root, receipt)
    return receipt


def exists(path):
    return os.path.lexists(path)


def cache_path(root, receipt):
    return Path(root) / 'source' / '.cache' / 'semantic_answer_v2' / receipt['head']


def absent_artifacts(root, receipt):
    for name in ('consumed.json', 'outcome.json', 'console.log', 'staging'):
        if exists(Path(root) / name):
            raise ValueError('Attempt artifacts already exist; no retry permitted')
    if exists(cache_path(root, receipt)):
        raise ValueError('Attempt cache already exists; no retry permitted')


@contextmanager
def attempt_lock(root):
    private_directory(root)
    fd = os.open(Path(root) / 'attempt.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError('Invalid attempt lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Contract:
    root: Path
    cwd: Path
    interpreter: str
    argv: tuple[str, ...]
    env: MappingProxyType
    metadata: bytes

    def record(self):
        return json.loads(self.metadata)


def child_argv(root, cwd, interpreter=INTERPRETER):
    return (str(interpreter), '-u', '-c', UNBLOCK_EXEC, str(cwd / LAUNCHER),
            '--approval', str(QUALITY), '--integration-approval', str(INTEGRATION),
            '--workload-control-v2', 'B_CONSECUTIVE_10',
            '--out-root', str(Path(root) / 'staging'), '--index-attestation', str(INDEX))


def validate_runtime_paths(env, cwd):
    if any(key in FORBIDDEN_GIT_ENVIRONMENT or key.startswith(('GIT_CONFIG_KEY_',
                                                               'GIT_CONFIG_VALUE_')) or key in (
            'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP', 'PYTHONPYCACHEPREFIX')
            or key.startswith('DYLD_') for key in env):
        raise ValueError('Runtime override environment is forbidden')
    for item in env.get('PYTHONPATH', '').split(os.pathsep):
        path = Path(item or cwd)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(cwd):
            raise ValueError('PYTHONPATH must resolve inside prepared source')


def registration(root, receipt):
    cwd = Path(root) / 'source'
    verify_prepared(root, receipt)
    if Path(__file__).resolve() != cwd / CONTROLLER:
        raise ValueError('Controller must execute from prepared source')
    if Path.cwd() != cwd:
        raise ValueError('Controller cwd must be prepared source')
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError('Pinned interpreter required')
    validate_runtime_paths(os.environ, cwd)
    approval_path = cwd / APPROVAL
    if not approval_path.is_file():
        raise ValueError('Candidate inactive: approval absent')
    approval = strict_json(approval_path.read_bytes())
    expected_keys = {
        'version', 'authorization_id', 'status', 'artifact_root',
        'effective_environment_contract_version', 'review_status', 'pull_request',
        'review_comment_id', 'review_url', 'review_body_sha256', 'reviewed_commit',
        'interpreter_path', 'interpreter_sha256', 'files_sha256',
    }
    expected_hashes = {str(path): sha(cwd / path) for path in BOUND_FILES}
    if (set(approval) != expected_keys or approval.get('version') != VERSION
            or approval.get('authorization_id') != AUTHORIZATION_ID
            or approval.get('status') != 'approved_candidate_not_execution'
            or approval.get('artifact_root') != str(root)
            or approval.get('effective_environment_contract_version') != '2'
            or approval.get('review_status') != 'completed_clean'
            or approval.get('interpreter_path') != str(INTERPRETER)
            or approval.get('interpreter_sha256') != sha(INTERPRETER)
            or approval.get('files_sha256') != expected_hashes):
        raise ValueError('Candidate approval mismatch')
    reviewed = approval.get('reviewed_commit', '')
    if not re.fullmatch(r'[0-9a-f]{40}', reviewed):
        raise ValueError('Candidate approval mismatch')
    git(cwd, 'merge-base', '--is-ancestor', reviewed, receipt['head'])
    changed = set(git(cwd, 'diff', '--name-only', reviewed, receipt['head']).decode().splitlines())
    if changed != {str(APPROVAL)}:
        raise ValueError('Candidate approval mismatch')
    return approval


def freeze(root, receipt, env_file):
    from scripts.operations import semantic_v7_environment as environment

    effective, provenance = environment.freeze_effective_child_environment(env_file)
    cwd = Path(root) / 'source'
    validate_runtime_paths(effective, cwd)
    argv = child_argv(root, cwd)
    metadata = {
        'version': VERSION, 'authorization_id': AUTHORIZATION_ID,
        'prepared': receipt, 'approval_sha256': sha(cwd / APPROVAL),
        'cwd': str(cwd), 'interpreter': str(INTERPRETER),
        'argv_sha256': digest(b'\0'.join(os.fsencode(item) for item in argv)),
        'environment': provenance,
    }
    return Contract(Path(root), cwd, str(INTERPRETER), argv, effective,
                    json.dumps(metadata, sort_keys=True).encode())


def authorization(contract, approval, remote):
    record, record_sha = read_record(contract.root / 'execution_authorization.json')
    expected_keys = {
        'version', 'authorization_id', 'status', 'explicit_user_authorization',
        'max_invocations', 'approval_sha256', 'reviewed_commit', 'prepared_tree',
        'prepared_files_sha256', 'artifact_root', 'review_status', 'pull_request',
        'review_comment_id', 'review_url', 'review_body_sha256',
    }
    metadata = contract.record()
    prepared = metadata['prepared']
    if (set(record) != expected_keys or record.get('version') != VERSION
            or record.get('authorization_id') != AUTHORIZATION_ID
            or record.get('status') != 'authorized_for_one_invocation'
            or record.get('explicit_user_authorization') is not True
            or type(record.get('max_invocations')) is not int
            or record.get('max_invocations') != 1
            or record.get('approval_sha256') != metadata['approval_sha256']
            or record.get('reviewed_commit') != prepared['head']
            or record.get('prepared_tree') != prepared['tree']
            or record.get('prepared_files_sha256') != prepared['files_sha256']
            or record.get('artifact_root') != str(contract.root)
            or record.get('review_status') != 'completed_clean'
            or record.get('pull_request') != approval.get('pull_request')):
        raise ValueError('Execution authorization mismatch')
    remote(record, 'execution')
    return {'record_sha256': record_sha, 'reviewed_commit': record['reviewed_commit'],
            'review_comment_id': record['review_comment_id']}


def probe(root):
    cwd = Path(root) / 'source'
    sys.path[0] = str(cwd / LAUNCHER.parent)
    validate_runtime_paths(os.environ, cwd)
    from scripts.operations import semantic_v7_environment as environment

    if os.environ.get('SEC_METRIC_FIXTURE_ROOT'):
        raise ValueError('Fixture mode forbidden')
    if not os.environ.get('SEC_USER_AGENT', '').strip() or not any(
            os.environ.get(key, '').strip()
            for key in ('DASHSCOPE_API_KEY', 'QWEN3_RERANK_API_KEY')):
        raise ValueError('Required credentials absent')
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and name.split('.')[0] in ('agents', 'evals', 'mcp_server', 'scripts'):
            if not Path(filename).resolve().is_relative_to(cwd):
                raise ValueError('Repository module escaped prepared source')
    approval = strict_json((cwd / APPROVAL).read_bytes())
    environment.remote_attestation(approval, 'candidate')
    from scripts.evals.agents import run_semantic_baseline_v2_3 as launcher

    args = argparse.Namespace(workload_control_v2='B_CONSECUTIVE_10',
                              integration_approval=INTEGRATION, approval=QUALITY,
                              index_attestation=INDEX)
    launcher.verify_opt_in(args)
    launcher.verify_frozen_launcher_v7(QUALITY)
    retirement = strict_json(launcher.RETIREMENT.read_bytes())
    for label in ('wrapper', 'approval'):
        if sha(retirement[label + '_path']) != retirement[label + '_sha256']:
            raise ValueError('Retired v6 identity differs')
    from qdrant_client import QdrantClient

    client = QdrantClient(host='127.0.0.1', port=6333, timeout=120)
    try:
        identity, records = launcher.canonical.verify_snapshot(client)
    finally:
        client.close()
    return {'ok': True, 'interpreter': sys.executable, 'cwd': str(Path.cwd()),
            'sys_path_0': sys.path[0], 'points': len(records),
            'fingerprint': identity['payload_vectors_sha256']}


def preflight(contract):
    verify_prepared(contract.root, contract.record()['prepared'])
    try:
        result = subprocess.run(
            [contract.interpreter, '-u', str(contract.cwd / CONTROLLER), '_probe'],
            cwd=contract.cwd, env=contract.env, capture_output=True, timeout=180, check=True,
        )
        record = strict_json(result.stdout)
    except subprocess.CalledProcessError as exc:
        try:
            reason = strict_json(exc.stdout).get('reason')
        except (ValueError, TypeError):
            reason = None
        if reason in SAFE_FAILURE_REASONS:
            raise RuntimeError(reason) from None
        raise RuntimeError('Runtime preflight failed') from None
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeError('Runtime preflight failed') from None
    if (record.get('ok') is not True or record.get('interpreter') != contract.interpreter
            or record.get('cwd') != str(contract.cwd)
            or record.get('sys_path_0') != str(contract.cwd / LAUNCHER.parent)):
        raise ValueError('Preflight execution identity differs')
    return record


class Redactor:
    def __init__(self, secrets):
        self.secrets = sorted({item.encode() for item in secrets if item}, key=len, reverse=True)
        self.pending = b''
        self.width = max(map(len, self.secrets), default=1)

    def feed(self, data, final=False):
        self.pending += data
        output = bytearray()
        while self.pending and (final or len(self.pending) >= self.width):
            match = next((item for item in self.secrets if self.pending.startswith(item)), None)
            if match:
                output.extend(b'[REDACTED]')
                self.pending = self.pending[len(match):]
            else:
                output.append(self.pending[0])
                self.pending = self.pending[1:]
        return bytes(output)


def sensitive_environment_values(env):
    return [value for name, value in env.items()
            if name == 'SEC_USER_AGENT' or SENSITIVE_ENVIRONMENT_NAME.search(name)]


def supervise(contract, auth, check, *, transition=lambda stage: None):
    result = dict(contract=contract.record(), authorization=auth, child_started=False,
                  child_returncode=None, wrapper_exit_code=None, error_type=None,
                  signals=[], stage='preflight')
    child = None
    received = []
    consumed = False
    handlers = {}
    stop_deadline = None
    receipt = contract.record()['prepared']

    def handle(number, frame):
        nonlocal stop_deadline
        received.append(number)
        if stop_deadline is None:
            stop_deadline = time.monotonic() + STOP_GRACE_SECONDS
        result['signals'].append({'number': number, 'received_at': now()})
        if child is not None and child.poll() is None:
            event = result['signals'][-1]
            event['forward_attempted_at'] = now()
            try:
                os.killpg(child.pid, number)
                event['forwarded_at'] = now()
            except OSError as exc:
                event['error_type'] = type(exc).__name__

    def latch_pending():
        for number in signal.sigpending() & {signal.SIGINT, signal.SIGTERM}:
            if number not in received:
                handle(number, None)

    for number in (signal.SIGINT, signal.SIGTERM):
        handlers[number] = signal.signal(number, handle)
    try:
        absent_artifacts(contract.root, receipt)
        result['preflight'] = check(contract)
        transition('after_preflight')
        absent_artifacts(contract.root, receipt)
        verify_prepared(contract.root, receipt)
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
        try:
            latch_pending()
            if received:
                return 128 + received[0]
            result['stage'] = 'consumption'
            try:
                write_once(contract.root / 'consumed.json',
                           dict(result, consumed_at=now(), status='consumed'))
            finally:
                consumed = exists(contract.root / 'consumed.json')
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        transition('after_marker')
        if received:
            return 128 + received[0]
        fd = os.open(contract.root / 'console.log',
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as log:
            transition('before_spawn')
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
            try:
                latch_pending()
                if not received:
                    child = subprocess.Popen(
                        contract.argv, cwd=contract.cwd, env=contract.env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    result['child_started'] = True
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            if child is not None:
                latch_pending()
                for number in set(received):
                    if child.poll() is None:
                        os.killpg(child.pid, number)
                redactor = Redactor(sensitive_environment_values(contract.env))
                with child.stdout, selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    while selector.get_map() or child.poll() is None:
                        if stop_deadline is not None and time.monotonic() >= stop_deadline:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                                result['forced_kill_at'] = now()
                            except ProcessLookupError:
                                pass
                            stop_deadline = None
                        for key, _ in selector.select(timeout=0.1):
                            data = os.read(key.fd, 65536)
                            if data:
                                log.write(redactor.feed(data))
                            else:
                                selector.unregister(key.fileobj)
                    log.write(redactor.feed(b'', final=True))
                result['child_returncode'] = child.wait()
                log.flush()
                os.fsync(log.fileno())
        result['stage'] = 'finished'
        code = result['child_returncode']
        result['wrapper_exit_code'] = 128 - code if code is not None and code < 0 else code
    except BaseException as exc:
        result['error_type'] = type(exc).__name__
        result['failure'] = safe_failure(exc)
        result['wrapper_exit_code'] = 1
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
        try:
            latch_pending()
            if received:
                result.update(wrapper_exit_code=128 + received[0], error_type='Interrupted',
                              stage='terminated' if child else 'terminated_prelaunch')
            if consumed:
                result['finished_at'] = now()
                result['signal_observation_closed_at'] = now()
                write_once(contract.root / 'outcome.json', result)
        finally:
            for number in handlers:
                signal.signal(number, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            for number, previous in handlers.items():
                signal.signal(number, previous)
    if not consumed and result['error_type']:
        print(json.dumps(result['failure']))
    return result['wrapper_exit_code'] if result['wrapper_exit_code'] is not None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--repository', type=Path, required=True)
    prepare_parser.add_argument('--head', required=True)
    for operation in ('preflight', 'execute'):
        current = subparsers.add_parser(operation)
        current.add_argument('--env-file', type=Path)
    subparsers.add_parser('_probe')
    args = parser.parse_args(argv)
    try:
        if args.operation == 'prepare':
            print(json.dumps(prepare(args.repository, args.head)))
            return 0
        if args.operation == '_probe':
            receipt, _ = read_record(ROOT / 'prepared.json')
            registration(ROOT, receipt)
            print(json.dumps(probe(ROOT)))
            return 0
        with attempt_lock(ROOT):
            receipt, _ = read_record(ROOT / 'prepared.json')
            absent_artifacts(ROOT, receipt)
            approval = registration(ROOT, receipt)
            contract = freeze(ROOT, receipt, args.env_file)
            if args.operation == 'preflight':
                print(json.dumps({'contract': contract.record(), 'preflight': preflight(contract)}))
                return 0
            from scripts.operations import semantic_v7_environment as environment

            auth = authorization(contract, approval, environment.remote_attestation)
            return supervise(contract, auth, preflight)
    except Exception as exc:
        print(json.dumps(safe_failure(exc)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Revision-2 v7 controller. No revision-2 approval or execution authority exists yet."""
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
import selectors
import signal
import stat
import subprocess
import sys
import time
from types import MappingProxyType

if __package__:
    from . import semantic_v7_snapshot as snapshot
else:
    import semantic_v7_snapshot as snapshot

ROOT = snapshot.ROOT
VERSION = '2'
AUTHORIZATION_ID = 'SEMANTIC-V2-FRESH-V7-R2'
APPROVAL = Path('docs/evals/semantic_answer_v2_fresh_v7_r2_approval.json')
CONTROLLER = Path('scripts/operations/run_semantic_v7.py')
LAUNCHER = Path('scripts/evals/agents/run_semantic_baseline_v2_3.py')
INTEGRATION = Path('docs/evals/semantic_answer_v2_control_v2_fresh_v7_approval.json')
QUALITY = Path('docs/evals/semantic_answer_v2_quality_approval.json')
INDEX = Path('docs/evals/semantic_answer_v2_fresh_v7_qdrant_candidate.json')
INTERPRETER = Path('/Users/shicheny/miniforge3/envs/finsearch-arm/bin/python')
BOUND_FILES = (CONTROLLER, Path('scripts/operations/semantic_v7_snapshot.py'),
               Path('scripts/operations/semantic_v7_environment.py'), LAUNCHER,
               INTEGRATION, QUALITY, INDEX)
REQUIRED_MODULES = ('requests', 'qdrant_client', 'dotenv', 'evals.semantic_dataset_v2',
                    'scripts.evals.agents.run_semantic_baseline_v2_3',
                    'scripts.evals.agents.semantic_workload_control_v2',
                    'scripts.diagnostics.run_workload_control_v2_calibration',
                    'scripts.diagnostics.observe_semantic_workload',
                    'agents.planner.interactive_target_resolution',
                    'agents.orchestrator.agent_orchestrator',
                    'agents.retrieval.mcp_client', 'mcp_server.server')
UNBLOCK_EXEC = (
    'import os,signal,sys; '
    'signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM}); '
    'os.execv(sys.executable,[sys.executable,"-u",*sys.argv[1:]])'
)
STOP_GRACE_SECONDS = 30
SAFE_FAILURE_REASONS = frozenset({
    'Revision-2 candidate inactive: approval absent', 'Revision-2 approval mismatch',
    'Execution authorization mismatch', 'Required credentials absent', 'Fixture mode forbidden',
    'Sealed runtime preflight failed', 'Preflight execution identity differs',
    'Source filesystem is not read-only', 'Mounted source identity differs',
    'Image digest differs', 'Pinned interpreter required',
    'Controller must execute from sealed source', 'Controller cwd must be sealed source',
    'PYTHONPATH must resolve inside sealed source', 'Runtime override environment is forbidden',
    'Unsealed interpreter import path', 'Repository module escaped sealed source',
    'Attempt artifacts already exist; no retry permitted',
    'Attempt cache is not empty; no retry permitted', 'Retired v6 identity differs',
})


def safe_failure(exc):
    reason = str(exc)
    return {'status': 'failed', 'error_type': type(exc).__name__,
            'reason': reason if reason in SAFE_FAILURE_REASONS else 'Validation or operation failed'}


def sha(path):
    return snapshot.digest(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def exists(path):
    return os.path.lexists(path)


def absent_artifacts(root):
    if any(exists(root / name) for name in
           ('consumed.json', 'outcome.json', 'console.log', 'staging')):
        raise ValueError('Attempt artifacts already exist; no retry permitted')
    if any((root / 'cache').iterdir()):
        raise ValueError('Attempt cache is not empty; no retry permitted')


@contextmanager
def attempt_lock(root):
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700 or root.resolve() != root):
        raise ValueError('Owned external mode-0700 attempt directory required')
    fd = os.open(root / 'attempt.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
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
    # Popen inherits the parent's spawn mask. Unblock in the new process before
    # execing the file-path launcher, without thread-unsafe preexec_fn callbacks.
    return (str(interpreter), '-u', '-c', UNBLOCK_EXEC, str(cwd / LAUNCHER), '--approval', str(QUALITY),
            '--integration-approval', str(INTEGRATION), '--workload-control-v2',
            'B_CONSECUTIVE_10', '--out-root', str(root / 'staging'),
            '--index-attestation', str(INDEX))


def validate_paths(env, cwd, prefix):
    # Preserve PYTHONPATH bytes, but refuse redirection into another checkout.
    if any(k.startswith('GIT_') or k in ('PYTHONHOME', 'PYTHONUSERBASE',
                                       'PYTHONSTARTUP', 'PYTHONPYCACHEPREFIX')
           or k.startswith('DYLD_') for k in env):
        raise ValueError('Runtime override environment is forbidden')
    for item in env.get('PYTHONPATH', '').split(os.pathsep):
        path = Path(item)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(cwd) or path.is_relative_to(cwd / '.cache'):
            raise ValueError('PYTHONPATH must resolve inside sealed source')
    for item in sys.path:
        path = Path(item or os.getcwd()).resolve()
        if not (path.is_relative_to(cwd) or path.is_relative_to(prefix)):
            raise ValueError('Unsealed interpreter import path')


def registration(root, receipt):
    cwd = root / 'source'
    snapshot.verify_mount(root, receipt)
    if Path(__file__).resolve() != cwd / CONTROLLER:
        raise ValueError('Controller must execute from sealed source')
    if Path.cwd() != cwd:
        raise ValueError('Controller cwd must be sealed source')
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError('Pinned interpreter required')
    validate_paths(os.environ, cwd, Path(sys.prefix).resolve())
    if not (cwd / APPROVAL).is_file():
        raise ValueError('Revision-2 candidate inactive: approval absent')
    approval = snapshot.strict_json((cwd / APPROVAL).read_bytes())
    if (approval.get('status') != 'approved_candidate_not_execution'
            or approval.get('execution_contract_version') != VERSION
            or approval.get('effective_environment_contract_version') != '2'
            or approval.get('preflight_implementation_sha256') != sha(cwd / CONTROLLER)
            or approval.get('authorization_id') != AUTHORIZATION_ID
            or approval.get('artifact_root') != str(root)
            or approval.get('interpreter_sha256') != sha(INTERPRETER)
            or approval.get('interpreter_path') != str(INTERPRETER)
            or approval.get('review_status') != 'completed_clean'
            or approval.get('files_sha256') != {str(p): sha(cwd / p) for p in BOUND_FILES}):
        raise ValueError('Revision-2 approval mismatch')
    reviewed = approval.get('reviewed_commit', '')
    if not snapshot.re.fullmatch('[0-9a-f]{40}', reviewed):
        raise ValueError('Full candidate review commit required')
    snapshot.git(cwd, 'merge-base', '--is-ancestor', reviewed, receipt['head'])
    if snapshot.git(cwd, 'diff', reviewed, receipt['head'], '--', 'src', 'scripts', 'data'):
        raise ValueError('Source changed after candidate review')
    return approval


def freeze(root, receipt, env_file):
    from scripts.operations import semantic_v7_environment as environment
    effective, provenance = environment.freeze_effective_child_environment(env_file)
    cwd = root / 'source'
    validate_paths(effective, cwd, Path(sys.prefix).resolve())
    argv = child_argv(root, cwd)
    metadata = dict(version=VERSION, authorization_id=AUTHORIZATION_ID,
                    prepared=receipt, approval_sha256=sha(cwd / APPROVAL),
                    cwd=str(cwd), interpreter=str(INTERPRETER),
                    argv_sha256=snapshot.digest(b'\0'.join(os.fsencode(x) for x in argv)),
                    environment=provenance)
    return Contract(root, cwd, str(INTERPRETER), argv, effective,
                    json.dumps(metadata, sort_keys=True).encode())


def authorization(contract, approval, remote):
    record, record_sha = snapshot.read_record(contract.root / 'execution_authorization.json')
    expected = {'version', 'authorization_id', 'status', 'explicit_user_authorization',
                'max_invocations', 'approval_sha256', 'reviewed_commit', 'image_sha256',
                'artifact_root', 'review_status', 'pull_request', 'review_comment_id',
                'review_url', 'review_body_sha256'}
    metadata = contract.record()
    if (set(record) != expected or record['version'] != VERSION
            or record['authorization_id'] != AUTHORIZATION_ID
            or record['status'] != 'authorized_for_one_invocation'
            or record['explicit_user_authorization'] is not True
            or type(record['max_invocations']) is not int or record['max_invocations'] != 1
            or record['approval_sha256'] != metadata['approval_sha256']
            or record['reviewed_commit'] != metadata['prepared']['head']
            or record['image_sha256'] != metadata['prepared']['image_sha256']
            or record['artifact_root'] != str(contract.root)
            or record['review_status'] != 'completed_clean'
            or record['pull_request'] != approval['pull_request']):
        raise ValueError('Execution authorization mismatch')
    remote(record, 'execution')
    return {'record_sha256': record_sha, 'reviewed_commit': record['reviewed_commit'],
            'review_comment_id': record['review_comment_id']}


def probe(root):
    """Separate process: imports and read-only provenance checks, no case execution."""
    cwd = root / 'source'
    sys.path[0] = str(cwd / LAUNCHER.parent)
    validate_paths(os.environ, cwd, Path(sys.prefix).resolve())
    from scripts.operations import semantic_v7_environment as environment
    if os.environ.get('SEC_METRIC_FIXTURE_ROOT'):
        raise ValueError('Fixture mode forbidden')
    if not os.environ.get('SEC_USER_AGENT', '').strip() or not any(
            os.environ.get(k, '').strip() for k in ('DASHSCOPE_API_KEY', 'QWEN3_RERANK_API_KEY')):
        raise ValueError('Required credentials absent')
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and name.split('.')[0] in ('agents', 'evals', 'mcp_server', 'scripts'):
            if not Path(filename).resolve().is_relative_to(cwd):
                raise ValueError('Repository module escaped sealed source')
    approval = snapshot.strict_json((cwd / APPROVAL).read_bytes())
    environment.remote_attestation(approval, 'candidate')
    from scripts.evals.agents import run_semantic_baseline_v2_3 as launcher
    args = argparse.Namespace(workload_control_v2='B_CONSECUTIVE_10',
                              integration_approval=INTEGRATION, approval=QUALITY,
                              index_attestation=INDEX)
    launcher.verify_opt_in(args)
    launcher.verify_frozen_launcher_v7(QUALITY)
    retirement = snapshot.strict_json(launcher.RETIREMENT.read_bytes())
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
    snapshot.verify_mount(contract.root, contract.record()['prepared'])
    try:
        result = subprocess.run([contract.interpreter, '-u', str(contract.cwd / CONTROLLER),
                                 '_probe'], cwd=contract.cwd, env=contract.env,
                                capture_output=True, timeout=180, check=True)
        record = snapshot.strict_json(result.stdout)
    except subprocess.CalledProcessError as exc:
        try:
            failure = snapshot.strict_json(exc.stdout)
            reason = failure.get('reason')
        except (ValueError, TypeError):
            reason = None
        if reason in SAFE_FAILURE_REASONS:
            raise RuntimeError(reason) from None
        raise RuntimeError('Sealed runtime preflight failed') from None
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RuntimeError('Sealed runtime preflight failed') from None
    if (record.get('ok') is not True or record.get('interpreter') != contract.interpreter
            or record.get('cwd') != str(contract.cwd)
            or record.get('sys_path_0') != str(contract.cwd / LAUNCHER.parent)):
        raise ValueError('Preflight execution identity differs')
    return record


class Redactor:
    def __init__(self, secrets):
        self.secrets = sorted({s.encode() for s in secrets if s}, key=len, reverse=True)
        self.pending = b''
        self.width = max(map(len, self.secrets), default=1)

    def feed(self, data, final=False):
        self.pending += data
        out = bytearray()
        while self.pending and (final or len(self.pending) >= self.width):
            match = next((s for s in self.secrets if self.pending.startswith(s)), None)
            if match:
                out.extend(b'[REDACTED]')
                self.pending = self.pending[len(match):]
            else:
                out.append(self.pending[0])
                self.pending = self.pending[1:]
        return bytes(out)


def supervise(contract, auth, check, *, transition=lambda stage: None):
    """Caller holds attempt lock. Tests inject transitions, never CLI options."""
    result = dict(contract=contract.record(), authorization=auth, child_started=False,
                  child_returncode=None, wrapper_exit_code=None, error_type=None,
                  signals=[], stage='preflight')
    child = None
    received = []
    consumed = False
    handlers = {}
    stop_deadline = None

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
        absent_artifacts(contract.root)
        result['preflight'] = check(contract)
        transition('after_preflight')
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
        try:
            latch_pending()
            if received:
                return 128 + received[0]
            result['stage'] = 'consumption'
            try:
                snapshot.write_once(contract.root / 'consumed.json',
                                    dict(result, consumed_at=now(), status='consumed'))
            finally:
                consumed = exists(contract.root / 'consumed.json')
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        transition('after_marker')
        if received:
            return 128 + received[0]
        fd = os.open(contract.root / 'console.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as log:
            transition('before_spawn')
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
            try:
                latch_pending()
                if not received:
                    child = subprocess.Popen(contract.argv, cwd=contract.cwd, env=contract.env,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    result['child_started'] = True
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            if child is not None:
                # Masked signals can arrive during Popen; deliver them once PID is known.
                latch_pending()
                for number in set(received):
                    if child.poll() is None:
                        os.killpg(child.pid, number)
                secrets = [v for k, v in contract.env.items()
                           if k not in ('PATH', 'PYTHONPATH', 'HOME', 'LANG', 'SHELL')]
                redactor = Redactor(secrets)
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
                snapshot.write_once(contract.root / 'outcome.json', result)
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
    parser.add_argument('operation', choices=('preflight', 'execute', '_probe'))
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == '_probe':
            # Internal non-consuming entrypoint; still refuses mutable source execution.
            receipt, _ = snapshot.read_record(ROOT / 'prepared.json')
            registration(ROOT, receipt)
            print(json.dumps(probe(ROOT)))
            return 0
        with attempt_lock(ROOT):
            absent_artifacts(ROOT)
            receipt, _ = snapshot.read_record(ROOT / 'prepared.json')
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

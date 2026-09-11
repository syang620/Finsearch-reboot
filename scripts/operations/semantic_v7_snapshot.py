"""Standard-library-only sealed source preparation. This module grants no authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import stat
import subprocess
import sys

ROOT = Path('/Users/shicheny/.local/share/finsearch/semantic-baseline/fresh-v7-r2')
VERSION = '2'


def digest(data):
    return hashlib.sha256(data).hexdigest()


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
    """Descriptor metadata, parse and hash all refer to the same opened bytes."""
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
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def command(argv, cwd=None, *, data=None, timeout=180):
    # Preparation/provenance commands cannot inherit Git config or object overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    try:
        return subprocess.run(argv, cwd=cwd, input=data, capture_output=True,
                              check=True, env=env, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError('Sealed-source command failed') from None


def git(root, *args, data=None):
    return command(['/usr/bin/git', '--no-replace-objects', '-c',
                    'core.hooksPath=/dev/null', *args], root, data=data)


def verify_tree(root, head, cache):
    root = Path(root)
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('Full commit required')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('Snapshot HEAD differs')
    if not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise ValueError('Independent Git directory required')
    for name in ('objects/info/alternates', 'info/grafts', 'shallow'):
        if (root / '.git' / name).exists():
            raise ValueError('External or incomplete Git history forbidden')
    if git(root, 'for-each-ref', 'refs/replace').strip():
        raise ValueError('Git replacement objects forbidden')
    entries = []
    expected = set()
    for entry in git(root, 'ls-tree', '-rz', head).split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        path = os.fsdecode(raw_path)
        if mode not in ('100644', '100755') or kind != 'blob':
            raise ValueError('Only regular tracked files are supported')
        target = root / path
        if target.is_symlink() or not target.is_file():
            raise ValueError('Tracked file type differs')
        content = target.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if blob != oid or bool(target.stat().st_mode & 0o111) != (mode == '100755'):
            raise ValueError('Tracked content or mode differs')
        entries.append((path, mode, digest(content)))
        expected.add(path)
    for directory, dirs, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            dirs[:] = [d for d in dirs if d not in ('.git', '.cache')]
        for name in dirs + files:
            target = Path(directory) / name
            relative = target.relative_to(root).as_posix()
            if relative in ('.git', '.cache'):
                continue
            if target.is_symlink() or (target.is_file() and relative not in expected):
                raise ValueError('Unexpected snapshot content')
    if not (root / '.cache').is_symlink() or os.readlink(root / '.cache') != str(cache):
        raise ValueError('External cache binding differs')
    if (root / '.git/info/exclude').read_bytes() != b'/.cache\n':
        raise ValueError('Snapshot cache exclusion differs')
    if git(root, 'status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('Snapshot must be a clean checkout')
    return {'head': head, 'tree': git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
            'files_sha256': digest(json.dumps(sorted(entries)).encode())}


def image_sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('Owned image required')
        if info.st_mode & 0o222 or not info.st_flags & stat.UF_IMMUTABLE:
            raise ValueError('Backing image must be immutable and read-only')
        result = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def verify_mount(root, receipt):
    if receipt.get('version') != VERSION:
        raise ValueError('Prepared source contract version differs')
    mount = root / 'source'
    if not os.statvfs(mount).f_flag & os.ST_RDONLY:
        raise ValueError('Source filesystem is not read-only')
    info = plistlib.loads(command(['/usr/bin/hdiutil', 'info', '-plist']))
    matches = [item for item in info.get('images', [])
               if any(e.get('mount-point') == str(mount)
                      for e in item.get('system-entities', []))]
    if len(matches) != 1 or Path(matches[0]['image-path']).resolve() != root / 'source.dmg':
        raise ValueError('Mounted image identity differs')
    if image_sha(root / 'source.dmg') != receipt['image_sha256']:
        raise ValueError('Image digest differs')
    identity = verify_tree(mount, receipt['head'], root / 'cache')
    if any(receipt.get(k) != v for k, v in identity.items()):
        raise ValueError('Mounted source identity differs')
    return identity


def prepare(repository, head, root=ROOT):
    if sys.platform != 'darwin':
        raise RuntimeError('Native macOS required')
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('Full commit required')
    root = Path(root).absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    build = root / 'build'
    build.mkdir(mode=0o700)
    (root / 'cache').mkdir(mode=0o700)
    (root / 'source').mkdir(mode=0o700)
    template = root / 'empty-template'
    template.mkdir(mode=0o700)
    git(build, 'init', '--quiet', '--template=' + str(template))
    git(build, 'fetch', '--no-tags', str(Path(repository).resolve()), head)
    git(build, 'checkout', '--detach', head)
    # A directory-only .cache/ ignore pattern does not ignore a symlink. Keep
    # the tracked .gitignore frozen and bind this one metadata exclusion instead.
    (build / '.git/info').mkdir(exist_ok=True)
    (build / '.git/info/exclude').write_bytes(b'/.cache\n')
    (build / '.cache').symlink_to(root / 'cache', target_is_directory=True)
    identity = verify_tree(build, head, root / 'cache')
    command(['/usr/bin/hdiutil', 'create', '-srcfolder', str(build), '-format',
             'UDRO', '-fs', 'Case-sensitive HFS+', '-nospotlight', str(root / 'source.dmg')])
    (root / 'source.dmg').chmod(0o400)
    os.chflags(root / 'source.dmg', stat.UF_IMMUTABLE)
    receipt = dict(identity, version=VERSION, image_sha256=image_sha(root / 'source.dmg'))
    command(['/usr/bin/hdiutil', 'attach', str(root / 'source.dmg'), '-readonly',
             '-nobrowse', '-noautoopen', '-mountpoint', str(root / 'source'), '-plist'])
    verify_mount(root, receipt)
    write_once(root / 'prepared.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare'])
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--head', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.repository, args.head)))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

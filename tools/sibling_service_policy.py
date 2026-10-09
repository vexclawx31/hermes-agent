"""Protected, target-aware admission for managed sibling service helpers."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

POLICY_PATH = '/Library/Hermes/managed/sibling-services.json'
PATTERN = re.compile(r'(?:system|gui/(?:0|[1-9][0-9]*))/[A-Za-z0-9_.-]+')
ENV = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LANG': 'C'}


def validate_policy(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'profile', 'own_target', 'helper', 'targets'}:
        raise ValueError('invalid policy shape')
    if type(value['schema']) is not int or value['schema'] != 2:
        raise ValueError('invalid schema')
    if not isinstance(value['profile'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', value['profile']):
        raise ValueError('invalid profile')
    own, targets, helper = value['own_target'], value['targets'], value['helper']
    if not isinstance(own, str) or not PATTERN.fullmatch(own):
        raise ValueError('invalid hosting target')
    if not isinstance(helper, str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+', helper) or '..' in Path(helper).parts:
        raise ValueError('invalid helper')
    if not isinstance(targets, dict) or not targets:
        raise ValueError('empty targets')
    for name, target in targets.items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
            raise ValueError('invalid name')
        if not isinstance(target, str) or not PATTERN.fullmatch(target):
            raise ValueError('invalid target')
        if target.rsplit('/', 1)[-1] == own.rsplit('/', 1)[-1]:
            raise ValueError('hosting label included')
    if len(set(targets.values())) != len(targets):
        raise ValueError('duplicate target')
    return value


def load_policy(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute fixed policy path required')
    directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory);directory=child
            st=os.fstat(directory)
            if st.st_uid != 0 or st.st_mode & 0o022:
                raise ValueError('unsafe policy ancestor')
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
    finally:
        os.close(directory)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1 or st.st_mode & 0o222 or st.st_size > 65536:
            raise ValueError('unsafe policy file')
        with os.fdopen(fd,'r',closefd=False) as f:
            return validate_policy(json.load(f))
    finally:
        os.close(fd)

def _protected_helper(path):
    path = Path(path)
    for parent in reversed(path.parents):
        st = parent.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
            raise ValueError('unsafe helper ancestor')
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1 or st.st_mode & 0o022 or not st.st_mode & 0o111:
        raise ValueError('unsafe helper')


def _request(policy, name, operation):
    if name not in policy['targets'] or operation not in {'status', 'restart'}:
        raise ValueError('unsupported request')
    return ['/usr/bin/sudo', '-n', policy['helper'], operation, name]


def canonical_request(policy_path, *, runtime_profile, runtime_service, name, operation):
    policy = load_policy(policy_path)
    if runtime_profile != policy['profile'] or runtime_service != policy['own_target']:
        raise ValueError('runtime identity mismatch')
    return _request(policy, name, operation)


def is_candidate(command):
    # Generic absolute sudo helper syntax; fleet paths remain in protected data.
    return sys.platform == 'darwin' and bool(re.fullmatch(
        r'/usr/bin/sudo -n /[A-Za-z0-9_./-]+ (?:status|restart) [a-z][a-z0-9_-]*', command))


def _hosting_identity(own):
    if own.startswith('gui/') and own.split('/')[1] != str(os.getuid()):
        return False
    result = subprocess.run(['/bin/launchctl', 'print', own], shell=False,
                            capture_output=True, text=True, timeout=3, cwd='/', env=ENV)
    if result.returncode:
        return False
    # launchctl's top-level properties have exactly one tab of indentation.
    pids = re.findall(r'^\tpid = ([1-9][0-9]*)$', result.stdout, re.MULTILINE)
    if len(pids) != 1:
        return False
    import psutil
    job = int(pids[0])
    pid = os.getpid()
    seen = set()
    for _ in range(8):
        if pid in seen or pid <= 1:
            return False
        seen.add(pid)
        parent = psutil.Process(pid).ppid()
        if pid == job:
            return parent == 1
        pid = parent
    return False


def admit_terminal_command(command):
    if not is_candidate(command):
        return False
    try:
        policy = load_policy(POLICY_PATH)
        argv = next((_request(policy, name, op)
                     for name in policy['targets'] for op in ('status', 'restart')
                     if command == ' '.join(_request(policy, name, op))), None)
        if argv is None:
            return False
        from hermes_cli.profiles import get_active_profile_name
        from gateway.status import get_running_pid
        if get_active_profile_name() != policy['profile'] or get_running_pid(cleanup_stale=False) != os.getpid():
            return False
        _protected_helper(policy['helper'])
        if not _hosting_identity(policy['own_target']):
            return False
        return argv
    except Exception:
        return False

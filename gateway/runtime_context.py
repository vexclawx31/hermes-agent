"""Write-only run authority, adapted from Acelro Engineering Agent's PR #81976.

Only the terminal foreground sink consumes the environment. Never merge this into
profile secrets or the generic child environment factory (browser/MCP/helpers).
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import os

PROTOCOL = 'trusted-local-foreground-v1'
ROUTE = '/v1/trusted-local-runs'


def environment_capability():
    from hermes_cli.config import load_config
    from tools.terminal_tool import _get_env_config
    try:
        validate_environment({'PAPERCLIP_API_KEY': 'capability-probe'}, load_config(), _get_env_config()['env_type'])
    except (ValueError, TypeError, KeyError):
        return {'enabled': False, 'protocol': None}
    return {'enabled': True, 'protocol': PROTOCOL, 'route': ROUTE, 'sessions': 'fresh-only'}

_current = ContextVar('run_environment', default=None)
_terminal = ContextVar('run_environment_terminal', default=False)


def validate_environment(raw, config, backend):
    # Explicit trusted-local opt-in, NOT malicious descendant containment.
    policy = config
    for key in ('gateway', 'api_server'):
        if not isinstance(policy, dict):
            raise ValueError('Invalid run environment policy')
        policy = policy.get(key, {})
    if not isinstance(policy, dict):
        raise ValueError('Invalid run environment policy')
    policy = policy.get('run_environment_allowlist', [])
    def safe(name):
        # Bounded sender contract, not a generic application-env/auth broker.
        # An operator allowlist cannot widen this to loader/runtime controls.
        return isinstance(name, str) and name in {
            'PAPERCLIP_API_KEY', 'PAPERCLIP_API_URL', 'PAPERCLIP_RUN_ID',
            'PAPERCLIP_AGENT_ID', 'PAPERCLIP_COMPANY_ID',
            'PAPERCLIP_TASK_ID', 'PAPERCLIP_WAKE_REASON'}
    if (os.name != 'posix' or backend != 'local' or not isinstance(policy, list) or not policy
            or not all(safe(k) for k in policy)):
        raise ValueError('Run environment is disabled or unsupported')
    if not isinstance(raw, dict) or not raw or len(raw) > 32:
        raise ValueError('Invalid run environment')
    if any(not safe(k) or k not in policy or not isinstance(v, str) or not v
           or len(v.encode('utf-8')) > 8192 or any(ord(c) < 32 or ord(c) == 127 for c in v)
           for k, v in raw.items()):
        raise ValueError('Invalid run environment name or value')
    if sum(len(k.encode()) + len(v.encode()) for k, v in raw.items()) > 32768:
        raise ValueError('Run environment exceeds size limit')
    return RunEnvironment(dict(raw))


@dataclass(repr=False)
class RunEnvironment:
    values: dict = field(repr=False)
    _closed: bool = False
    _processes: list = field(default_factory=list, repr=False)
    _lock: object = field(default_factory=threading.RLock, repr=False)

    def redact(self, value):
        if isinstance(value, str):
            for secret in sorted(self.values.values(), key=len, reverse=True):
                value = value.replace(secret, '[REDACTED]')
            return value
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.redact(v) for v in value)
        return value

    def spawn(self, spawn, kill):
        with self._lock:
            if self._closed:
                raise RuntimeError('Run credential authority has ended')
            proc = spawn(dict(self.values))
            self._processes.append((proc, kill))
            return proc

    def forget_process(self, proc):
        with self._lock:
            self._processes = [(p, kill) for p, kill in self._processes if p is not proc]

    def close(self):
        with self._lock:
            self._closed = True
            processes, self._processes = self._processes, []
        errors = []
        for proc, kill in processes:
            try:
                kill(proc)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('Run subprocess cleanup failed') from errors[0]
        # Keep redaction values until surviving worker contexts are destroyed.
        # Revocation blocks spawns, not cleanup/error redaction.


def current_environment():
    return _current.get()


@contextmanager
def bind_environment(scope):
    token = _current.set(scope)
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def terminal_environment():
    token = _terminal.set(True)
    try:
        yield
    finally:
        _terminal.reset(token)


def terminal_scope():
    return _current.get() if _terminal.get() else None

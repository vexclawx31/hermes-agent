"""Real registry/terminal execution proof for the gated implementation substrate.

This is NOT evidence that HTTP credential admission is safe; that stays disabled.
"""
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import shlex
import sys
import threading

from gateway.runtime_context import bind_environment, validate_environment, current_environment

KEY = 'PAPERCLIP_API_KEY'
CONFIG = {'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}


def test_real_dispatch_concurrent_snapshot_and_redaction(tmp_path, monkeypatch):
    import model_tools
    import tools.terminal_tool as terminal
    from tools.environments.local import LocalEnvironment
    from tools.thread_context import propagate_context_to_thread
    from hermes_logging import _NonFormattingQueueHandler

    # Real initialized LocalEnvironments include a preexisting shell snapshot.
    envs = {name: LocalEnvironment(cwd=str(tmp_path)) for name in ('alpha', 'beta')}
    monkeypatch.setattr(terminal, '_acquire_env', lambda plan, task_id: envs[task_id])
    events = []
    monkeypatch.setattr(model_tools, '_emit_post_tool_call_hook', lambda **kw: events.append(kw))
    # Complete the existing single-profile config-to-env bridge before comparing
    # run execution; scoped credentials must not cause any further global writes.
    terminal._get_env_config()
    before = dict(os.environ)
    barrier = threading.Barrier(2)
    secrets = ['synthetic-dispatch-alpha-12345', 'synthetic-dispatch-beta-67890']
    def run(name, secret):
        scope = validate_environment({KEY: secret}, CONFIG, 'local')
        # Fingerprint proves the exact child value, not merely its presence.
        code = 'import os,hashlib; s=os.environ.get("PAPERCLIP_API_KEY", "missing"); print(hashlib.sha256(s.encode()).hexdigest()); print(s)'
        command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
        with bind_environment(scope):
            barrier.wait(timeout=10)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(propagate_context_to_thread(lambda: model_tools.handle_function_call(
                    'terminal', {'command': command, 'timeout': 10}, task_id=name))).result(timeout=30)
            parsed = json.loads(result)
            assert parsed['exit_code'] == 0, result
            assert hashlib.sha256(secret.encode()).hexdigest() in parsed['output']
            assert secret not in result
            assert '[REDACTED]' in parsed['output']
            for flags in ({'background': True}, {'background': True, 'pty': True}, {'timeout': 601}):
                refused = model_tools.handle_function_call('terminal', {'command': 'true', **flags}, task_id=name)
                assert 'bounded local foreground' in refused
            # Queue preparation must scrub on the producing context, including errors.
            record = logging.LogRecord('test', logging.ERROR, __file__, 1, 'value=%s', (secret,), None)
            record.stack_info = 'stack: ' + secret
            prepared = _NonFormattingQueueHandler(queue.SimpleQueue()).prepare(record)
            scope.close()
        assert current_environment() is None
        assert secret not in prepared.getMessage()
        assert secret not in logging.Formatter().format(prepared)
        later = json.loads(model_tools.handle_function_call('terminal', {'command': command, 'timeout': 10}, task_id=name))
        assert 'missing' in later['output'], later
        assert secret not in open(envs[name]._snapshot_path).read()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, envs, secrets))
        assert dict(os.environ) == before
        assert all(secret not in json.dumps(events) for secret in secrets)
    finally:
        for env in envs.values():
            env.cleanup()


def test_dispatch_transforms_cannot_reintroduce_literal(monkeypatch):
    import model_tools
    secret = 'synthetic-transform-secret-12345'
    scope = validate_environment({KEY: secret}, CONFIG, 'local')
    observed = []
    monkeypatch.setattr(model_tools.registry, 'dispatch', lambda *a, **kw: json.dumps({'output': secret}))
    monkeypatch.setattr(model_tools, '_emit_post_tool_call_hook', lambda **kw: observed.append(kw))
    monkeypatch.setattr(model_tools, '_apply_transform_tool_result_hook', lambda *a: json.dumps({'output': secret}))
    with bind_environment(scope):
        result = model_tools.handle_function_call('terminal', {'command': 'true'})
    assert secret not in result
    assert secret not in json.dumps(observed)

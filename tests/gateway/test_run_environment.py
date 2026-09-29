"""Scoped authority contracts; synthetic credentials and real foreground subprocesses."""
import concurrent.futures
import json
import os
import shlex
import sys
import pytest
from gateway.runtime_context import validate_environment, bind_environment, terminal_environment, current_environment

KEY='PAPERCLIP_API_KEY'
CONFIG={'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}


def test_policy_and_concurrent_real_terminal_workers(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment, build_subprocess_env
    from tools.thread_context import propagate_context_to_thread
    from agent.redact import redact_sensitive_text
    monkeypatch.setattr(LocalEnvironment, 'init_session', lambda self: None)
    before=dict(os.environ)
    def run(secret):
        scope=validate_environment({KEY:secret},CONFIG,'local')
        env=LocalEnvironment(cwd=str(tmp_path))
        # Test child receives value without returning its literal through model output.
        cmd=f'{shlex.quote(sys.executable)} -c '+shlex.quote('import os; print(os.environ.get("'+KEY+'", "missing") == '+repr(secret)+')')
        with bind_environment(scope):
            assert KEY not in build_subprocess_env()
            def tool_worker():
                with terminal_environment():
                    return env.execute(cmd, timeout=10)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result=pool.submit(propagate_context_to_thread(tool_worker)).result()
            assert result['returncode']==0, result
            assert result['output'].strip()=='True'
            assert secret not in redact_sensitive_text(secret, force=True)
            assert not os.path.exists(env._snapshot_path)
            scope.close()
        assert current_environment() is None
        missing=env.execute(f'{shlex.quote(sys.executable)} -c '+shlex.quote('import os; print(os.environ.get("'+KEY+'", "missing"))'),timeout=10)
        assert 'missing' in missing['output']
        env.cleanup()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,['synthetic-authority-alpha-123','synthetic-authority-beta-456']))
    assert dict(os.environ)==before
    for raw, policy, backend in [({KEY:'x'}, {}, 'local'), ({'PATH':'x'},CONFIG,'local'), ({KEY:3},CONFIG,'local'), ({KEY:'a\0b'},CONFIG,'local'), ({KEY:'x'},CONFIG,'ssh')]:
        with pytest.raises(ValueError): validate_environment(raw,policy,backend)


def test_revocation_blocks_worker_and_retains_cleanup_redaction(tmp_path):
    import subprocess
    from tools.environments.local import LocalEnvironment
    scope=validate_environment({KEY:'synthetic-cleanup-token'},CONFIG,'local')
    with bind_environment(scope):
        proc=scope.spawn(lambda values: subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True,env={**os.environ,**values}),lambda p: (p.kill(),p.wait(timeout=5)))
        scope.close()
        assert proc.poll() is not None
        with pytest.raises(RuntimeError): scope.spawn(lambda _: None,lambda _:None)
        assert scope.redact('synthetic-cleanup-token')=='[REDACTED]'
    assert current_environment() is None

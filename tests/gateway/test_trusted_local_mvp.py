"""Synthetic API -> executor -> real registry/foreground process acceptance."""
import asyncio
import hashlib
import json
import logging
import queue
import shlex
import sys
import threading
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from gateway.runtime_context import PROTOCOL, ROUTE, current_environment, RunEnvironment
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app

KEY = 'PAPERCLIP_API_KEY'
AUTH = {'Authorization': 'Bearer synthetic-receiver-key'}


@pytest.mark.asyncio
async def test_api_executor_terminal_concurrent_cancel_and_readback(tmp_path, monkeypatch):
    import model_tools
    import hermes_cli.config
    import tools.terminal_tool as terminal
    from tools.environments.local import LocalEnvironment
    from agent.trajectory import save_trajectory
    from agent.session_persistence import SessionPersistenceMixin
    from hermes_state import SessionDB
    from types import SimpleNamespace
    db = SessionDB(tmp_path / 'proof.db')
    from hermes_logging import _NonFormattingQueueHandler
    config = {'approvals': {'unattended_mode': 'approve'}, 'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}
    monkeypatch.setattr(hermes_cli.config, 'load_config', lambda: config)
    monkeypatch.setattr(hermes_cli.config, 'load_config_readonly', lambda: config)
    terminal._get_env_config()
    envs = {}
    env_lock = threading.Lock()
    def acquire(plan, task_id):
        with env_lock:
            if task_id not in envs:
                envs[task_id] = LocalEnvironment(cwd=str(tmp_path))
            return envs[task_id]
    monkeypatch.setattr(terminal, '_acquire_env', acquire)
    barrier = threading.Barrier(2)
    results, scopes, records = {}, {}, []
    class Agent:
        model = 'deterministic'
        provider = 'test'
        def __init__(self, **kwargs):
            self.session_id = kwargs['session_id']
        def interrupt(self, *args, **kwargs):
            pass
        def run_conversation(self, user_message, task_id, **kwargs):
            scope = current_environment()
            scopes[user_message] = scope
            if user_message in ('alpha', 'beta'):
                barrier.wait(timeout=10)
            code = 'import os,hashlib; s=os.environ.get("PAPERCLIP_API_KEY", "missing"); print(hashlib.sha256(s.encode()).hexdigest()); print(s)'
            if user_message == 'cancel':
                code = 'import time; time.sleep(60)'
            command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
            result = model_tools.handle_function_call('terminal', {'command': command, 'timeout': 90}, task_id=task_id)
            results[user_message] = result
            if scope:
                secret = scope.values[KEY]
                db.create_session(task_id, source='api')
                persist = SimpleNamespace(_session_db=db, _session_db_created=True,
                    _persist_disabled=False, session_id=task_id, _session_persist_lock=None,
                    _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
                    _last_flushed_db_idx=0)
                persist._flush_messages_to_session_db_unlocked = SessionPersistenceMixin._flush_messages_to_session_db_unlocked.__get__(persist)
                assert SessionPersistenceMixin._flush_messages_to_session_db(persist, [{'role': 'assistant', 'content': secret}])
                from agent.agent_runtime_helpers import dump_api_request_debug
                assert dump_api_request_debug(self, {'messages': [secret]}, reason='test') is None
                save_trajectory([{'content': secret}], 'test', True, str(tmp_path / (user_message + '.jsonl')))
                record = logging.LogRecord('test', logging.ERROR, __file__, 1, '%s', (secret,), None)
                record.stack_info = secret
                records.append(_NonFormattingQueueHandler(queue.SimpleQueue()).prepare(record))
            return {'completed': True, 'final_response': result}
    adapter = _make_adapter('synthetic-receiver-key')
    app = _create_runs_app(adapter)
    app.router.add_post(ROUTE, adapter._handle_runs)
    app.router.add_get('/v1/capabilities', adapter._handle_capabilities)
    secrets = {'alpha': 'synthetic-alpha-mvp-12345', 'beta': 'synthetic-beta-mvp-67890', 'cancel': 'synthetic-cancel-mvp-54321'}
    async def wait_for(predicate):
        async with asyncio.timeout(15):
            while not predicate():
                await asyncio.sleep(.01)
    try:
        with patch.object(adapter, '_create_agent', side_effect=lambda **kw: Agent(**kw)):
            async with TestClient(TestServer(app)) as cli:
                async def start(name):
                    response = await cli.post(ROUTE, headers={**AUTH, 'Idempotency-Key': name}, json={'input': name, 'environment': {KEY: secrets[name]}, 'required_environment_capability': PROTOCOL})
                    assert response.status == 202, await response.text()
                    payload = await response.json()
                    assert payload['environment_capability'] == PROTOCOL
                    return payload['run_id']
                runs = await asyncio.gather(start('alpha'), start('beta'))
                await wait_for(lambda: all(r not in adapter._active_run_tasks for r in runs))
                for name, run in zip(('alpha', 'beta'), runs):
                    assert hashlib.sha256(secrets[name].encode()).hexdigest() in results[name]
                    assert all(secret not in results[name] for secret in secrets.values())
                    status = await (await cli.get('/v1/runs/' + run, headers=AUTH)).text()
                    assert all(secret not in status for secret in secrets.values())
                    assert scopes[name]._closed
                    assert secrets[name] not in json.dumps(db.get_messages_as_conversation(run))
                    assert await start(name) == run
                    # Subsequent uncredentialed API turn in the prior session.
                    response = await cli.post('/v1/runs', headers=AUTH, json={'input': 'later-' + name, 'session_id': run})
                    assert response.status == 202
                await wait_for(lambda: 'later-alpha' in results and 'later-beta' in results)
                assert 'missing' in results['later-alpha'] and 'missing' in results['later-beta']
                run = await start('cancel')
                await wait_for(lambda: scopes.get('cancel') and bool(scopes['cancel']._processes))
                proc = scopes['cancel']._processes[0][0]
                response = await cli.post('/v1/runs/' + run + '/stop', headers=AUTH)
                assert response.status == 200
                await wait_for(lambda: scopes['cancel']._closed and proc.poll() is not None)
                await wait_for(lambda: 'cancel' in results)
                assert scopes['cancel']._processes == []
                for extra in ({'session_id': runs[0]}, {'previous_response_id': 'old'}, {'conversation_history': []}):
                    response = await cli.post(ROUTE, headers=AUTH, json={'input': 'refuse', 'environment': {KEY: secrets['alpha']}, 'required_environment_capability': PROTOCOL, **extra})
                    assert response.status == 400
        for path in tmp_path.glob('*.jsonl'):
            assert all(secret not in path.read_text() for secret in secrets.values())
        assert all(secret not in logging.Formatter().format(record) for record in records for secret in secrets.values())
    finally:
        db.close()
        for env in envs.values():
            env.cleanup()


def test_close_drains_all_callbacks_on_error_and_revokes():
    scope = RunEnvironment({KEY: 'synthetic-close-secret'})
    visited = []
    def broken(proc):
        visited.append(proc)
        raise ValueError('cleanup')
    scope.spawn(lambda values: 'first', broken)
    scope.spawn(lambda values: 'second', lambda proc: visited.append(proc))
    with pytest.raises(RuntimeError, match='cleanup failed'):
        scope.close()
    assert visited == ['first', 'second']
    scope.close()
    with pytest.raises(RuntimeError, match='authority has ended'):
        scope.spawn(lambda values: None, lambda proc: None)

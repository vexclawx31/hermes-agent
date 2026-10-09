"""Cleanup failures must not skip lifecycle work or leak approval literals."""
import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.platforms import api_server_runs as runs
from gateway.platforms import api_server
from gateway.runtime_context import RunEnvironment, current_environment
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app, _claim_run
from aiohttp.test_utils import TestClient, TestServer


SECRET = 'synthetic-approval-literal-012345'


def launch(adapter, scope):
    q = asyncio.Queue()
    adapter._run_streams['run_cleanup'] = q
    return runs._RunLaunch(
        owner=adapter, run_id='run_cleanup', queue=q, session_id='session_cleanup',
        gateway_session_key=None, declared_selected=False, user_message='test',
        conversation_history=[], session_history_delivery=True,
        agent_kwargs={'room_dispatch': None}, request_profile=None,
        browser_control_principal=None, browser_control_transport_family=None,
        environment=scope)


def failing_scope():
    scope = RunEnvironment({'PAPERCLIP_API_KEY': SECRET})
    visited = []
    def fail(proc):
        visited.append(proc)
        raise RuntimeError(SECRET)
    scope.spawn(lambda env: 'failed', fail)
    scope.spawn(lambda env: 'other', visited.append)
    return scope, visited


@pytest.mark.asyncio
async def test_cleanup_failure_still_interrupts_reaps_and_unwinds(monkeypatch, caplog):
    from tools import approval
    from tools.approval_context import get_current_session_key
    from tools.environments.base import BaseEnvironment
    scope, visited = failing_scope()
    adapter = _make_adapter()
    run = launch(adapter, scope)
    agent = SimpleNamespace(_api_run_environment=scope)
    monkeypatch.setattr(runs, '_load_owned_run', lambda *a, **kw: (
        run.run_id, {'status': 'running'}, agent, None, None))
    interrupt, reap = Mock(), Mock()
    facade = SimpleNamespace(_openai_error=api_server._openai_error,
        request_hard_interrupt=interrupt, _reap_disconnected_agent_processes=reap)
    response = await runs._handle_stop_run(adapter, None, _api_server=facade)
    assert response.status == 200
    interrupt.assert_called_once()
    reap.assert_called_once()
    assert visited == ['failed', 'other']
    with pytest.raises(RuntimeError, match='authority has ended'):
        scope.spawn(lambda env: None, lambda proc: None)

    scope, visited = failing_scope()
    run.environment = scope
    run.declared_selected = True
    profile = []
    @contextmanager
    def profile_scope(value):
        profile.append('enter')
        try:
            yield
        finally:
            profile.append('exit')
    monkeypatch.setattr(adapter, '_profile_scope', profile_scope)
    monkeypatch.setattr(adapter, '_bind_api_server_session', lambda **kw: None)
    bind = Mock()
    monkeypatch.setattr(adapter, '_bind_declared_conversation', bind)
    register, unregister = Mock(), Mock()
    monkeypatch.setattr(approval, 'register_gateway_notify', register)
    monkeypatch.setattr(approval, 'unregister_gateway_notify', unregister)
    prior = get_current_session_key()
    def conversation(**kw):
        assert current_environment() is scope
        assert get_current_session_key() == run.run_id
        return {'completed': True}
    agent.run_conversation = conversation
    publish, clear = Mock(), Mock()
    facade = SimpleNamespace(_publish_turn_process_ownership=publish,
                             _clear_turn_process_ownership=clear)
    result, _, _ = runs._run_agent_sync(adapter, run, agent, Mock(), _api_server=facade)
    assert result == {'completed': True}
    assert visited == ['failed', 'other']
    clear.assert_called_once_with(agent)
    assert unregister.called and bind.called
    assert profile == ['enter', 'exit']
    assert current_environment() is None
    assert get_current_session_key() == prior

    kill = Mock(side_effect=RuntimeError(SECRET))
    proc = SimpleNamespace(_hermes_scoped=True, pid=123)
    fallback = Mock()
    monkeypatch.setattr('agent.deadline.kill_process_tree', fallback)
    BaseEnvironment._kill_spawned_tree(SimpleNamespace(_kill_process=kill), proc)
    kill.assert_called_once_with(proc)
    fallback.assert_not_called()
    assert 'teardown unconfirmed' in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['completed', 'cancelled'])
async def test_execute_run_cleanup_failure_keeps_terminal_status_and_secret(monkeypatch, caplog, outcome):
    import logging
    caplog.set_level(logging.DEBUG)
    adapter = _make_adapter()
    scope, visited = failing_scope()
    run = launch(adapter, scope)
    persisted = []
    monkeypatch.setattr(adapter, '_set_run_status', lambda run_id, status, **kw: persisted.append(
        json.dumps({'status': status, **kw}, default=str)))
    started = asyncio.Event()

    async def scoped(self, run, *, _api_server):
        assert current_environment() is scope
        if outcome == 'completed':
            self._set_run_status(run.run_id, 'completed', output='ok')
            return
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._set_run_status(run.run_id, 'cancelled')
            raise
    monkeypatch.setattr(runs, '_execute_run_scoped', scoped)
    loop_errors = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: loop_errors.append(ctx))

    task = asyncio.ensure_future(runs._execute_run(adapter, run, _api_server=api_server))
    if outcome == 'completed':
        assert await task is None
    else:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    assert visited == ['failed', 'other']
    with pytest.raises(RuntimeError, match='authority has ended'):
        scope.spawn(lambda env: None, lambda proc: None)
    assert [json.loads(p)['status'] for p in persisted] == [outcome]
    assert 'teardown unconfirmed' in caplog.text
    assert SECRET not in caplog.text
    assert not any(r.exc_info for r in caplog.records)
    assert SECRET not in ''.join(persisted)
    assert not loop_errors
    assert current_environment() is None


@pytest.mark.asyncio
async def test_approval_literal_redacted_before_status_and_sse(monkeypatch):
    adapter = _make_adapter()
    scope = RunEnvironment({'PAPERCLIP_API_KEY': SECRET})
    run = launch(adapter, scope)
    _claim_run(adapter, run.run_id)
    persisted = []
    original = adapter._set_run_status
    def persist(*args, **kwargs):
        assert SECRET not in json.dumps(kwargs)
        persisted.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(adapter, '_set_run_status', persist)
    notify = runs._make_approval_notify(adapter, run, _api_server=api_server)
    event = {'command': 'printf ' + SECRET, 'reason': SECRET,
             'nested': {SECRET: [SECRET]}}
    notify(event)
    await asyncio.sleep(0)
    assert persisted and SECRET in event['reason']
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        status = await client.get('/v1/runs/' + run.run_id)
        assert status.status == 200
        text = await status.text()
        assert SECRET not in text and '[REDACTED]' in text
        adapter._set_run_status(run.run_id, 'completed')
        run.put_event(None)
        stream = await client.get('/v1/runs/' + run.run_id + '/events')
        assert stream.status == 200
        text = await stream.text()
        assert 'approval.request' in text and '[REDACTED]' in text
        assert SECRET not in text
    # A late callback must not publish into transport it no longer owns.
    adapter._run_streams.pop(run.run_id, None)
    notify(event)
    await asyncio.sleep(0)
    assert run.queue.empty()

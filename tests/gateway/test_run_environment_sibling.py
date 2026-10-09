"""Credential runs refuse sibling service control; uncredentialed admission is unchanged.

Synthetic API -> executor -> registry -> terminal_tool. The protected policy, helper and
hosting identity are faked to "would admit" so the only variable is the run credential scope.
"""
import asyncio
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from gateway.runtime_context import (
    PROTOCOL, ROUTE, bind_environment, current_environment, terminal_environment,
    terminal_scope as run_terminal_scope, validate_environment)
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app

KEY = 'PAPERCLIP_API_KEY'
AUTH = {'Authorization': 'Bearer synthetic-receiver-key'}
POLICY = dict(schema=2, profile='fixture', own_target='gui/501/example.host',
              helper='/usr/local/sbin/example-service', targets={'sibling': 'system/example.sibling'})
COMMAND = '/usr/bin/sudo -n /usr/local/sbin/example-service restart sibling'
SECRET = 'synthetic-sibling-refusal-12345'


@pytest.mark.asyncio
async def test_credential_run_refuses_sibling_control_end_to_end(monkeypatch):
    import model_tools
    import hermes_cli.config
    import tools.terminal_tool as terminal
    from tools import sibling_service_policy as policy
    config = {'approvals': {'unattended_mode': 'approve'}, 'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}
    monkeypatch.setattr(hermes_cli.config, 'load_config', lambda: config)
    monkeypatch.setattr(hermes_cli.config, 'load_config_readonly', lambda: config)
    terminal._get_env_config()
    monkeypatch.setattr(terminal, '_acquire_env', lambda plan, task_id: object())
    monkeypatch.setattr(policy.sys, 'platform', 'darwin')
    loader = Mock(return_value=POLICY.copy())
    monkeypatch.setattr(policy, 'load_policy', loader)
    monkeypatch.setattr(policy, '_protected_helper', lambda path: None)
    monkeypatch.setattr(policy, '_hosting_identity', lambda own: True)
    monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: 'fixture')
    monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: os.getpid())
    approval = Mock(return_value=terminal._ApprovalVerdict())
    monkeypatch.setattr(terminal, '_run_approval_guards', approval)
    sink = Mock(return_value=SimpleNamespace(stdout='ok', stderr='', returncode=0))
    monkeypatch.setattr(subprocess, 'run', sink)
    results, scopes = {}, {}

    class Agent:
        model = 'deterministic'
        provider = 'test'
        def __init__(self, **kwargs):
            self.session_id = kwargs['session_id']
        def interrupt(self, *args, **kwargs):
            pass
        def run_conversation(self, user_message, task_id, **kwargs):
            scopes[user_message] = current_environment()
            results[user_message] = model_tools.handle_function_call(
                'terminal', {'command': COMMAND, 'timeout': 30}, task_id=task_id)
            return {'completed': True, 'final_response': results[user_message]}

    adapter = _make_adapter('synthetic-receiver-key')
    app = _create_runs_app(adapter)
    app.router.add_post(ROUTE, adapter._handle_runs)

    async def wait_for(predicate):
        async with asyncio.timeout(15):
            while not predicate():
                await asyncio.sleep(.01)

    with patch.object(adapter, '_create_agent', side_effect=lambda **kw: Agent(**kw)):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ROUTE, headers=AUTH, json={
                'input': 'scoped', 'environment': {KEY: SECRET}, 'required_environment_capability': PROTOCOL})
            assert response.status == 202, await response.text()
            await wait_for(lambda: 'scoped' in results)
            refused = json.loads(results['scoped'])
            assert scopes['scoped'] is not None
            assert refused['status'] == 'blocked'
            assert 'credential runs' in refused['error']
            assert 'service_control' not in refused
            # Refusal precedes protected-policy admission, approval and the helper sink.
            loader.assert_not_called()
            approval.assert_not_called()
            sink.assert_not_called()

            # Same command, no credential scope: sibling admission still runs the helper.
            response = await cli.post('/v1/runs', headers=AUTH, json={'input': 'plain'})
            assert response.status == 202, await response.text()
            await wait_for(lambda: 'plain' in results)
            admitted = json.loads(results['plain'])
            assert scopes['plain'] is None
            assert admitted['service_control'] and not admitted['health_verified']
            loader.assert_called_once()
            sink.assert_called_once()
            assert sink.call_args.args[0] == ['/usr/bin/sudo', '-n', POLICY['helper'], 'restart', 'sibling']
            assert sink.call_args.kwargs['env'] == policy.ENV
    assert SECRET not in json.dumps(results)


def test_profile_terminal_scope_and_run_scope_bind_independently():
    from tools.terminal_scope import get_terminal_scope, reset_terminal_scope, set_terminal_scope
    profile = {'TERMINAL_ENV': 'local'}
    scope = validate_environment({KEY: SECRET}, {'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}, 'local')
    token = set_terminal_scope(profile)
    try:
        with bind_environment(scope):
            # Profile TERMINAL_* policy is visible; run credentials reach only the terminal sink.
            assert get_terminal_scope() is profile
            assert run_terminal_scope() is None
            with terminal_environment():
                assert run_terminal_scope() is scope
                assert get_terminal_scope() is profile
                assert KEY not in get_terminal_scope()
            assert run_terminal_scope() is None
        assert current_environment() is None
        assert get_terminal_scope() is profile
    finally:
        reset_terminal_scope(token)
        scope.close()
    assert get_terminal_scope() is None

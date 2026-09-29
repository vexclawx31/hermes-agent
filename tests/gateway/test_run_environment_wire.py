"""Fail-closed wire admission; synthetic tokens, no production network."""
from unittest.mock import AsyncMock, patch
import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app

KEY = 'PAPERCLIP_API_KEY'
AUTH = {'Authorization': 'Bearer synthetic-receiver-key'}


@pytest.mark.asyncio
async def test_environment_gate_precedes_every_session_admission(monkeypatch):
    import hermes_cli.config
    import tools.terminal_tool
    config = {'gateway': {'api_server': {'run_environment_allowlist': [KEY]}}}
    monkeypatch.setattr(hermes_cli.config, 'load_config', lambda: config)
    monkeypatch.setattr(tools.terminal_tool, '_get_env_config', lambda: {'env_type': 'local'})
    adapter = _make_adapter('synthetic-receiver-key')
    app = _create_runs_app(adapter)
    app.router.add_get('/v1/capabilities', adapter._handle_capabilities)
    secret = 'synthetic-wire-secret-123456'
    with patch.object(adapter, '_create_agent') as create, \
         patch.object(adapter, '_ensure_session_db_async', new_callable=AsyncMock) as db, \
         patch.object(adapter, '_admit_to_live_bot_chat', new_callable=AsyncMock) as mailbox:
        async with TestClient(TestServer(app)) as cli:
            capability = await cli.get('/v1/capabilities', headers=AUTH)
            assert capability.status == 200
            assert (await capability.json())['features']['run_environment']['enabled'] is True
            # New sessions, explicit existing/live-owner sessions, response history,
            # and header affinity all fail before a DB/mailbox/task is touched.
            for extra, headers in [({}, AUTH), ({'session_id': 'live-owner'}, AUTH),
                    ({'previous_response_id': 'old-response'}, AUTH),
                    ({}, {**AUTH, 'X-Hermes-Session-Key': 'live-affinity'})]:
                response = await cli.post('/v1/runs', headers=headers, json={
                    'input': 'hello', 'environment': {KEY: secret}, **extra})
                assert response.status == 422, await response.text()
                text = await response.text()
                assert 'run_environment_unsupported' in text
                assert secret not in text
            unauthorized = await cli.post('/v1/runs', json={'input': 'hello', 'environment': {KEY: secret}})
            assert unauthorized.status in (401, 403)
    create.assert_not_called()
    db.assert_not_called()
    mailbox.assert_not_called()
    assert not adapter._active_run_tasks
    assert not adapter._run_owners


@pytest.mark.asyncio
async def test_environment_policy_malformed_and_default_denied(monkeypatch):
    import hermes_cli.config
    import tools.terminal_tool
    config = {}
    monkeypatch.setattr(hermes_cli.config, 'load_config', lambda: config)
    monkeypatch.setattr(tools.terminal_tool, '_get_env_config', lambda: {'env_type': 'local'})
    adapter = _make_adapter('synthetic-receiver-key')
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        for config in ({}, None, [], {'gateway': None}, {'gateway': []},
                       {'gateway': {'api_server': None}},
                       {'gateway': {'api_server': {'run_environment_allowlist': 'PAPERCLIP_API_KEY'}}}):
            response = await cli.post('/v1/runs', headers=AUTH, json={
                'input': 'hello', 'environment': {KEY: 'synthetic-denied-token'}})
            assert response.status == 400, await response.text()
        assert not adapter._active_run_tasks


@pytest.mark.asyncio
async def test_production_profile_middleware_policy_a_b_a(tmp_path, monkeypatch):
    from aiohttp import web
    from gateway.config import GatewayConfig
    from agent import secret_scope
    import hermes_cli.profiles
    homes = {name: tmp_path / name for name in ('allowed', 'denied')}
    for name, home in homes.items():
        home.mkdir()
        (home / '.env').write_text('API_SERVER_KEY=' + name + '-synthetic-key\n')
        policy = '[PAPERCLIP_API_KEY]' if name == 'allowed' else '[]'
        (home / 'config.yaml').write_text('terminal:\n  backend: local\ngateway:\n  api_server:\n    run_environment_allowlist: ' + policy + '\n')
    monkeypatch.setattr(hermes_cli.profiles, 'profiles_to_serve', lambda multiplex: list(homes.items()))
    monkeypatch.setattr(hermes_cli.profiles, 'get_profile_dir', lambda name: homes[name])
    adapter = _make_adapter('default-synthetic-key')
    adapter.gateway_runner = type('Runner', (), {'config': GatewayConfig(multiplex_profiles=True)})()
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post('/p/{profile}/v1/runs', adapter._handle_runs)
    app.router.add_get('/p/{profile}/v1/capabilities', adapter._handle_capabilities)
    secret_scope.set_multiplex_active(True)
    try:
        async with TestClient(TestServer(app)) as cli:
            for name, status in [('allowed', 422), ('denied', 400), ('allowed', 422)]:
                caps = await cli.get('/p/' + name + '/v1/capabilities',
                    headers={'Authorization': 'Bearer ' + name + '-synthetic-key'})
                assert caps.status == 200
                assert (await caps.json())['features']['run_environment']['enabled'] == (name == 'allowed')
                response = await cli.post('/p/' + name + '/v1/runs',
                    headers={'Authorization': 'Bearer ' + name + '-synthetic-key'},
                    json={'input': 'hello', 'environment': {KEY: 'synthetic-profile-token'}})
                assert response.status == status, await response.text()
            wrong = await cli.post('/p/allowed/v1/runs',
                headers={'Authorization': 'Bearer default-synthetic-key'},
                json={'input': 'hello', 'environment': {KEY: 'synthetic-profile-token'}})
            assert wrong.status in (401, 403)
        assert not adapter._active_run_tasks
    finally:
        secret_scope.set_multiplex_active(False)

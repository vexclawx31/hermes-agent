"""``auth: client_credentials`` for remote MCP servers (Linear hosted MCP + existing app grant).

Offline: the token endpoint, the Linear GraphQL identity check and the MCP endpoint are all an
in-process ``MockTransport``. Values below are fakes; the assertions are behaviour contracts:
fresh in-memory mint with the exact scope, fail-closed identity/scope/resource checks, a bounded
re-mint after 401, replay only for side-effect-free requests, and secret-free errors.
"""

import asyncio
import base64
import json

import pytest

from tools import mcp_client_credentials as cc
from tools import mcp_tool as _core

httpx = _core.sdk_httpx()

TOKEN_URL = "https://api.linear.app/oauth/token"
GRAPHQL_URL = "https://api.linear.app/graphql"
MCP_URL = "https://mcp.linear.app/mcp"
SECRET = "fake-client-secret-DO-NOT-LEAK"
SCOPE = "read,write,app:assignable"
VERIFY = {"kind": "linear_graphql", "url": GRAPHQL_URL, "actor_id": "app-user-1",
          "organization_id": "org-aligned", "team_keys": ["WEB2"]}


def _server_config(**overrides):
    block = {"token_url": TOKEN_URL, "client_id": "client-1", "client_secret": SECRET, "scope": SCOPE,
             "verify": VERIFY, **overrides}
    return {"url": MCP_URL, "auth": "client_credentials", "client_credentials": block}


class FakeLinear:
    """Token endpoint + GraphQL + MCP endpoint. Each mint returns a new token."""

    def __init__(self, *, token_status=200, token_body=None, granted_scope=SCOPE, viewer="app-user-1",
                 org="org-aligned", teams=("WEB2",), mcp_401=0):
        self.token_status, self.token_body, self.granted_scope = token_status, token_body, granted_scope
        self.viewer, self.org, self.teams, self.mcp_401 = viewer, org, teams, mcp_401
        self.mints, self.token_requests, self.mcp_requests = 0, [], []

    def __call__(self, request):
        url = str(request.url)
        if url == TOKEN_URL:
            self.token_requests.append(request)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json=self.token_body)
            self.mints += 1
            return httpx.Response(200, json={"access_token": f"tok-{self.mints}", "token_type": "Bearer",
                                             "expires_in": 2591999, "scope": self.granted_scope})
        if url == GRAPHQL_URL:
            return httpx.Response(200, json={"data": {
                "viewer": {"id": self.viewer}, "organization": {"id": self.org},
                "teams": {"nodes": [{"key": k} for k in self.teams]}}})
        assert url == MCP_URL
        self.mcp_requests.append((request.headers.get("authorization"), json.loads(request.content)))
        if self.mcp_401:
            self.mcp_401 -= 1
            return httpx.Response(401, json={"error": "invalid_token"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})


def _provider(fake, clock=None, **cfg_overrides):
    cfg = cc.parse_config("linear_aligned", _server_config(**cfg_overrides))
    kwargs = {"clock": clock} if clock else {}
    return cc.ClientCredentialsTokenProvider("linear_aligned", cfg, httpx_mod=httpx,
                                             transport=httpx.MockTransport(fake), **kwargs)


def _rpc(client, method, **params):
    return client.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def _run_with_auth(fake, provider, *calls):
    async def _go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake), auth=cc.make_auth(provider, httpx)) as c:
            return [await _rpc(c, m, **p) for m, p in calls]
    return asyncio.run(_go())


def test_mints_once_in_memory_with_exact_scope_and_verified_identity():
    fake = FakeLinear()
    provider = _provider(fake)
    first = asyncio.run(provider.token())
    assert asyncio.run(provider.token()) == first and fake.mints == 1  # reused, not re-minted per request
    req = fake.token_requests[0]
    form = dict(x.split("=", 1) for x in req.content.decode().split("&"))
    assert form["grant_type"] == "client_credentials" and form["scope"] == "read%2Cwrite%2Capp%3Aassignable"
    assert req.headers["authorization"] == "Basic " + base64.b64encode(f"client-1:{SECRET}".encode()).decode()
    assert SECRET not in repr(provider.cfg) and first not in repr(provider._token)


@pytest.mark.parametrize("fake_kwargs, code", [
    ({"token_status": 401, "token_body": {"error": "invalid_client", "error_description": f"bad {SECRET}"}},
     "invalid_client"),
    ({"token_status": 400, "token_body": {"error": "Error", "error_description":
                                          "Client does not support the client_credentials grant type"}},
     "unsupported_grant"),
    ({"token_status": 400, "token_body": {"error": "unsupported_grant_type"}}, "unsupported_grant"),
    ({"token_status": 500, "token_body": {"html": "<h1>tok-leak</h1>"}}, "token_endpoint_error"),
    ({"granted_scope": "read"}, "scope_mismatch"),
    ({"viewer": "some-human"}, "actor_mismatch"),
    ({"org": "other-workspace"}, "workspace_mismatch"),
    ({"teams": ("ENG",)}, "resource_access_missing"),
])
def test_rejections_fail_closed_with_secret_safe_codes(fake_kwargs, code):
    fake = FakeLinear(**fake_kwargs)
    provider = _provider(fake)
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(provider.token())
    assert info.value.code == code
    message = str(info.value)
    for leaked in (SECRET, "tok-", "<h1>", "bad "):
        assert leaked not in message
    assert provider._token is None  # a rejected/mismatched token is never cached or used


def test_unset_env_placeholder_and_missing_fields_are_config_errors():
    for overrides in ({"client_secret": "${LINEAR_CLIENT_SECRET}"}, {"scope": ""},
                      {"token_url": "http://api.linear.app/oauth/token"}, {"verify": {"kind": "nope"}}):
        with pytest.raises(cc.ClientCredentialsError) as info:
            cc.parse_config("linear_aligned", _server_config(**overrides))
        assert info.value.code == "config_error" and "LINEAR_CLIENT_SECRET}" not in str(info.value)


def test_401_remints_once_and_replays_only_side_effect_free_requests():
    now = [1000.0]
    fake = FakeLinear(mcp_401=1)
    provider = _provider(fake, clock=lambda: now[0])
    (resp,) = _run_with_auth(fake, provider, ("tools/list", {}))
    assert resp.status_code == 200 and fake.mints == 2
    assert [auth for auth, _ in fake.mcp_requests] == ["Bearer tok-1", "Bearer tok-2"]

    # A write is re-minted for later calls but never replayed by the auth layer.
    now[0] += 120
    fake.mcp_401, fake.mcp_requests = 1, []
    (resp,) = _run_with_auth(fake, provider, ("tools/call", {"name": "save_issue", "arguments": {}}))
    assert resp.status_code == 401 and len(fake.mcp_requests) == 1 and fake.mints == 3


def test_401_remint_is_bounded_when_server_keeps_rejecting():
    now = [1000.0]
    fake = FakeLinear(mcp_401=99)
    provider = _provider(fake, clock=lambda: now[0])
    responses = _run_with_auth(fake, provider, ("tools/list", {}), ("tools/list", {}))
    assert [r.status_code for r in responses] == [401, 401]
    assert fake.mints == 2  # initial + exactly one re-mint; the second 401 inside the cooldown mints nothing
    with pytest.raises(cc.ClientCredentialsError) as info:  # fail fast: no mint, no rejected token resent
        _run_with_auth(fake, provider, ("ping", {}))
    assert info.value.code == "token_rejected" and fake.mints == 2 and len(fake.mcp_requests) == 3
    now[0] += cc.DEFAULT_REMINT_COOLDOWN_SECONDS
    fake.mcp_401 = 0
    (resp,) = _run_with_auth(fake, provider, ("ping", {}))
    assert resp.status_code == 200 and fake.mints == 3  # recovers after the cooldown


def test_tool_handler_does_not_replay_write_after_401_but_retries_read_only(monkeypatch):
    from tools import mcp_tool_handlers as handlers
    srv = type("Srv", (), {"_config": _server_config(), "_reconnect_event": None})()
    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *a, **k: srv)
    monkeypatch.setattr(_core, "_bump_server_error", lambda *a, **k: None)
    retries = []
    monkeypatch.setattr(handlers, "_retry_once", lambda *a, **k: retries.append(a) or '{"result": "ok"}')
    exc = cc.ClientCredentialsError("linear_aligned", "invalid_token", "rejected")

    write = json.loads(handlers._handle_auth_error_and_retry(
        "linear_aligned", exc, lambda: None, "tools/call save_issue", call_may_have_side_effects=True))
    assert retries == [] and write["outcome_uncertain"] is True and "NOT replayed" in write["error"]
    assert "hermes mcp login" not in write["error"]

    read = handlers._handle_auth_error_and_retry("linear_aligned", exc, lambda: None, "tools/call list_issues")
    assert read == '{"result": "ok"}' and len(retries) == 1


def test_transport_builds_client_credentials_auth_and_errors_park_as_auth():
    """Red on base: ``auth: client_credentials`` previously connected with no credentials at all."""
    from tools.mcp_tool_errors import _classify_mcp_failure
    task = _core.MCPServerTask("linear_aligned")
    task._auth_type = "client_credentials"
    auth = task._build_oauth_auth(MCP_URL, _server_config())
    assert isinstance(auth, httpx.Auth)
    with pytest.raises(cc.ClientCredentialsError):
        task._build_oauth_auth(MCP_URL, _server_config(client_secret="${UNSET}"))
    assert _classify_mcp_failure(cc.ClientCredentialsError("linear_aligned", "invalid_client", "x")) == "permanent"

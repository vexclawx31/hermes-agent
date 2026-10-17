"""``auth: client_credentials`` for remote MCP servers (Linear hosted MCP + existing app grant).

Offline: the token endpoint, the Linear GraphQL identity check and the MCP endpoint are one
in-process ``MockTransport``; socket connects are refused for the whole module. Values are fakes.
The end-to-end tests drive the real installed MCP SDK (``streamable_http_client`` +
``ClientSession``) inside a real ``MCPServerTask`` on the real MCP loop and call the exported
registry handler, so the run binding, the per-operation recovery allowance and the outcome mapping
are exercised across the actual auth -> SDK -> handler boundary.

Contracts: a fresh token per admitted agent run (profile + run identity + destination), released at
run end; connection maintenance never lends its token to a run; one 401 recovery per logical
operation, terminal on a failed re-mint or a second rejection; read-only replay only; writes are
never replayed and report an unconfirmed outcome; complete verifier pins and fail-closed
scope/expiry/GraphQL validation; secret-free errors.
"""

import asyncio
import base64
import contextvars
import json
import math
import socket
import threading
from collections import deque

import pytest
from unittest.mock import patch

from tools import mcp_client_credentials as cc
from tools import mcp_tool as _core

httpx = _core.sdk_httpx()

NAME = "linear_aligned"
TOKEN_URL = "https://api.linear.app/oauth/token"
GRAPHQL_URL = "https://api.linear.app/graphql"
MCP_URL = "https://mcp.linear.app/mcp"
SECRET = "fake-client-secret-DO-NOT-LEAK"
SCOPE = "read,write,app:assignable"
VERIFY = {"kind": "linear_graphql", "url": GRAPHQL_URL, "actor_id": "app-user-1",
          "organization_id": "org-aligned", "team_keys": ["WEB2"]}


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch):
    def _refuse(*_a, **_k):
        raise OSError("network access is not allowed in this test module")
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


def _server_config(secret=SECRET, **overrides):
    block = {"token_url": TOKEN_URL, "client_id": "client-1", "client_secret": secret, "scope": SCOPE,
             "verify": dict(VERIFY), **overrides}
    return {"url": MCP_URL, "auth": "client_credentials", "client_credentials": block,
            "protocol": "legacy", "skip_preflight": True, "connect_timeout": 5}


def _basic_secret(request) -> str:
    return base64.b64decode(request.headers["authorization"].split()[1]).decode().split(":", 1)[1]


class FakeLinear:
    """Token endpoint + GraphQL + a JSON-RPC MCP endpoint. Each successful mint is a new token.

    ``token_plan``: per token request, ``"ok"`` / an HTTP status / a payload override dict.
    ``call_plan``: per ``tools/call``, ``200`` / ``401`` / ``500`` (non-JSON error).
    ``reject``: tokens the MCP endpoint answers with 401 for every request."""

    TOOLS = [{"name": "write_tool", "inputSchema": {"type": "object"}},
             {"name": "read_tool", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]

    def __init__(self, *, token_plan=(), call_plan=(), graphql=None, call_delay=0.0, mint_gate=None):
        self.token_plan, self.call_plan = deque(token_plan), deque(call_plan)
        self.graphql, self.call_delay, self.mint_gate = graphql, call_delay, mint_gate
        self.mints, self.token_requests, self.graphql_requests, self.mcp = 0, [], [], []
        self.reject = set()

    async def __call__(self, request):
        url = str(request.url)
        if url == TOKEN_URL:
            return await self._token(request)
        if url == GRAPHQL_URL:
            body = json.loads(request.content)
            self.graphql_requests.append(body["variables"])
            if self.graphql is not None:
                return self.graphql(body["variables"])
            return httpx.Response(200, json={"data": _page(teams=("WEB2",))})
        assert url == MCP_URL, url
        return await self._mcp(request)

    async def _token(self, request):
        self.token_requests.append(request)
        if self.mint_gate is not None:
            await self.mint_gate()
        step = self.token_plan.popleft() if self.token_plan else "ok"
        if isinstance(step, int):
            return httpx.Response(step, text="upstream body that must never be echoed")
        self.mints += 1
        payload = {"access_token": f"tok-{self.mints}", "token_type": "Bearer", "expires_in": 2591999,
                   "scope": SCOPE}
        if isinstance(step, dict):
            payload.update(step)
            payload = {k: v for k, v in payload.items() if v is not _ABSENT}
        return httpx.Response(200, content=json.dumps(payload).encode(),
                              headers={"content-type": "application/json"})

    async def _mcp(self, request):
        bearer = (request.headers.get("authorization") or "").removeprefix("Bearer ") or None
        if request.method != "POST":
            self.mcp.append((request.method, bearer, None))
            return httpx.Response(405 if request.method == "GET" else 200)
        message = json.loads(request.content)
        method = message.get("method")
        tool = (message.get("params") or {}).get("name") if method == "tools/call" else None
        self.mcp.append((method, bearer, tool))
        if bearer is None or bearer in self.reject:
            return httpx.Response(401, text="invalid_token")
        if "id" not in message:
            return httpx.Response(202)
        if method == "tools/call":
            if self.call_delay:
                await asyncio.sleep(self.call_delay)
            status = self.call_plan.popleft() if self.call_plan else 200
            if status != 200:
                return httpx.Response(status, text="error page")
            result = {"content": [{"type": "text", "text": f"{tool} ok"}], "isError": False}
        elif method == "initialize":
            result = {"protocolVersion": message["params"].get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake-linear", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.TOOLS}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def calls(self, tool=None):
        return [(bearer, t) for m, bearer, t in self.mcp if m == "tools/call" and (tool is None or t == tool)]

    def maintenance_tokens(self):
        return {bearer for m, bearer, _ in self.mcp if m in ("initialize", "tools/list") and bearer}


_ABSENT = object()


def _page(*, viewer="app-user-1", org="org-aligned", teams=("WEB2",), next_cursor=None):
    return {"viewer": {"id": viewer}, "organization": {"id": org},
            "teams": {"nodes": [{"key": k} for k in teams],
                      "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor}}}


def _cfg(**overrides):
    return cc.parse_config(NAME, _server_config(**overrides))


def _provider(fake, clock=None, **overrides):
    kwargs = {"clock": clock} if clock else {}
    return cc.ClientCredentialsTokenProvider(NAME, _cfg(**overrides), httpx_mod=httpx,
                                             transport=httpx.MockTransport(fake), **kwargs)


def _connection(fake, secret=SECRET):
    return cc.open_connection(NAME, MCP_URL, _server_config(secret=secret), httpx_mod=httpx,
                              token_transport=httpx.MockTransport(fake))


class _Server:
    """What :func:`begin_operation` reads off an ``MCPServerTask``."""

    def __init__(self, connection):
        self.name, self._cc_connection, self._config = NAME, connection, _server_config()


async def _request(connection, fake, op, method, *, tool=None):
    """One JSON-RPC POST through the connection's auth, as *op* (None = maintenance)."""
    params = {"name": tool, "arguments": {}} if tool else {}
    async def _send():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake), auth=connection.auth(httpx)) as client:
            return await client.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if op is None:
        return await _send()
    return await cc.run_operation(op, _Server(connection), _send)


# --------------------------------------------------------------------------- F5: config + verification

@pytest.mark.parametrize("verify", [
    None,                                                        # verifier omitted
    {"kind": "linear_graphql", "url": GRAPHQL_URL},              # kind/url only
    {**VERIFY, "actor_id": ""},
    {**VERIFY, "organization_id": None},
    {**VERIFY, "team_keys": []},
    {**VERIFY, "team_keys": ["WEB2", ""]},
    {**VERIFY, "actor_id": "${LINEAR_APP_ACTOR}"},
    {**VERIFY, "url": "http://api.linear.app/graphql"},
    {**VERIFY, "kind": "none"},
])
def test_incomplete_verifier_pins_fail_closed_at_config(verify):
    config = _server_config()
    if verify is None:
        del config["client_credentials"]["verify"]
    else:
        config["client_credentials"]["verify"] = verify
    with pytest.raises(cc.ClientCredentialsError) as info:
        cc.parse_config(NAME, config)
    assert info.value.code == "config_error"


@pytest.mark.parametrize("override,code", [
    ({"scope": _ABSENT}, "scope_unverified"),            # omission is not verification
    ({"scope": "read"}, "scope_mismatch"),
    ({"scope": ["read", "write", "app:assignable", "admin"]}, "scope_mismatch"),
    ({"token_type": "mac"}, "malformed_token_response"),
    ({"access_token": ""}, "malformed_token_response"),
])
def test_token_response_scope_and_shape_fail_closed(override, code):
    fake = FakeLinear(token_plan=[override])
    provider = _provider(fake)
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(provider.token())
    assert info.value.code == code and provider._token is None
    assert fake.graphql_requests == []  # rejected before any identity call with the token


def test_granted_scope_as_list_or_spaces_matches_configured_set():
    for granted in (["read", "write", "app:assignable"], "app:assignable write read"):
        assert asyncio.run(_provider(FakeLinear(token_plan=[{"scope": granted}])).token()) == "tok-1"


@pytest.mark.parametrize("graphql,code", [
    (lambda v: httpx.Response(200, json={"data": _page(), "errors": [{"message": "forbidden"}]}), "verify_failed"),
    (lambda v: httpx.Response(200, json={"data": {**_page(), "viewer": [{"id": "app-user-1"}]}}), "verify_failed"),
    (lambda v: httpx.Response(200, json={"data": {**_page(), "teams": {"nodes": [{"key": "WEB2"}]}}}),
     "verify_failed"),                                                       # no pageInfo: completeness unknown
    (lambda v: httpx.Response(200, json={"data": _page(next_cursor="")}), "verify_failed"),
    (lambda v: httpx.Response(500, json={"data": _page()}), "verify_failed"),
    (lambda v: httpx.Response(200, text="not json"), "verify_failed"),
    (lambda v: httpx.Response(200, json={"data": _page(viewer="someone-else")}), "actor_mismatch"),
    (lambda v: httpx.Response(200, json={"data": _page(org="other-org")}), "workspace_mismatch"),
    (lambda v: httpx.Response(200, json={"data": _page(teams=())}), "resource_access_missing"),
])
def test_verifier_validates_shape_errors_and_pins(graphql, code):
    provider = _provider(FakeLinear(graphql=graphql))
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(provider.token())
    assert info.value.code == code and provider._token is None
    assert SECRET not in str(info.value) and "tok-1" not in str(info.value)


def test_verifier_paginates_team_listing_with_cursor():
    pages = {None: _page(teams=("OPS",), next_cursor="c1"), "c1": _page(teams=("WEB2",))}
    fake = FakeLinear(graphql=lambda v: httpx.Response(200, json={"data": pages[v["after"]]}))
    assert asyncio.run(_provider(fake).token()) == "tok-1"
    assert [v["after"] for v in fake.graphql_requests] == [None, "c1"]
    assert all(v["keys"] == ["WEB2"] and v["first"] > 0 for v in fake.graphql_requests)


def test_verifier_page_bound_fails_closed():
    fake = FakeLinear(graphql=lambda v: httpx.Response(200, json={"data": _page(teams=(), next_cursor="more")}))
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(_provider(fake).token())
    assert info.value.code == "verify_failed" and len(fake.graphql_requests) == cc._VERIFY_MAX_PAGES


# --------------------------------------------------------------------------- F6: expiry

@pytest.mark.parametrize("expires_in", [-1, 0, -0.5, True, False, "3600", None, [60], math.nan, math.inf, -math.inf])
def test_unusable_expiry_is_malformed_and_never_cached(expires_in):
    fake = FakeLinear(token_plan=[{"expires_in": expires_in}])
    provider = _provider(fake)
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(provider.token())
    assert info.value.code == "malformed_token_response" and provider._token is None
    assert fake.graphql_requests == []  # validated before verification and cache publication
    with pytest.raises(cc.ClientCredentialsError):  # terminal: no second mint for this owner
        asyncio.run(provider.token())
    assert len(fake.token_requests) == 1


def test_lifetime_below_usable_minimum_is_refused():
    with pytest.raises(cc.ClientCredentialsError) as info:
        asyncio.run(_provider(FakeLinear(token_plan=[{"expires_in": 1}])).token())
    assert info.value.code == "token_lifetime_too_short"


def test_short_ttl_skew_is_bounded_and_long_ttl_skew_capped():
    now = [1000.0]
    fake = FakeLinear(token_plan=[{"expires_in": 60}, {"expires_in": 60}])
    provider = _provider(fake, clock=lambda: now[0])

    async def _go():
        first = await provider.token()
        now[0] += 44  # skew is 15s for a 60s token: still fresh, no mint per lookup
        assert await provider.token() == first
        now[0] += 2
        return first, await provider.token()
    first, second = asyncio.run(_go())
    assert (first, second) == ("tok-1", "tok-2") and len(fake.token_requests) == 2
    assert cc.expiry_skew(2591999) == cc.MAX_EXPIRY_SKEW_SECONDS and cc.expiry_skew(60) == 15


# --------------------------------------------------------------------------- F1/F2: run + profile binding

def test_calls_outside_an_admitted_run_fail_closed():
    fake = FakeLinear()
    server = _Server(_connection(fake))
    with pytest.raises(cc.ClientCredentialsError) as info:
        cc.begin_operation(server, read_only=True)
    assert info.value.code == "no_run_binding"

    async def _unbound_tools_call():  # defence in depth below the handler: no token is lent
        return await _request(server._cc_connection, fake, None, "tools/call", tool="write_tool")
    assert asyncio.run(_unbound_tools_call()).status_code == 401
    assert fake.calls() == [(None, "write_tool")] and fake.token_requests == []


def test_each_run_gets_a_fresh_token_released_at_run_end():
    fake = FakeLinear()
    connection = _connection(fake)
    server = _Server(connection)
    seen = []
    for run_id in ("run-1", "run-2", "run-3"):
        with cc.agent_run(run_id) as admission:
            op = cc.begin_operation(server, read_only=True)
            asyncio.run(_request(connection, fake, op, "tools/call", tool="read_tool"))
            asyncio.run(_request(connection, fake, op, "resources/read"))
            provider = op.provider(connection)
        assert admission.closed and provider._token is None
        with pytest.raises(cc.ClientCredentialsError) as info:
            asyncio.run(provider.token())
        assert info.value.code == "binding_released"
        seen.append({bearer for m, bearer, _ in fake.mcp[-2:]})
    assert seen == [{"tok-1"}, {"tok-2"}, {"tok-3"}]  # one fresh mint per run, reused within the run
    assert cc.current_run() is None


def test_connection_maintenance_token_is_never_lent_to_a_run():
    fake = FakeLinear()
    connection = _connection(fake)

    async def _go():
        await _request(connection, fake, None, "initialize")
        with cc.agent_run("run-1"):
            op = cc.begin_operation(_Server(connection), read_only=False)
            await _request(connection, fake, op, "tools/call", tool="write_tool")
        await _request(connection, fake, None, "tools/list")
    asyncio.run(_go())
    assert [(m, b) for m, b, _ in fake.mcp] == [("initialize", "tok-1"), ("tools/call", "tok-2"),
                                                 ("tools/list", "tok-1")]


def test_concurrent_runs_are_independent_and_release_without_touching_each_other():
    fake = FakeLinear()
    connection = _connection(fake)
    server = _Server(connection)

    async def _run(run_id, entered, release):
        with cc.agent_run(run_id):
            op = cc.begin_operation(server, read_only=True)
            await _request(connection, fake, op, "tools/call", tool="read_tool")
            entered.set()
            await release.wait()
            await _request(connection, fake, op, "tools/call", tool="read_tool")
            return op.provider(connection)

    async def _go():
        a_in, b_in, a_go, b_go = (asyncio.Event() for _ in range(4))
        task_a = asyncio.create_task(_run("run-a", a_in, a_go), context=contextvars.copy_context())
        task_b = asyncio.create_task(_run("run-b", b_in, b_go), context=contextvars.copy_context())
        await a_in.wait(), await b_in.wait()
        a_go.set()
        provider_a = await task_a  # run A ends while run B is still mid-run
        b_go.set()
        provider_b = await task_b
        return provider_a, provider_b
    provider_a, provider_b = asyncio.run(_go())
    tokens = [bearer for bearer, _ in fake.calls()]
    assert len(set(tokens[:2])) == 2 and tokens[2] == tokens[0] and tokens[3] == tokens[1]
    assert provider_a is not provider_b and provider_a._closed and provider_b._closed
    assert not connection.provider._closed and fake.mints == 2  # no connection token was minted or rotated


def test_profile_a_b_a_isolation(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    homes = {p: tmp_path / f"profile-{p}" for p in "AB"}
    for home in homes.values():
        home.mkdir()
    fake = FakeLinear()

    def _in(profile, fn):
        token = set_hermes_home_override(str(homes[profile]))
        try:
            return fn()
        finally:
            reset_hermes_home_override(token)

    conn_a = _in("A", lambda: _connection(fake))
    conn_b = _in("B", lambda: _connection(fake))

    def _run_call(connection, run_id):
        with cc.agent_run(run_id):
            op = cc.begin_operation(_Server(connection), read_only=True)
            asyncio.run(_request(connection, fake, op, "tools/call", tool="read_tool"))
            return fake.calls()[-1][0]

    first_a = _in("A", lambda: _run_call(conn_a, "a-1"))
    token_b = _in("B", lambda: _run_call(conn_b, "b-1"))
    second_a = _in("A", lambda: _run_call(conn_a, "a-2"))
    assert len({first_a, token_b, second_a}) == 3  # A, B, A again: three owners, three tokens

    def _cross():  # a profile-B run must never use profile A's connection
        with cc.agent_run("b-2"):
            cc.begin_operation(_Server(conn_a), read_only=True)
    with pytest.raises(cc.ClientCredentialsError) as info:
        _in("B", _cross)
    assert info.value.code == "profile_mismatch"

    def _switch():  # a run admitted under A cannot be driven from B
        with cc.agent_run("a-3"):
            _in("B", lambda: cc.begin_operation(_Server(conn_a), read_only=True))
    with pytest.raises(cc.ClientCredentialsError) as info:
        _in("A", _switch)
    assert info.value.code == "profile_mismatch"


def test_secret_rotation_rebinds_the_run_to_the_new_app_config():
    fake = FakeLinear()
    old_conn, new_conn = _connection(fake, secret="old-secret"), _connection(fake, secret="rotated-secret")

    async def _go():
        with cc.agent_run("run-1"):
            op = cc.begin_operation(_Server(old_conn), read_only=True)
            await _request(old_conn, fake, op, "tools/call", tool="read_tool")
            await _request(new_conn, fake, op, "tools/call", tool="read_tool")  # reconnect after rotation
            await _request(new_conn, fake, op, "tools/call", tool="read_tool")
    asyncio.run(_go())
    assert [_basic_secret(r) for r in fake.token_requests] == ["old-secret", "rotated-secret"]
    assert [b for b, _ in fake.calls()] == ["tok-1", "tok-2", "tok-2"]


def test_run_cancelled_mid_mint_publishes_nothing():
    gate, started = asyncio.Event(), asyncio.Event()

    async def _hold():
        started.set()
        await gate.wait()
    fake = FakeLinear(mint_gate=_hold)
    connection = _connection(fake)

    async def _go():
        with cc.agent_run("run-1") as admission:
            op = cc.begin_operation(_Server(connection), read_only=True)
            provider = op.provider(connection)
            task = asyncio.create_task(provider.token())
            await started.wait()
        assert admission.closed
        gate.set()
        with pytest.raises(cc.ClientCredentialsError) as info:
            await task
        return provider, info.value
    provider, exc = asyncio.run(_go())
    assert exc.code == "binding_released" and provider._token is None and fake.mcp == []


# --------------------------------------------------------------------------- F3: one recovery per operation

def test_failed_remint_is_terminal_for_the_run():
    """Review repro A: mint -> 401 -> token endpoint 503 -> immediate next request."""
    fake = FakeLinear(token_plan=["ok", 503])
    fake.reject.add("tok-1")
    connection = _connection(fake)

    async def _go():
        with cc.agent_run("run-1"):
            server = _Server(connection)
            op1 = cc.begin_operation(server, read_only=True)
            await _request(connection, fake, op1, "tools/call", tool="read_tool")
            op2 = cc.begin_operation(server, read_only=True)
            with pytest.raises(cc.ClientCredentialsError) as info:
                await _request(connection, fake, op2, "tools/call", tool="read_tool")
            return op1, info.value
    op1, exc = asyncio.run(_go())
    assert op1.recovery_used and op1.outcome == "token_endpoint_error"
    assert exc.code == "token_endpoint_error"
    assert len(fake.token_requests) == 2 and len(fake.calls()) == 1  # initial + one recovery, nothing more


def test_slow_401_retry_of_the_same_operation_cannot_remint_again():
    """Review repro B: the clock moving past any cooldown does not refresh an operation's allowance."""
    now = [0.0]
    fake = FakeLinear()
    fake.reject.update({"tok-1", "tok-2"})
    connection = cc.open_connection(NAME, MCP_URL, _server_config(), httpx_mod=httpx,
                                    token_transport=httpx.MockTransport(fake))

    async def _go():
        with cc.agent_run("run-1"):
            op = cc.begin_operation(_Server(connection), read_only=False)
            op.provider(connection)._clock = lambda: now[0]
            for _ in range(2):  # first attempt + a handler-level retry of the same logical operation
                now[0] += 61
                await _request(connection, fake, op, "tools/call", tool="write_tool")
            return op
    op = asyncio.run(_go())
    assert op.outcome == "token_rejected" and op.recovery_used
    assert fake.mints == 2 and [b for b, _ in fake.calls()] == ["tok-1", "tok-2"]


def test_second_rejection_invalidates_the_fresh_token_and_ends_recovery():
    fake = FakeLinear()
    fake.reject.update({"tok-1", "tok-2"})
    connection = _connection(fake)

    async def _go():
        with cc.agent_run("run-1"):
            server = _Server(connection)
            op = cc.begin_operation(server, read_only=True)
            response = await _request(connection, fake, op, "tools/call", tool="read_tool")
            provider = op.provider(connection)
            later = cc.begin_operation(server, read_only=True)
            with pytest.raises(cc.ClientCredentialsError):
                await _request(connection, fake, later, "tools/call", tool="read_tool")
            return op, provider, response
    op, provider, response = asyncio.run(_go())
    assert response.status_code == 401 and op.outcome == "token_rejected"
    assert provider._token is None and provider.terminal_code == "token_rejected"
    assert [b for b, _ in fake.calls()] == ["tok-1", "tok-2"] and fake.mints == 2


def test_concurrent_401s_in_one_run_share_a_single_recovery():
    fake = FakeLinear()
    fake.reject.add("tok-1")
    connection = _connection(fake)

    async def _go():
        with cc.agent_run("run-1"):
            server = _Server(connection)
            ops = [cc.begin_operation(server, read_only=True) for _ in range(3)]
            await ops[0].provider(connection).token()
            responses = await asyncio.gather(*(_request(connection, fake, op, "resources/read") for op in ops))
            return ops, responses
    ops, responses = asyncio.run(_go())
    # Whichever operations saw tok-1 rejected share the one recovery mint; later senders use tok-2.
    assert fake.mints == 2 and all(r.status_code == 200 for r in responses)
    assert {op.outcome for op in ops} <= {"recovered", None} and any(op.outcome == "recovered" for op in ops)
    assert [b for m, b, _ in fake.mcp].count("tok-1") >= 1 and fake.mcp[-1][1] == "tok-2"


def test_connection_maintenance_401_recovers_once_and_replays_only_safe_requests():
    fake = FakeLinear()
    fake.reject.add("tok-1")
    connection = _connection(fake)
    response = asyncio.run(_request(connection, fake, None, "tools/list"))
    assert response.status_code == 200
    assert [(m, b) for m, b, _ in fake.mcp] == [("tools/list", "tok-1"), ("tools/list", "tok-2")]


# --------------------------------------------------------------------------- F4: real SDK + exported handler

@pytest.fixture
def live(monkeypatch, tmp_path):
    """A real ``MCPServerTask`` (real SDK client + session) on the real MCP loop, wired to FakeLinear."""
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_handlers as handlers
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_transport as transport

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    fake = FakeLinear()
    monkeypatch.setattr(transport, "_make_mcp_body_cap_transport", lambda httpx_mod, *_a, **_k:
                        httpx_mod.MockTransport(fake))
    monkeypatch.setattr(transport, "_mcp_proxy_mounts", lambda *_a, **_k: None)
    real_open = cc.open_connection
    monkeypatch.setattr(cc, "open_connection", lambda name, url, config: real_open(
        name, url, config, token_transport=httpx.MockTransport(fake)))
    loop._ensure_mcp_loop()
    server = loop._run_on_mcp_loop(lambda: discovery._connect_server(NAME, _server_config()), timeout=15)
    with _core._lock:
        _core._servers[NAME] = server
    _core._tool_read_only_hints[NAME] = {"read_tool": True}

    class Live:
        def __init__(self):
            self.fake, self.server = fake, server

        def call(self, tool, run_id="run-1"):
            handler = handlers._make_tool_handler(NAME, tool, 10)
            if run_id is None:
                return json.loads(handler({}))
            with cc.agent_run(run_id):
                return json.loads(handler({}))

    try:
        yield Live()
    finally:
        loop._run_on_mcp_loop(server.shutdown, timeout=15)
        with _core._lock:
            _core._servers.pop(NAME, None)
        for ledger in (_core._tool_read_only_hints, _core._server_error_counts,
                       _core._server_breaker_opened_at, _core._server_errors_all_application):
            ledger.pop(NAME, None)
        loop._stop_mcp_loop()


def test_sdk_handler_uses_a_fresh_run_token_per_run_and_never_the_connection_token(live):
    assert live.call("read_tool", "run-1") == {"result": "read_tool ok"}
    assert live.call("write_tool", "run-2") == {"result": "write_tool ok"}
    connection_tokens = live.fake.maintenance_tokens()
    run_tokens = [b for b, _ in live.fake.calls()]
    assert len(connection_tokens) == 1 and len(set(run_tokens)) == 2
    assert not connection_tokens & set(run_tokens)


def test_sdk_handler_outside_a_run_fails_closed_before_any_request(live):
    result = live.call("write_tool", run_id=None)
    assert result["code"] == "no_run_binding" and result["auth_failed"] is True
    assert live.fake.calls() == []


def test_sdk_read_only_401_is_replayed_once_with_the_fresh_token(live):
    live.fake.call_plan.extend([401])
    assert live.call("read_tool") == {"result": "read_tool ok"}
    (first, _), (second, _) = live.fake.calls()
    assert first != second and live.fake.mints == 3  # connection + run + one recovery


def test_sdk_write_401_is_not_replayed_and_requires_readback(live):
    live.fake.call_plan.extend([401])
    result = live.call("write_tool")
    assert result["outcome_uncertain"] is True and result["readback_required"] is True
    assert "NOT replayed" in result["error"] and "read back" in result["error"]
    assert "never" not in result["error"].lower()  # does not claim the server never executed it
    assert len(live.fake.calls("write_tool")) == 1


def test_sdk_persistent_401_ends_the_run_without_further_mints(live):
    live.fake.call_plan.extend([401, 401])
    result = live.call("read_tool")
    assert result["auth_failed"] is True and result["code"] == "token_rejected"
    assert len(live.fake.calls()) == 2 and live.fake.mints == 3


def test_sdk_failed_recovery_mint_is_reported_and_terminal(live):
    live.fake.call_plan.extend([401])
    live.fake.token_plan.extend(["ok", 503])  # run mint ok, recovery mint fails
    with cc.agent_run("run-1"):
        from tools import mcp_tool_handlers as handlers
        handler = handlers._make_tool_handler(NAME, "read_tool", 10)
        first, second = json.loads(handler({})), json.loads(handler({}))
    assert first["auth_failed"] is True and first["code"] == "token_endpoint_error"
    assert second["code"] == "token_endpoint_error" and len(live.fake.calls()) == 1
    assert len(live.fake.token_requests) == 3  # connection + run + one recovery attempt
    assert "upstream body" not in json.dumps([first, second])


def test_sdk_slow_401_still_gets_exactly_one_recovery(live):
    live.fake.call_delay = 0.3
    live.fake.call_plan.extend([401, 401])
    result = live.call("write_tool")
    assert result["outcome_uncertain"] is True
    assert len(live.fake.calls()) == 1 and live.fake.mints == 3


def test_sdk_generic_error_is_not_classified_as_auth(live):
    live.fake.call_plan.extend([500])
    result = live.call("write_tool")
    assert "outcome_uncertain" not in result and "auth_failed" not in result
    assert "MCP call failed" in result["error"] and live.fake.mints == 2


def test_sdk_reconnect_keeps_the_run_token_and_mints_a_new_connection_token(live):
    from tools import mcp_tool_handlers as handlers
    from tools import mcp_tool_loop as loop
    handler = handlers._make_tool_handler(NAME, "read_tool", 10)
    with cc.agent_run("run-1"):
        assert json.loads(handler({})) == {"result": "read_tool ok"}
        assert loop._signal_reconnect_and_wait(NAME, live.server, op_description="test", timeout=15)
        assert json.loads(handler({})) == {"result": "read_tool ok"}
    run_tokens = {b for b, _ in live.fake.calls()}
    assert len(run_tokens) == 1 and len(live.fake.maintenance_tokens()) == 2
    assert not run_tokens & live.fake.maintenance_tokens()


def test_sdk_concurrent_runs_in_threads_use_their_own_tokens(live):
    barrier, results = threading.Barrier(2), {}

    def _worker(run_id):
        from tools import mcp_tool_handlers as handlers
        handler = handlers._make_tool_handler(NAME, "read_tool", 10)
        with cc.agent_run(run_id):
            handler({})
            barrier.wait(timeout=10)
            handler({})
            results[run_id] = cc.current_run().run_id
    threads = [threading.Thread(target=_worker, args=(r,)) for r in ("run-a", "run-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    tokens = [b for b, _ in live.fake.calls()]
    assert len(tokens) == 4 and len(set(tokens)) == 2
    assert results == {"run-a": "run-a", "run-b": "run-b"}
    assert cc.current_run() is None


# --------------------------------------------------------------------------- secret safety

def test_errors_and_logs_never_carry_secret_tokens_or_bodies(caplog):
    failures = [FakeLinear(token_plan=[401]), FakeLinear(token_plan=[{"scope": _ABSENT}]),
                FakeLinear(token_plan=[{"expires_in": math.inf}]),
                FakeLinear(graphql=lambda v: httpx.Response(200, json={"data": _page(viewer="x")}))]
    messages = []
    with caplog.at_level("DEBUG", logger="tools.mcp_tool"):
        for fake in failures:
            with pytest.raises(cc.ClientCredentialsError) as info:
                asyncio.run(_provider(fake).token())
            messages.append(str(info.value))
    text = "\n".join(messages) + caplog.text
    assert SECRET not in text and "tok-1" not in text and "upstream body" not in text
    assert "client_secret" not in repr(_cfg()) and SECRET not in repr(_cfg())


# --- Cross-profile adoption (v0.22.2 merge, security-critical) ---------------------------------
# Upstream v0.22.2 lets a profile adopt another profile's live connection when the resolved
# connection inputs hash equal. A client_credentials connection's tokens are bound to the
# owner's runs and secret source, so it must never be adopted, even with identical digests.


def _route_server(config, digest):
    from types import SimpleNamespace
    return SimpleNamespace(_config=config, _resolved_identity=digest)


from tests.tools.test_mcp_multiplex_connection_keys import two_profiles  # the multiplex-registry fixture


def test_same_server_route_refuses_cross_profile_client_credentials():
    from tools.mcp_tool_registration import _same_server_route
    config = _server_config()
    server = _route_server(_server_config(), "digest-1")
    assert not _same_server_route(server, config, cross_profile=True, resolved_identity="digest-1")
    # Upper-case / padded auth spelling normalises to the same type and is refused too.
    loud = {**_server_config(), "auth": " Client_Credentials "}
    assert not _same_server_route(_route_server(loud, "digest-1"), loud, cross_profile=True,
                                  resolved_identity="digest-1")
    # The owning profile keeps using its own connection.
    assert _same_server_route(server, config, cross_profile=False)


def test_same_server_route_cross_profile_control_without_client_credentials():
    """Control: the same comparison on an unauthenticated route adopts only on an equal digest,
    so the refusal above comes from the auth type, not from a mismatched fixture."""
    from tools.mcp_tool_registration import _same_server_route
    plain = {"url": MCP_URL}
    server = _route_server(plain, "digest-1")
    assert _same_server_route(server, dict(plain), cross_profile=True, resolved_identity="digest-1")
    assert not _same_server_route(server, dict(plain), cross_profile=True, resolved_identity="digest-2")
    assert not _same_server_route(server, dict(plain), cross_profile=True, resolved_identity=None)


def test_client_credentials_connection_is_not_adopted_by_another_profile(two_profiles):  # health: allow F811 -- pytest injects the imported two_profiles fixture by parameter name
    """End to end through the multiplex registry: profile B never sees A's client_credentials
    tools, so B must open (and authenticate) its own connection."""
    from tests.tools.test_mcp_multiplex_connection_keys import _server
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    config = _server_config()
    two_profiles("a")
    owner = _server("x", config)
    owner._resolved_identity = "same-digest"
    disc._adopt_server("x", owner)
    owner._registered_tool_names = reg._register_server_tools("x", owner, config)
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    two_profiles("b")
    # Even an adopter that resolves the identical digest is refused.
    with patch.object(reg, "_adopter_identity_digest", lambda name, cfg: "same-digest"):
        assert reg.register_connected_into_current_scope({"x": _server_config()}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []

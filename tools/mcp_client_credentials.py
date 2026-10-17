"""OAuth 2.0 client-credentials bearer auth for remote (Streamable HTTP) MCP servers: ``auth: client_credentials``.

For servers that accept an app access token from an authorization server that is NOT the one their
protected-resource metadata advertises — e.g. Linear's hosted MCP (``mcp.linear.app`` advertises its
own DCR authorization server, which has no ``client_credentials`` grant) accepts
``Authorization: Bearer <app token>`` minted at ``https://api.linear.app/oauth/token``. The SDK's
``ClientCredentialsOAuthProvider`` discovers the token endpoint from that metadata, so it cannot be used.

Contract (memory-only; tokens are never written to disk, config, env or logs and are registered for
redaction):

- **Run-bound tokens.** ``AIAgent.run_conversation`` (``agent/turn_facade.py``, the turn-admission
  boundary) opens :func:`agent_run` with an explicit run identity and the authenticated profile. Every
  MCP request a run sends — ``tools/call``, ``resources/*``, ``prompts/*`` — carries a token owned by
  THAT run: minted and verified fresh before the run's first request to the server, keyed by
  (server, destination URL, app config) inside the run, routed through reconnects and dropped when
  the run ends (normal, error, interrupt or cancellation). Runs never share a token, so concurrent runs
  and profiles are independent; a call outside an admitted run, or from another profile than the
  run's or the connection's, fails closed (``no_run_binding`` / ``profile_mismatch``).
- **Connection token.** The long-lived connection mints its own token at each connect, used only for
  protocol maintenance (``initialize``, ``notifications/*``, list calls during discovery, the SSE GET
  stream, session DELETE, JSON-RPC responses); it is never presented for a run's requests and a
  run-originated method without a run binding is refused.
- **One recovery per logical operation.** A tool/utility call is one :class:`Operation` carried across
  the HTTP auth flow, the SDK, the exported handler and its retries/reconnects. The first 401 consumes
  its single recovery allowance *before* the token endpoint is awaited; the rejected token is never
  presented again. A read-only request is replayed once with the fresh token; a write-capable
  ``tools/call`` is never replayed and is reported ``outcome_uncertain`` (read back before any manual
  retry). A failed re-mint, or a 401 for a token that a recovery minted, ends the provider for its
  owner (``token_rejected`` / the mint code) — no further mints until a new run or connection.
- **Fail-closed verification.** Every mint must state the granted scope, equal to the configured one
  (an omitted ``scope`` is ``scope_unverified``: the provider contract does not prove it), a finite,
  positive, non-boolean ``expires_in`` of at least :data:`MIN_TOKEN_LIFETIME_SECONDS`, and pass the
  configured verifier with complete pins (actor, workspace, required teams) before it is cached.
- Errors carry a stable code and fixed text — never response bodies, the client secret or a token.

Config::

    mcp_servers:
      linear_aligned:
        url: https://mcp.linear.app/mcp
        auth: client_credentials
        client_credentials:
          token_url: https://api.linear.app/oauth/token
          client_id: ${LINEAR_CLIENT_ID}
          client_secret: ${LINEAR_CLIENT_SECRET}
          scope: read,write            # exactly the app's approved scopes (Linear: comma-separated)
          verify:                      # required, fail-closed after every mint
            kind: linear_graphql
            url: https://api.linear.app/graphql
            actor_id: <app user id>
            organization_id: <workspace id>
            team_keys: [WEB2]
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("tools.mcp_tool")

AUTH_TYPE = "client_credentials"
MAX_EXPIRY_SKEW_SECONDS = 300.0
MIN_TOKEN_LIFETIME_SECONDS = 10.0
_TOKEN_TIMEOUT_SECONDS = 20.0
_VERIFY_PAGE_SIZE = 50
_VERIFY_MAX_PAGES = 5
_OAUTH_ERROR_CODE = re.compile(r"^[a-z_]{1,64}$")
_UNRESOLVED_PLACEHOLDER = re.compile(r"\$\{[^}]*\}")
# JSON-RPC methods with no server-side effects: safe to replay once after a 401 + re-mint.
_REPLAY_SAFE_METHODS = frozenset({
    "initialize", "ping", "tools/list", "resources/list", "resources/templates/list", "resources/read",
    "prompts/list", "prompts/get", "completion/complete", "logging/setLevel"})


class ClientCredentialsError(PermissionError):
    """Secret-safe client-credentials failure. ``code`` is stable for health checks and tests.

    A ``PermissionError`` so the run loop treats it as an auth failure rather than retrying it."""

    def __init__(self, server_name: str, code: str, detail: str):
        self.server_name, self.code = server_name, code
        super().__init__(f"MCP server '{server_name}': client_credentials {code}: {detail}")


def _split_scope(value: Any) -> frozenset:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return frozenset(s for s in re.split(r"[,\s]+", str(value or "")) if s)


def _origin(url: Any) -> tuple:
    parts = urlsplit(str(url))
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port


@dataclass(frozen=True)
class VerifySpec:
    kind: str
    url: str
    actor_id: str
    organization_id: str
    team_keys: tuple


@dataclass(frozen=True)
class ClientCredentialsConfig:
    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    scope: str
    verify: VerifySpec


def _pin(server_name: str, verify: dict, key: str) -> str:
    value = verify.get(key)
    if not isinstance(value, str) or not value.strip() or _UNRESOLVED_PLACEHOLDER.search(value):
        raise ClientCredentialsError(server_name, "config_error", f"verify.{key} is required")
    return value.strip()


def parse_config(server_name: str, config: dict) -> ClientCredentialsConfig:
    """Validate ``config['client_credentials']``; fail closed (``config_error``) on anything missing,
    including any verifier pin: an unpinned actor, workspace or required team is never "verified"."""
    def _bad(detail: str) -> ClientCredentialsError:
        return ClientCredentialsError(server_name, "config_error", detail)

    block = config.get("client_credentials")
    if not isinstance(block, dict):
        raise _bad("missing client_credentials block")
    values = {}
    for key in ("token_url", "client_id", "client_secret", "scope"):
        value = block.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _bad(f"{key} is required")
        if _UNRESOLVED_PLACEHOLDER.search(value):
            raise _bad(f"{key} references an unset environment variable")
        values[key] = value.strip()
    if not values["token_url"].startswith("https://"):
        raise _bad("token_url must be https")
    verify = block.get("verify")
    if not isinstance(verify, dict) or verify.get("kind") not in _VERIFIERS:
        raise _bad(f"verify is required; verify.kind must be one of {sorted(_VERIFIERS)}")
    if not str(verify.get("url") or "").startswith("https://"):
        raise _bad("verify.url must be https")
    keys = verify.get("team_keys")
    if (not isinstance(keys, (list, tuple)) or not keys
            or not all(isinstance(k, str) and k.strip() and not _UNRESOLVED_PLACEHOLDER.search(k) for k in keys)):
        raise _bad("verify.team_keys must list at least one required team key")
    spec = VerifySpec(kind=verify["kind"], url=verify["url"], actor_id=_pin(server_name, verify, "actor_id"),
                      organization_id=_pin(server_name, verify, "organization_id"),
                      team_keys=tuple(sorted({k.strip() for k in keys})))
    return ClientCredentialsConfig(verify=spec, **values)


def expiry_skew(lifetime: float) -> float:
    """Refresh margin: a quarter of the lifetime, capped, so a short valid TTL stays usable."""
    return min(MAX_EXPIRY_SKEW_SECONDS, lifetime / 4.0)


def _parse_lifetime(value: Any) -> Optional[float]:
    """Finite positive number of seconds; None for anything else (bool, str, NaN, inf, <= 0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    lifetime = float(value)
    return lifetime if math.isfinite(lifetime) and lifetime > 0 else None


@dataclass
class _Token:
    value: str = field(repr=False)
    expires_at: float
    skew: float


def _oauth_error_code(response: Any) -> str:
    """Classify a token-endpoint rejection from the OAuth ``error`` field only (never echoed raw)."""
    try:
        payload = response.json()
    except Exception:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    description = str(payload.get("error_description") or "") if isinstance(payload, dict) else ""
    # Linear answers a disabled grant with error "Error" + this description instead of the RFC code.
    if error == "unsupported_grant_type" or "does not support the client_credentials grant" in description:
        return "unsupported_grant"
    if isinstance(error, str) and _OAUTH_ERROR_CODE.match(error):
        return error
    return "token_endpoint_error"


class ClientCredentialsTokenProvider:
    """One in-memory app token for one owner: a single agent run, or a single connection.

    Bounded by construction: one mint, plus at most one 401 recovery mint, plus expiry refreshes.
    Any mint failure, and a 401 for a recovery-minted token, is terminal for the owner."""

    def __init__(self, server_name: str, cfg: ClientCredentialsConfig, *, owner: str = "connection",
                 httpx_mod: Any = None, transport: Any = None, clock: Callable[[], float] = time.monotonic):
        if httpx_mod is None:
            from tools import mcp_tool as _core
            httpx_mod = _core.sdk_httpx()
        self.server_name, self.cfg, self.owner = server_name, cfg, owner
        self._httpx, self._transport, self._clock = httpx_mod, transport, clock
        self._token: Optional[_Token] = None
        self._recovery_value: Optional[str] = None
        self._recoveries = 0
        self._terminal: Optional[ClientCredentialsError] = None
        self._closed = False
        self.mint_count = 0
        self._lock = asyncio.Lock()

    def _error(self, code: str, detail: str) -> ClientCredentialsError:
        return ClientCredentialsError(self.server_name, code, detail)

    @property
    def terminal_code(self) -> Optional[str]:
        return self._terminal.code if self._terminal is not None else None

    def _client(self):
        return self._httpx.AsyncClient(timeout=_TOKEN_TIMEOUT_SECONDS, follow_redirects=False,
                                       **({"transport": self._transport} if self._transport else {}))

    def _fresh(self) -> Optional[str]:
        tok = self._token
        return tok.value if tok and tok.expires_at - tok.skew > self._clock() else None

    def _check_usable(self) -> None:
        if self._closed:
            raise self._error("binding_released", f"the {self.owner} that owned this token has ended")
        if self._terminal is not None:
            raise self._error(self._terminal.code, f"token unavailable for this {self.owner} after an "
                                                    "earlier failure; not minting again")

    def _fail(self, code: str, detail: str) -> ClientCredentialsError:
        """Record a terminal failure (no further mints for this owner) and return the error."""
        self._token = None
        self._terminal = self._error(code, detail)
        return self._terminal

    async def token(self) -> str:
        self._check_usable()
        if (value := self._fresh()) is not None:
            return value
        async with self._lock:
            self._check_usable()
            if (value := self._fresh()) is not None:
                return value
            return await self._mint_locked()

    async def recover(self, rejected: str) -> Optional[str]:
        """The server answered *rejected* with 401; the caller already consumed its operation's
        allowance. Returns the token to retry with, or None (terminal; see ``terminal_code``)."""
        async with self._lock:
            if self._closed or self._terminal is not None:
                return None
            current = self._token
            if current is not None and current.value != rejected and self._fresh() is not None:
                return current.value  # a concurrent operation already replaced it
            self._token = None  # never present a token the server has rejected
            if rejected == self._recovery_value or self._recoveries >= 1:
                self._fail("token_rejected", "server rejected a freshly re-minted token; not minting again")
                logger.warning("%s", self._terminal)
                return None
            self._recoveries += 1  # consumed before the token endpoint is awaited
            try:
                value = await self._mint_locked()
            except ClientCredentialsError as exc:
                logger.warning("%s", exc)
                return None
            self._recovery_value = value
            return value

    async def reject(self, rejected: str) -> None:
        """A token presented after this operation's recovery was rejected again: terminal."""
        async with self._lock:
            if self._token is not None and self._token.value == rejected:
                self._token = None
            if self._terminal is None and not self._closed:
                logger.warning("%s", self._fail("token_rejected", "server rejected the token again after "
                                                                    "recovery; not minting again"))

    def close(self) -> None:
        """Drop the token (owner ended). An in-flight mint completes but is never published."""
        self._closed = True
        self._token = None

    async def _mint_locked(self) -> str:
        cfg = self.cfg
        if self._closed:
            raise self._error("binding_released", f"the {self.owner} that owned this token has ended")
        self.mint_count += 1
        issued_at = self._clock()
        try:
            async with self._client() as client:
                response = await client.post(
                    cfg.token_url, auth=(cfg.client_id, cfg.client_secret),
                    data={"grant_type": "client_credentials", "scope": cfg.scope},
                    headers={"Accept": "application/json"})
        except Exception as exc:
            raise self._fail("token_endpoint_unreachable", type(exc).__name__) from None
        if response.status_code >= 400:
            raise self._fail(_oauth_error_code(response),
                             f"token endpoint rejected the request (HTTP {response.status_code})")
        try:
            payload = response.json()
        except Exception:
            payload = None
        value = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise self._fail("malformed_token_response", "no access_token in token response")
        value = value.strip()
        _register_redaction(value)
        token_type = payload.get("token_type")
        if token_type is not None and (not isinstance(token_type, str) or token_type.lower() != "bearer"):
            raise self._fail("malformed_token_response", "token_type is not Bearer; token discarded")
        lifetime = _parse_lifetime(payload.get("expires_in"))
        if lifetime is None:
            raise self._fail("malformed_token_response", "expires_in is not a finite positive number; "
                                                         "token discarded")
        if lifetime < MIN_TOKEN_LIFETIME_SECONDS:
            raise self._fail("token_lifetime_too_short", "token lifetime too short to use; token discarded")
        granted = payload.get("scope")
        if granted is None:
            raise self._fail("scope_unverified", "token response did not state the granted scope; "
                                                 "token discarded")
        if _split_scope(granted) != _split_scope(cfg.scope):
            raise self._fail("scope_mismatch", "granted scope differs from the configured scope; token discarded")
        try:
            await _VERIFIERS[cfg.verify.kind](self, value, cfg.verify)
        except ClientCredentialsError as exc:
            self._token, self._terminal = None, exc
            raise
        if self._closed:
            raise self._error("binding_released", f"the {self.owner} ended while its token was being minted")
        self._token = _Token(value, issued_at + lifetime, expiry_skew(lifetime))
        logger.info("MCP server '%s': minted client_credentials token for %s (expires in %ds)",
                    self.server_name, self.owner, int(lifetime))
        return value


def _register_redaction(value: str) -> None:
    try:
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(value)
    except Exception:  # redaction registry unavailable (minimal installs): token is still never logged
        pass


_LINEAR_VERIFY_QUERY = (
    "query HermesAppTokenVerify($keys: [String!], $first: Int!, $after: String) { viewer { id } "
    "organization { id } teams(first: $first, after: $after, filter: { key: { in: $keys } }) "
    "{ nodes { key } pageInfo { hasNextPage endCursor } } }")


def _linear_page(data: Any) -> tuple:
    """``(viewer_id, org_id, team_keys, next_cursor)`` from one response page; ValueError on any
    shape the schema does not promise."""
    if not isinstance(data, dict):
        raise ValueError("data")
    viewer, org, teams = data.get("viewer"), data.get("organization"), data.get("teams")
    if not (isinstance(viewer, dict) and isinstance(viewer.get("id"), str)
            and isinstance(org, dict) and isinstance(org.get("id"), str) and isinstance(teams, dict)):
        raise ValueError("identity")
    nodes, page = teams.get("nodes"), teams.get("pageInfo")
    if not isinstance(nodes, list) or not all(isinstance(n, dict) and isinstance(n.get("key"), str) for n in nodes):
        raise ValueError("nodes")
    if not isinstance(page, dict) or not isinstance(page.get("hasNextPage"), bool):
        raise ValueError("pageInfo")
    cursor = page.get("endCursor")
    if page["hasNextPage"] and not (isinstance(cursor, str) and cursor):
        raise ValueError("cursor")
    return viewer["id"], org["id"], {n["key"] for n in nodes}, cursor if page["hasNextPage"] else None


async def _verify_linear_graphql(provider: ClientCredentialsTokenProvider, token: str, spec: VerifySpec) -> None:
    """Read-only identity check: app actor, workspace and every required team must match. The team
    listing is paginated (bounded); an incomplete or malformed answer, or any GraphQL ``errors``,
    fails closed."""
    visible: set = set()
    cursor = None
    for _ in range(_VERIFY_MAX_PAGES):
        try:
            async with provider._client() as client:
                response = await client.post(spec.url, headers={"Authorization": f"Bearer {token}"},
                                             json={"query": _LINEAR_VERIFY_QUERY, "variables": {
                                                 "keys": list(spec.team_keys), "first": _VERIFY_PAGE_SIZE,
                                                 "after": cursor}})
            body = response.json() if response.status_code == 200 else None
            if not isinstance(body, dict) or body.get("errors"):
                raise ValueError("errors")
            viewer_id, org_id, keys, cursor = _linear_page(body.get("data"))
        except Exception:
            raise provider._error("verify_failed", "identity query failed or returned an unexpected "
                                                   "shape; token discarded") from None
        if viewer_id != spec.actor_id:
            raise provider._error("actor_mismatch", "token actor is not the configured app actor; token discarded")
        if org_id != spec.organization_id:
            raise provider._error("workspace_mismatch", "token workspace is not the configured workspace; "
                                                        "token discarded")
        visible |= keys
        if cursor is None or visible >= set(spec.team_keys):
            break
    else:
        if not visible >= set(spec.team_keys):
            raise provider._error("verify_failed", "team listing exceeded the page bound; token discarded")
    missing = sorted(set(spec.team_keys) - visible)
    if missing:
        raise provider._error("resource_access_missing", f"required team access missing: {', '.join(missing)}")


_VERIFIERS = {"linear_graphql": _verify_linear_graphql}


def _jsonrpc_methods(request: Any) -> Optional[list]:
    try:
        body = json.loads(request.content or b"null")
    except Exception:
        return None
    messages = body if isinstance(body, list) else [body]
    if not messages or not all(isinstance(m, dict) for m in messages):
        return None
    return [m.get("method") for m in messages]


def is_replay_safe(request: Any) -> bool:
    """Whether *request* may be resent after a 401: session GET/DELETE, notifications, responses and
    side-effect-free JSON-RPC methods. ``tools/call`` and unknown methods are never replayed here."""
    if request.method in ("GET", "DELETE"):
        return True
    methods = _jsonrpc_methods(request)
    if methods is None:
        return False
    return all(m is None or m in _REPLAY_SAFE_METHODS or str(m).startswith("notifications/") for m in methods)


def _is_tools_call(request: Any) -> bool:
    return _jsonrpc_methods(request) == ["tools/call"]


def _current_profile_key() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


# ---------------------------------------------------------------- run / connection / operation binding

_RUN: ContextVar[Optional["RunAdmission"]] = ContextVar("hermes_mcp_cc_run", default=None)
_OPERATION: ContextVar[Optional["Operation"]] = ContextVar("hermes_mcp_cc_operation", default=None)


class RunAdmission:
    """One admitted agent run: its identity, authenticated profile and run-owned providers."""

    def __init__(self, run_id: str, profile_key: str):
        self.run_id, self.profile_key = run_id, profile_key
        self._providers: dict = {}
        self._lock = threading.Lock()
        self.closed = False

    def provider_for(self, connection: "ConnectionBinding") -> ClientCredentialsTokenProvider:
        with self._lock:
            if self.closed:
                raise ClientCredentialsError(connection.server_name, "binding_released", "the agent run has ended")
            provider = self._providers.get(connection.key)
            if provider is None:
                provider = self._providers[connection.key] = ClientCredentialsTokenProvider(
                    connection.server_name, connection.cfg, owner="agent run",
                    httpx_mod=connection.httpx_mod, transport=connection.token_transport)
            return provider

    def close(self) -> None:
        with self._lock:
            self.closed = True
            providers, self._providers = list(self._providers.values()), {}
        for provider in providers:
            provider.close()


@contextmanager
def agent_run(run_id: str) -> Iterator[RunAdmission]:
    """Admit one agent run: run-owned tokens for every ``client_credentials`` server it calls, all
    released on exit. Context-local, so concurrent runs (threads, tasks) never see each other's."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("agent_run requires an explicit run identity")
    admission = RunAdmission(run_id, _current_profile_key())
    reset = _RUN.set(admission)
    try:
        yield admission
    finally:
        _RUN.reset(reset)
        admission.close()


def current_run() -> Optional[RunAdmission]:
    return _RUN.get()


class ConnectionBinding:
    """One connect of one server: its destination, app config, owning profile and the
    connection-maintenance token."""

    def __init__(self, server_name: str, url: str, cfg: ClientCredentialsConfig, *, profile_key: str,
                 httpx_mod: Any = None, token_transport: Any = None):
        self.server_name, self.url, self.cfg, self.profile_key = server_name, url, cfg, profile_key
        self.httpx_mod, self.token_transport = httpx_mod, token_transport
        self.origin = _origin(url)
        self.key = (server_name, url, cfg)  # run providers are keyed by destination + app config
        self.provider = ClientCredentialsTokenProvider(server_name, cfg, owner="connection",
                                                       httpx_mod=httpx_mod, transport=token_transport)

    def close(self) -> None:
        self.provider.close()

    def auth(self, httpx_mod: Any = None):
        return make_auth(self, httpx_mod or self.httpx_mod)


def sdk_supports_run_binding() -> bool:
    """The run's token reaches the HTTP auth flow only because the SDK runs each outbound POST in
    the sending task's context (``mcp.shared._context_streams``, mcp >= 2.0)."""
    try:
        from mcp.shared import _context_streams  # noqa: F401
    except Exception:
        return False
    return True


def open_connection(server_name: str, url: str, config: dict, *, httpx_mod: Any = None,
                    token_transport: Any = None) -> ConnectionBinding:
    """Validate and bind one connect attempt; fails closed (``config_error``) when run binding
    cannot be honoured on this transport/SDK."""
    cfg = parse_config(server_name, config)
    if config.get("transport") == "sse":
        raise ClientCredentialsError(server_name, "config_error", "requires the Streamable HTTP transport")
    if not sdk_supports_run_binding():
        raise ClientCredentialsError(server_name, "config_error",
                                     "installed mcp SDK cannot carry run-bound tokens (needs mcp >= 2.0)")
    return ConnectionBinding(server_name, url, cfg, profile_key=_current_profile_key(),
                             httpx_mod=httpx_mod, token_transport=token_transport)


@dataclass
class Operation:
    """One logical tool/utility operation: its run, replay policy and single recovery allowance."""
    server_name: str
    run: RunAdmission
    read_only: bool
    recovery_used: bool = False
    unauthorized: bool = False
    outcome: Optional[str] = None  # None | "recovered" | "write_unconfirmed" | terminal error code

    def provider(self, connection: ConnectionBinding) -> ClientCredentialsTokenProvider:
        return self.run.provider_for(connection)


def begin_operation(server: Any, *, read_only: bool) -> Operation:
    """Bind a handler call to the current admitted run; fail closed without one."""
    name = getattr(server, "name", "?")
    connection = getattr(server, "_cc_connection", None)
    if not isinstance(connection, ConnectionBinding):
        raise ClientCredentialsError(name, "connection_unbound", "server connection has no app binding yet")
    run = _RUN.get()
    if run is None or run.closed:
        raise ClientCredentialsError(name, "no_run_binding", "MCP calls with client_credentials auth must come "
                                                             "from an admitted agent run")
    profile = _current_profile_key()
    if profile != run.profile_key or profile != connection.profile_key:
        raise ClientCredentialsError(name, "profile_mismatch", "the calling profile does not own this run "
                                                               "or connection")
    return Operation(name, run, read_only)


async def run_operation(op: Operation, server: Any, call: Callable[[], Any]) -> Any:
    """One attempt of *op* on the MCP loop: mint/verify the run's token for the server's current
    connection BEFORE any request is sent, then run *call* with *op* visible to the auth flow."""
    connection = getattr(server, "_cc_connection", None)
    if not isinstance(connection, ConnectionBinding):
        raise ClientCredentialsError(op.server_name, "connection_unbound", "server connection has no app binding")
    await op.provider(connection).token()
    reset = _OPERATION.set(op)
    try:
        return await call()
    finally:
        _OPERATION.reset(reset)


def make_auth(connection: ConnectionBinding, httpx_mod: Any = None):
    """An ``httpx.Auth`` (SDK httpx flavour) choosing the run's or the connection's token per request,
    with the one-recovery-per-operation 401 contract."""
    if httpx_mod is None:
        from tools import mcp_tool as _core
        httpx_mod = _core.sdk_httpx()

    class _ClientCredentialsAuth(httpx_mod.Auth):
        requires_request_body = True

        async def async_auth_flow(self, request):
            if _origin(request.url) != connection.origin:
                yield request  # never present the app token to another origin
                return
            op = _OPERATION.get()
            if op is None and not is_replay_safe(request):
                # A run-originated method with no admitted run: never lend it the connection token.
                # Raising here would tear down the transport shared by other runs, so the request
                # goes out without credentials and fails on its own.
                logger.warning("MCP server '%s': client_credentials request without an admitted run sent "
                               "unauthenticated (no_run_binding)", connection.server_name)
                yield request
                return
            if op is None:
                token = await connection.provider.token()  # connection-scoped: a failure reconnects
                provider = connection.provider
            else:
                try:
                    provider = op.provider(connection)
                    token = await provider.token()
                except ClientCredentialsError as exc:  # run-local: fail this request, not the transport
                    op.unauthorized, op.outcome = True, exc.code
                    yield request
                    return
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            if response.status_code != 401:
                return
            if op is None:  # connection maintenance: one recovery per request, replay-safe only
                fresh = await provider.recover(token)
                if fresh is None or not is_replay_safe(request):
                    return
                request.headers["Authorization"] = f"Bearer {fresh}"
                response = yield request
                if response.status_code == 401:
                    await provider.reject(fresh)
                return
            op.unauthorized = True
            if op.recovery_used:
                await provider.reject(token)
                op.outcome = "token_rejected"
                return
            op.recovery_used = True  # consumed before awaiting the token endpoint
            fresh = await provider.recover(token)
            if fresh is None:
                op.outcome = provider.terminal_code or "token_rejected"
                return
            if not (is_replay_safe(request) or (op.read_only and _is_tools_call(request))):
                op.outcome = "write_unconfirmed"  # never auto-replay a write-capable call
                return
            request.headers["Authorization"] = f"Bearer {fresh}"
            response = yield request
            if response.status_code == 401:
                await provider.reject(fresh)
                op.outcome = "token_rejected"
                return
            op.outcome = "recovered"

    return _ClientCredentialsAuth()


def is_client_credentials(config: Optional[dict]) -> bool:
    return isinstance(config, dict) and str(config.get("auth") or "").lower().strip() == AUTH_TYPE


NOT_REPLAYED_MSG = (
    "MCP server '{s}' answered '{t}' with HTTP 401 (app token rejected). This call was NOT replayed because "
    "the tool is not read-only. A 401 is normally returned before execution, but Hermes cannot confirm that "
    "(a proxy or server may reject after acting), so treat the outcome as UNCONFIRMED: read back the current "
    "state with a read-only tool before any manual retry of this change.")
AUTH_FAILED_MSG = (
    "MCP server '{s}' could not authenticate this run with its client_credentials app token ({code}). Do NOT "
    "retry in this run — the app grant, scopes or resource access need operator attention.")

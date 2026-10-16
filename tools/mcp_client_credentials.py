"""OAuth 2.0 client-credentials bearer auth for remote (HTTP/SSE) MCP servers: ``auth: client_credentials``.

For servers that accept an app access token from an authorization server that is NOT the one their
protected-resource metadata advertises — e.g. Linear's hosted MCP (``mcp.linear.app`` advertises its
own DCR authorization server, which has no ``client_credentials`` grant) accepts
``Authorization: Bearer <app token>`` minted at ``https://api.linear.app/oauth/token``. The SDK's
``ClientCredentialsOAuthProvider`` discovers the token endpoint from that metadata, so it cannot be used.

Contract (process-local, memory-only):

- The first request of a process mints a fresh token from the configured app with the configured,
  unchanged scope; the token is never written to disk, config, env or logs and is registered for
  redaction. A token whose granted scope differs from the requested scope, or that fails the optional
  post-mint ``verify`` (actor / workspace / required team access), is discarded and the connect fails.
- A 401 triggers at most one re-mint per request. A second 401-triggered re-mint within the cooldown is
  refused and the provider fails fast (``token_rejected``) until it ends, so a server persistently
  rejecting fresh tokens cannot drain the provider's active-token cap. Only JSON-RPC
  methods without side effects are replayed here; ``tools/call`` is never replayed by the auth layer
  (the tool handler replays only ``readOnlyHint`` tools and reports write tools as not replayed).
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
          verify:                      # optional, fail-closed after every mint
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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("tools.mcp_tool")

AUTH_TYPE = "client_credentials"
DEFAULT_EXPIRY_SKEW_SECONDS = 300.0
DEFAULT_REMINT_COOLDOWN_SECONDS = 60.0
_TOKEN_TIMEOUT_SECONDS = 20.0
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


@dataclass(frozen=True)
class VerifySpec:
    kind: str
    url: str
    actor_id: str = ""
    organization_id: str = ""
    team_keys: tuple = ()


@dataclass(frozen=True)
class ClientCredentialsConfig:
    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    scope: str
    verify: Optional[VerifySpec] = None

    @property
    def identity(self) -> tuple:
        """Cache/connection identity without the secret value."""
        return (self.token_url, self.client_id, tuple(sorted(_split_scope(self.scope))), self.verify)


def parse_config(server_name: str, config: dict) -> ClientCredentialsConfig:
    """Validate ``config['client_credentials']``; fail closed (``config_error``) on anything missing."""
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
    spec = None
    if verify is not None:
        if not isinstance(verify, dict) or verify.get("kind") not in _VERIFIERS:
            raise _bad(f"verify.kind must be one of {sorted(_VERIFIERS)}")
        if not str(verify.get("url") or "").startswith("https://"):
            raise _bad("verify.url must be https")
        spec = VerifySpec(kind=verify["kind"], url=verify["url"], actor_id=str(verify.get("actor_id") or ""),
                          organization_id=str(verify.get("organization_id") or ""),
                          team_keys=tuple(str(k) for k in verify.get("team_keys") or ()))
    return ClientCredentialsConfig(verify=spec, **values)


@dataclass
class _Token:
    value: str = field(repr=False)
    expires_at: float


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
    """Mints and caches one app access token in memory for one server identity."""

    def __init__(self, server_name: str, cfg: ClientCredentialsConfig, *, httpx_mod: Any = None,
                 transport: Any = None, clock: Callable[[], float] = time.monotonic,
                 expiry_skew: float = DEFAULT_EXPIRY_SKEW_SECONDS,
                 remint_cooldown: float = DEFAULT_REMINT_COOLDOWN_SECONDS):
        if httpx_mod is None:
            from tools import mcp_tool as _core
            httpx_mod = _core.sdk_httpx()
        self.server_name, self.cfg, self._httpx, self._transport = server_name, cfg, httpx_mod, transport
        self._clock, self._skew, self._cooldown = clock, expiry_skew, remint_cooldown
        self._token: Optional[_Token] = None
        self._last_remint: Optional[float] = None
        self._rejected_until = 0.0
        self.mint_count = 0
        self._lock = asyncio.Lock()

    def _error(self, code: str, detail: str) -> ClientCredentialsError:
        return ClientCredentialsError(self.server_name, code, detail)

    def _client(self):
        return self._httpx.AsyncClient(timeout=_TOKEN_TIMEOUT_SECONDS, follow_redirects=False,
                                       **({"transport": self._transport} if self._transport else {}))

    def _fresh(self) -> Optional[str]:
        tok = self._token
        return tok.value if tok and tok.expires_at - self._skew > self._clock() else None

    async def token(self) -> str:
        if (value := self._fresh()) is not None:
            return value
        async with self._lock:
            if (value := self._fresh()) is not None:
                return value
            if self._clock() < self._rejected_until:
                raise self._error("token_rejected", "server rejected a freshly re-minted token; "
                                  "not minting again until the cooldown ends")
            return await self._mint()

    async def remint_after_401(self, rejected: str) -> bool:
        """Replace *rejected* once. True if a different token is now current (minted here or by a
        concurrent request); False when the cooldown bounds another mint or the mint failed."""
        async with self._lock:
            if self._token is not None and self._token.value != rejected:
                return True
            self._token = None  # never present a token the server has rejected
            if self._last_remint is not None and self._clock() - self._last_remint < self._cooldown:
                self._rejected_until = self._last_remint + self._cooldown
                logger.warning("MCP server '%s': client_credentials token rejected again within %.0fs of a "
                               "re-mint; not re-minting (bounded)", self.server_name, self._cooldown)
                return False
            self._last_remint = self._clock()
            try:
                await self._mint()
            except ClientCredentialsError as exc:
                logger.warning("%s", exc)
                return False
            return True

    async def _mint(self) -> str:
        cfg = self.cfg
        self.mint_count += 1
        try:
            async with self._client() as client:
                response = await client.post(
                    cfg.token_url, auth=(cfg.client_id, cfg.client_secret),
                    data={"grant_type": "client_credentials", "scope": cfg.scope},
                    headers={"Accept": "application/json"})
        except Exception as exc:
            raise self._error("token_endpoint_unreachable", type(exc).__name__) from None
        if response.status_code >= 400:
            code = _oauth_error_code(response)
            raise self._error(code, f"token endpoint rejected the request (HTTP {response.status_code})")
        try:
            payload = response.json()
        except Exception:
            payload = None
        value = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise self._error("malformed_token_response", "no access_token in token response")
        value = value.strip()
        _register_redaction(value)
        granted = payload.get("scope")
        if granted is not None and _split_scope(granted) != _split_scope(cfg.scope):
            raise self._error("scope_mismatch", "granted scope differs from the configured scope; token discarded")
        try:
            expires_in = float(payload.get("expires_in"))
        except (TypeError, ValueError):
            raise self._error("malformed_token_response", "no valid expires_in in token response") from None
        if cfg.verify is not None:
            await _VERIFIERS[cfg.verify.kind](self, value, cfg.verify)
        self._token = _Token(value, self._clock() + max(0.0, expires_in))
        logger.info("MCP server '%s': minted client_credentials token (expires in %ds)",
                    self.server_name, int(expires_in))
        return value


def _register_redaction(value: str) -> None:
    try:
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(value)
    except Exception:  # redaction registry unavailable (minimal installs): token is still never logged
        pass


_LINEAR_VERIFY_QUERY = ("query HermesAppTokenVerify($keys: [String!]) { viewer { id } organization { id } "
                        "teams(filter: { key: { in: $keys } }) { nodes { key } } }")


async def _verify_linear_graphql(provider: ClientCredentialsTokenProvider, token: str, spec: VerifySpec) -> None:
    """Read-only identity check: app actor, workspace and required team access must all match."""
    try:
        async with provider._client() as client:
            response = await client.post(spec.url, headers={"Authorization": f"Bearer {token}"},
                                         json={"query": _LINEAR_VERIFY_QUERY,
                                               "variables": {"keys": list(spec.team_keys)}})
        data = response.json().get("data") if response.status_code < 400 else None
    except Exception:
        data = None
    if not isinstance(data, dict):
        raise provider._error("verify_failed", "identity query did not return data; token discarded")
    viewer_id = (data.get("viewer") or {}).get("id")
    org_id = (data.get("organization") or {}).get("id")
    if spec.actor_id and viewer_id != spec.actor_id:
        raise provider._error("actor_mismatch", "token actor is not the configured app actor; token discarded")
    if spec.organization_id and org_id != spec.organization_id:
        raise provider._error("workspace_mismatch", "token workspace is not the configured workspace; token discarded")
    visible = {(n or {}).get("key") for n in ((data.get("teams") or {}).get("nodes") or [])}
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


def make_auth(provider: ClientCredentialsTokenProvider, httpx_mod: Any = None):
    """An ``httpx.Auth`` (SDK httpx flavour) bearing *provider*'s token with the bounded 401 contract."""
    if httpx_mod is None:
        from tools import mcp_tool as _core
        httpx_mod = _core.sdk_httpx()

    class _ClientCredentialsAuth(httpx_mod.Auth):
        requires_request_body = True

        async def async_auth_flow(self, request):
            token = await provider.token()
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            if response.status_code != 401:
                return
            # Re-mint even when not replaying so the handler's (read-only) retry and later calls work.
            if not await provider.remint_after_401(token) or not is_replay_safe(request):
                return
            request.headers["Authorization"] = f"Bearer {await provider.token()}"
            yield request

    return _ClientCredentialsAuth()


_PROVIDERS: dict = {}


def build_mcp_auth(server_name: str, config: dict):
    """Process-wide provider per (server, app identity) so reconnects reuse the in-memory token."""
    cfg = parse_config(server_name, config)
    key = (server_name, cfg.identity)
    provider = _PROVIDERS.get(key)
    if provider is None or provider.cfg != cfg:
        provider = _PROVIDERS[key] = ClientCredentialsTokenProvider(server_name, cfg)
    return make_auth(provider)


def is_client_credentials(config: Optional[dict]) -> bool:
    return isinstance(config, dict) and str(config.get("auth") or "").lower().strip() == AUTH_TYPE


NOT_REPLAYED_MSG = (
    "MCP server '{s}' rejected the app token (HTTP 401) before running '{t}'. A fresh token was requested "
    "(at most once), but this call was NOT replayed because the tool is not read-only. The request was "
    "refused at authentication, yet treat its outcome as unconfirmed: read back the current state with a "
    "read-only tool before retrying this change.")
AUTH_FAILED_MSG = (
    "MCP server '{s}' rejected its client_credentials app token even after one bounded re-mint. Do NOT "
    "retry — the app grant, scopes or resource access need operator attention.")

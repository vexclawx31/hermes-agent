"""Tests for the capability-gated ``ctx.call_mcp`` plugin surface (#64204).

The gate: ``plugins.entries.<plugin_id>.mcp_allowlist`` — a list of MCP
server names. Absent key = no MCP access (default-deny). Calls to unlisted
servers raise PermissionError naming the config key. A granted server only
exposes what the model sees: names its connection advertises that pass that
connection's ``tools.include``/``exclude``. All calls route through the
existing tools.mcp_tool handler machinery (mocked here — no live MCP servers).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManifest


def _make_ctx(plugin_key: str = "my-plugin") -> PluginContext:
    manifest = PluginManifest(name=plugin_key, key=plugin_key)
    manager = MagicMock()
    return PluginContext(manifest, manager)


def _patch_config(monkeypatch, entries: dict) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda *a, **k: {"plugins": {"entries": entries}},
    )


def _patch_handler(monkeypatch, response: str, captured: dict | None = None):
    """Replace tools.mcp_tool_handlers._make_tool_handler with a transport mock."""
    from tools import mcp_tool_handlers as _mcp_handlers

    def _fake_make_handler(server_name, tool_name, tool_timeout):
        if captured is not None:
            captured["server"] = server_name
            captured["tool"] = tool_name
            captured["timeout"] = tool_timeout

        def _handler(args, **kwargs):
            if captured is not None:
                captured["args"] = args
            return response

        return _handler

    monkeypatch.setattr(_mcp_handlers, "_make_tool_handler", _fake_make_handler)


def _connect(monkeypatch, name: str, tools: list[str], config: dict | None = None) -> None:
    """Register a fake live connection advertising ``tools``, started with ``config``."""
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_scope import _resolve_server_key

    server = SimpleNamespace(
        name=name, session=None, _config=config if config is not None else {"command": "x"},
        _tools=[SimpleNamespace(name=t, description="") for t in tools],
    )
    monkeypatch.setitem(_core._servers, _resolve_server_key(name), server)


def _connect_lazy(monkeypatch, name: str, tools: list[str], config: dict) -> None:
    """Register a fake lazy (schema-cache) server: no live connection yet."""
    from tools import mcp_schema_cache
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_scope import _server_key

    key = _server_key(name)
    monkeypatch.setitem(_core._lazy_server_configs, key, dict(config))
    monkeypatch.setitem(_core._lazy_server_fingerprints, key, "fp-1")
    entry = {"fingerprint": "fp-1", "tools": [{"name": t} for t in tools]}
    monkeypatch.setattr(
        mcp_schema_cache, "get_cached_entry",
        lambda server, fp: entry if (server, fp) == (name, "fp-1") else None,
    )


# ---------------------------------------------------------------------------
# Default-deny and allowlist enforcement
# ---------------------------------------------------------------------------


def test_default_deny_when_key_absent(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {}})
    ctx = _make_ctx()
    with pytest.raises(PermissionError) as exc:
        ctx.call_mcp("github", "create_issue", {"title": "x"})
    # Error message names the exact config key the operator must set.
    assert "plugins.entries.my-plugin.mcp_allowlist" in str(exc.value)
    assert "github" in str(exc.value)


def test_default_deny_when_plugin_has_no_entry(monkeypatch):
    _patch_config(monkeypatch, {})
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_default_deny_when_config_unreadable(monkeypatch):
    import hermes_cli.config as config_mod

    def _boom(*a, **k):
        raise OSError("config torn mid-edit")

    monkeypatch.setattr(config_mod, "load_config", _boom)
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_unlisted_server_denied_even_with_other_grants(monkeypatch):
    _patch_config(
        monkeypatch, {"my-plugin": {"mcp_allowlist": ["knowledge_rag"]}}
    )
    ctx = _make_ctx()
    with pytest.raises(PermissionError) as exc:
        ctx.call_mcp("github", "create_issue")
    assert "github" in str(exc.value)


def test_non_list_allowlist_is_denied(monkeypatch):
    """A scalar/'*' value must not grant ambient access."""
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": "*"}})
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")


def test_denied_call_never_touches_transport(monkeypatch):
    _patch_config(monkeypatch, {})
    called = {}
    _patch_handler(monkeypatch, '{"result": "hi"}', called)
    ctx = _make_ctx()
    with pytest.raises(PermissionError):
        ctx.call_mcp("github", "create_issue")
    assert called == {}


# ---------------------------------------------------------------------------
# Allowed calls route through the existing MCP handler machinery
# ---------------------------------------------------------------------------


def test_allowed_call_routes_through_existing_handler(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}})
    _connect(monkeypatch, "github", ["create_issue"])
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "issue #7 created"}), captured)

    ctx = _make_ctx()
    result = ctx.call_mcp("github", "create_issue", {"title": "bug"})

    assert captured["server"] == "github"
    assert captured["tool"] == "create_issue"
    assert captured["args"] == {"title": "bug"}
    assert result == {"ok": True, "result": "issue #7 created"}


def test_error_result_maps_to_ok_false(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}})
    _connect(monkeypatch, "github", ["create_issue"])
    _patch_handler(monkeypatch, json.dumps({"error": "MCP server 'github' is not connected"}))

    ctx = _make_ctx()
    result = ctx.call_mcp("github", "create_issue")
    assert result["ok"] is False
    assert "not connected" in result["error"]


def test_structured_content_passthrough(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["rag"]}})
    _connect(monkeypatch, "rag", ["query"])
    _patch_handler(
        monkeypatch,
        json.dumps({"result": "text part", "structuredContent": {"hits": 3}}),
    )

    ctx = _make_ctx()
    result = ctx.call_mcp("rag", "query")
    assert result["ok"] is True
    assert result["result"] == "text part"
    assert result["structuredContent"] == {"hits": 3}


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------




def test_timeout_defaults_and_bounds(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["s"]}})
    _connect(monkeypatch, "s", ["t"])
    captured = {}
    _patch_handler(monkeypatch, '{"result": ""}', captured)
    ctx = _make_ctx()

    ctx.call_mcp("s", "t")
    assert captured["timeout"] == 30.0

    ctx.call_mcp("s", "t", timeout=0)  # below floor → clamped to 1s
    assert captured["timeout"] == 1.0

    ctx.call_mcp("s", "t", timeout=99999)  # above ceiling → clamped to 600s
    assert captured["timeout"] == 600.0

    ctx.call_mcp("s", "t", timeout="nonsense")  # unparseable → default
    assert captured["timeout"] == 30.0


# ---------------------------------------------------------------------------
# Result size cap
# ---------------------------------------------------------------------------


def test_oversized_result_is_truncated(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["big"]}})
    _connect(monkeypatch, "big", ["dump"])
    huge = "x" * (PluginContext._MCP_RESULT_CHAR_CAP + 5000)
    _patch_handler(monkeypatch, huge)

    ctx = _make_ctx()
    result = ctx.call_mcp("big", "dump")
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["result"]) <= PluginContext._MCP_RESULT_CHAR_CAP + 20
    assert result["result"].endswith("… [truncated]")


# ---------------------------------------------------------------------------
# A plugin sees what the model sees: advertised names through the connection's
# own mcp_servers.<x>.tools.include / exclude
# ---------------------------------------------------------------------------

_GITHUB_TOOLS = ["list_issues", "get_issue", "create_issue", "delete_repo"]
_INCLUDE = {"command": "gh-mcp", "tools": {"include": ["list_issues", "get_*"]}}
_EXCLUDE = {"command": "gh-mcp", "tools": {"exclude": ["delete_*"]}}
_INCLUDE_NONE = {"command": "gh-mcp", "tools": {"include": []}}


def _allow_github(monkeypatch) -> None:
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}})


def _assert_refused(monkeypatch, tool: str) -> None:
    called = {}
    _patch_handler(monkeypatch, json.dumps({"result": "should not run"}), called)
    with pytest.raises(PermissionError) as exc:
        _make_ctx().call_mcp("github", tool)
    assert repr(tool) in str(exc.value)
    assert "mcp_servers.github.tools" in str(exc.value)
    assert called == {}  # refused before any handler is built or transport touched


@pytest.mark.parametrize("tool", ["list_issues", "get_issue"])
def test_included_advertised_tool_is_callable(monkeypatch, tool):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _INCLUDE)
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    result = _make_ctx().call_mcp("github", tool, {"n": 1})

    assert captured["tool"] == tool
    assert captured["args"] == {"n": 1}
    assert result == {"ok": True, "result": "ok"}


@pytest.mark.parametrize(
    ("config", "tool"),
    [(_INCLUDE, "create_issue"), (_EXCLUDE, "delete_repo"), (_INCLUDE_NONE, "list_issues")],
)
def test_filtered_out_tool_is_refused_before_any_server_call(monkeypatch, config, tool):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, config)
    _assert_refused(monkeypatch, tool)


def test_exclude_list_leaves_other_tools_callable(monkeypatch):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _EXCLUDE)
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    assert _make_ctx().call_mcp("github", "list_issues")["ok"] is True
    assert captured["tool"] == "list_issues"


def test_server_without_filter_is_unchanged(monkeypatch):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, {"command": "gh-mcp"})
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    assert _make_ctx().call_mcp("github", "delete_repo")["ok"] is True
    assert captured["tool"] == "delete_repo"


def test_failing_filter_fails_closed(monkeypatch):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _INCLUDE)
    from tools import mcp_tool_registration as _registration

    def _boom(name, config):
        raise RuntimeError("filter exploded")

    monkeypatch.setattr(_registration, "_make_tool_filter", _boom)
    _assert_refused(monkeypatch, "list_issues")


def test_unreadable_mcp_config_does_not_lift_the_filter(monkeypatch):
    """The filter comes from the connection, so an unreadable mcp_servers can't open it."""
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _EXCLUDE)
    from tools import mcp_tool_config

    def _boom():
        raise OSError("config torn mid-edit")

    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", _boom)
    _assert_refused(monkeypatch, "delete_repo")


def test_safe_mode_does_not_lift_the_filter(monkeypatch):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _EXCLUDE)
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    _assert_refused(monkeypatch, "delete_repo")


def test_server_without_connection_is_refused(monkeypatch):
    """No live or lazy connection: nothing to check the name against, so fail closed."""
    _allow_github(monkeypatch)
    _assert_refused(monkeypatch, "list_issues")


@pytest.mark.parametrize("tool", ["DELETE_REPO", "Delete_Repo"])
def test_case_variant_of_excluded_tool_is_refused(monkeypatch, tool):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, _EXCLUDE)
    _assert_refused(monkeypatch, tool)


@pytest.mark.parametrize("config", [_EXCLUDE, {"command": "gh-mcp"}])
def test_name_the_server_does_not_advertise_is_refused(monkeypatch, config):
    _allow_github(monkeypatch)
    _connect(monkeypatch, "github", _GITHUB_TOOLS, config)
    _assert_refused(monkeypatch, "admin_backdoor")


def test_lazy_server_uses_its_cached_manifest_and_config(monkeypatch):
    _allow_github(monkeypatch)
    _connect_lazy(monkeypatch, "github", _GITHUB_TOOLS, _EXCLUDE)
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    assert _make_ctx().call_mcp("github", "list_issues")["ok"] is True
    assert captured["tool"] == "list_issues"
    _assert_refused(monkeypatch, "delete_repo")
    _assert_refused(monkeypatch, "admin_backdoor")

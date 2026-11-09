"""Tests for the capability-gated ``ctx.call_mcp`` plugin surface (#64204).

The gate: ``plugins.entries.<plugin_id>.mcp_allowlist`` — a list of MCP
server names. Absent key = no MCP access (default-deny). Calls to unlisted
servers raise PermissionError naming the config key. All calls route
through the existing tools.mcp_tool handler machinery (mocked here — no
live MCP servers).
"""

import json
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
    _patch_handler(monkeypatch, json.dumps({"error": "MCP server 'github' is not connected"}))

    ctx = _make_ctx()
    result = ctx.call_mcp("github", "create_issue")
    assert result["ok"] is False
    assert "not connected" in result["error"]


def test_structured_content_passthrough(monkeypatch):
    _patch_config(monkeypatch, {"my-plugin": {"mcp_allowlist": ["rag"]}})
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
    huge = "x" * (PluginContext._MCP_RESULT_CHAR_CAP + 5000)
    _patch_handler(monkeypatch, huge)

    ctx = _make_ctx()
    result = ctx.call_mcp("big", "dump")
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["result"]) <= PluginContext._MCP_RESULT_CHAR_CAP + 20
    assert result["result"].endswith("… [truncated]")


# ---------------------------------------------------------------------------
# mcp_servers.<x>.tools.include / exclude apply to plugins too
# ---------------------------------------------------------------------------


def _patch_config_with_servers(monkeypatch, entries: dict, mcp_servers: dict) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda *a, **k: {"plugins": {"entries": entries}, "mcp_servers": mcp_servers},
    )


_GITHUB_INCLUDE = {"github": {"command": "gh-mcp", "tools": {"include": ["list_issues", "get_*"]}}}
_GITHUB_EXCLUDE = {"github": {"command": "gh-mcp", "tools": {"exclude": ["delete_*"]}}}


@pytest.mark.parametrize("tool", ["list_issues", "get_issue"])
def test_included_tool_is_callable(monkeypatch, tool):
    _patch_config_with_servers(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}}, _GITHUB_INCLUDE)
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    result = _make_ctx().call_mcp("github", tool, {"n": 1})

    assert captured["tool"] == tool
    assert captured["args"] == {"n": 1}
    assert result == {"ok": True, "result": "ok"}


@pytest.mark.parametrize(
    ("servers", "tool"),
    [(_GITHUB_INCLUDE, "create_issue"), (_GITHUB_EXCLUDE, "delete_repo"),
     ({"github": {"command": "gh-mcp", "tools": {"include": []}}}, "list_issues")],
)
def test_filtered_out_tool_is_refused_before_any_server_call(monkeypatch, servers, tool):
    _patch_config_with_servers(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}}, servers)
    called = {}
    _patch_handler(monkeypatch, json.dumps({"result": "should not run"}), called)

    with pytest.raises(PermissionError) as exc:
        _make_ctx().call_mcp("github", tool)

    assert tool in str(exc.value)
    assert "mcp_servers.github.tools" in str(exc.value)
    assert called == {}  # no handler built, no transport touched


def test_exclude_list_leaves_other_tools_callable(monkeypatch):
    _patch_config_with_servers(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}}, _GITHUB_EXCLUDE)
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    assert _make_ctx().call_mcp("github", "list_issues")["ok"] is True
    assert captured["tool"] == "list_issues"


def test_server_without_filter_is_unchanged(monkeypatch):
    _patch_config_with_servers(
        monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}}, {"github": {"command": "gh-mcp"}})
    captured = {}
    _patch_handler(monkeypatch, json.dumps({"result": "ok"}), captured)

    assert _make_ctx().call_mcp("github", "delete_repo")["ok"] is True
    assert captured["tool"] == "delete_repo"


def test_failing_filter_fails_closed(monkeypatch):
    _patch_config_with_servers(monkeypatch, {"my-plugin": {"mcp_allowlist": ["github"]}}, _GITHUB_INCLUDE)
    from tools import mcp_tool_registration as _registration

    def _boom(name, config):
        raise RuntimeError("filter exploded")

    monkeypatch.setattr(_registration, "_make_tool_filter", _boom)
    called = {}
    _patch_handler(monkeypatch, json.dumps({"result": "should not run"}), called)

    with pytest.raises(PermissionError):
        _make_ctx().call_mcp("github", "list_issues")
    assert called == {}

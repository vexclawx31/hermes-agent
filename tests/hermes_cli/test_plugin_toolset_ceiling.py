"""Plugin toolset ceiling: config ``plugin_toolset_platforms`` caps where plugin toolsets exist.

Plugin toolsets used to resolve ON on every platform without a ``known_plugin_toolsets`` record (local, relay,
sms, dormant adapters) and on saved platforms whose last save predates the plugin. With the key set, a plugin
toolset is available on a platform only when its list names that platform; the normal resolution still narrows
inside the ceiling, and every session path (gateway, cron job lists, delegates, refreshes) goes through the same
gate at agent build.
"""

import pytest

import hermes_cli.config as config_mod
import hermes_cli.plugins as plugins_mod
import hermes_cli.tools_config as tc
import model_tools
import toolsets as toolsets_mod
from hermes_cli.toolset_scope import ceiling_blocked_toolsets, parse_plugin_toolset_ceiling
from tools.registry import ToolRegistry

PLUGIN_TS = "ceilfake"
PLUGIN_TOOL = "ceilfake_read"
OTHER_TS = "ceilother"
OTHER_TOOL = "ceilother_call"


def _handler(args, **kwargs):
    return "{}"


def _schema(name):
    return {"name": name, "description": "test tool", "parameters": {"type": "object", "properties": {}}}


@pytest.fixture
def plugin_env(monkeypatch):
    """Two fake plugin toolsets in a fresh registry, reported as plugin-provided."""
    reg = ToolRegistry()
    reg.register(name=PLUGIN_TOOL, toolset=PLUGIN_TS, schema=_schema(PLUGIN_TOOL), handler=_handler)
    reg.register(name=OTHER_TOOL, toolset=OTHER_TS, schema=_schema(OTHER_TOOL), handler=_handler)
    monkeypatch.setattr("tools.registry.registry", reg)
    keys = {PLUGIN_TS, OTHER_TS}
    monkeypatch.setattr(plugins_mod, "get_plugin_toolsets", lambda: [(k, k, k) for k in sorted(keys)])
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda: None)
    monkeypatch.setattr(tc, "_get_plugin_toolset_keys", lambda: set(keys))
    return reg, keys


def _use_config(monkeypatch, cfg):
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)


# --- parsing -------------------------------------------------------------------------------------------------


def test_parse_absent_means_no_ceiling():
    assert parse_plugin_toolset_ceiling({}) is None
    assert parse_plugin_toolset_ceiling(None) is None


def test_parse_valid_mapping():
    assert parse_plugin_toolset_ceiling({"plugin_toolset_platforms": {PLUGIN_TS: ["slack", " whatsapp "]}}) == {
        PLUGIN_TS: frozenset({"slack", "whatsapp"})}


@pytest.mark.parametrize("bad", ["slack", 3, None, ["slack"]])
def test_parse_non_mapping_fails_closed(bad):
    assert parse_plugin_toolset_ceiling({"plugin_toolset_platforms": bad}) == {}


@pytest.mark.parametrize("bad", ["slack", None, [1, 2], {"slack": True}])
def test_parse_malformed_list_fails_closed(bad):
    assert parse_plugin_toolset_ceiling({"plugin_toolset_platforms": {PLUGIN_TS: bad}}) == {PLUGIN_TS: frozenset()}


# --- blocked set -----------------------------------------------------------------------------------------------


def test_blocked_none_without_ceiling(plugin_env):
    assert ceiling_blocked_toolsets("local", ceiling=None) == []


def test_blocked_outside_listed_platforms(plugin_env):
    ceiling = {PLUGIN_TS: frozenset({"slack"})}
    assert ceiling_blocked_toolsets("slack", ceiling=ceiling) == [OTHER_TS]  # missing from mapping = off
    assert ceiling_blocked_toolsets("local", ceiling=ceiling) == sorted([PLUGIN_TS, OTHER_TS])
    assert ceiling_blocked_toolsets("whatsapp", ceiling={PLUGIN_TS: frozenset()}) == sorted([PLUGIN_TS, OTHER_TS])


def test_unknown_platform_fails_closed(plugin_env):
    ceiling = {PLUGIN_TS: frozenset({"slack"}), OTHER_TS: frozenset({"slack"})}
    assert ceiling_blocked_toolsets(None, ceiling=ceiling) == sorted([PLUGIN_TS, OTHER_TS])


def test_delegated_child_inherits_parent_ceiling(plugin_env):
    assert ceiling_blocked_toolsets("subagent", ceiling={}) == []


def test_alias_key_counts_and_aliases_of_blocked_are_blocked(plugin_env):
    reg, _keys = plugin_env
    reg.register_toolset_alias("ceilalias", PLUGIN_TS)
    assert ceiling_blocked_toolsets("slack", ceiling={"ceilalias": frozenset({"slack"}), OTHER_TS: frozenset({"slack"})}) == []
    assert "ceilalias" in ceiling_blocked_toolsets("cron", ceiling={PLUGIN_TS: frozenset({"slack"})})


def test_never_names_a_builtin_toolset(plugin_env):
    reg, _keys = plugin_env
    reg.register_toolset_alias("web", PLUGIN_TS)  # an alias shadowing a built-in toolset
    blocked = ceiling_blocked_toolsets("cron", ceiling={})
    assert "web" not in blocked
    assert PLUGIN_TS in blocked


# --- resolver (_get_platform_tools): what `hermes tools` and the gateway compute -----------------------------


def test_resolver_unchanged_without_ceiling(plugin_env):
    cfg = {"platform_toolsets": {"slack": ["hermes-slack"]}}
    assert PLUGIN_TS in tc._get_platform_tools(cfg, "local")  # the old implicit-ON path, untouched
    assert PLUGIN_TS in tc._get_platform_tools(cfg, "slack")


def test_resolver_removes_plugin_toolset_outside_ceiling(plugin_env):
    cfg = {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}}
    assert PLUGIN_TS in tc._get_platform_tools(cfg, "slack")
    for platform in ("local", "relay", "sms", "cron", "cli", "telegram"):
        assert PLUGIN_TS not in tc._get_platform_tools(cfg, platform), platform
    assert OTHER_TS not in tc._get_platform_tools(cfg, "slack")


def test_saved_list_cannot_widen_past_ceiling(plugin_env):
    cfg = {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]},
           "platform_toolsets": {"telegram": ["hermes-telegram", PLUGIN_TS]}}
    assert PLUGIN_TS not in tc._get_platform_tools(cfg, "telegram")


def test_untick_inside_ceiling_stays_off(plugin_env):
    cfg = {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]},
           "platform_toolsets": {"slack": ["hermes-slack"]},
           "known_plugin_toolsets": {"slack": [PLUGIN_TS]}}
    assert PLUGIN_TS not in tc._get_platform_tools(cfg, "slack")


def test_builtins_unchanged_with_ceiling(plugin_env):
    _reg, keys = plugin_env
    base = {"platform_toolsets": {"slack": ["hermes-slack"]}}
    capped = {**base, "plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}}
    for platform in ("slack", "cron", "cli", "local", "telegram"):
        before = tc._get_platform_tools(base, platform) - keys
        after = tc._get_platform_tools(capped, platform) - keys
        assert before == after, platform


# --- session gate: every agent build (gateway turn, cron, webhook, api_server, CLI) -------------------------


def test_session_gate_adds_blocked_plugin_toolsets(plugin_env, monkeypatch):
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack", "cron"]}})
    assert PLUGIN_TS not in (toolsets_mod.session_disabled_toolsets(None, "cron") or [])
    assert OTHER_TS in toolsets_mod.session_disabled_toolsets(None, "cron")
    assert PLUGIN_TS in toolsets_mod.session_disabled_toolsets(["memory"], "local")
    assert "memory" in toolsets_mod.session_disabled_toolsets(["memory"], "local")  # explicit disables kept


def test_session_gate_no_ceiling_is_unchanged(plugin_env, monkeypatch):
    _use_config(monkeypatch, {})
    assert toolsets_mod.session_disabled_toolsets(["memory"], "local") == ["memory"] + [
        n for n in toolsets_mod.TOOLSET_SESSION_PLATFORMS if "local" not in toolsets_mod.TOOLSET_SESSION_PLATFORMS[n]]


def test_cron_job_list_cannot_widen_past_ceiling(plugin_env, monkeypatch):
    """A cron job's own enabled_toolsets naming a plugin toolset is stripped at agent build unless cron is in
    its list (approved scheduled reads keep it)."""
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["whatsapp", "cron"], OTHER_TS: ["slack"]}})
    disabled = toolsets_mod.session_disabled_toolsets(None, "cron")
    tools = model_tools._select_tool_names([PLUGIN_TS, OTHER_TS], disabled, quiet_mode=True)
    assert PLUGIN_TOOL in tools
    assert OTHER_TOOL not in tools


@pytest.mark.parametrize("enabled", [None, ["all"], ["*"]])
def test_wildcards_cannot_bring_blocked_tools_back(plugin_env, monkeypatch, enabled):
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}})
    disabled = toolsets_mod.session_disabled_toolsets(None, "webhook")
    tools = model_tools._select_tool_names(enabled, disabled, quiet_mode=True)
    assert PLUGIN_TOOL not in tools and OTHER_TOOL not in tools


def test_custom_composite_cannot_bring_blocked_tools_back(plugin_env, monkeypatch):
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}})
    toolsets_mod.create_custom_toolset("_ceil_bundle", "bundle", tools=[PLUGIN_TOOL], includes=[OTHER_TS])
    try:
        disabled = toolsets_mod.session_disabled_toolsets(None, "api_server")
        tools = model_tools._select_tool_names(["_ceil_bundle"], disabled, quiet_mode=True)
        assert PLUGIN_TOOL not in tools and OTHER_TOOL not in tools
    finally:
        toolsets_mod.TOOLSETS.pop("_ceil_bundle", None)


def test_delegate_child_inherits_parent_block(plugin_env, monkeypatch):
    from types import SimpleNamespace
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}})
    parent_disabled = toolsets_mod.session_disabled_toolsets(None, "telegram")
    parent = SimpleNamespace(enabled_toolsets=["web", PLUGIN_TS], disabled_toolsets=parent_disabled)
    child_enabled, child_disabled = _resolve_child_toolsets(parent, [PLUGIN_TS, "web"], "leaf")
    child_disabled = toolsets_mod.session_disabled_toolsets(child_disabled, "subagent")
    tools = model_tools._select_tool_names(child_enabled, child_disabled, quiet_mode=True)
    assert PLUGIN_TOOL not in tools


def test_refresh_drops_late_registered_plugin_toolset(plugin_env, monkeypatch):
    """session_platform_tool_drops (load, MCP refresh, prefix restore) is computed fresh, so a plugin toolset
    registered after the agent was built is dropped at the next refresh."""
    reg, keys = plugin_env
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"], OTHER_TS: ["slack"]}})
    assert "late_tool" not in toolsets_mod.session_platform_tool_drops("slack")
    reg.register(name="late_tool", toolset="ceillate", schema=_schema("late_tool"), handler=_handler)
    keys.add("ceillate")
    drops = toolsets_mod.session_platform_tool_drops("slack")
    assert "late_tool" in drops
    assert PLUGIN_TOOL not in drops


def test_real_config_file_is_read(plugin_env, tmp_path, monkeypatch):
    """End to end through the real loader: the active profile's config.yaml sets the ceiling."""
    import os

    home = os.environ["HERMES_HOME"]
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"plugin_toolset_platforms:\n  {PLUGIN_TS}: [slack]\n")
    if hasattr(config_mod, "_invalidate_config_cache"):
        config_mod._invalidate_config_cache()
    blocked = toolsets_mod.session_disabled_toolsets(None, "local") or []
    assert PLUGIN_TS in blocked and OTHER_TS in blocked
    assert PLUGIN_TS not in (toolsets_mod.session_disabled_toolsets(None, "slack") or [])

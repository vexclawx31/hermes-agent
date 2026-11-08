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


def test_platform_named_subagent_gets_no_exemption(plugin_env):
    """Review finding: a client can set source "subagent"; the name alone must not lift the ceiling."""
    assert ceiling_blocked_toolsets("subagent", ceiling={}) == sorted([PLUGIN_TS, OTHER_TS])


def test_inherited_ceiling_platform_only_inside_delegation():
    from hermes_cli.toolset_scope import ceiling_platform_for_new_agent, inherited_ceiling_platform

    assert ceiling_platform_for_new_agent("subagent") == "subagent"
    with inherited_ceiling_platform("slack"):
        assert ceiling_platform_for_new_agent("subagent") == "slack"
    assert ceiling_platform_for_new_agent("subagent") == "subagent"


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
    parent = SimpleNamespace(enabled_toolsets=["web", PLUGIN_TS], disabled_toolsets=parent_disabled,
                             platform="telegram", _plugin_ceiling_platform="telegram")
    child_enabled, child_disabled = _resolve_child_toolsets(parent, [PLUGIN_TS, "web"], "leaf")
    child_disabled = toolsets_mod.session_disabled_toolsets(child_disabled, "subagent", "telegram")
    tools = model_tools._select_tool_names(child_enabled, child_disabled, quiet_mode=True)
    assert PLUGIN_TOOL not in tools


def test_delegate_child_of_allowed_parent_keeps_plugin(plugin_env, monkeypatch):
    from types import SimpleNamespace
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}})
    parent = SimpleNamespace(enabled_toolsets=["web", PLUGIN_TS], platform="slack", _plugin_ceiling_platform="slack",
                             disabled_toolsets=toolsets_mod.session_disabled_toolsets(None, "slack"))
    child_enabled, child_disabled = _resolve_child_toolsets(parent, [PLUGIN_TS, "web"], "leaf")
    child_disabled = toolsets_mod.session_disabled_toolsets(child_disabled, "subagent", "slack")
    assert PLUGIN_TOOL in model_tools._select_tool_names(child_enabled, child_disabled, quiet_mode=True)


def test_late_plugin_cannot_reach_bridge_or_children_before_refresh(plugin_env, monkeypatch):
    """Review finding: disabled_toolsets is fixed at build; dispatch-time readers re-apply the gate."""
    from types import SimpleNamespace

    reg, keys = plugin_env
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"], OTHER_TS: ["slack"]}})
    agent = SimpleNamespace(platform="slack", _plugin_ceiling_platform="slack", enabled_toolsets=None,
                            disabled_toolsets=toolsets_mod.session_disabled_toolsets(None, "slack"))
    reg.register(name="late_tool", toolset="ceillate", schema=_schema("late_tool"), handler=_handler)
    keys.add("ceillate")
    fresh = toolsets_mod.agent_disabled_toolsets(agent)
    assert "ceillate" in fresh
    assert "late_tool" not in model_tools._select_tool_names(None, fresh, quiet_mode=True)


def test_refresh_reapplies_gate_without_override(plugin_env, monkeypatch):
    from types import SimpleNamespace
    from tools.mcp_tool_agent import _resolve_refresh_toolsets

    reg, keys = plugin_env
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"], OTHER_TS: ["slack"]}})
    agent = SimpleNamespace(platform="slack", _plugin_ceiling_platform="slack", enabled_toolsets=None,
                            disabled_toolsets=["memory"])
    reg.register(name="late_tool", toolset="ceillate", schema=_schema("late_tool"), handler=_handler)
    keys.add("ceillate")
    _enabled, disabled = _resolve_refresh_toolsets(agent, None, None)
    assert "ceillate" in disabled and "memory" in disabled
    assert agent.disabled_toolsets == disabled


def test_mcp_toolset_is_never_blocked(plugin_env, monkeypatch):
    reg, _keys = plugin_env
    reg.register(name="mcp__srv_call", toolset="mcp-srv", schema=_schema("mcp__srv_call"), handler=_handler)
    _use_config(monkeypatch, {"plugin_toolset_platforms": {}})
    disabled = toolsets_mod.session_disabled_toolsets(None, "cron") or []
    assert "mcp-srv" not in disabled
    assert "mcp__srv_call" not in toolsets_mod.session_platform_tool_drops("cron")


def test_two_profiles_do_not_share_a_ceiling(plugin_env, monkeypatch):
    """The ceiling is read per call from the active profile's config: switching configs switches it."""
    profiles = {"a": {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}}, "b": {}}
    active = {"name": "a"}
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: profiles[active["name"]])
    assert PLUGIN_TS in toolsets_mod.session_disabled_toolsets(None, "local")
    active["name"] = "b"
    assert PLUGIN_TS not in (toolsets_mod.session_disabled_toolsets(None, "local") or [])
    active["name"] = "a"
    assert PLUGIN_TS in toolsets_mod.session_disabled_toolsets(None, "local")


def test_tools_save_keeps_choices_the_ceiling_hides(plugin_env, monkeypatch):
    """Review finding: saving a platform must not rewrite plugin choices the checklist did not offer."""
    saved = {}
    monkeypatch.setattr(tc, "save_config", lambda cfg: saved.update(cfg))
    cfg = {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"], OTHER_TS: ["telegram"]},
           "platform_toolsets": {"telegram": ["hermes-telegram", PLUGIN_TS]},
           "known_plugin_toolsets": {"telegram": []}}
    tc._save_platform_tools(cfg, "telegram", {"web", OTHER_TS})
    assert PLUGIN_TS in saved["platform_toolsets"]["telegram"]  # hidden choice kept as saved
    assert OTHER_TS in saved["platform_toolsets"]["telegram"]
    assert PLUGIN_TS not in saved["known_plugin_toolsets"]["telegram"]  # not recorded as an untick
    assert OTHER_TS in saved["known_plugin_toolsets"]["telegram"]


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


# --- background review fork: keeps the ceiling platform of the session it reviews -------------------------


def _build_review_fork(monkeypatch, parent):
    """agent.background_review.build_cache_parity_fork with AIAgent replaced by a stand-in that runs the real
    build-time gate (agent/agent_init.py: ``new_agent_disabled_toolsets``) on its constructor kwargs."""
    from types import SimpleNamespace

    import agent.background_review as br
    import run_agent

    class _ForkAgent(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(platform=kwargs["platform"], _oneshot_session=False)
            self.disabled_toolsets = toolsets_mod.new_agent_disabled_toolsets(self, kwargs["disabled_toolsets"])

    monkeypatch.setattr(run_agent, "AIAgent", _ForkAgent)
    monkeypatch.setattr(br, "_resolve_review_runtime", lambda agent, task_cfg=None: {"routed": True})
    fork, _rt, _routed = br.build_cache_parity_fork(parent, {}, max_iterations=1)
    return fork


def _parent(platform, ceiling_platform, disabled):
    from types import SimpleNamespace

    return SimpleNamespace(
        platform=platform, _plugin_ceiling_platform=ceiling_platform, model="m", provider="p", session_id="s",
        enabled_toolsets=None, disabled_toolsets=disabled, _memory_store=None, _memory_enabled=False,
        _user_profile_enabled=False, _oneshot_session=False)


def test_review_fork_of_delegated_child_keeps_parent_ceiling(plugin_env, monkeypatch):
    """A delegated child (platform "subagent", ceiling platform of its slack parent): its review fork is judged
    by slack, at build and for plugin toolsets registered after the fork was built, never by "subagent"."""
    reg, keys = plugin_env
    _use_config(monkeypatch, {"plugin_toolset_platforms": {
        PLUGIN_TS: ["slack"], OTHER_TS: ["subagent"], "ceillate": ["slack"], "ceilsub": ["subagent"]}})
    child = _parent("subagent", "slack", toolsets_mod.session_disabled_toolsets(None, "subagent", "slack"))
    fork = _build_review_fork(monkeypatch, child)
    assert toolsets_mod.ceiling_platform_of(fork) == "slack"
    assert PLUGIN_TS not in fork.disabled_toolsets  # was wrongly OFF (judged by "subagent")
    assert OTHER_TS in fork.disabled_toolsets
    for name, toolset in (("late_tool", "ceillate"), ("late_sub", "ceilsub")):
        reg.register(name=name, toolset=toolset, schema=_schema(name), handler=_handler)
        keys.add(toolset)
    fresh = toolsets_mod.agent_disabled_toolsets(fork)
    assert "ceillate" not in fresh  # late plugin allowed for slack: ON, as in the reviewed session
    assert "ceilsub" in fresh  # late plugin allowed only for "subagent": OFF (was wrongly ON)
    names = model_tools._select_tool_names(None, fresh, quiet_mode=True)
    assert "late_tool" in names and "late_sub" not in names


def test_review_fork_of_parent_keeps_own_platform(plugin_env, monkeypatch):
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"]}})
    fork = _build_review_fork(monkeypatch, _parent("slack", "slack", toolsets_mod.session_disabled_toolsets(None, "slack")))
    assert toolsets_mod.ceiling_platform_of(fork) == "slack"
    assert PLUGIN_TS not in fork.disabled_toolsets and OTHER_TS in fork.disabled_toolsets


def test_review_fork_without_ceiling_is_unchanged(plugin_env, monkeypatch):
    _use_config(monkeypatch, {})
    fork = _build_review_fork(monkeypatch, _parent("subagent", "slack", ["memory"]))
    assert fork.disabled_toolsets == toolsets_mod.session_disabled_toolsets(["memory"], "subagent")
    assert PLUGIN_TS not in fork.disabled_toolsets and OTHER_TS not in fork.disabled_toolsets


# --- CLI banner: counts tools with the same ceiling gate as the agent ----------------------------------------


class _BannerStop(Exception):
    pass


def _banner_cli(disabled):
    from types import SimpleNamespace

    from hermes_cli.cli_info_mixin import CLIInfoMixin

    class _Cli(CLIInfoMixin):
        pass

    cli = _Cli()
    cli.console = SimpleNamespace(clear=lambda: None)
    cli.enabled_toolsets, cli.disabled_toolsets = None, disabled
    return cli


def _banner_tool_names(monkeypatch, cli):
    """The tool names show_banner's cold path counts (get_tool_definitions' selection)."""
    import cli as cli_mod
    import hermes_cli.banner as banner_mod
    import hermes_cli.cli_info_mixin as info_mod

    captured = {}

    def _defs(enabled_toolsets=None, disabled_toolsets=None, **kwargs):
        captured["names"] = model_tools._select_tool_names(enabled_toolsets, disabled_toolsets, quiet_mode=True)
        raise _BannerStop

    monkeypatch.setattr(info_mod, "banner_plan", lambda cli, console: info_mod.BannerPlan(None, None, None, {}))
    monkeypatch.setattr(banner_mod, "load_banner_snapshot", lambda enabled_toolsets=None: None)
    monkeypatch.setattr(cli_mod, "get_tool_definitions", _defs)
    with pytest.raises(_BannerStop):
        cli.show_banner()
    return captured["names"]


def test_banner_count_drops_plugin_toolsets_outside_ceiling(plugin_env, monkeypatch):
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["cli"]}})
    names = _banner_tool_names(monkeypatch, _banner_cli(None))
    assert PLUGIN_TOOL in names and OTHER_TOOL not in names


def test_banner_count_without_ceiling_is_unchanged(plugin_env, monkeypatch):
    from hermes_cli.cli_info_mixin import banner_disabled_toolsets

    _use_config(monkeypatch, {})
    names = _banner_tool_names(monkeypatch, _banner_cli(None))
    assert PLUGIN_TOOL in names and OTHER_TOOL in names
    configured = [OTHER_TS]
    assert banner_disabled_toolsets(_banner_cli(configured)) is configured


def test_banner_gate_keeps_configured_disables(plugin_env, monkeypatch):
    from hermes_cli.cli_info_mixin import banner_disabled_toolsets

    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["cli"], OTHER_TS: ["cli"]}})
    assert banner_disabled_toolsets(_banner_cli(["memory"])) == ["memory"]
    _use_config(monkeypatch, {"plugin_toolset_platforms": {PLUGIN_TS: ["slack"], OTHER_TS: ["cli"]}})
    assert banner_disabled_toolsets(_banner_cli(["memory"])) == ["memory", PLUGIN_TS]

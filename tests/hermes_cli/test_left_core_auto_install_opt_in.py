"""``plugins.auto_install_left_core`` (default off) gates the left-core plugin migration on agent and
gateway start and on ``hermes update``, and it overrides ``security.allow_lazy_installs``: off, a home
that uses Home Assistant, Spotify or Langfuse without its plugin gets one notice per feature with the
install command, and no catalog lookup, no install and no config.yaml write (not even the toolset
scope). Package installs through pm keep following ``security.allow_lazy_installs`` alone."""

from __future__ import annotations

import io

import pytest

import hermes_cli.left_core_migration as lcm
import hermes_cli.memory_provider_migration as mpm

# Uses all three rows: a stored Spotify login, the bundled Langfuse plugin enabled, and the Spotify
# toolset on Telegram (a scope the migration would record in known_plugin_toolsets when on).
_CONFIG = (
    "memory:\n"
    "  provider: hindsight\n"
    "security:\n"
    "  allow_lazy_installs: true\n"
    "plugins:\n"
    "  enabled: [observability/langfuse]\n"
    "platform_toolsets:\n"
    "  telegram: [hermes-telegram, spotify]\n"
)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_undelivered", {})
    monkeypatch.setattr(mpm, "_attempted", set())
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    # The hermetic harness disables lazy installs by env; these homes allow them in config.yaml.
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)


@pytest.fixture
def no_network(monkeypatch):
    """Every way either migration reaches the catalog or the installer fails the test."""
    from hermes_cli import plugin_catalog, plugins_cmd

    def forbidden(what):
        def _fail(*_a, **_kw):
            pytest.fail(f"{what} called with plugins.auto_install_left_core off")
        return _fail

    monkeypatch.setattr(plugin_catalog, "load_catalog_live", forbidden("live catalog"))
    monkeypatch.setattr(mpm, "catalog_source", forbidden("catalog lookup"))
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", forbidden("installer"))
    monkeypatch.setattr(lcm, "_install_into", lambda home, **_kw: forbidden("installer"))
    monkeypatch.setattr(mpm, "_install_into", lambda home, **_kw: forbidden("installer"))


def _home(tmp_path, name="home"):
    home = tmp_path / name
    home.mkdir(parents=True)
    (home / ".env").write_text("HASS_TOKEN=abc\n", encoding="utf-8")
    (home / "auth.json").write_text('{"providers": {"spotify": {"refresh_token": "r"}}}', encoding="utf-8")
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    return home


def _snapshot(home):
    """What a migration could change: the home's own files (config.yaml first), an installed plugin
    and a failed-install backoff stamp. Profile scaffolding a gateway scope seeds (SOUL.md) is not."""
    files = {name: (home / name).read_bytes() for name in ("config.yaml", ".env", "auth.json")}
    return files, sorted((home / "plugins").rglob("*")), sorted(home.rglob("*.failed"))


def test_default_config_ships_auto_install_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["plugins"]["auto_install_left_core"] is False


def test_the_homes_use_every_row_and_allow_lazy_installs(tmp_path, monkeypatch):
    """Precondition for the tests below: without the new switch, these homes would migrate."""
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import pm.install
    assert pm.install.lazy_installs_allowed() is True
    assert [f.plugin for f in lcm._missing(home)] == ["homeassistant", "spotify", "langfuse"]


def test_agent_and_gateway_start_off_install_nothing_and_leave_config_identical(
        tmp_path, monkeypatch, no_network):
    from gateway.run_startup import recover_left_core_in

    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = _snapshot(home)

    recover_left_core_in(home, hydrate_secrets=False)   # gateway start: queued for the first agent
    said: list[str] = []
    assert lcm.recover_at_startup(say=said.append) == []  # first agent: delivered, not re-attempted
    assert mpm.recover_at_startup("hindsight", say=said.append) is False

    assert _snapshot(home) == before  # config.yaml byte-identical; no plugin, no backoff stamp
    assert len(said) == 4
    for line, plugin in zip(said, ["homeassistant", "spotify", "langfuse", "hindsight"]):
        assert "plugins.auto_install_left_core" in line and f"`hermes plugins install {plugin}`" in line


def test_update_off_installs_nothing_and_leaves_every_profile_config_identical(
        tmp_path, monkeypatch, no_network):
    root = tmp_path / ".hermes"
    homes = [_home(tmp_path, ".hermes"), _home(root, "profiles/work")]
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)
    before = [_snapshot(h) for h in homes]

    said: list[str] = []
    assert lcm.migrate_all_homes(say=said.append) == []
    assert mpm.migrate_all_homes(say=said.append) == []

    assert [_snapshot(h) for h in homes] == before
    assert len(said) == 8
    assert sum("[profile 'work']" in line and "`hermes -p work plugins install" in line for line in said) == 4


def test_a_plugin_already_installed_is_not_reported(tmp_path, monkeypatch, no_network):
    home = _home(tmp_path)
    for plugin in ("homeassistant", "spotify", "langfuse"):
        (home / "plugins" / plugin).mkdir(parents=True)
        (home / "plugins" / plugin / "plugin.yaml").write_text(f"name: {plugin}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = _snapshot(home)
    said: list[str] = []
    assert lcm.recover_at_startup(say=said.append) == []
    assert said == [] and _snapshot(home) == before


def test_opt_in_runs_the_migration_as_before(tmp_path, monkeypatch):
    home = _home(tmp_path)
    (home / "config.yaml").write_text(
        _CONFIG.replace("plugins:\n", "plugins:\n  auto_install_left_core: true\n"), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)
    installs = []
    monkeypatch.setattr(lcm, "_install_into",
                        lambda h, consent=None: lambda name: installs.append(name) or {"ok": True})
    assert lcm.recover_at_startup(say=lambda m: None) == ["homeassistant", "spotify", "langfuse"]
    assert installs == ["homeassistant", "spotify", "langfuse"]


def test_package_lazy_install_still_runs_with_the_switch_off(tmp_path, monkeypatch, no_network):
    """The switch covers catalog plugins only: with allow_lazy_installs on, a package install through
    pm (edge-tts for TTS here) still runs while the catalog plugin install in the same home does not."""
    import pm.client
    import pm.extras
    import pm.paths

    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    synced = []
    monkeypatch.setattr(pm.extras, "available", lambda extra: False)
    monkeypatch.setattr(pm.extras, "extra_supported", lambda extra: True)
    monkeypatch.setattr(pm.client, "sync_venv", lambda extras: synced.append(list(extras)))
    monkeypatch.setattr(pm.paths, "runtime_facts_path", lambda: tmp_path / "absent")
    monkeypatch.setattr("sys.stdin", io.StringIO())  # no terminal: no install prompt

    pm.extras.ensure_import("edge-tts")
    assert synced == [["edge-tts"]]
    assert lcm.recover_at_startup(say=lambda m: None) == []  # no_network: no plugin install


# Home Assistant's core-era scope: core never ran its toolset on ACP or webhook, nor on a platform
# whose saved list leaves it out. The migration records that in known_plugin_toolsets; with the
# switch off nothing is written, so the resolver applies the same scope read-only.
_HA_PLATFORMS = {"telegram": ["web"], "acp": ["hermes-acp"], "webhook": ["hermes-webhook"],
                 "discord": ["hermes-discord"]}


def _ha_home(tmp_path, monkeypatch, config, *, token=True):
    home = tmp_path / "ha"
    (home / "plugins" / "homeassistant").mkdir(parents=True)  # installed by hand or before the update
    (home / ".env").write_text("HASS_TOKEN=abc\n" if token else "", encoding="utf-8")
    (home / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _ha_states(home):
    from hermes_cli.tools_config import _enabled_plugin_toolsets

    config = lcm._read_config(home)
    return {platform: "homeassistant" in _enabled_plugin_toolsets(config, platform, names, {"homeassistant"})
            for platform, names in _HA_PLATFORMS.items()}


_CORE_SCOPE = {"telegram": False, "acp": False, "webhook": False, "discord": True}


@pytest.mark.parametrize("opt_in", [False, True])
def test_preinstalled_homeassistant_keeps_core_scope_either_way(tmp_path, monkeypatch, opt_in):
    text = (f"plugins:\n  auto_install_left_core: {str(opt_in).lower()}\n"
            "platform_toolsets:\n  telegram: [web]\n")
    home = _ha_home(tmp_path, monkeypatch, text)
    said = []
    assert lcm.recover_at_startup(say=said.append) == []
    assert said == []
    assert _ha_states(home) == _CORE_SCOPE
    if not opt_in:
        assert (home / "config.yaml").read_text(encoding="utf-8") == text


def test_recorded_scope_is_not_overridden(tmp_path, monkeypatch):
    """Once recorded, known_plugin_toolsets decides: a user who enabled it on ACP since keeps it."""
    home = _ha_home(tmp_path, monkeypatch, "_left_core_scoped: [homeassistant]\n"
                                           "platform_toolsets:\n  telegram: [web]\n"
                                           "known_plugin_toolsets:\n  telegram: [homeassistant]\n")
    assert _ha_states(home) == {"telegram": False, "acp": True, "webhook": True, "discord": True}


def test_core_scope_applies_only_to_a_home_that_used_home_assistant(tmp_path, monkeypatch):
    home = _ha_home(tmp_path, monkeypatch, "platform_toolsets:\n  telegram: [web]\n", token=False)
    assert _ha_states(home) == dict.fromkeys(_HA_PLATFORMS, True)


def test_a_saved_list_naming_home_assistant_keeps_it_on(tmp_path, monkeypatch):
    home = _ha_home(tmp_path, monkeypatch, "platform_toolsets:\n  acp: [hermes-acp, homeassistant]\n")
    from hermes_cli.tools_config import _enabled_plugin_toolsets

    config = lcm._read_config(home)
    assert _enabled_plugin_toolsets(config, "acp", ["hermes-acp", "homeassistant"], {"homeassistant"})

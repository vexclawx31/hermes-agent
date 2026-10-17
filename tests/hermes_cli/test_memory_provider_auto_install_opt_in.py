"""Installing a memory provider that left core is opt-in (``plugins.auto_install_left_core``, default
off). Off, a missing provider fails closed on agent init and ``hermes update``: one warning with the
install command, no catalog lookup, no install, the agent runs without it. On, the catalog install
runs as before. A provider bundled at ``plugins/memory/<name>`` is used with no install attempt."""

import logging
import sys

import pytest

from agent.agent_init import _init_memory
from agent.status_output import StatusOutputMixin
from hermes_cli import memory_provider_migration as mig


class StartupAgent(StatusOutputMixin):
    enabled_toolsets = disabled_toolsets = tools = []
    valid_tool_names = set()
    log_prefix = ""
    quiet_mode = False
    status_callback = notice_callback = tool_progress_callback = None

    def __init__(self, platform):
        self.platform = platform
        self.printed = []

    def _vprint(self, message, **kwargs):
        self.printed.append(message)


def _write_config(home, provider, auto_install=None):
    home.mkdir(parents=True, exist_ok=True)
    text = f"memory:\n  provider: {provider}\n"
    if auto_install is not None:
        text += f"plugins:\n  auto_install_left_core: {auto_install}\n"
    (home / "config.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def no_network(monkeypatch):
    """Every way a migration reaches the network or the installer fails the test."""
    from hermes_cli import plugin_catalog, plugins_cmd

    calls: list[str] = []

    def forbidden(what):
        def _fail(*_a, **_kw):
            calls.append(what)
            pytest.fail(f"{what} called with plugins.auto_install_left_core off")
        return _fail

    monkeypatch.setattr(plugin_catalog, "load_catalog_live", forbidden("live catalog"))
    monkeypatch.setattr(mig, "catalog_source", forbidden("catalog lookup"))
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", forbidden("installer"))
    monkeypatch.setattr(mig, "_install_into", lambda home, **_kw: forbidden("installer"))
    return calls


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(mig, "_attempted", set())
    # Prove the new switch is what refuses: the older lazy-install policy allows installs here.
    monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: True)


@pytest.fixture
def fake_installer(monkeypatch):
    """The catalog install path, recording each call and creating the plugin dir it would."""
    from hermes_cli import plugins_cmd
    from hermes_constants import get_hermes_home

    seen: list[dict] = []

    def install(*_a, catalog_name, **kw):
        seen.append({"home": get_hermes_home(), "catalog_name": catalog_name, **kw})
        plugin = get_hermes_home() / "plugins" / catalog_name
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text("class P(MemoryProvider): ...\n", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(mig, "catalog_source", lambda name: name)
    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    return seen


def test_default_config_ships_auto_install_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["plugins"]["auto_install_left_core"] is False


@pytest.mark.parametrize(("value", "enabled"), [
    (None, False),      # key absent: the shipped default
    ("false", False),
    ("true", True),
    ('"yes"', False),   # not a YAML boolean: fail closed
])
def test_switch_is_read_from_the_home_config(tmp_path, monkeypatch, value, enabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))  # the home argument wins
    _write_config(tmp_path, "hindsight", value)
    assert mig.auto_install_enabled(tmp_path) is enabled



def test_unreadable_config_counts_as_off(tmp_path, monkeypatch):
    def broken():
        raise ValueError("config.yaml: mapping values are not allowed here")

    _write_config(tmp_path, "hindsight", "true")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", broken)
    assert mig.auto_install_enabled(tmp_path) is False


def test_malformed_config_counts_as_off_even_with_a_last_known_good_opt_in(tmp_path, monkeypatch):
    """The loader does not raise on bad YAML: it serves the last-known-good config, tagged
    FailedConfigRead. An opt-in from that fallback is not what the file says now."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.config_read_errors import FailedConfigRead

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("plugins:\n  auto_install_left_core: true\n", encoding="utf-8")
    assert mig.auto_install_enabled(tmp_path) is True
    path.write_text("plugins:\n  auto_install_left_core: false\nbroken: [\n", encoding="utf-8")
    fallback = load_config_readonly()
    assert isinstance(fallback, FailedConfigRead)
    assert fallback["plugins"]["auto_install_left_core"] is True  # the stale opt-in is still served
    assert mig.auto_install_enabled(tmp_path) is False

def test_agent_init_default_off_warns_once_and_makes_no_network_call(
        tmp_path, monkeypatch, caplog, fresh, no_network):
    _write_config(tmp_path, "hindsight")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = StartupAgent("cli")
    monkeypatch.setattr(agent, "_warning_presentation_enabled", lambda: True)
    with caplog.at_level(logging.WARNING, logger=mig.__name__):
        _init_memory(agent, {"memory": {"provider": "hindsight"}}, False, "cli")
        assert mig.recover_at_startup("hindsight") is False  # same process: not repeated

    assert agent._memory_manager is None
    assert no_network == []
    assert not (tmp_path / "plugins").exists()
    assert not (tmp_path / "cache" / "left-core-hindsight.failed").exists()  # no failed-install backoff
    warnings = [r.getMessage() for r in caplog.records if r.name == mig.__name__]
    assert len(warnings) == 1
    assert "'hindsight'" in warnings[0] and "plugins.auto_install_left_core" in warnings[0]
    assert "`hermes plugins install hindsight`" in warnings[0]
    assert any("hermes plugins install hindsight" in line for line in agent.printed)


def test_update_default_off_names_each_profile_and_installs_nothing(tmp_path, monkeypatch, no_network):
    root = tmp_path / ".hermes"
    homes = [root, root / "profiles" / "work"]
    for profile_home in homes:
        _write_config(profile_home, "honcho")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: homes)

    said: list[str] = []
    assert mig.migrate_all_homes(say=said.append) == []
    assert no_network == []
    assert not any((h / "plugins").exists() for h in homes)
    assert len(said) == 2
    assert said[0].startswith("  [profile 'default']") and "`hermes plugins install honcho`" in said[0]
    assert said[1].startswith("  [profile 'work']") and "`hermes -p work plugins install honcho`" in said[1]
    assert all("plugins.auto_install_left_core" in line for line in said)


def test_agent_init_opt_in_installs_from_the_catalog(tmp_path, monkeypatch, fresh, fake_installer):
    _write_config(tmp_path, "hindsight", "true")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    said: list[str] = []
    assert mig.recover_at_startup("hindsight", say=said.append) is True
    assert [(c["catalog_name"], c["assume_deps_consent"]) for c in fake_installer] == [("hindsight", True)]
    assert "installed its plugin from the catalog" in said[0]


def test_update_installs_only_for_profiles_that_opted_in(tmp_path, monkeypatch, fake_installer):
    import io

    root = tmp_path / ".hermes"
    default, work = root, root / "profiles" / "work"
    _write_config(default, "honcho")
    _write_config(work, "honcho", "true")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: [default, work])
    monkeypatch.setattr("sys.stdin", io.StringIO())  # scripted update: no terminal
    monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: True)

    said: list[str] = []
    assert mig.migrate_all_homes(say=said.append) == ["honcho"]
    assert [c["home"] for c in fake_installer] == [work]
    assert not (default / "plugins").exists()
    assert "[profile 'default']" in said[0] and "plugins.auto_install_left_core" in said[0]


@pytest.fixture
def bundled_hindsight(tmp_path, monkeypatch):
    """A Hindsight provider at the bundled ``plugins/memory/hindsight`` path."""
    import plugins.memory as memory_pkg

    bundled = tmp_path / "bundled"
    (bundled / "hindsight").mkdir(parents=True)
    (bundled / "hindsight" / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n\n\n"
        "class BundledHindsight(MemoryProvider):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'hindsight'\n\n"
        "    def is_available(self):\n"
        "        return True\n\n"
        "    def initialize(self, session_id, **kwargs):\n"
        "        pass\n\n"
        "    def get_tool_schemas(self):\n"
        "        return []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_pkg, "_MEMORY_PLUGINS_DIR", bundled)
    monkeypatch.delitem(sys.modules, "plugins.memory.hindsight", raising=False)
    yield bundled
    sys.modules.pop("plugins.memory.hindsight", None)


@pytest.mark.parametrize("auto_install", [None, "true"])
def test_bundled_provider_is_used_without_an_install_attempt(
        tmp_path, monkeypatch, fresh, no_network, bundled_hindsight, auto_install):
    home = tmp_path / "home"
    _write_config(home, "hindsight", auto_install)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(mig, "recover_at_startup", lambda *_a, **_kw: pytest.fail("recovery attempted"))
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: [home])
    # Session scoping kwargs read agent state this stub does not carry; the provider ignores them.
    monkeypatch.setattr("agent.agent_init._memory_provider_init_kwargs",
                        lambda agent, platform: {"session_id": "s", "platform": platform})

    agent = StartupAgent("telegram")
    _init_memory(agent, {"memory": {"provider": "hindsight"}}, False, "telegram")
    assert [p.name for p in agent._memory_manager.providers] == ["hindsight"]

    said: list[str] = []
    assert mig.migrate_all_homes(say=said.append) == []
    assert said == [] and no_network == []
    assert not (home / "plugins").exists()

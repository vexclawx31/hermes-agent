"""``PluginContext.inject_message(author=...)``: capability, host routing and strict validation."""

from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import hermes_yaml as yaml

from agent.plugin_author import PluginAuthor
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "switchboard"
AUTHOR = {"id": "research", "name": "Research Agent"}
ORIGIN = {"platform": "telegram", "chat_id": "100", "chat_type": "group", "user_id": "u1"}


def _context(key: str = PLUGIN) -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    return PluginContext(PluginManifest(name=key, key=key, source="user"), manager), manager


def _config(tmp_path, monkeypatch, entry: dict, key: str = PLUGIN) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {key: entry}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))


GRANTED = {"allow_gateway_injection": True, "allow_injection_author": True}


def _gateway(tmp_path, monkeypatch, entry=GRANTED, key: str = PLUGIN):
    _config(tmp_path, monkeypatch, entry, key)
    ctx, manager = _context(key)
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)
    return ctx, manager, injector


def test_author_reaches_the_gateway_as_a_frozen_host_object(tmp_path, monkeypatch):
    ctx, _, injector = _gateway(tmp_path, monkeypatch)
    assert ctx.inject_message("result", origin=ORIGIN, author=AUTHOR) is True
    kwargs = injector.call_args.kwargs
    assert kwargs["author"] == PluginAuthor(PLUGIN, "research", "Research Agent")
    assert type(kwargs["author"]) is PluginAuthor
    assert kwargs["content"] == "result" and kwargs["plugin_id"] == PLUGIN and kwargs["origin"] == ORIGIN


def test_author_works_on_the_session_key_path_too(tmp_path, monkeypatch):
    ctx, _, injector = _gateway(tmp_path, monkeypatch)
    assert ctx.inject_message("result", session_key="agent:main:telegram:dm:1", author=AUTHOR) is True
    assert injector.call_args.kwargs == {"session_key": "agent:main:telegram:dm:1", "content": "result",
                                         "plugin_id": PLUGIN, "author": PluginAuthor(PLUGIN, "research",
                                                                                     "Research Agent")}


@pytest.mark.parametrize("target", [{"origin": ORIGIN}, {"session_key": "agent:main:telegram:dm:1"}])
def test_author_absent_sends_exactly_todays_gateway_kwargs(tmp_path, monkeypatch, target):
    """Vex condition 1: without ``author`` (or with ``author=None``) the gateway injector receives
    exactly today's keyword set, so existing hosts and relays see no difference."""
    ctx, manager, injector = _gateway(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    assert ctx.inject_message("hello", **target) is True
    assert ctx.inject_message("hello", **target, author=None) is True
    expected = {**target, "content": "hello", "plugin_id": PLUGIN}
    if "origin" in target:
        expected["plugin_home"] = manager.home_path
    assert [c.kwargs for c in injector.call_args_list] == [expected, expected]


@pytest.mark.parametrize("entry", [
    {"allow_gateway_injection": True},
    {"allow_gateway_injection": True, "allow_injection_author": "true"},
    {"allow_gateway_injection": True, "allow_injection_author": 1},
    {"allow_injection_author": True},
    {"allow_gateway_injection": False, "allow_injection_author": True},
    {},
])
def test_author_without_both_flags_is_refused(tmp_path, monkeypatch, entry):
    ctx, _, injector = _gateway(tmp_path, monkeypatch, entry)
    assert ctx.inject_message("result", origin=ORIGIN, author=AUTHOR) is False
    injector.assert_not_called()


def test_author_fails_closed_when_the_config_cannot_be_read(tmp_path, monkeypatch):
    ctx, _, injector = _gateway(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.plugins.load_config_readonly", MagicMock(side_effect=OSError("boom")))
    assert ctx.inject_message("result", origin=ORIGIN, author=AUTHOR) is False
    injector.assert_not_called()


@pytest.mark.parametrize("key", ["memory/honcho", "Switchboard", "1switch", "a" * 33])
def test_plugin_whose_manifest_key_fails_cannot_use_the_capability(tmp_path, monkeypatch, key):
    ctx, _, injector = _gateway(tmp_path, monkeypatch, GRANTED, key=key)
    assert ctx.inject_message("result", origin=ORIGIN, author=AUTHOR) is False
    injector.assert_not_called()
    assert ctx.inject_message("result", origin=ORIGIN) is True  # the plain injection is unaffected


@pytest.mark.parametrize("author", [
    {"id": "Research", "name": "R"}, {"id": "research ", "name": "R"}, {"id": "a.b", "name": "R"},
    {"id": "research", "name": "Evil]"}, {"id": "research", "name": "R", "is_bot": True},
    {"id": "research"}, "research", ["research", "R"],
])
def test_invalid_author_is_refused_never_cleaned(tmp_path, monkeypatch, author):
    ctx, _, injector = _gateway(tmp_path, monkeypatch)
    assert ctx.inject_message("result", origin=ORIGIN, author=author) is False
    injector.assert_not_called()


def test_author_requires_the_user_role(tmp_path, monkeypatch):
    ctx, _, injector = _gateway(tmp_path, monkeypatch)
    assert ctx.inject_message("result", "system", origin=ORIGIN, author=AUTHOR) is False
    injector.assert_not_called()


@pytest.mark.parametrize("running", [False, True])
def test_cli_with_author_returns_false_and_queues_nothing(tmp_path, monkeypatch, running):
    _config(tmp_path, monkeypatch, GRANTED)
    ctx, manager = _context()
    cli = SimpleNamespace(_agent_running=running, _pending_input=SimpleQueue(), _interrupt_queue=SimpleQueue())
    manager._cli_ref = cli
    assert ctx.inject_message("result", author=AUTHOR) is False
    assert cli._pending_input.empty() and cli._interrupt_queue.empty()


def test_tui_host_never_receives_an_authored_message(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, GRANTED)
    ctx, manager = _context()
    tui = MagicMock(return_value=True)
    manager.set_tui_message_injector(object(), tui)
    assert ctx.inject_message("result", session_key="ses_1", author=AUTHOR) is False  # no gateway either
    tui.assert_not_called()
    assert ctx.inject_message("result", session_key="ses_1") is True  # today's TUI path is unchanged
    tui.assert_called_once()


def test_old_gateway_host_without_author_support_fails_closed(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, GRANTED)
    ctx, manager = _context()

    def legacy_host(*, content, plugin_id, session_key=None, origin=None, plugin_home=None):
        return True

    manager.set_gateway_message_injector(object(), legacy_host)
    assert ctx.inject_message("result", origin=ORIGIN, author=AUTHOR) is False

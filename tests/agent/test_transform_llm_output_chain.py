"""``apply_llm_output_transform`` runs the whole priority-ordered ``transform_llm_output`` chain.

Real plugin manager, real lifecycle dispatch: a guard registered with a low priority sees (and has
the last word on) the text every earlier transform produced; ``pre_transform`` is the text before
the whole chain.
"""

import logging
from types import SimpleNamespace

import pytest

from agent.turn_finalizer import apply_llm_output_transform
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

_LOG = logging.getLogger("test.transform_chain")


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: mgr)
    return mgr


def _agent():
    return SimpleNamespace(session_id="s1", model="m", platform="telegram")


def _ctx(manager, name):
    return PluginContext(manager=manager, manifest=PluginManifest(name=name, source="user"))


def test_guard_with_low_priority_gets_the_last_word(manager):
    seen = []

    def guard(response_text, **_):
        seen.append(response_text)
        return "[withheld]" if "secret" in response_text else None

    _ctx(manager, "guard").register_hook("transform_llm_output", guard, priority=-1000)
    _ctx(manager, "style").register_hook("transform_llm_output", lambda response_text, **_: response_text + " secret")

    final, transformed, pre = apply_llm_output_transform(_agent(), "hello", turn_id="t1", logger=_LOG)
    assert (final, transformed, pre) == ("[withheld]", True, "hello")
    assert seen == ["hello secret"]


def test_one_hook_unchanged_and_idempotent_per_turn(manager):
    calls = []

    def shout(response_text, **_):
        calls.append(response_text)
        return response_text.upper()

    _ctx(manager, "style").register_hook("transform_llm_output", shout)
    agent = _agent()
    assert apply_llm_output_transform(agent, "hi", turn_id="t1", logger=_LOG) == ("HI", True, "hi")
    # A later seam in the same turn gets the recorded outcome, not a second firing.
    assert apply_llm_output_transform(agent, "HI", turn_id="t1", logger=_LOG) == ("HI", True, "hi")
    assert calls == ["hi"]


def test_chain_of_no_ops_leaves_the_response_untouched(manager):
    _ctx(manager, "a").register_hook("transform_llm_output", lambda **_: None)
    _ctx(manager, "b").register_hook("transform_llm_output", lambda **_: "")
    assert apply_llm_output_transform(_agent(), "hi", turn_id="t1", logger=_LOG) == ("hi", False, None)

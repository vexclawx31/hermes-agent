"""Hook priority and the ``transform_llm_output`` chain.

Before this, core kept the first ``transform_llm_output`` string and stopped, so any transform
registered earlier pre-empted a guard plugin that had to check the text the user receives. Now every
callback runs as a chain in priority order (higher first, ties in registration order), each one
receiving the previous stage's output.
"""

import logging
import threading

import pytest

from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: mgr)
    return mgr


def _ctx(manager, name="probe"):
    return PluginContext(manager=manager, manifest=PluginManifest(name=name, source="user"))


def _appender(tag, seen=None):
    def transform(response_text, **kwargs):
        if seen is not None:
            seen.append((tag, response_text, kwargs.get("chain_position"), kwargs.get("chain_length")))
        return f"{response_text}|{tag}"
    transform.__name__ = f"transform_{tag}"
    return transform


def test_single_hook_behaves_as_before(manager):
    _ctx(manager).register_hook("transform_llm_output", lambda response_text, **_: response_text.upper())
    assert manager.invoke_hook("transform_llm_output", response_text="hello") == ["HELLO"]


def test_single_hook_returning_none_leaves_text_unchanged(manager):
    _ctx(manager).register_hook("transform_llm_output", lambda response_text, **_: None)
    assert manager.invoke_hook("transform_llm_output", response_text="hello") == []


def test_narrow_legacy_signature_still_works(manager):
    def legacy(response_text):  # no **kwargs: never sees chain_position/chain_length
        return response_text + "!"
    _ctx(manager).register_hook("transform_llm_output", legacy)
    assert manager.invoke_hook("transform_llm_output", response_text="hi") == ["hi!"]


def test_chain_feeds_each_stage_the_previous_output_in_registration_order(manager):
    seen = []
    ctx = _ctx(manager)
    ctx.register_hook("transform_llm_output", _appender("a", seen))
    ctx.register_hook("transform_llm_output", _appender("b", seen))
    ctx.register_hook("transform_llm_output", _appender("c", seen))

    assert manager.invoke_hook("transform_llm_output", response_text="x") == ["x|a|b|c"]
    assert seen == [("a", "x", 0, 3), ("b", "x|a", 1, 3), ("c", "x|a|b", 2, 3)]


def test_priority_orders_the_chain_and_a_low_priority_guard_sees_the_final_text(manager):
    seen = []
    guard_ctx, other_ctx = _ctx(manager, "guard"), _ctx(manager, "other")
    # The guard registers FIRST but with a very low priority: it must still run last.
    guard_ctx.register_hook("transform_llm_output", _appender("guard", seen), priority=-1000)
    other_ctx.register_hook("transform_llm_output", _appender("plain", seen))
    other_ctx.register_hook("transform_llm_output", _appender("early", seen), priority=10)

    assert manager.invoke_hook("transform_llm_output", response_text="x") == ["x|early|plain|guard"]
    assert seen[-1] == ("guard", "x|early|plain", 2, 3)
    assert [cb.__name__ for cb in manager.iter_hook_callbacks("transform_llm_output")] == [
        "transform_early", "transform_plain", "transform_guard"]


def test_priority_orders_ordinary_hooks_too(manager):
    order = []
    ctx = _ctx(manager)
    ctx.register_hook("post_llm_call", lambda **_: order.append("default"))
    ctx.register_hook("post_llm_call", lambda **_: order.append("high"), priority=5)
    ctx.register_hook("post_llm_call", lambda **_: order.append("low"), priority=-5)
    manager.invoke_hook("post_llm_call", session_id="s")
    assert order == ["high", "default", "low"]


def test_priority_must_be_an_int(manager):
    with pytest.raises(TypeError):
        _ctx(manager).register_hook("transform_llm_output", _appender("a"), priority="last")
    with pytest.raises(TypeError):
        _ctx(manager).register_hook("transform_llm_output", _appender("a"), priority=True)


def test_failing_stage_is_skipped_and_later_stages_still_run(manager, caplog):
    seen = []

    def broken(response_text, **_):
        raise RuntimeError("boom")

    ctx = _ctx(manager)
    ctx.register_hook("transform_llm_output", _appender("a", seen))
    ctx.register_hook("transform_llm_output", broken)
    ctx.register_hook("transform_llm_output", _appender("guard", seen), priority=-1000)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        assert manager.invoke_hook("transform_llm_output", response_text="x") == ["x|a|guard"]
    assert seen[-1] == ("guard", "x|a", 2, 3)
    assert any("broken" in r.getMessage() and "boom" in r.getMessage() for r in caplog.records)


def test_timed_out_stage_is_skipped_and_later_stages_still_run(manager, monkeypatch):
    monkeypatch.setattr(plugins_module, "_resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()

    def hung(response_text, **_):
        release.wait(30)
        return "LATE"

    ctx = _ctx(manager)
    ctx.register_hook("transform_llm_output", hung)
    ctx.register_hook("transform_llm_output", _appender("guard"), priority=-1000)
    try:
        assert manager.invoke_hook("transform_llm_output", response_text="x", turn_id="t1") == ["x|guard"]
    finally:
        release.set()


def test_unregistering_drops_the_priority(manager):
    ctx = _ctx(manager)
    guard = _appender("guard")
    handle = ctx.register_hook("transform_llm_output", guard, priority=-1000)
    handle.dispose()
    assert manager._hook_priorities == {}
    assert manager.iter_hook_callbacks("transform_llm_output") == ()


WITHHELD = "[reply withheld]"


def _register_guard(manager, callback, **options):
    _ctx(manager, "guard").register_hook("transform_llm_output", callback, priority=-1000, **options)


def test_fail_closed_stage_that_raises_replaces_the_text(manager):
    def guard(response_text, **_):
        raise RuntimeError("judge unavailable")

    _ctx(manager, "style").register_hook("transform_llm_output", _appender("a"))
    _register_guard(manager, guard, fail_closed_text=WITHHELD)
    assert manager.invoke_hook("transform_llm_output", response_text="secret") == [WITHHELD]


def test_fail_closed_stage_that_times_out_replaces_the_text(manager, monkeypatch):
    monkeypatch.setattr(plugins_module, "_resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()

    def guard(response_text, **_):
        release.wait(30)
        return response_text

    _register_guard(manager, guard, fail_closed_text=WITHHELD)
    try:
        assert manager.invoke_hook("transform_llm_output", response_text="secret", turn_id="t1") == [WITHHELD]
    finally:
        release.set()


def test_fail_closed_stage_is_retried_on_the_next_call_after_a_timeout(manager, monkeypatch):
    monkeypatch.setattr(plugins_module, "_resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()
    calls = []

    def guard(response_text, **_):
        calls.append(response_text)
        if len(calls) == 1:
            release.wait(30)  # first call hangs past the timeout
        return f"checked:{response_text}"

    _register_guard(manager, guard, fail_closed_text=WITHHELD)
    try:
        assert manager.invoke_hook("transform_llm_output", response_text="one", turn_id="t1") == [WITHHELD]
        # Well inside the 60 s suppression window: a fail-closed stage still runs.
        assert manager.invoke_hook("transform_llm_output", response_text="two", turn_id="t2") == ["checked:two"]
    finally:
        release.set()
    assert calls == ["one", "two"]


def test_non_fail_closed_stage_keeps_the_suppression_window(manager, monkeypatch):
    monkeypatch.setattr(plugins_module, "_resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()
    calls = []

    def stage(response_text, **_):
        calls.append(response_text)
        if len(calls) == 1:
            release.wait(30)
        return f"styled:{response_text}"

    _ctx(manager).register_hook("transform_llm_output", stage)
    try:
        assert manager.invoke_hook("transform_llm_output", response_text="one", turn_id="t1") == []
        assert manager.invoke_hook("transform_llm_output", response_text="two", turn_id="t2") == []
    finally:
        release.set()
    assert calls == ["one"]  # suppressed: the second call never reached the callback


def test_fail_closed_text_is_validated(manager):
    with pytest.raises(ValueError):
        _ctx(manager).register_hook("post_llm_call", lambda **_: None, fail_closed_text=WITHHELD)
    with pytest.raises(TypeError):
        _register_guard(manager, _appender("g"), fail_closed_text="")


def test_unregistering_drops_the_fail_closed_text(manager):
    handle = _ctx(manager).register_hook("transform_llm_output", _appender("g"), fail_closed_text=WITHHELD)
    handle.dispose()
    assert manager._hook_fail_closed_texts == {}


def test_module_level_dispatch_uses_the_chain(manager):
    ctx = _ctx(manager)
    ctx.register_hook("transform_llm_output", _appender("a"))
    ctx.register_hook("transform_llm_output", _appender("b"))
    assert plugins_module.invoke_hook("transform_llm_output", response_text="x") == ["x|a|b"]

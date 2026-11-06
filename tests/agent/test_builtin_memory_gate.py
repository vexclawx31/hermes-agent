"""``filter_builtin_memory``: a plugin can keep MEMORY.md / USER.md out of a chat, failing closed.

Real plugin manager and a real on-disk memory store under a temp ``HERMES_HOME``: the system-prompt
memory block and the ``memory`` tool both go through the same hook.
"""

import json
import threading
from types import SimpleNamespace

import pytest

from agent.builtin_memory_gate import OMITTED_NOTE
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.system_prompt import _memory_parts
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.memory_tool_store import MemoryStore

FACT = "The household safe code lives in the blue binder."


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: mgr)
    return mgr


@pytest.fixture()
def agent(manager):
    seed = MemoryStore(memory_char_limit=500, user_char_limit=300)
    seed.load_from_disk()
    assert seed.add("memory", FACT)["success"]
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()  # the prompt snapshot is taken at load time
    return SimpleNamespace(
        _memory_store=store, _memory_enabled=True, _user_profile_enabled=True, _memory_manager=None,
        session_id="s1", platform="slack", _chat_id="C123", _chat_type="group", _user_id="U9",
        _thread_id=None, _gateway_session_key="agent:main:slack:group:C123",
        _build_memory_write_metadata=lambda **kwargs: kwargs,
    )


def _register(manager, callback):
    PluginContext(manager=manager, manifest=PluginManifest(name="gate", source="user")).register_hook(
        "filter_builtin_memory", callback)


def _prompt(agent):
    return "\n\n".join(_memory_parts(agent))


def _memory_tool(agent, **args):
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")
    return json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, args, ctx))


def test_no_hook_leaves_the_block_unchanged(agent):
    assert FACT in _prompt(agent)
    assert OMITTED_NOTE not in _prompt(agent)


def test_allow_keeps_the_block_and_sees_the_chat_identity(manager, agent):
    calls = []

    def gate(**kwargs):
        calls.append(kwargs)
        return "allow"

    _register(manager, gate)
    assert FACT in _prompt(agent)
    seen = calls[0]
    assert (seen["session_id"], seen["platform"], seen["chat_id"], seen["chat_type"], seen["sender_id"]) == (
        "s1", "slack", "C123", "group", "U9")
    assert seen["surface"] == "system_prompt"


def test_none_is_no_opinion(manager, agent):
    _register(manager, lambda **_: None)
    assert FACT in _prompt(agent)


def test_omit_replaces_the_block_with_a_note(manager, agent):
    _register(manager, lambda chat_type, **_: "omit" if chat_type == "group" else "allow")
    prompt = _prompt(agent)
    assert FACT not in prompt
    assert prompt == OMITTED_NOTE


def test_any_omit_wins_over_allow(manager, agent):
    _register(manager, lambda **_: "allow")
    _register(manager, lambda **_: "omit")
    assert FACT not in _prompt(agent)


def test_exception_fails_closed(manager, agent):
    def gate(**_):
        raise RuntimeError("membership lookup failed")

    _register(manager, gate)
    assert _prompt(agent) == OMITTED_NOTE


def test_unknown_answer_fails_closed(manager, agent):
    _register(manager, lambda **_: "maybe")
    assert _prompt(agent) == OMITTED_NOTE


def test_timeout_fails_closed(manager, agent, monkeypatch):
    monkeypatch.setattr(plugins_module, "_resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()

    def gate(**_):
        release.wait(30)
        return "allow"

    _register(manager, gate)
    try:
        assert _prompt(agent) == OMITTED_NOTE
    finally:
        release.set()


def test_dispatch_failure_fails_closed(manager, agent, monkeypatch):
    _register(manager, lambda **_: "allow")

    def broken(*_args, **_kwargs):
        raise RuntimeError("plugin manager unavailable")

    monkeypatch.setattr(plugins_module, "invoke_hook", broken)
    assert _prompt(agent) == OMITTED_NOTE


def test_memory_tool_refuses_on_omit_without_touching_the_store(manager, agent):
    surfaces = []

    def gate(surface, **_):
        surfaces.append(surface)
        return "omit"

    _register(manager, gate)
    result = _memory_tool(agent, action="remove", target="memory", old_text="nothing like this")
    assert result.get("success") is False
    assert OMITTED_NOTE in result["error"]
    assert FACT not in json.dumps(result)
    assert surfaces == ["memory_tool"]
    agent._memory_store.load_from_disk()
    assert FACT in agent._memory_store.format_for_system_prompt("memory")


def test_memory_tool_works_on_allow(manager, agent):
    _register(manager, lambda **_: "allow")
    assert _memory_tool(agent, action="add", target="user", content="Prefers tea.")["success"]

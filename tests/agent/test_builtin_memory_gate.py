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


# --- circles: per-entry filter and write circle --------------------------------------------------

HOUSEHOLD = "[circle: household]\nThe family dentist is Dr. Moreau."
PRIVATE = "[circle: private_to:jamal]\nJamal is planning a surprise party."
EVERYONE = "[circle: everyone]\nAnswer in short paragraphs."
GROUP = "[circle: group:whatsapp:123@g.us, household]\nThe book club meets on Thursdays."


@pytest.fixture()
def circle_agent(agent):
    """The base agent's store plus one entry per circle shape (FACT stays untagged)."""
    store = MemoryStore(memory_char_limit=1000, user_char_limit=300)
    store.load_from_disk()
    for entry in (HOUSEHOLD, PRIVATE, EVERYONE, GROUP):
        assert store.add("memory", entry)["success"]
    store.load_from_disk()
    agent._memory_store = store
    return agent


def _keep_only(*allowed):
    """A keep callback that keeps entries whose tags all lie in *allowed* (and untagged ones never)."""
    return lambda tags: bool(tags) and tags <= set(allowed)


def _disk_entries():
    fresh = MemoryStore(memory_char_limit=1000, user_char_limit=300)
    fresh.load_from_disk()
    return fresh.memory_entries


def test_keep_renders_only_kept_entries_with_their_markers(manager, circle_agent):
    seen = []

    def gate(**kwargs):
        seen.append(kwargs["surface"])
        return {"keep": _keep_only("everyone", "household")}

    _register(manager, gate)
    prompt = _prompt(circle_agent)
    assert "[circle: household]\nThe family dentist" in prompt
    assert "[circle: everyone]\nAnswer in short" in prompt
    assert "surprise party" not in prompt and "book club" not in prompt and FACT not in prompt
    assert seen == ["system_prompt"]


def test_keep_receives_tags_as_frozensets(manager, circle_agent):
    received = []

    def keep(tags):
        received.append(tags)
        return True

    _register(manager, lambda **_: {"keep": keep})
    _prompt(circle_agent)
    assert all(isinstance(tags, frozenset) for tags in received)
    assert frozenset() in received  # the untagged FACT
    assert frozenset({"group:whatsapp:123@g.us", "household"}) in received


def test_every_keep_must_keep_an_entry(manager, circle_agent):
    _register(manager, lambda **_: {"keep": _keep_only("everyone", "household")})
    _register(manager, lambda **_: {"keep": lambda tags: "everyone" in tags})
    prompt = _prompt(circle_agent)
    assert "Answer in short" in prompt and "dentist" not in prompt


def test_omit_still_wins_over_keep(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: True})
    _register(manager, lambda **_: "omit")
    assert _prompt(circle_agent) == OMITTED_NOTE


def test_a_raising_keep_hides_the_entry(manager, circle_agent):
    def keep(tags):
        if "household" in tags:
            raise RuntimeError("roster lookup failed")
        return "everyone" in tags

    _register(manager, lambda **_: {"keep": keep})
    prompt = _prompt(circle_agent)
    assert "Answer in short" in prompt
    assert "dentist" not in prompt and "book club" not in prompt


def test_a_truthy_non_true_keep_hides_the_entry(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: "yes"})
    assert _prompt(circle_agent) == ""


@pytest.mark.parametrize("answer", [
    {"keep": "not callable"}, {"write_circle": ""}, {"write_circle": ["ok", 3]}, {"write_circle": "a]b"},
    {"write_circle": []}, {"action": "block", "message": "timed out"}, {},
])
def test_invalid_dict_verdicts_fail_closed(manager, circle_agent, answer):
    _register(manager, lambda **_: answer)
    assert _prompt(circle_agent) == OMITTED_NOTE


def test_hidden_entries_are_invisible_to_the_memory_tool(manager, circle_agent):
    _register(manager, lambda **_: {"keep": _keep_only("everyone", "household")})
    missing = _memory_tool(circle_agent, action="remove", target="memory", old_text="surprise party")
    assert missing["success"] is False
    dumped = json.dumps(missing)
    # The error echoes the model's own old_text, never a hidden entry.
    assert "Jamal is planning" not in dumped and "book club" not in dumped and FACT not in dumped
    assert PRIVATE in _disk_entries()


def test_ambiguity_is_judged_on_visible_entries_only(manager, circle_agent):
    _register(manager, lambda **_: {"keep": _keep_only("everyone", "household")})
    # "The" matches the dentist, party, book club and FACT entries; only the dentist one is visible.
    result = _memory_tool(circle_agent, action="patch", target="memory", old_text="Dr. Moreau",
                          content="Dr. Martin")
    assert result["success"] is True
    entries = _disk_entries()
    assert "[circle: household]\nThe family dentist is Dr. Martin." in entries
    assert PRIVATE in entries and GROUP in entries and FACT in entries  # hidden entries untouched
    assert entries.index(PRIVATE) == 2  # and kept in place


def test_patch_cannot_change_a_marker(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: True})
    result = _memory_tool(circle_agent, action="patch", target="memory", old_text="[circle: household]",
                          content="[circle: everyone]")
    assert result["success"] is False and "marker" in result["error"]
    assert HOUSEHOLD in _disk_entries()


def test_batch_remove_of_every_visible_entry_keeps_hidden_ones(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: tags == frozenset({"everyone"})})
    result = _memory_tool(circle_agent, target="memory",
                          operations=[{"action": "remove", "old_text": "short paragraphs"}])
    assert result["success"] is True
    assert _disk_entries() == [FACT, HOUSEHOLD, PRIVATE, GROUP]


def test_char_limit_counts_hidden_entries(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: tags == frozenset({"everyone"}),
                                    "write_circle": "everyone"})
    used = len("\n§\n".join(_disk_entries()))
    filler = "x" * (1000 - used)  # fits if hidden entries were not counted
    result = _memory_tool(circle_agent, action="add", target="memory", content=filler)
    assert result["success"] is False
    assert all("surprise party" not in entry for entry in result.get("current_entries", []))


def test_add_stamps_the_write_circle_and_strips_model_markers(manager, circle_agent):
    surfaces = []

    def gate(surface, **_):
        surfaces.append(surface)
        return {"keep": lambda tags: True, "write_circle": "group:whatsapp:123@g.us"}

    _register(manager, gate)
    result = _memory_tool(circle_agent, action="add", target="memory",
                          content="[circle: everyone] [circle: household]\nMovie night is Friday.")
    assert result["success"] is True
    assert surfaces == ["memory_tool_write"]
    assert "[circle: group:whatsapp:123@g.us]\nMovie night is Friday." in _disk_entries()


def test_batch_adds_are_stamped_too(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: True, "write_circle": ["household", "private_to:jamal"]})
    result = _memory_tool(circle_agent, target="memory", operations=[
        {"action": "add", "new_text": "[circle: everyone]\nGift idea: a kite."},
        {"action": "patch", "old_text": "short paragraphs", "content": "short answers"},
    ])
    assert result["success"] is True
    entries = _disk_entries()
    assert "[circle: household, private_to:jamal]\nGift idea: a kite." in entries
    assert "[circle: everyone]\nAnswer in short answers." in entries


def test_keep_without_write_circle_still_strips_model_markers(manager, circle_agent):
    _register(manager, lambda **_: {"keep": lambda tags: True})
    assert _memory_tool(circle_agent, action="add", target="memory",
                        content="[circle: everyone]\nLikes jazz.")["success"]
    assert "Likes jazz." in _disk_entries()


def test_old_plugin_add_is_stored_verbatim(manager, circle_agent):
    _register(manager, lambda **_: "allow")
    assert _memory_tool(circle_agent, action="add", target="memory",
                        content="[circle: everyone]\nLikes jazz.")["success"]
    assert "[circle: everyone]\nLikes jazz." in _disk_entries()


def test_omit_still_refuses_writes(manager, circle_agent):
    _register(manager, lambda **_: "omit")
    result = _memory_tool(circle_agent, action="add", target="memory", content="New fact.")
    assert result["success"] is False and OMITTED_NOTE in result["error"]
    assert "New fact." not in _disk_entries()

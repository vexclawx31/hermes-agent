"""Circle markers in MEMORY.md / USER.md entries and the store's per-call entry view.

``[circle: <tag>[, <tag>...]]`` at the start of an entry's first line carries its circle tags; core
parses it and a ``filter_builtin_memory`` plugin decides what the tags mean
(agent/builtin_memory_gate.py).
"""

import pytest

from tools.memory_tool_store import (
    MemoryStore, circle_marker, entry_circles, entry_view, strip_circle_markers)


@pytest.mark.parametrize("entry, tags", [
    ("[circle: household]\nDentist is Dr. Moreau.", {"household"}),
    ("[circle: group:whatsapp:1@g.us, private_to:jamal]\nx", {"group:whatsapp:1@g.us", "private_to:jamal"}),
    ("[circle:everyone] Answer briefly.", {"everyone"}),
    ("  [circle: household]\nleading space", {"household"}),
    ("[circle: ]\nempty marker", set()),
    ("No marker here.", set()),
    ("Text first [circle: everyone]", set()),  # only a leading marker counts
    ("First line\n[circle: everyone]", set()),
])
def test_entry_circles(entry, tags):
    assert entry_circles(entry) == frozenset(tags)


def test_strip_and_build_markers():
    assert strip_circle_markers("[circle: everyone] [circle: x]\n  Body text ") == "Body text"
    assert strip_circle_markers("Body [circle: x]") == "Body [circle: x]"
    assert circle_marker(["household", "private_to:jamal", "household"]) == "[circle: household, private_to:jamal]"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    seed = MemoryStore(memory_char_limit=500, user_char_limit=300)
    seed.load_from_disk()
    for entry in ("[circle: everyone]\nShort answers.", "[circle: household]\nDentist: Dr. Moreau.", "Untagged."):
        assert seed.add("memory", entry)["success"]
    fresh = MemoryStore(memory_char_limit=500, user_char_limit=300)
    fresh.load_from_disk()
    return fresh


def test_prompt_without_keep_is_the_frozen_snapshot(store):
    assert store.format_for_system_prompt("memory") == store._system_prompt_snapshot["memory"]
    assert store.format_for_system_prompt("memory", keep=lambda entry: True) == store._system_prompt_snapshot["memory"]


def test_prompt_with_keep_renders_kept_entries_and_full_usage(store):
    block = store.format_for_system_prompt("memory", keep=lambda entry: "everyone" in entry_circles(entry))
    assert "[circle: everyone]\nShort answers." in block
    assert "Dr. Moreau" not in block and "Untagged." not in block
    full = store._system_prompt_snapshot["memory"]
    assert block.splitlines()[1] == full.splitlines()[1]  # same usage header: the whole file counts
    assert store.format_for_system_prompt("memory", keep=lambda entry: False) is None


def test_prompt_keep_that_raises_omits_the_entry(store):
    def keep(entry):
        raise RuntimeError("boom")

    assert store.format_for_system_prompt("memory", keep=keep) is None


def test_entry_view_scopes_one_call_and_resets(store):
    with entry_view(lambda entry: "everyone" in entry_circles(entry)):
        result = store.remove("memory", "Dr. Moreau")
        assert result["success"] is False
        assert result["current_entries"] == ["[circle: everyone]\nShort answers."]
    assert store.remove("memory", "Dr. Moreau")["success"] is True  # no view after the block

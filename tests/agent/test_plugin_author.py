"""Plugin author validation, provenance and display readers (agent/plugin_author.py).

Rules: reject, never clean, truncate or case-fold. Trust lives in the host object and the
gateway-built turn_author, never in the ``plugin:`` label; the persisted record is display-only.
"""

import json
import time
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor, _summary_author_label
from agent.plugin_author import (
    PluginAuthor, build_plugin_author, display_plugin_author, plugin_author_from_record, valid_author_name,
    valid_plugin_key,
)
from agent.recall_scope import UNKNOWN_LABEL, plugin_author_label, sender_label
from agent.turn_author import parse_turn_author, turn_author_env, turn_author_from_env
from hermes_state import SessionDB
from tools.session_search_tool import session_search

PLUGIN = "switchboard"
RECORD = {"v": 1, "plugin": PLUGIN, "id": "research", "name": "Research Agent"}
GATEWAY_AUTHOR = {"id": "plugin:switchboard/research", "name": "Research Agent", "is_bot": True,
                  "verified": False, "origin": "plugin", "plugin": PLUGIN}
HUMAN_SENDER = {"v": 1, "platform": "slack", "user_id": "UJAMAL0001", "chat_id": "C1", "chat_type": "group",
                "trusted": True, "name": "Jamal"}


def _author(**overrides):
    return build_plugin_author(PLUGIN, {"id": "research", "name": "Research Agent", **overrides})


# ── identity validation ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_id", [
    "a.b", "a/b", "a:b", " research", "research ", "Research", "r" + chr(0xE9) + "sum" + chr(0xE9),
    "r" + chr(0x0435) + "search",  # Cyrillic ie
    chr(0xFF52) + "esearch",  # fullwidth r
    "re" + chr(0x200B) + "search",  # zero-width space
    "re" + chr(0x202E) + "search",  # bidi override
    "research\n", "", "-research", "_research", "a" * 49, 7, None, b"research",
])
def test_bad_author_ids_are_rejected_not_cleaned(bad_id):
    with pytest.raises(ValueError):
        _author(id=bad_id)


def test_48_char_id_is_accepted_and_49_rejected_never_truncated():
    assert _author(id="a" * 48).id == "a" * 48
    with pytest.raises(ValueError):
        _author(id="a" * 49)


@pytest.mark.parametrize("bad_name", [
    "", " ", "\t", "a" * 65, "Bad\nName", "Bad]Name", "[Bad", "<b>", "Bad>", "Evil" + chr(0x202E) + "eman",
    "Evil" + chr(0x2066) + "x", "nul\x00", "bell\x07", "cr\r", "line" + chr(0x2028) + "sep",
    "para" + chr(0x2029) + "sep", "surrogate" + chr(0xD800), "private" + chr(0xE000),
    "unassigned" + chr(0xE0080), 3, None,
])
def test_bad_author_names_are_rejected(bad_name):
    with pytest.raises(ValueError):
        _author(name=bad_name)


@pytest.mark.parametrize("good_name", ["R", "Research Agent", "a" * 64, "Résumé bot 🤖", "Agent (beta): v2"])
def test_good_author_names_are_kept_verbatim(good_name):
    assert _author(name=good_name).name == good_name


@pytest.mark.parametrize("author", [
    {"id": "research"}, {"name": "Research"}, {"id": "research", "name": "R", "is_bot": True},
    {"id": "research", "name": "R", "verified": True}, {"id": "research", "name": "R", "origin": "plugin"},
    "research", ["research", "R"], None,
])
def test_author_must_be_exactly_id_and_name(author):
    with pytest.raises(ValueError):
        build_plugin_author(PLUGIN, author)


@pytest.mark.parametrize("pair", [
    ("research", "research "), ("research", "re" + chr(0x200B) + "search"), ("Research", "research"),
    ("research", "r" + chr(0x0435) + "search"), ("research", chr(0xFF52) + "esearch"),
])
def test_values_a_cleaner_would_collapse_yield_one_accepted_and_one_rejected(pair):
    results = []
    for value in pair:
        try:
            results.append(_author(id=value).id)
        except ValueError:
            results.append(None)
    assert results.count(None) == 1
    accepted = [r for r in results if r is not None]
    assert len(set(accepted)) == 1 and accepted[0] in pair


@pytest.mark.parametrize("key, ok", [
    ("switchboard", True), ("a", True), ("my-plugin_2", True), ("a" * 32, True), ("a" * 33, False),
    ("Switchboard", False), ("1plugin", False), ("memory/honcho", False), ("my.plugin", False), ("", False),
    (None, False),
])
def test_plugin_key_eligibility(key, ok):
    assert valid_plugin_key(key) is ok


def test_plugin_author_is_frozen_and_validated_on_construction():
    author = _author()
    with pytest.raises(AttributeError):
        author.id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError):
        PluginAuthor(plugin="Bad Key", id="research", name="R")
    assert author.label == "plugin:switchboard/research"
    assert author.name_tag() == "Research Agent (agent plugin:switchboard/research)"
    assert author.turn_author() == GATEWAY_AUTHOR
    assert author.display_record() == RECORD
    assert valid_author_name("Research Agent")


# ── the display-only record ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("record", [
    dict(RECORD, v=2), dict(RECORD, v=True), dict(RECORD, v="1"), dict(RECORD, extra=1),
    {k: v for k, v in RECORD.items() if k != "name"}, dict(RECORD, id="Research"), dict(RECORD, id="a/b"),
    dict(RECORD, name="Evil]Name"), dict(RECORD, plugin="memory/honcho"), "nope", None, [],
])
def test_malformed_records_read_as_unknown(record):
    assert plugin_author_from_record(record) is None
    assert display_plugin_author({"plugin_author": record}) is None
    assert plugin_author_label({"plugin_author": record}) is None


def test_record_with_a_sender_is_ambiguous_and_unknown():
    assert display_plugin_author({"plugin_author": RECORD}) == _author()
    assert display_plugin_author({"plugin_author": RECORD, "sender": HUMAN_SENDER}) is None
    assert display_plugin_author(json.dumps({"plugin_author": RECORD})) == _author()
    assert display_plugin_author("not json") is None


def test_a_record_never_becomes_a_trusted_sender_label():
    """A plugin row stays outside sender-based recall: unknown label, never in any scope."""
    assert sender_label({"plugin_author": RECORD}) == UNKNOWN_LABEL


# ── parse_turn_author hardening and bot-relay origin compatibility (Vex condition 1) ──────────


@pytest.mark.parametrize("raw", [
    {"id": "plugin:switchboard/research", "name": "R"},
    {"id": "PLUGIN:switchboard/research", "name": "R"},
    {"id": "\x00plugin:switchboard/research", "name": "R"},
    {"id": " plugin:x/y ", "name": "R", "is_bot": True},
    dict(GATEWAY_AUTHOR),  # the exact gateway shape, but from a client: still refused
])
def test_api_and_env_can_never_mint_a_plugin_author(raw):
    assert parse_turn_author(raw) is None
    assert turn_author_from_env(turn_author_env(raw)) is None


def test_api_server_body_with_a_plugin_author_is_dropped():
    from gateway.platforms.api_server import _request_turn_author
    assert _request_turn_author({"author": dict(GATEWAY_AUTHOR)}) is None
    assert _request_turn_author({"author": {"id": "bot:dixie", "name": "d", "origin": "plugin"}}) == {
        "id": "bot:plugin/dixie", "name": "d", "is_bot": False}


@pytest.mark.parametrize("raw, expected", [
    ({"id": "bot:dixie", "name": "dixie", "is_bot": True, "origin": "cloud-1"},
     {"id": "bot:cloud-1/dixie", "name": "dixie", "is_bot": True}),
    ({"id": "bot:dixie", "name": "dixie", "is_bot": True, "origin": "plugin"},
     {"id": "bot:plugin/dixie", "name": "dixie", "is_bot": True}),
    ({"id": "bot:host/dixie", "name": "dixie", "is_bot": True, "origin": "cloud-1"},
     {"id": "bot:host/dixie", "name": "dixie", "is_bot": True}),
    ({"id": "U123", "name": "Jamal", "origin": "cloud-1"}, {"id": "U123", "name": "Jamal", "is_bot": False}),
])
def test_bot_relay_origin_qualification_is_unchanged(raw, expected):
    """``origin`` keeps its existing meaning on every client path (qualifying a bare ``bot:`` id),
    and the output never carries an ``origin`` or ``plugin`` key."""
    assert parse_turn_author(raw) == expected
    assert parse_turn_author(raw, keep_verified=True) == expected
    assert parse_turn_author(json.dumps(raw)) == expected


def test_relay_delivery_authors_pass_through_unchanged():
    from tools.bot_relay import delivery_turn_author, relaying_principal_author
    author = delivery_turn_author("dixie", "Dixie", "conn-1")
    assert author == {"id": "bot:conn-1/dixie", "name": "Dixie", "is_bot": True}
    assert parse_turn_author(author) == author
    assert turn_author_from_env(turn_author_env(author)) == author
    principal = relaying_principal_author("abc")
    assert parse_turn_author(principal) == principal


def test_gateway_plugin_author_survives_the_trusted_parse_exactly():
    assert parse_turn_author(dict(GATEWAY_AUTHOR), keep_verified=True) == GATEWAY_AUTHOR


@pytest.mark.parametrize("forged", [
    dict(GATEWAY_AUTHOR, verified=True), dict(GATEWAY_AUTHOR, is_bot=False), dict(GATEWAY_AUTHOR, is_bot="true"),
    dict(GATEWAY_AUTHOR, plugin="other"), dict(GATEWAY_AUTHOR, id="plugin:switchboard/Research"),
    dict(GATEWAY_AUTHOR, name="Evil]"), dict(GATEWAY_AUTHOR, plugin=None),
    {k: v for k, v in GATEWAY_AUTHOR.items() if k != "verified"},
])
def test_malformed_gateway_plugin_author_is_refused(forged):
    assert parse_turn_author(forged, keep_verified=True) is None


def test_origin_plugin_without_a_plugin_id_is_never_a_plugin_author():
    """``origin: "plugin"`` alone marks nothing: a non-``plugin:`` id parses as before, without
    ``origin``/``plugin`` keys, so no provider can read it as plugin-authored."""
    parsed = parse_turn_author(dict(GATEWAY_AUTHOR, id="U123"), keep_verified=True)
    assert parsed == {"id": "U123", "name": "Research Agent", "is_bot": True, "verified": False}


def test_human_gateway_author_parse_is_unchanged():
    human = {"id": "UJAMAL0001", "name": "Jamal", "is_bot": False, "verified": True, "label_id": "user_x"}
    assert parse_turn_author(human, keep_verified=True) == human
    assert parse_turn_author(human) == {"id": "UJAMAL0001", "name": "Jamal", "is_bot": False}


# ── provenance: display metadata never makes a turn plugin-authored ───────────────────────────


def _turn_author_for(display_metadata, plugin_author, monkeypatch):
    from agent import recall_scope
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(recall_scope, "attribution_enabled", lambda config=None: True)
    monkeypatch.setattr(recall_scope, "author_id_redactor", lambda config=None: (lambda p, u: str(u or "")))
    seen = {}

    class _Agent:
        def run_conversation(self, message, *, turn_author=None, **_kw):
            seen["turn_author"] = turn_author
            return {}

    agent = _Agent()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1",
                           user_name="Requester")
    ctx = TurnContext(source=source, message="m", session_key="k", session_id="s",
                      persist_user_display_metadata=display_metadata, plugin_author=plugin_author)
    TurnRunner(SimpleNamespace(_consume_pending_native_image_paths=lambda key: []), ctx) \
        ._run_conversation_with_approval(agent, [], None, None, None)
    return seen["turn_author"], getattr(agent, "_recall_scope", "unset")


def test_forged_or_replayed_record_on_a_human_turn_grants_nothing(monkeypatch):
    """A spool-replayed / branched / recovered row carrying a forged ``plugin_author`` record is
    display-only: the turn runner ignores it (no plugin origin, no verified, no scope change)."""
    author, scope = _turn_author_for({"plugin_author": RECORD}, None, monkeypatch)
    assert author == {"id": "u1", "name": "Requester", "is_bot": False, "verified": False}
    assert "origin" not in author
    assert scope is None  # no native ingress receipt in this test: unchanged human path, no plugin scope


def test_only_the_host_object_makes_a_plugin_turn(monkeypatch):
    look_alike = SimpleNamespace(plugin=PLUGIN, id="research", name="Research Agent",
                                 turn_author=lambda: dict(GATEWAY_AUTHOR, verified=True))
    author, _ = _turn_author_for({}, look_alike, monkeypatch)
    assert author["id"] == "u1"
    author, scope = _turn_author_for({}, _author(), monkeypatch)
    assert author == GATEWAY_AUTHOR and scope is None


# ── display readers: compressor, session_search, replay ───────────────────────────────────────


def _compressor():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._CONTENT_MAX = 10_000
    return compressor


def test_compression_labels_the_agent_never_the_human(monkeypatch):
    monkeypatch.setattr("agent.recall_scope.attribution_enabled", lambda config=None: True)
    rows = [
        {"role": "user", "content": "result", "display_metadata": {"plugin_author": RECORD}},
        {"role": "user", "content": "forged", "display_metadata": {"plugin_author": dict(RECORD, v=9)}},
        {"role": "user", "content": "both", "display_metadata": {"plugin_author": RECORD, "sender": HUMAN_SENDER}},
        {"role": "user", "content": "replayed", "_plugin_author": RECORD},
        {"role": "user", "content": "human", "display_metadata": {"sender": HUMAN_SENDER}},
    ]
    parts = _compressor()._serialize_records_for_summary(rows)
    assert parts == [
        "[USER agent plugin:switchboard/research, shown as Research Agent]: result",
        "[USER author unknown]: forged",
        "[USER author unknown]: both",
        "[USER agent plugin:switchboard/research, shown as Research Agent]: replayed",
        "[USER slack:UJAMAL0001 verified, shown as Jamal]: human",
    ]


def test_compression_label_strips_brackets_from_a_name():
    row = {"role": "user", "_plugin_author": dict(RECORD, name="Agent (beta)")}
    assert _summary_author_label(row) == "agent plugin:switchboard/research, shown as Agent beta"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_joint", source="slack", user_id="UJAMAL0001")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (int(time.time()) - 600, "s_joint"))
    db.append_message("s_joint", role="user", content="relay: agent result",
                      display_metadata={"plugin_author": RECORD})
    db.append_message("s_joint", role="user", content="relay: forged record",
                      display_metadata={"plugin_author": dict(RECORD, id="Bad Id")})
    db.append_message("s_joint", role="user", content="relay: both",
                      display_metadata={"plugin_author": RECORD, "sender": HUMAN_SENDER})
    db.append_message("s_joint", role="user", content="relay: human",
                      display_metadata={"sender": HUMAN_SENDER})
    db._conn.commit()
    return db


def test_session_search_tags_plugin_rows_as_unverified_agents(db, monkeypatch):
    monkeypatch.setattr("agent.recall_scope.attribution_enabled", lambda config=None: True)
    out = json.loads(session_search(db=db, session_id="s_joint", enforce_scope=False))
    by_content = {m["content"]: m for m in out["messages"]}
    assert by_content["relay: agent result"]["author"] == {
        "agent": "plugin:switchboard/research", "name": "Research Agent", "origin": "plugin", "verified": False}
    assert by_content["relay: forged record"]["author"] == {"verified": False}
    assert by_content["relay: both"]["author"] == {"verified": False}
    assert by_content["relay: human"]["author"] == {
        "platform": "slack", "user_id": "UJAMAL0001", "name": "Jamal", "chat_type": "group", "verified": True}


def test_sender_filtered_search_never_returns_plugin_rows(db):
    out = json.loads(session_search(db=db, session_id="s_joint", sender_platform="slack",
                                    sender_user_id="UJAMAL0001", enforce_scope=False))
    contents = [m["content"] for m in out.get("messages", [])]
    assert "relay: agent result" not in contents


def test_replay_carries_a_view_only_plugin_author_next_to_sender():
    from gateway.run import _build_replay_entry
    plugin_row = {"role": "user", "content": "r", "display_metadata": json.dumps({"plugin_author": RECORD})}
    entry = _build_replay_entry("user", "r", plugin_row)
    assert entry["_plugin_author"] == RECORD and "_sender" not in entry
    forged = _build_replay_entry("user", "r", {"display_metadata": {"plugin_author": dict(RECORD, v=0)}})
    assert "_plugin_author" not in forged and "_sender" not in forged
    both = _build_replay_entry("user", "r", {"display_metadata": {"plugin_author": RECORD, "sender": HUMAN_SENDER}})
    assert both["_sender"] == HUMAN_SENDER and "_plugin_author" not in both
    human = _build_replay_entry("user", "r", {"display_metadata": {"sender": HUMAN_SENDER}})
    assert human["_sender"] == HUMAN_SENDER and "_plugin_author" not in human
    assistant = _build_replay_entry("assistant", "r", plugin_row)
    assert "_plugin_author" not in assistant

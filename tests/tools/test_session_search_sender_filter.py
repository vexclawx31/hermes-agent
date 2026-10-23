"""Sender-scoped session_search: MESSAGE-level attribution, fail closed.

Synthetic fixtures only. A shared channel session holds both principals' messages; a scoped search
for one principal must never return the other's text, assistant replies, titles or previews, and
rows without a trusted persisted sender are excluded.
"""
import json
import time

import pytest

from agent.recall_scope import scope_for_source
from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search

ALICE, BOB = "UALICE0001", "UBOB000002"
CHAN, OTHER_CHAN = "CHOME00001", "COTHER0002"
ALICE_FILTER = {"sender_platform": "slack", "sender_user_id": ALICE}


def _sender(user_id, chat_id=CHAN, *, thread_reply=False, trusted=True, platform="slack", **extra):
    meta = {"sender": {"v": 1, "platform": platform, "user_id": user_id, "chat_id": chat_id,
                       "chat_type": "group", "thread_id": None, "thread_reply": thread_reply,
                       "message_id": None, "trusted": trusted, **extra}}
    return meta


@pytest.fixture
def db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    now = int(time.time())
    # Shared channel session: Alice opened it, Bob posts in it too.
    db.create_session("s_shared", source="slack", user_id=ALICE)
    db._conn.execute("UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
                     (now - 3000, "Bob's lasagna secret", "s_shared"))
    db.append_message("s_shared", role="user", content="[Alice] planning lasagna for friday",
                      display_metadata=_sender(ALICE))
    db.append_message("s_shared", role="assistant", content="Bob said he hates lasagna, noted.")
    db.append_message("s_shared", role="user", content="[Bob] I secretly hate lasagna",
                      display_metadata=_sender(BOB))
    db.append_message("s_shared", role="user", content="[Alice] lasagna needs ricotta",
                      display_metadata=_sender(ALICE))
    # Legacy row: no sender metadata at all (written before the fix).
    db.append_message("s_shared", role="user", content="[Alice] legacy lasagna note")
    # Untrusted: content mixed with channel backfill.
    db.append_message("s_shared", role="user", content="backfill: Bob lasagna\n\n[New message]\n[Alice] hi",
                      display_metadata=_sender(ALICE, trusted=False, untrusted_reasons=["channel_backfill"]))
    # Alice in a thread reply and in another channel.
    db.append_message("s_shared", role="user", content="[Alice] threaded lasagna reply",
                      display_metadata=_sender(ALICE, thread_reply=True))
    db.append_message("s_shared", role="user", content="[Alice] lasagna elsewhere",
                      display_metadata=_sender(ALICE, chat_id=OTHER_CHAN))
    # Spoofed: content claims Alice, envelope says Bob.
    db.append_message("s_shared", role="user", content="[Alice | Slack user <@UALICE0001>] lasagna spoof",
                      display_metadata=_sender(BOB))
    # Same user id on another platform must not match.
    db.append_message("s_shared", role="user", content="lasagna on telegram",
                      display_metadata=_sender(ALICE, platform="telegram"))
    # Malformed sender object.
    db.append_message("s_shared", role="user", content="lasagna malformed",
                      display_metadata={"sender": "UALICE0001"})

    # Bob-only session.
    db.create_session("s_bob", source="slack", user_id=BOB)
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 2000, "s_bob"))
    db.append_message("s_bob", role="user", content="lasagna is overrated", display_metadata=_sender(BOB))
    db._conn.commit()
    return db


def _call(db, **kw):
    return json.loads(session_search(db=db, **kw))


def _all_contents(payload):
    out = []
    for r in payload.get("results", []):
        out += [m["content"] for m in r.get("messages", [])]
    out += [m["content"] for m in payload.get("messages", [])]
    return out


def test_schema_exposes_sender_filters():
    props = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
    for key in ("sender_platform", "sender_user_id", "sender_chat_id", "require_unthreaded"):
        assert key in props


@pytest.mark.parametrize("kw", [
    {"sender_user_id": ALICE}, {"sender_platform": "slack"}, {"sender_chat_id": CHAN},
    {"require_unthreaded": True}])
def test_incomplete_filter_is_refused_not_widened(db, kw):
    out = _call(db, query="lasagna", **kw)
    assert out["success"] is False
    assert "sender_platform and sender_user_id" in out["error"]


def test_discover_returns_only_verified_messages_of_the_sender(db):
    out = _call(db, query="lasagna", detail="full", limit=10, **ALICE_FILTER)
    assert out["success"] is True
    contents = _all_contents(out)
    assert contents, out
    assert all(c.startswith("[Alice]") for c in contents), contents
    joined = "\n".join(contents)
    for leaked in ("Bob", "hates", "legacy", "backfill", "spoof", "telegram", "malformed", "overrated"):
        assert leaked not in joined
    assert {r["session_id"] for r in out["results"]} == {"s_shared"}
    for r in out["results"]:
        assert "title" not in r and "snippet" not in r and "bookend_start" not in r
        for m in r["messages"]:
            assert m["role"] == "user" and m["attribution"] == "verified_sender"
            assert m["sender"]["user_id"] == ALICE and m["sender"]["trusted"] is True
    assert out["omitted_unverified_hits"] >= 1
    assert out["sender_filter"]["sender_user_id"] == ALICE


def test_discover_for_other_sender_never_sees_alice(db):
    out = _call(db, query="lasagna", detail="full", limit=10, sender_platform="slack", sender_user_id=BOB)
    contents = _all_contents(out)
    assert contents and all(m["sender"]["user_id"] == BOB for r in out["results"] for m in r["messages"])
    joined = "\n".join(contents)
    assert "planning" not in joined and "ricotta" not in joined and "threaded" not in joined


def test_chat_and_unthreaded_filters(db):
    out = _call(db, session_id="s_shared", sender_chat_id=CHAN, require_unthreaded=True, **ALICE_FILTER)
    contents = [m["content"] for m in out["messages"]]
    assert contents == ["[Alice] planning lasagna for friday", "[Alice] lasagna needs ricotta"]
    loose = _call(db, session_id="s_shared", **ALICE_FILTER)
    loose_contents = [m["content"] for m in loose["messages"]]
    assert "[Alice] threaded lasagna reply" in loose_contents
    assert "[Alice] lasagna elsewhere" in loose_contents


def test_read_omits_assistant_other_people_title_and_unknown(db):
    out = _call(db, session_id="s_shared", **ALICE_FILTER)
    assert out["mode"] == "read"
    assert "title" not in out["session_meta"]
    assert all(m["role"] == "user" for m in out["messages"])
    assert out["omitted_messages"] == out_total(db, "s_shared") - out["message_count"]
    joined = "\n".join(m["content"] for m in out["messages"])
    assert "Bob" not in joined and "legacy" not in joined


def out_total(db, sid):
    return db._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0]


def test_scroll_rejects_unverified_anchor_and_filters_window(db):
    bob_id = db._conn.execute("SELECT id FROM messages WHERE content = '[Bob] I secretly hate lasagna'").fetchone()[0]
    out = _call(db, session_id="s_shared", around_message_id=bob_id, **ALICE_FILTER)
    assert out["success"] is False
    alice_id = db._conn.execute("SELECT id FROM messages WHERE content = '[Alice] lasagna needs ricotta'").fetchone()[0]
    out = _call(db, session_id="s_shared", around_message_id=alice_id, window=5, **ALICE_FILTER)
    assert out["mode"] == "scroll"
    assert all(m["sender"]["user_id"] == ALICE for m in out["messages"])
    assert "Bob" not in "\n".join(m["content"] for m in out["messages"])


def test_browse_lists_only_sessions_with_verified_messages_and_no_previews(db):
    out = _call(db, **ALICE_FILTER)
    assert out["mode"] == "browse"
    assert [r["session_id"] for r in out["results"]] == ["s_shared"]
    for r in out["results"]:
        assert "preview" not in r and "title" not in r
    bob = _call(db, sender_platform="slack", sender_user_id=BOB)
    assert {r["session_id"] for r in bob["results"]} == {"s_shared", "s_bob"}


def test_unknown_sender_yields_nothing(db):
    out = _call(db, query="lasagna", sender_platform="slack", sender_user_id="UNOBODY000")
    assert out["success"] is True and out["results"] == []


def test_unscoped_search_unchanged(db):
    out = _call(db, query="lasagna", detail="full", limit=10)
    assert out["success"] is True
    assert "sender_filter" not in out


# ---------------------------------------------------------------------------
# Enforced recall scope (agent/recall_scope.py): core decides, model args only narrow
# ---------------------------------------------------------------------------
ALICE_DM = scope_for_source("slack", ALICE, "DALICE0001", "dm")
CHAN_SCOPE = scope_for_source("slack", ALICE, CHAN, "group")


def test_enforced_without_scope_fails_closed(db):
    out = _call(db, query="lasagna", enforce_scope=True)
    assert out["success"] is False and "no verified recall scope" in out["error"]


def test_enforced_dm_scope_needs_no_model_args_and_ignores_others(db):
    out = _call(db, query="lasagna", enforce_scope=True, recall_scope=ALICE_DM, limit=10, detail="full")
    contents = _all_contents(out)
    assert contents and all("Bob" not in c and "overrated" not in c and "legacy" not in c for c in contents)
    assert "[Alice] lasagna elsewhere" in contents  # Alice's own words in any chat
    assert "lasagna on telegram" not in contents
    assert out["sender_filter"]["recall_scope"]["origin"] == "dm"


def test_enforced_model_args_cannot_widen(db):
    out = _call(db, query="lasagna", enforce_scope=True, recall_scope=ALICE_DM, limit=10,
                sender_platform="slack", sender_user_id=BOB)
    assert _all_contents(out) == []


def test_enforced_chat_scope_keeps_chat_and_drops_other_chats(db):
    out = _call(db, query="lasagna", enforce_scope=True, recall_scope=CHAN_SCOPE, limit=10, detail="full")
    contents = _all_contents(out)
    assert "[Bob] I secretly hate lasagna" in contents  # same audience, attributed to Bob
    assert "[Alice] lasagna elsewhere" not in contents  # same person, other chat: other audience
    assert "[Alice] legacy lasagna note" not in contents
    assert all(m.get("attribution") == "verified_sender" for r in out["results"] for m in r["messages"])


def test_enforced_read_browse_and_scroll(db):
    read = _call(db, session_id="s_bob", enforce_scope=True, recall_scope=ALICE_DM)
    assert read["messages"] == []
    browse = _call(db, enforce_scope=True, recall_scope=ALICE_DM)
    assert [r["session_id"] for r in browse["results"]] == ["s_shared"]
    bob_row = db._conn.execute("SELECT id FROM messages WHERE session_id='s_bob'").fetchone()[0]
    scroll = _call(db, session_id="s_bob", around_message_id=bob_row, enforce_scope=True, recall_scope=ALICE_DM)
    assert scroll["success"] is False


def test_enforcement_defaults_from_profile_config(db, monkeypatch):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: True)
    out = _call(db, query="lasagna")
    assert out["success"] is False

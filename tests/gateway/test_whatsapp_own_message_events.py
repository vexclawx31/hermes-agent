"""Observer-only WhatsApp events for this account's own messages: bridge-marked own edits/revokes
(``observerOnly``, delivered only as ``event_kind="own_mutation"``) and in-process
``direction="outbound"`` events after successful group sends/finalized edits."""

import asyncio
import re
import time
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig

OWNER = "15550000001@s.whatsapp.net"
GUEST = "15550000002@s.whatsapp.net"
BOT_IDS = ["15551230000@s.whatsapp.net", "15551230000@lid"]
GROUP = "120363000000000001@g.us"
DM = "15550000002@s.whatsapp.net"


def _make_adapter(responses=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = {"dm_policy": "allowlist", "allow_from": [OWNER], "group_policy": "open", "require_mention": False,
             "free_response_chats": [GROUP], "owners": [OWNER]}
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "allowlist"
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list([OWNER])
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._groups = {}
    adapter._owners = WhatsAppAdapter._coerce_allow_list([OWNER])
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._reply_prefix = None
    adapter._bridge_unavailable = AsyncMock(return_value=None)
    adapter.posted = []
    queue = list(responses or [])

    @asynccontextmanager
    async def fake_req(method, path, timeout, **kwargs):
        adapter.posted.append((path, kwargs.get("json")))
        status, body = queue.pop(0)
        resp = AsyncMock()
        resp.status = status
        resp.json = AsyncMock(return_value=body)
        resp.text = AsyncMock(return_value=str(body))
        yield resp

    adapter._bridge_req = fake_req
    return adapter


def _capture(monkeypatch, fail=False):
    from hermes_cli import lifecycle, plugins

    seen = []

    def fake_invoke(name, **kwargs):
        seen.append((name, kwargs))
        if fail:
            raise RuntimeError("observer failed")
        return []

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "whatsapp_transport_event")
    monkeypatch.setattr(plugins, "invoke_hook", fake_invoke)
    return seen


async def _drain(adapter):
    for bucket in ("_observer_tasks", "_own_observer_tasks"):
        await asyncio.gather(*list(adapter.__dict__.get(bucket, ())))


def _run(adapter, coro):
    async def go():
        result = await coro
        await _drain(adapter)
        return result

    return asyncio.run(go())


def _outbound(seen):
    return [kw["raw_message"] for _, kw in seen if kw.get("direction") == "outbound"]


def _mutation(kind="revoke", **overrides):
    data = {"observerOnly": True, "ownMutation": kind, "chatId": GROUP, "isGroup": True, "fromMe": True,
            "senderId": OWNER, "messageId": "P1", "targetMessageId": "T1", "body": "/reset @alex",
            "mentionedIds": [BOT_IDS[0]], "botIds": list(BOT_IDS), "timestamp": 1700000000}
    data.update(overrides)
    return data


# --- inbound: bridge-marked own mutations reach observers under event_kind only, never intake ---------

@pytest.mark.parametrize("mutation", ["edit", "revoke"])
def test_observer_only_event_is_observed_but_never_a_turn(monkeypatch, mutation):
    seen = _capture(monkeypatch)
    adapter = _make_adapter()
    # Worst case for intake: owner sender, native mention of the bot, a free-response group, a slash command.
    data = _mutation(mutation)
    assert adapter._should_process_message(data) is False
    assert adapter._should_process_message({**data, "observerOnly": False}) is True  # the marker alone decides
    assert _run(adapter, adapter._build_message_event(data)) is None
    assert len(seen) == 1
    name, kwargs = seen[0]
    assert name == "whatsapp_transport_event"
    assert kwargs["event_kind"] == "own_mutation"
    assert "direction" not in kwargs  # bridge events are never "outbound"
    raw = kwargs["raw_message"]
    assert raw["ownMutation"] == mutation and raw["targetMessageId"] == "T1"
    assert "messageId" not in raw  # second guard for observers keyed by messageId
    assert data["messageId"] == "P1"  # the bridge event itself is not mutated


# whatsapp-archive 0.2.0's observer guard, verbatim in behaviour: it archives any whatsapp group event
# with a messageId and ignores extra keywords.
_GROUP_RE = re.compile(r"[0-9]{6,}(-[0-9]+)?@g\.us")


def _archive_0_2_0(recorded):
    def on_transport_event(raw_message: Any = None, platform: Any = None, **_: Any) -> None:
        if str(getattr(platform, "value", platform) or "").lower() != "whatsapp" or not isinstance(raw_message, dict):
            return
        chat_id = str(raw_message.get("chatId") or "")
        if raw_message.get("isGroup") is not True or not _GROUP_RE.fullmatch(chat_id):
            return
        message_id = str(raw_message.get("messageId") or "").strip()
        if not message_id:
            return
        recorded.append((chat_id, message_id, raw_message.get("body")))

    return on_transport_event


def test_archive_0_2_0_observer_records_no_own_mutation_or_outbound_event(monkeypatch):
    from hermes_cli import lifecycle, plugins
    from hermes_cli.plugins import PluginManager

    recorded = []
    manager = PluginManager()
    manager._discovered = True
    manager._hooks = {"whatsapp_transport_event": [_archive_0_2_0(recorded)]}
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "whatsapp_transport_event")

    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]}),
                             (200, {"success": True, "messageIds": []})])
    _run(adapter, adapter._build_message_event(_mutation("revoke")))
    _run(adapter, adapter._build_message_event(_mutation("edit", body="edited")))
    assert _run(adapter, adapter.send(GROUP, "reply")).success
    assert _run(adapter, adapter.edit_message(GROUP, "A", "reply v2", finalize=True)).success
    assert recorded == []

    # Positive control: an ordinary inbound group message still reaches the same observer.
    _run(adapter, adapter._build_message_event({"chatId": GROUP, "isGroup": True, "senderId": GUEST,
                                                "messageId": "IN1", "body": "hello", "botIds": list(BOT_IDS)}))
    assert recorded == [(GROUP, "IN1", "hello")]


# --- outbound: successful group sends/finalized edits fire direction="outbound" -----------------------

def test_multi_chunk_group_send_reports_every_chunk_id(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageId": "B", "messageIds": ["A", "B"]}),
                             (200, {"success": True, "messageId": "C", "messageIds": ["C"]})])
    adapter.truncate_message = lambda content, limit: ["part one", "part two"]
    before = int(time.time())
    result = _run(adapter, adapter.send(GROUP, "long reply"))
    assert result.success and result.message_id == "C"
    events = _outbound(seen)
    assert [e["messageIds"] for e in events] == [["A", "B"], ["C"]]
    assert [e["body"] for e in events] == ["part one", "part two"]
    for event in events:
        assert {k: event[k] for k in ("chatId", "isGroup", "fromMe", "kind")} == {
            "chatId": GROUP, "isGroup": True, "fromMe": True, "kind": "text"}
        assert before <= event["timestamp"] <= int(time.time()) + 1  # seconds, like bridge events
        assert "targetMessageId" not in event and "messageId" not in event


def test_media_and_poll_sends_are_observed_location_is_not(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageId": "M1"}),
                             (200, {"success": True, "messageId": "Q1"}),
                             (200, {"success": True, "messageId": "L1"})])
    _run(adapter, adapter._post_bridge_message(
        "send-media", {"chatId": GROUP, "filePath": "/x.png", "mediaType": "image", "caption": "look"}, timeout=1))
    _run(adapter, adapter.send_poll(GROUP, "Lunch?", ["yes", "no"]))
    _run(adapter, adapter.send_location(GROUP, 1.0, 2.0))
    events = _outbound(seen)
    assert [(e["kind"], e["messageIds"], e["body"]) for e in events] == [("media", ["M1"], "look"), ("poll", ["Q1"], "Lunch?")]


def test_only_finalized_group_edits_are_observed(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageIds": []}),
                             (200, {"success": True, "messageIds": ["S1"]})])
    assert _run(adapter, adapter.edit_message(GROUP, "T9", "streaming...")).success
    assert _outbound(seen) == []
    result = _run(adapter, adapter.edit_message(GROUP, "T9", "new text", finalize=True))
    assert result.success and result.message_id == "T9"
    (event,) = _outbound(seen)
    assert (event["kind"], event["targetMessageId"], event["messageIds"], event["body"]) == ("edit", "T9", ["S1"], "new text")


def test_dm_sends_and_edits_and_failed_sends_are_never_observed(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageId": "D1", "messageIds": ["D1"]}),
                             (200, {"success": True, "messageIds": []}),
                             (500, {"error": "boom"}),
                             (500, {"error": "boom"})])
    assert _run(adapter, adapter.send(DM, "hi")).success
    assert _run(adapter, adapter.edit_message(DM, "D1", "hi again", finalize=True)).success
    assert not _run(adapter, adapter.send(GROUP, "hi")).success
    assert not _run(adapter, adapter.edit_message(GROUP, "T1", "x", finalize=True)).success
    assert seen == []


def test_failing_observer_never_changes_the_send_result(monkeypatch):
    _capture(monkeypatch, fail=True)
    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]})])
    result = _run(adapter, adapter.send(GROUP, "hello"))
    assert result.success and result.message_id == "A"


def test_fire_raising_synchronously_never_breaks_a_send(monkeypatch):
    _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]}),
                             (200, {"success": True, "messageId": "Q1"}),
                             (200, {"success": True, "messageId": "M1"}),
                             (200, {"success": True, "messageIds": []}),
                             (200, ["not", "a", "dict"])])

    def boom(*args, **kwargs):
        raise RuntimeError("fire failed")

    adapter._fire_outbound_observers = boom
    assert _run(adapter, adapter.send(GROUP, "hello")).message_id == "A"
    assert _run(adapter, adapter.send_poll(GROUP, "Lunch?", ["yes", "no"])).message_id == "Q1"
    media = _run(adapter, adapter._post_bridge_message("send-media", {"chatId": GROUP, "caption": "c"}, timeout=1))
    assert media.success and media.message_id == "M1"
    assert _run(adapter, adapter.edit_message(GROUP, "T1", "x", finalize=True)).success
    odd = _run(adapter, adapter._post_bridge_message("send-poll", {"chatId": GROUP, "question": "q"}, timeout=1))
    assert odd.success and odd.message_id is None


def test_own_events_have_their_own_cap_and_never_crowd_out_inbound(monkeypatch):
    from plugins.platforms.whatsapp import adapter as adapter_mod

    seen = _capture(monkeypatch)
    monkeypatch.setattr(adapter_mod, "_MAX_PENDING_OWN_EVENTS", 2)
    monkeypatch.setattr(adapter_mod, "_MAX_PENDING_OBSERVER_EVENTS", 2)
    adapter = _make_adapter()

    async def go():
        for i in range(5):  # a burst of own events: only 2 fit their backlog, the rest are dropped
            adapter._fire_outbound_observers(GROUP, "text", [f"O{i}"], "x")
        await adapter._fire_transport_observers(_mutation())  # also own: dropped, backlog still full
        for i in range(2):  # inbound keeps its own, untouched capacity
            await adapter._fire_transport_observers({"chatId": GROUP, "isGroup": True, "messageId": f"I{i}"})
        await _drain(adapter)

    asyncio.run(go())
    outbound = [kw["raw_message"]["messageIds"] for _, kw in seen if kw.get("direction") == "outbound"]
    inbound = [kw["raw_message"]["messageId"] for _, kw in seen if "direction" not in kw and "event_kind" not in kw]
    assert outbound == [["O0"], ["O1"]]
    assert not any(kw.get("event_kind") for _, kw in seen)
    assert inbound == ["I0", "I1"]


def test_no_outbound_event_without_a_subscribed_observer(monkeypatch):
    from hermes_cli import lifecycle, plugins

    calls = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: calls.append(name))
    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]})])
    assert _run(adapter, adapter.send(GROUP, "hello")).success
    assert calls == []

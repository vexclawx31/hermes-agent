"""Observer-only WhatsApp events for this account's own messages: bridge-marked own edits/revokes
(``observerOnly``) and in-process ``direction="outbound"`` events after successful group sends/edits."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig

OWNER = "15550000001@s.whatsapp.net"
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


def _run(adapter, coro):
    async def go():
        result = await coro
        await asyncio.gather(*list(adapter.__dict__.get("_observer_tasks", ())))
        return result

    return asyncio.run(go())


def _outbound(seen):
    return [kw["raw_message"] for _, kw in seen if kw.get("direction") == "outbound"]


# --- inbound: bridge-marked own mutations reach observers, never intake --------------------------

@pytest.mark.parametrize("mutation", ["edit", "revoke"])
def test_observer_only_event_is_observed_but_never_a_turn(monkeypatch, mutation):
    seen = _capture(monkeypatch)
    adapter = _make_adapter()
    # Worst case for intake: owner sender, native mention of the bot, a free-response group, a slash command.
    data = {"observerOnly": True, "ownMutation": mutation, "chatId": GROUP, "isGroup": True, "fromMe": True,
            "senderId": OWNER, "messageId": "P1", "targetMessageId": "T1", "body": "/reset @alex",
            "mentionedIds": [BOT_IDS[0]], "botIds": list(BOT_IDS), "timestamp": 1700000000}
    assert adapter._should_process_message(data) is False
    assert adapter._should_process_message({**data, "observerOnly": False}) is True  # the marker alone decides
    assert _run(adapter, adapter._build_message_event(data)) is None
    assert len(seen) == 1
    name, kwargs = seen[0]
    assert name == "whatsapp_transport_event"
    assert kwargs["raw_message"]["ownMutation"] == mutation
    assert "direction" not in kwargs  # bridge events are never "outbound"


# --- outbound: successful group sends/edits fire direction="outbound" ---------------------------

def test_multi_chunk_group_send_reports_every_chunk_id(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageId": "B", "messageIds": ["A", "B"]}),
                             (200, {"success": True, "messageId": "C", "messageIds": ["C"]})])
    adapter.truncate_message = lambda content, limit: ["part one", "part two"]
    result = _run(adapter, adapter.send(GROUP, "long reply"))
    assert result.success and result.message_id == "C"
    events = _outbound(seen)
    assert [e["messageIds"] for e in events] == [["A", "B"], ["C"]]
    assert [e["body"] for e in events] == ["part one", "part two"]
    for event in events:
        assert {k: event[k] for k in ("chatId", "isGroup", "fromMe", "kind")} == {
            "chatId": GROUP, "isGroup": True, "fromMe": True, "kind": "text"}
        assert isinstance(event["timestamp"], int) and event["timestamp"] > 1_600_000_000_000  # milliseconds
        assert "targetMessageId" not in event


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


def test_group_edit_reports_target_and_spill_ids(monkeypatch):
    seen = _capture(monkeypatch)
    adapter = _make_adapter([(200, {"success": True, "messageIds": ["S1"]})])
    result = _run(adapter, adapter.edit_message(GROUP, "T9", "new text"))
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
    assert _run(adapter, adapter.edit_message(DM, "D1", "hi again")).success
    assert not _run(adapter, adapter.send(GROUP, "hi")).success
    assert not _run(adapter, adapter.edit_message(GROUP, "T1", "x")).success
    assert seen == []


def test_failing_observer_never_changes_the_send_result(monkeypatch):
    _capture(monkeypatch, fail=True)
    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]})])
    result = _run(adapter, adapter.send(GROUP, "hello"))
    assert result.success and result.message_id == "A"


def test_no_outbound_event_without_a_subscribed_observer(monkeypatch):
    from hermes_cli import lifecycle, plugins

    calls = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: calls.append(name))
    adapter = _make_adapter([(200, {"success": True, "messageId": "A", "messageIds": ["A"]})])
    assert _run(adapter, adapter.send(GROUP, "hello")).success
    assert calls == []

"""Outbound read-back for sends that bypass the live adapter (cron's standalone lane, worker send_message).

* WhatsApp ``_standalone_send`` fires the same observer-only ``whatsapp_transport_event``
  (``direction="outbound"``) as the live adapter (#37) for each successful group post, and
  returns every message ID it put on screen, also when a later post fails.
* ``_send_chunks`` keeps every chunk's IDs, not only the last one's.
* The scheduler keeps per-target IDs from both lanes as ``last_delivery_message_ids``.
"""

import asyncio
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cron.scheduler_delivery as sd
from gateway.config import GatewayConfig, Platform
from plugins.platforms.whatsapp.adapter import _standalone_send
from tools.send_message_tool import _send_chunks, result_message_ids

GROUP = "120363000000000001@g.us"


def _resp(status, json_data=None, text_data=None):
    r = AsyncMock()
    r.status = status
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value=text_data or "")
    return r


def _session_with(responses):
    calls = []
    idx = [0]

    def _ctx(value):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=value)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        r = responses[idx[0]] if idx[0] < len(responses) else responses[-1]
        idx[0] += 1
        return _ctx(r)

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session.get = MagicMock(side_effect=lambda url, **kw: _ctx(_resp(200)))
    return _ctx(session), calls


def _pconfig():
    return SimpleNamespace(token="", extra={"bridge_port": 3000})


@pytest.fixture
def hook():
    """Subscribed ``whatsapp_transport_event``; records every invoke_hook call."""
    events = []

    def _invoke(name, **kwargs):
        events.append((name, kwargs))

    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=_invoke):
        yield events


def _join_observers(timeout=5.0):
    for thread in threading.enumerate():
        if thread.name == "whatsapp-outbound-observer":
            thread.join(timeout)


def _send(responses, *args, **kwargs):
    session_ctx, calls = _session_with(responses)
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(_standalone_send(_pconfig(), *args, **kwargs))
    _join_observers()
    return res, calls


def test_group_text_send_fires_outbound_event_and_returns_ids(hook):
    res, _ = _send([_resp(200, {"messageId": "A2", "messageIds": ["A1", "A2"]})], GROUP, "water the plants")
    assert res["success"] is True
    assert res["message_id"] == "A2"
    assert res["message_ids"] == ["A1", "A2"]
    assert len(hook) == 1
    name, kwargs = hook[0]
    assert name == "whatsapp_transport_event"
    assert kwargs["direction"] == "outbound"
    assert kwargs["platform"] == "whatsapp"
    raw = kwargs["raw_message"]
    assert raw["chatId"] == GROUP and raw["isGroup"] is True and raw["fromMe"] is True
    assert raw["kind"] == "text" and raw["messageIds"] == ["A1", "A2"] and raw["body"] == "water the plants"
    assert isinstance(raw["timestamp"], int)


def test_group_media_send_fires_media_event_with_caption(hook):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"x")
    f.close()
    try:
        res, calls = _send([_resp(200, {"messageId": "M1"})], GROUP, "", media_files=[(f.name, False)],
                           caption="the plan")
    finally:
        os.unlink(f.name)
    assert res["message_ids"] == ["M1"]
    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["send-media"]  # POSTs only; the caption rides on it
    raw = hook[0][1]["raw_message"]
    assert raw["kind"] == "media" and raw["messageIds"] == ["M1"] and raw["body"] == "the plan"


def test_direct_chat_send_fires_no_event_but_keeps_ids(hook):
    res, _ = _send([_resp(200, {"messageId": "D1"})], "15551234567", "hi")
    assert res["message_ids"] == ["D1"]
    assert hook == []


def test_partial_failure_keeps_ids_already_on_screen_and_archives_them(hook):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"x")
    f.close()
    try:
        res, _ = _send([_resp(200, {"messageId": "T1"}), _resp(500, text_data="boom")], GROUP, "text first",
                       media_files=[(f.name, False)])
    finally:
        os.unlink(f.name)
    assert res.get("error") and "boom" in res["error"]
    assert res["message_ids"] == ["T1"]
    assert [e[1]["raw_message"]["messageIds"] for e in hook] == [["T1"]]


def test_failed_first_post_has_no_ids_and_no_event(hook):
    res, _ = _send([_resp(500, text_data="down")], GROUP, "hello")
    assert res.get("error")
    assert "message_ids" not in res
    assert hook == []


def test_observer_error_never_changes_the_send_result():
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("archive locked")):
        res, _ = _send([_resp(200, {"messageId": "A1"})], GROUP, "hello")
    assert res["success"] is True and res["message_ids"] == ["A1"]


def test_slow_observer_never_delays_the_send():
    """A slow archive (SQLite lock) must not hold the send: the standalone lane bounds it with a timeout,
    and a false timeout after a delivered post would lose its IDs or queue a resend."""
    release = threading.Event()
    seen = []

    def _slow(name, **kwargs):
        release.wait(5)
        seen.append(kwargs["raw_message"]["messageIds"])

    session_ctx, _ = _session_with([_resp(200, {"messageId": "A1"})])
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=_slow), \
            patch("aiohttp.ClientSession", return_value=session_ctx):
        started = time.monotonic()
        res = asyncio.run(asyncio.wait_for(_standalone_send(_pconfig(), GROUP, "hello"), timeout=1.0))
        elapsed = time.monotonic() - started
        release.set()
        _join_observers()
    assert res["success"] is True and res["message_ids"] == ["A1"]
    assert elapsed < 1.0
    assert seen == [["A1"]]


def test_cancelled_send_still_archives_what_went_out(hook):
    """A send cancelled after its first post (outer timeout, shutdown) cancels promptly; the post that is
    already on screen is still observed."""
    async def _run():
        never = asyncio.Event()  # the media upload hangs until cancelled

        async def _hang(*_):
            await never.wait()

        def _ctx(**enter):
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(**enter)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session = MagicMock()
        session.post = MagicMock(side_effect=lambda url, **kw: _ctx(
            return_value=_resp(200, {"messageId": "T1"})) if url.endswith("/send") else _ctx(side_effect=_hang))
        session.get = MagicMock(side_effect=lambda url, **kw: _ctx(return_value=_resp(200)))
        outer = _ctx(return_value=session)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(b"x")
        f.close()
        try:
            with patch("aiohttp.ClientSession", return_value=outer):
                await asyncio.wait_for(_standalone_send(_pconfig(), GROUP, "text", media_files=[(f.name, False)]),
                                       timeout=0.3)
        finally:
            os.unlink(f.name)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(_run())
    assert time.monotonic() - started < 2.0
    _join_observers()
    assert [e[1]["raw_message"]["messageIds"] for e in hook] == [["T1"]]


def test_no_subscriber_means_no_dispatch():
    invoke = MagicMock()
    with patch("hermes_cli.lifecycle.has_hook", return_value=False), patch("hermes_cli.plugins.invoke_hook", invoke):
        res, _ = _send([_resp(200, {"messageId": "A1"})], GROUP, "hello")
    assert res["success"] is True
    invoke.assert_not_called()


# ---------------------------------------------------------------------------
# _send_chunks: every chunk's IDs
# ---------------------------------------------------------------------------


def test_send_chunks_keeps_every_chunk_id():
    results = iter([{"success": True, "message_id": "c1"}, {"success": True, "message_id": "c2",
                                                             "message_ids": ["c2", "c3"]}])

    async def send_one(chunk, is_last):
        return next(results)

    res = asyncio.run(_send_chunks(["a", "b"], send_one))
    assert res["message_id"] == "c2"
    assert res["message_ids"] == ["c1", "c2", "c3"]


def test_send_chunks_keeps_ids_when_a_later_chunk_fails():
    results = iter([{"success": True, "message_id": "c1"}, {"error": "rate limited"}, {"success": True}])

    async def send_one(chunk, is_last):
        return next(results)

    res = asyncio.run(_send_chunks(["a", "b", "c"], send_one))
    assert res["error"] == "rate limited"
    assert res["message_ids"] == ["c1"]


def test_result_message_ids_shapes():
    assert result_message_ids(None) == []
    assert result_message_ids({"message_id": "x"}) == ["x"]
    assert result_message_ids({"message_id": "x", "message_ids": ["x", "y"]}) == ["x", "y"]
    assert result_message_ids({"error": "e"}) == []


# ---------------------------------------------------------------------------
# Scheduler: last_delivery_message_ids from the standalone lane
# ---------------------------------------------------------------------------


def _target(job):
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job=job, platform=Platform.WHATSAPP, platform_name="whatsapp", chat_id=GROUP, thread_id=None,
                  config=GatewayConfig(), target_adapters={}, mirror_text="", origin={}, mirror_this_target=False)
    return sd._TargetDelivery(**fields)


@pytest.fixture
def persisted(monkeypatch):
    writes = []
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    monkeypatch.setattr("cron.jobs.update_job", lambda job_id, values: writes.append((job_id, values)))
    return writes


def test_standalone_delivery_records_message_ids(monkeypatch, persisted):
    job = {"id": "water"}
    monkeypatch.setattr(sd, "_standalone_send", lambda t, c, m: (
        {"success": True, "message_id": "A2", "message_ids": ["A1", "A2"]}, None))
    t = _target(job)
    delivery_errors = []
    sd._deliver_standalone(t, "water the plants", [], [], delivery_errors)
    assert delivery_errors == []
    sd._record_delivery_verification(job, [])
    assert job["last_delivery_message_ids"] == {t.where: ["A1", "A2"]}
    assert persisted == [("water", {"last_delivery_message_ids": {t.where: ["A1", "A2"]}})]


def test_standalone_partial_failure_still_records_sent_ids(monkeypatch, persisted):
    job = {"id": "water"}
    monkeypatch.setattr(sd, "_standalone_send", lambda t, c, m: ({"error": "media failed", "message_ids": ["T1"]}, None))
    monkeypatch.setattr(sd, "_queue_for_live_reconnect", lambda *a, **k: None)
    t = _target(job)
    delivery_errors = []
    sd._deliver_standalone(t, "text", [], [], delivery_errors)
    assert delivery_errors and "media failed" in delivery_errors[0]
    assert job["_delivery_message_ids"] == {t.where: ["T1"]}


def test_standalone_timeout_records_nothing(monkeypatch, persisted):
    job = {"id": "water"}
    monkeypatch.setattr(sd, "_standalone_send", lambda t, c, m: (None, "timed out (the send may still be in flight)"))
    monkeypatch.setattr(sd, "_queue_for_live_reconnect", lambda *a, **k: None)
    sd._deliver_standalone(_target(job), "text", [], [], [])
    sd._record_delivery_verification(job, [])
    assert "last_delivery_message_ids" not in job
    assert persisted == []


def test_live_lane_records_message_id(monkeypatch, persisted):
    import threading

    from gateway.platforms.base import SendResult

    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        class Transport:
            adapter = object()
            is_relay = False

            async def send(self, platform, chat_id, content, metadata=None):
                return SendResult(success=True, message_id="L1", raw_response={"messageId": "L1"})

        monkeypatch.setattr(sd, "_seed_live_delivery_sessions", lambda *a, **k: None)
        job = {"id": "water"}
        t = _target(job)
        t.transport, t.loop, t.runtime_adapter = Transport(), loop, Transport.adapter
        assert sd._deliver_via_live_adapter(
            t, "water the plants", [], target_errors=[], delivery_errors=[], unverified_targets=[])
        assert job["_delivery_message_ids"] == {t.where: ["L1"]}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
        loop.close()


def test_next_run_clears_previous_ids(persisted):
    job = {"id": "water", "last_delivery_message_ids": {"whatsapp:old@g.us": ["OLD"]}}
    sd._record_delivery_verification(job, [])
    assert job["last_delivery_message_ids"] is None
    assert persisted == [("water", {"last_delivery_message_ids": None})]

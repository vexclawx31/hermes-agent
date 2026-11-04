"""Deferred-action receipts (gateway.deferred): mint in a native turn, verified notice, link, activation by a signed
single-use marker at dispatch, redeem once, settle, cancel, expiry, recovery and integrity.

TrustedIngress is stubbed at the module boundary for mint/link/cancel (its own binding is covered by the ingress
tests); activation and redemption run through the real dispatch path in the last tests.
"""

import asyncio
import base64
import gc
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import gateway.ingress_identity as ingress
import gateway.run as gateway_run
from gateway import deferred
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.injected_turn import TurnSpec, attach_to_event, bind_for_event, clear_injected_turn, get_injected_turn
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "slack-joined-history"
LINE = "switchboard-line"
KEY = Ed25519PrivateKey.generate()
PUB = base64.b64encode(KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
OTHER_KEY = Ed25519PrivateKey.generate()
B, M = "hd-1111111111111111", "hd-2222222222222222"


class _NoticeAdapter:
    def __init__(self, verified=True):
        self.verified, self.notices = verified, []

    async def deferred_notice(self, ti, text):
        self.notices.append(text)
        return {"verified": True, "ts": "1792000000.000300"} if self.verified else {"verified": False}


def _ti(adapter=None, user="U0OWNER001", message="1792000000.000100", profile="default"):
    return ingress.TrustedIngress("slack", profile, message, "C0CHAN0001", "group", "T0TEAM0001", user,
                                  adapter or _NoticeAdapter())


def _home(tmp_path, enabled=True, signer=PUB, extra_entries=None):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    cfg = {"plugins": {"entries": {LINE: {"allow_gateway_injection": True, "allow_injection_author": True,
                                          "allow_injected_turn_context": True}, **(extra_entries or {})}}}
    if enabled:
        cfg["gateway"] = {"deferred": {"enabled": True, "resume_signer": signer, "target": TARGET}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home


def _ctx(home, plugin=PLUGIN):
    return PluginContext(PluginManifest(name=plugin, key=plugin, source="user"), PluginManager(scope_key=str(home)))


TARGET = "tenant:aligned:default"  # this profile's exact Switchboard identity (gateway.deferred.target)


def _marker(receipt, event_ref=M, blocker=B, target=TARGET, marker="a1b2c3d4e5f60718", key=KEY, expires=None):
    m = {"v": 1, "event_ref": event_ref, "blocker": blocker, "receipt": receipt, "target": target, "marker": marker,
         "expires_at": time.time() + 3600 if expires is None else expires}
    m["sig"] = base64.b64encode(key.sign(deferred.marker_payload(m))).decode()
    return m


@pytest.fixture(autouse=True)
def _fresh_stores():
    deferred._STORES.clear()
    yield
    deferred._STORES.clear()
    clear_injected_turn()


async def _minted(home, ti=None, linked=True):
    api = _ctx(home).deferred
    api.register("slack_post_joined_message", ["post_enabled", "thread_reply_enabled"])
    ti = ti or _ti()
    with patch.object(ingress, "get_trusted_ingress", return_value=ti), \
            patch("gateway.session_context.get_session_env", return_value=""):
        out = await api.mint("slack_post_joined_message", {"channel_id": "C0ANN00001", "text": "Launch!"},
                             "post_enabled", "post 'Launch!' to #announcements")
        if linked:
            _ctx(home, LINE).deferred.link(out["id"], B)
    return api, out


def _activate(home, rid, **kw):
    return deferred.activate(home, plugin_id=kw.pop("plugin_id", LINE), profile=kw.pop("profile", "default"),
                             event_ref=kw.pop("event_ref", M), origin_chat=kw.pop("chat", "C0CHAN0001"),
                             origin_thread=kw.pop("thread", ""), marker=_marker(rid, **kw))


class _Event:
    pass


def _bind_resume(rid, plugin=LINE):
    event = _Event()
    attach_to_event(event, TurnSpec(plugin, "message", M, False, {}), (rid, lambda: None))
    bind_for_event(event)
    return event


# ---- mint, notice, link ----

@pytest.mark.asyncio
async def test_mint_needs_a_native_turn_and_a_declared_switch(tmp_path):
    home = _home(tmp_path)
    api = _ctx(home).deferred
    api.register("slack_post_joined_message", ["post_enabled"])
    with patch.object(ingress, "get_trusted_ingress", return_value=None):
        with pytest.raises(deferred.DeferredError, match="native turn"):
            await api.mint("slack_post_joined_message", {"x": 1}, "post_enabled", "s")
    with patch.object(ingress, "get_trusted_ingress", return_value=_ti()):
        with pytest.raises(deferred.DeferredError, match="not a resumable switch"):
            await api.mint("slack_post_joined_message", {"x": 1}, "admin_mode", "s")


@pytest.mark.asyncio
async def test_off_unless_enabled_with_a_signer(tmp_path):
    for home in (_home(tmp_path / "a", enabled=False), _home(tmp_path / "b", signer="not-base64!")):
        api = _ctx(home).deferred
        api.register("t", ["s"])
        assert api.enabled is False
        with patch.object(ingress, "get_trusted_ingress", return_value=_ti()), pytest.raises(deferred.DeferredError):
            await api.mint("t", {"x": 1}, "s", "summary")


@pytest.mark.asyncio
async def test_identity_comes_from_trusted_ingress_and_the_notice_is_verified(tmp_path):
    home = _home(tmp_path)
    adapter = _NoticeAdapter()
    _api, out = await _minted(home, _ti(adapter))
    assert out["state"] == "noticed"
    rec = deferred.store_for(home).get(out["id"])
    assert (rec["requester"], rec["workspace"], rec["origin_chat"], rec["message_id"]) == \
        ("U0OWNER001", "T0TEAM0001", "C0CHAN0001", "1792000000.000100")
    assert rec["action"] == {"channel_id": "C0ANN00001", "text": "Launch!"} and rec["blocker"] == B
    (notice,) = adapter.notices
    assert out["id"] in notice and f'cancel {out["id"]}' in notice
    assert "post 'Launch!' to #announcements" in notice and "post_enabled" in notice


@pytest.mark.asyncio
async def test_an_unverified_notice_stays_pending_and_cannot_resume(tmp_path):
    home = _home(tmp_path)
    _api, out = await _minted(home, _ti(_NoticeAdapter(verified=False)))
    assert out["state"] == "pending"
    with pytest.raises(deferred.DeferredError, match="pending"):
        _activate(home, out["id"])


@pytest.mark.asyncio
async def test_link_only_in_the_minting_turn_and_once(tmp_path):
    home = _home(tmp_path)
    _api, out = await _minted(home, linked=False)
    line = _ctx(home, LINE).deferred
    with patch.object(ingress, "get_trusted_ingress", return_value=_ti(message="1792000000.000999")), \
            pytest.raises(deferred.DeferredError, match="this turn"):
        line.link(out["id"], B)
    with patch.object(ingress, "get_trusted_ingress", return_value=_ti()):
        line.link(out["id"], B)
        with pytest.raises(deferred.DeferredError):
            line.link(out["id"], "hd-3333333333333333")


# ---- activation (the signed, single-use marker) ----

@pytest.mark.asyncio
async def test_a_valid_marker_activates_once(tmp_path):
    home = _home(tmp_path)
    _api, out = await _minted(home)
    assert _activate(home, out["id"]) == out["id"]
    assert deferred.store_for(home).get(out["id"])["state"] == "active"
    deferred.release_unredeemed(home, out["id"])
    with pytest.raises(deferred.DeferredError, match="already used"):  # same marker again, after release
        _activate(home, out["id"])
    assert _activate(home, out["id"], marker="0000000000000001") == out["id"]  # a new marker works


@pytest.mark.asyncio
@pytest.mark.parametrize("kw, why", [
    ({"key": OTHER_KEY}, "signature"),
    ({"event_ref": "hd-9999999999999999"}, "another Switchboard event"),
    ({"blocker": "hd-9999999999999999"}, "another blocker"),
    ({"target": "tenant:upwind:lynx"}, "another profile"),
    ({"target": "tenant:upwind:default"}, "another profile"),  # same profile name, another tenant
    ({"target": "asker:default"}, "another profile"),
    ({"profile": "lynx"}, "another profile"),
    ({"chat": "C0OTHER001"}, "origin"),
    ({"thread": "1792000000.000555"}, "origin"),
    ({"plugin_id": "other-plugin"}, "another plugin"),
    ({"expires": time.time() - 1}, "expired"),
])
async def test_marker_mismatches_activate_nothing(tmp_path, kw, why):
    home = _home(tmp_path)
    _api, out = await _minted(home)
    with pytest.raises(deferred.DeferredError, match=why):
        _activate(home, out["id"], **kw)
    assert deferred.store_for(home).get(out["id"])["state"] == "noticed"


@pytest.mark.asyncio
async def test_a_tampered_row_fails_its_integrity_check(tmp_path):
    import sqlite3
    home = _home(tmp_path)
    _api, out = await _minted(home)
    db = sqlite3.connect(deferred.store_for(home).path)
    doc = json.loads(db.execute("SELECT doc FROM receipts").fetchone()[0])
    doc["action"]["text"] = "something else"
    db.execute("UPDATE receipts SET doc = ?", (json.dumps(doc, sort_keys=True, separators=(",", ":")),))
    db.commit()
    db.close()
    with pytest.raises(deferred.DeferredError, match="integrity"):
        _activate(home, out["id"])


# ---- redeem, settle, cancel, races, expiry, recovery ----

@pytest.mark.asyncio
async def test_redeem_needs_this_turns_resume_and_claims_once(tmp_path):
    home = _home(tmp_path)
    api, out = await _minted(home)
    rid = out["id"]
    _activate(home, rid)
    with pytest.raises(deferred.DeferredError, match="doesn't resume"):
        api.redeem(rid, "slack_post_joined_message")  # no injected turn bound
    event = _bind_resume(rid)
    with pytest.raises(deferred.DeferredError, match="another tool"):
        api.redeem(rid, "slack_edit_message")
    got = api.redeem(rid, "slack_post_joined_message")
    assert got["action"] == {"channel_id": "C0ANN00001", "text": "Launch!"}
    with pytest.raises(deferred.DeferredError, match="claimed"):
        api.redeem(rid, "slack_post_joined_message")
    with pytest.raises(deferred.DeferredError, match="evidence"):
        api.settle(rid, got["claim"], "done")
    api.settle(rid, got["claim"], "done", {"ts": "1792000000.000400"})
    assert deferred.store_for(home).get(rid)["state"] == "done"
    del event


@pytest.mark.asyncio
async def test_cancel_and_claim_race_with_exactly_one_winner(tmp_path):
    home = _home(tmp_path)
    for _ in range(5):
        api, out = await _minted(home)
        rid = out["id"]
        _activate(home, rid, marker=f"{int(time.time() * 1e6) % 10**16:016x}")
        event = _bind_resume(rid)
        results = []

        def claim():
            try:
                api.redeem(rid, "slack_post_joined_message")
                results.append("claimed")
            except deferred.DeferredError:
                results.append("lost")

        def cancel():
            with patch.object(ingress, "get_trusted_ingress", return_value=_ti()):
                try:
                    api.cancel(rid)
                    results.append("cancelled")
                except deferred.DeferredError:
                    results.append("lost")
        ctx_claim = threading.Thread(target=lambda: __import__("contextvars").copy_context().run(claim))
        threads = [ctx_claim, threading.Thread(target=cancel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) in (["claimed", "lost"], ["cancelled", "lost"])
        clear_injected_turn()
        del event


@pytest.mark.asyncio
async def test_two_concurrent_redeems_have_one_winner(tmp_path):
    import contextvars
    home = _home(tmp_path)
    api, out = await _minted(home)
    rid = out["id"]
    _activate(home, rid)
    event = _bind_resume(rid)
    results = []

    def redeem():
        try:
            api.redeem(rid, "slack_post_joined_message")
            results.append("claimed")
        except deferred.DeferredError:
            results.append("lost")
    threads = [threading.Thread(target=contextvars.copy_context().run, args=(redeem,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["claimed", "lost", "lost", "lost"]
    del event


@pytest.mark.asyncio
async def test_the_turns_end_releases_an_unclaimed_receipt_but_a_new_scope_does_not(tmp_path):
    from gateway.session_context import clear_session_vars, set_session_vars
    home = _home(tmp_path)
    _api, out = await _minted(home)
    rid = out["id"]
    _activate(home, rid)
    event = _Event()
    attach_to_event(event, TurnSpec(LINE, "message", M, False, {}), (rid, lambda: deferred.release_unredeemed(home, rid)))
    tokens = set_session_vars(platform="telegram")
    bind_for_event(event)
    set_session_vars(platform="telegram")  # a nested scope starting: not the turn's end
    bind_for_event(event)
    assert deferred.store_for(home).get(rid)["state"] == "active"
    clear_session_vars(tokens)  # the turn ends
    assert deferred.store_for(home).get(rid)["state"] == "noticed"
    del event


@pytest.mark.asyncio
async def test_the_same_refused_call_in_one_turn_reuses_its_receipt(tmp_path):
    home = _home(tmp_path)
    adapter = _NoticeAdapter()
    _api, a = await _minted(home, _ti(adapter), linked=False)
    _api, b = await _minted(home, _ti(adapter), linked=False)
    assert a["id"] == b["id"] and len(adapter.notices) == 1


@pytest.mark.asyncio
async def test_link_checks_chat_requester_and_workspace(tmp_path):
    home = _home(tmp_path)
    _api, out = await _minted(home, linked=False)
    line = _ctx(home, LINE).deferred
    other_chat = ingress.TrustedIngress("slack", "default", "1792000000.000100", "C0OTHER001", "group", "T0TEAM0001",
                                        "U0OWNER001", _NoticeAdapter())
    for ti in (other_chat, _ti(user="U0SOMEONE1")):
        with patch.object(ingress, "get_trusted_ingress", return_value=ti), pytest.raises(deferred.DeferredError):
            line.link(out["id"], B)


@pytest.mark.asyncio
async def test_peek_reads_without_claiming(tmp_path):
    home = _home(tmp_path)
    api, out = await _minted(home)
    rid = out["id"]
    _activate(home, rid)
    with pytest.raises(deferred.DeferredError):
        api.peek(rid, "slack_post_joined_message")  # not in its turn
    event = _bind_resume(rid)
    assert api.peek(rid, "slack_post_joined_message")["action"]["text"] == "Launch!"
    assert api.in_turn(rid) is True
    assert deferred.store_for(home).get(rid)["state"] == "active"
    del event


@pytest.mark.asyncio
async def test_only_the_requester_cancels(tmp_path):
    home = _home(tmp_path)
    api, out = await _minted(home)
    with patch.object(ingress, "get_trusted_ingress", return_value=_ti(user="U0SOMEONE1")), \
            pytest.raises(deferred.DeferredError, match="requester"):
        api.cancel(out["id"])
    with patch.object(ingress, "get_trusted_ingress", return_value=_ti()):
        api.cancel(out["id"])
    assert deferred.store_for(home).get(out["id"])["state"] == "cancelled"
    with pytest.raises(deferred.DeferredError, match="cancelled"):
        _activate(home, out["id"])


@pytest.mark.asyncio
async def test_expiry_is_checked_at_claim(tmp_path):
    home = _home(tmp_path)
    api, out = await _minted(home)
    rid = out["id"]
    _activate(home, rid)
    event = _bind_resume(rid)
    st = deferred.store_for(home)
    with patch.object(st, "clock", lambda: time.time() + deferred.MAX_TTL + 1):
        with pytest.raises(deferred.DeferredError, match="expired"):
            api.redeem(rid, "slack_post_joined_message")
    del event


@pytest.mark.asyncio
async def test_restart_recovery_never_reruns_a_claim(tmp_path):
    home = _home(tmp_path)
    api, a = await _minted(home)
    _api2, b = await _minted(home, _ti(message="1792000000.000200"))  # another turn (one turn reuses one receipt)
    _activate(home, a["id"])
    _activate(home, b["id"], marker="0000000000000002")
    event = _bind_resume(a["id"])
    api.redeem(a["id"], "slack_post_joined_message")  # claimed, then the gateway "crashes"
    deferred._STORES.clear()
    st = deferred.store_for(home)  # a new process: recovery runs once
    assert st.get(a["id"])["state"] == "unknown"
    assert st.get(b["id"])["state"] == "noticed"
    del event


@pytest.mark.asyncio
async def test_the_receipt_returns_to_noticed_when_its_event_is_gone(tmp_path):
    home = _home(tmp_path)
    _api, out = await _minted(home)
    _activate(home, out["id"])
    event = _Event()
    attach_to_event(event, TurnSpec(LINE, "message", M, False, {}),
                    (out["id"], lambda: deferred.release_unredeemed(home, out["id"])))
    del event
    gc.collect()
    assert deferred.store_for(home).get(out["id"])["state"] == "noticed"


# ---- the Slack adapter's notice: post, then read that exact message back ----

class _SlackClient:
    def __init__(self, post=None, back=None, raise_post=False):
        self.post, self.back, self.raise_post, self.calls = post, back, raise_post, []

    async def chat_postMessage(self, **kw):
        self.calls.append(("post", kw))
        if self.raise_post:
            raise TimeoutError()
        return self.post

    async def conversations_replies(self, **kw):
        self.calls.append(("replies", kw))
        return self.back


def _notice(client, text="Deferred df-0123456789abcdef: & <ok>"):
    from plugins.platforms.slack.adapter import SlackAdapter
    fake = SimpleNamespace(_team_clients={"T0TEAM0001": client}, _team_bot_user_ids={"T0TEAM0001": "U0BOT00001"})
    with patch("gateway.session_context.get_session_env", return_value=""):
        return asyncio.run(SlackAdapter.deferred_notice(fake, _ti(), text)), text


POSTED = {"ok": True, "ts": "1792000000.000300"}


def test_slack_notice_verified_only_by_an_exact_read_back():
    stored = "Deferred df-0123456789abcdef: &amp; &lt;ok&gt;"  # as Slack stores it
    good = {"ok": True, "messages": [{"ts": "1792000000.000300", "user": "U0BOT00001", "text": stored,
                                      "thread_ts": "1792000000.000100"}]}
    client = _SlackClient(POSTED, good)
    (out, _text) = _notice(client)
    assert out == {"verified": True, "ts": "1792000000.000300"}
    post = client.calls[0][1]
    assert (post["unfurl_links"], post["unfurl_media"], post["thread_ts"]) == (False, False, "1792000000.000100")
    for back in ({"ok": True, "messages": [{**good["messages"][0], "user": "U0SOMEONE1"}]},
                 {"ok": True, "messages": [{**good["messages"][0], "text": "edited"}]},
                 {"ok": True, "messages": []}, {"ok": False}):
        assert _notice(_SlackClient(POSTED, back))[0]["verified"] is False


def test_slack_notice_refuses_broadcasts_and_never_resends_an_ambiguous_post():
    client = _SlackClient(POSTED, None)
    assert _notice(client, "hey <!channel> look")[0]["verified"] is False and client.calls == []
    client = _SlackClient(raise_post=True)
    out, _ = _notice(client)
    assert out["verified"] is False and [c[0] for c in client.calls] == ["post"]


# ---- real dispatch: a signed marker on an injected human-origin event ----

class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="tok"), Platform.TELEGRAM)
        self.sent = []
        self._running = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((str(chat_id), content))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


async def _settle(runner, adapter, rounds=200):
    for _ in range(rounds):
        await asyncio.sleep(0.01)
        if not runner._background_tasks and not adapter._session_tasks and not adapter._active_sessions:
            return


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "U0OWNER001")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    adapter = _Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    turns = []
    api = _ctx(home).deferred
    api.register("slack_post_joined_message", ["post_enabled"])

    async def _fake_turn(message, context_prompt, history, source, session_id, **_kwargs):
        it = get_injected_turn()
        out = {"resume": None if it is None else it.resume}
        if it is not None and it.resume:
            try:
                out["redeemed"] = api.redeem(it.resume, "slack_post_joined_message")["action"]
            except deferred.DeferredError as exc:
                out["redeemed"] = str(exc)
        turns.append(out)
        return {"final_response": "ok", "messages": [], "tools": [], "history_offset": len(history),
                "last_prompt_tokens": 0}
    runner._run_agent_inner = _fake_turn
    return SimpleNamespace(runner=runner, adapter=adapter, home=home, turns=turns, api=api)


async def _inject_resume(gw, marker):
    ctx = _ctx(gw.home, LINE)
    origin = {"platform": "telegram", "chat_id": "C0CHAN0001", "chat_type": "group", "user_id": "U0OWNER001"}
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[ctx._manager]):
        gw.runner._gateway_loop = asyncio.get_running_loop()
        gw.runner._running = True
        gw.runner._install_plugin_message_injector()
        ok = ctx.inject_message("Switchboard message from Vex: posting is on", origin=origin,
                                author={"id": "vex", "name": "Vex"},
                                turn={"kind": "message", "ref": M, "resume": marker})
        await _settle(gw.runner, gw.adapter)
    return ok


@pytest.mark.asyncio
async def test_a_signed_marker_on_the_injected_event_resumes_once(gateway):
    _api, out = await _minted(gateway.home)
    rid = out["id"]
    assert await _inject_resume(gateway, _marker(rid)) is True
    (t,) = gateway.turns
    assert t == {"resume": rid, "redeemed": {"channel_id": "C0ANN00001", "text": "Launch!"}}
    # the same marker on a second event activates nothing; the message itself still runs
    assert await _inject_resume(gateway, _marker(rid)) is True
    assert gateway.turns[-1] == {"resume": None}


@pytest.mark.asyncio
async def test_a_forged_marker_delivers_the_message_without_any_resume(gateway):
    _api, out = await _minted(gateway.home)
    assert await _inject_resume(gateway, _marker(out["id"], key=OTHER_KEY)) is True
    assert gateway.turns[-1] == {"resume": None}
    assert deferred.store_for(gateway.home).get(out["id"])["state"] == "noticed"

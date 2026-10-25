"""Plugin-supplied author for injected gateway turns (``ctx.inject_message(author=...)``).

Converted from the "today" repro (test_plugin_injection_author_repro.py on the design branch): the
same real ``GatewayRunner`` + real adapter ingress, with only the model call replaced. Each test
names the promise it proves (design-author-seam.md):

* P1 the injected text is recorded and shown as written by a plugin-named agent, never the human;
* P2 routing and authorization are exactly today's, on the human origin;
* P3 the event cannot run commands or answer approval / clarify / slash-confirm / update prompts;
* P4 plugin-authored turns are counted by the bot loop guard, once each;
* P5 zero change for human turns (``author`` absent = today's behaviour).

``turn_author`` is read from the real ``TurnRunner._run_conversation_with_approval`` fed the
TurnContext the turn actually ran with (source, display metadata, plugin author).
"""

import asyncio
import logging
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from agent.plugin_author import PluginAuthor, event_plugin_author
from agent.recall_scope import UNKNOWN_LABEL, plugin_author_label, sender_label
from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType, _event_sender_key
from gateway.run import GatewayRunner
from gateway.sender_attribution import plugin_author_metadata, sender_metadata
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "switchboard"
HUMAN_ID, HUMAN_NAME = "u1", "Requester"
AGENT_ID, AGENT_NAME = "research", "Research Agent"
AUTHOR = {"id": AGENT_ID, "name": AGENT_NAME}
AGENT_TAG = f"[{AGENT_NAME} (agent plugin:{PLUGIN}/{AGENT_ID})]"
AGENT_TURN_AUTHOR = {"id": f"plugin:{PLUGIN}/{AGENT_ID}", "name": AGENT_NAME, "is_bot": True,
                     "verified": False, "origin": "plugin", "plugin": PLUGIN}
AGENT_RECORD = {"v": 1, "plugin": PLUGIN, "id": AGENT_ID, "name": AGENT_NAME}


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, label: str):
        super().__init__(PlatformConfig(enabled=True, token=f"tok-{label}"), Platform.TELEGRAM)
        self.label = label
        self.sent: list[tuple[str, str]] = []
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


def _write_home(home, extra_env: str = "", *, author_flag: bool = True) -> None:
    home.mkdir(parents=True, exist_ok=True)
    entry = {"allow_gateway_injection": True}
    if author_flag:
        entry["allow_injection_author"] = True
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {PLUGIN: entry}}}), encoding="utf-8")
    (home / ".env").write_text(f"TELEGRAM_ALLOWED_USERS={HUMAN_ID}\n{extra_env}", encoding="utf-8")


def _origin(chat_id: str = "100", **overrides) -> dict:
    # A forum-topic thread in a group: a shared multi-user session, so the gateway name-tags speakers.
    return {"platform": "telegram", "chat_id": chat_id, "chat_type": "group", "thread_id": "7",
            "user_id": HUMAN_ID, "user_name": HUMAN_NAME, **overrides}


def _build_gateway(homes: dict, monkeypatch, alpha_env: str = "", author_flag: bool = True):
    """Multiplexed gateway serving ``default`` and ``alpha`` (its own bot); the plugin lives in alpha."""
    home = homes["default"]
    _write_home(homes["default"])
    _write_home(homes["alpha"], alpha_env, author_flag=author_flag)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions"))
    adapters = {name: _RecordingAdapter(name) for name in homes}
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {"alpha": {Platform.TELEGRAM: adapters["alpha"]}}
    for name, adapter in adapters.items():
        adapter.gateway_runner = runner
        if name == "default":
            adapter.set_message_handler(runner._handle_message)
        else:
            adapter.set_owner_profile(name)
            adapter.set_message_handler(runner._make_profile_message_handler(name))

    turns = []
    hold = asyncio.Event()
    hold.set()

    async def _fake_turn(message, context_prompt, history, source, session_id, **kwargs):
        turns.append(SimpleNamespace(message=message, source=source, session_id=session_id, kwargs=kwargs,
                                     plugin_author=kwargs.get("plugin_author"), history=history,
                                     display_metadata=kwargs.get("persist_user_display_metadata") or {}))
        await hold.wait()
        return {"final_response": f"reply-{len(turns)}", "messages": [], "tools": [],
                "history_offset": len(history), "last_prompt_tokens": 0}

    runner._run_agent_inner = _fake_turn
    manager = PluginManager(scope_key=str(homes["alpha"]))
    ctx = PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager)
    return SimpleNamespace(runner=runner, homes=homes, adapters=adapters, turns=turns, ctx=ctx,
                           manager=manager, hold=hold)


@pytest.fixture
def make_gateway(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    homes = {"default": home, "alpha": home / "profiles" / "alpha"}
    built = []

    def _make(alpha_env: str = "", author_flag: bool = True):
        gw = _build_gateway(homes, monkeypatch, alpha_env, author_flag)
        built.append(gw)
        return gw

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=list(homes.items())), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: homes[n]), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in homes):
        yield _make
    for gw in built:
        gw.runner._clear_plugin_message_injector()


def _send(gw, content: str, origin: dict, author=None) -> bool:
    """Call the real ``PluginContext.inject_message`` against this gateway (no waiting)."""
    runner = gw.runner
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[gw.manager]):
        runner._install_plugin_message_injector()
        if author is None:
            return gw.ctx.inject_message(content, origin=origin)
        return gw.ctx.inject_message(content, origin=origin, author=author)


async def _settle(gw, *, until=None) -> None:
    alpha = gw.adapters["alpha"]
    for _ in range(300):
        await asyncio.sleep(0.01)
        if until is not None:
            if until():
                return
            continue
        if not gw.runner._background_tasks and not alpha._session_tasks and not alpha._active_sessions:
            return


async def _inject(gw, content: str, origin: dict, author=None) -> bool:
    accepted = _send(gw, content, origin, author)
    await _settle(gw)
    return accepted


async def _human(gw, text: str, message_id: str) -> None:
    """A real human message from the origin's user through the plugin profile's adapter."""
    alpha = gw.adapters["alpha"]
    src = alpha.build_source(chat_id="100", chat_type="group", user_id=HUMAN_ID, user_name=HUMAN_NAME,
                             thread_id="7", message_id=message_id)
    await alpha.handle_message(MessageEvent(text=text, source=src, message_id=message_id))


def _turn_author(turn, monkeypatch) -> dict:
    """The ``turn_author`` the real gateway turn runner hands ``run_conversation`` for *turn*, with
    ``privacy.recall_scope: attribute`` on (fork PR #27's mode). Also returns the recall scope set."""
    from agent import recall_scope
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(recall_scope, "attribution_enabled", lambda config=None: True)
    monkeypatch.setattr(recall_scope, "author_id_redactor", lambda config=None: (lambda p, u: str(u or "")))
    seen = {}

    class _Agent:
        _recall_scope = "unset"

        def run_conversation(self, message, *, conversation_history=None, task_id=None, turn_author=None, **_kw):
            seen["turn_author"] = turn_author
            return {}

    agent = _Agent()
    ctx = TurnContext(source=turn.source, message=turn.message, session_key="k", session_id=turn.session_id,
                      persist_user_display_metadata=turn.display_metadata, plugin_author=turn.plugin_author)
    worker = TurnRunner(SimpleNamespace(_consume_pending_native_image_paths=lambda key: []), ctx)
    worker._run_conversation_with_approval(agent, [], None, None, None)
    seen["recall_scope"] = agent._recall_scope
    return seen


def _guard_counts(gw) -> list[int]:
    guard = getattr(gw.runner, "_bot_loop_guard", None)
    return [] if guard is None else [len(v) for v in guard._events.values()]


def _small_guard(gw, max_events: int = 2) -> BotLoopGuard:
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(max_events=max_events, window_seconds=300.0,
                                                               cooldown_seconds=600.0))
    gw.runner._bot_loop_guard = guard
    return guard


# ── P1 / P2: the converted repro ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_after_author_rides_separately_while_the_human_origin_authorizes(make_gateway, monkeypatch):
    """P1+P2. Was: the agent's id had to go in ``user_id`` and was refused, or the human's id made
    the turn the human's. Now routing/authorization read the human origin (unchanged) and the agent
    rides in ``author``: name tag, turn_author and the persisted record all name the agent."""
    gw = make_gateway()
    assert await _inject(gw, "Here is the research result", _origin(), AUTHOR) is True
    (turn,) = gw.turns
    assert gw.adapters["alpha"].sent == [("100", "reply-1")]
    # P2: the turn runs on the human origin exactly as before.
    assert (turn.source.user_id, turn.source.user_name, turn.source.is_bot) == (HUMAN_ID, HUMAN_NAME, False)
    # P1: every attribution surface names the agent.
    assert turn.message.startswith(f"{AGENT_TAG} Here is the research result")
    assert f"[{HUMAN_NAME}]" not in turn.message
    seen = _turn_author(turn, monkeypatch)
    assert seen["turn_author"] == AGENT_TURN_AUTHOR
    assert seen["recall_scope"] is None
    assert turn.plugin_author == PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME)


@pytest.mark.asyncio
async def test_after_authored_turn_persists_a_display_only_plugin_author_record(make_gateway):
    """P1. Was: no persisted sender, recall label ``?``. Now a separate ``plugin_author`` record
    (never ``sender``): the sender-based recall label stays unknown (no human, no trust), and the
    display label names the agent."""
    gw = make_gateway()
    await _inject(gw, "Here is the research result", _origin(), AUTHOR)
    (turn,) = gw.turns
    assert turn.display_metadata["plugin_author"] == AGENT_RECORD
    assert "sender" not in turn.display_metadata
    assert sender_label(turn.display_metadata) == UNKNOWN_LABEL
    assert plugin_author_label(turn.display_metadata) == f"agent plugin:{PLUGIN}/{AGENT_ID}, shown as {AGENT_NAME}"


@pytest.mark.asyncio
async def test_after_agent_user_id_origin_is_still_refused_with_an_author(make_gateway, caplog):
    """P2. The author never widens authorization: an origin naming an unlisted id is refused at
    dispatch exactly as today, author or not."""
    gw = make_gateway()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        accepted = await _inject(gw, "x", _origin(user_id="bot-research", user_name=AGENT_NAME), AUTHOR)
    assert accepted is True  # scheduled; refused later, at dispatch (unchanged)
    assert gw.turns == [] and gw.adapters["alpha"].sent == []
    assert "denied by current gateway authorization" in caplog.text
    assert _guard_counts(gw) == []  # refused before the loop guard is charged


@pytest.mark.asyncio
async def test_after_is_bot_origin_with_agent_id_is_still_refused_when_bots_are_allowed(make_gateway, caplog):
    """P2. ``is_bot`` on the origin is still dropped by the parser and the human allowlist still
    decides; the author seam does not go through bot admission."""
    gw = make_gateway(alpha_env="TELEGRAM_ALLOW_BOTS=all\n")
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _inject(gw, "agent says hi", _origin(user_id="bot-research", is_bot=True), AUTHOR)
    assert gw.turns == []
    assert "denied by current gateway authorization" in caplog.text


def test_origin_dict_still_drops_is_bot():
    """P2: the origin parser is unchanged; ``is_bot`` never reaches authz from an origin."""
    assert SessionSource.from_dict(_origin(is_bot=True)).is_bot is False


@pytest.mark.asyncio
async def test_author_without_the_flag_is_refused_and_nothing_runs(make_gateway):
    """Capability: ``allow_injection_author`` is required on top of ``allow_gateway_injection``."""
    gw = make_gateway(author_flag=False)
    assert await _inject(gw, "x", _origin(), AUTHOR) is False
    assert gw.turns == []


# ── P5: author absent = today's behaviour ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_author_absent_keeps_todays_human_origin_behaviour(make_gateway, monkeypatch):
    """P5 / Vex condition 1. Without ``author`` an origin injection is byte-for-byte today's: the
    turn runs as the human (name tag, turn_author), no sender (internal), no plugin record, no
    ``plugin_author`` turn kwarg, and the loop guard is never charged."""
    gw = make_gateway()
    assert await _inject(gw, "Here is the research result", _origin(is_bot=True)) is True
    (turn,) = gw.turns
    assert f"[{HUMAN_NAME}] Here is the research result" in turn.message
    assert "plugin_author" not in turn.kwargs
    assert "plugin_author" not in turn.display_metadata and "sender" not in turn.display_metadata
    assert sender_label(turn.display_metadata) == UNKNOWN_LABEL
    seen = _turn_author(turn, monkeypatch)
    assert seen["turn_author"] == {"id": HUMAN_ID, "name": HUMAN_NAME, "is_bot": False, "verified": False}
    assert _guard_counts(gw) == []


@pytest.mark.asyncio
async def test_author_absent_dispatch_event_is_unchanged():
    """Vex condition 1: the event the dispatch builds without an author has today's exact fields
    and carries no author or loop-guard marker."""
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u1")
    meta = {"hermes_plugin_id": PLUGIN, "hermes_plugin_injection": True, "gateway_session_key": "k"}
    built = GatewayRunner._plugin_injection_event(content="hi", source=source, metadata=dict(meta))
    today = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, internal=True,
                         allow_gateway_control=False, metadata=dict(meta), timestamp=built.timestamp)
    assert built == today
    assert built._plugin_author is None and not hasattr(built, "_bot_loop_admitted")


# ── Lifecycle (Vex condition 2) ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_authored_event_queued_behind_running_human_turn_keeps_author_on_drain(make_gateway, monkeypatch):
    """A human turn is running; the plugin injects; the authored event queues (internal events
    never steer or interrupt) and, when the human turn ends, runs as ITS OWN turn with its author."""
    gw = make_gateway()
    gw.hold.clear()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    gw.runner._running = True
    await _human(gw, "please research this", "m1")
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    assert _send(gw, "Here is the research result", _origin(), AUTHOR) is True
    await _settle(gw, until=lambda: not gw.runner._background_tasks)
    assert len(gw.turns) == 1  # queued, not merged into or steering the human turn
    gw.hold.set()
    await _settle(gw, until=lambda: len(gw.turns) == 2)
    await _settle(gw)

    human, authored = gw.turns
    assert human.plugin_author is None and human.display_metadata["sender"]["user_id"] == HUMAN_ID
    assert "plugin_author" not in human.display_metadata
    assert "please research this" in human.message and AGENT_TAG not in human.message
    assert authored.plugin_author == PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME)
    assert authored.message.startswith(f"{AGENT_TAG} Here is the research result")
    assert authored.display_metadata["plugin_author"] == AGENT_RECORD
    assert "sender" not in authored.display_metadata
    assert _turn_author(authored, monkeypatch)["turn_author"] == AGENT_TURN_AUTHOR


@pytest.mark.asyncio
async def test_authored_turn_interrupted_by_human_keeps_row_and_human_keeps_sender(make_gateway, monkeypatch):
    """An authored turn is running and a human writes: the authored turn keeps its plugin record
    and agent author; the human's message is its own turn with the human's trusted sender."""
    gw = make_gateway()
    gw.hold.clear()
    assert _send(gw, "Here is the research result", _origin(), AUTHOR) is True
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    await _human(gw, "stop, thanks", "m2")
    await asyncio.sleep(0.05)
    gw.hold.set()
    await _settle(gw, until=lambda: len(gw.turns) == 2)
    await _settle(gw)

    authored, human = gw.turns
    assert authored.display_metadata["plugin_author"] == AGENT_RECORD and "sender" not in authored.display_metadata
    assert _turn_author(authored, monkeypatch)["turn_author"] == AGENT_TURN_AUTHOR
    assert human.plugin_author is None and "plugin_author" not in human.kwargs
    assert "plugin_author" not in human.display_metadata
    sender = human.display_metadata["sender"]
    assert (sender["user_id"], sender["trusted"]) == (HUMAN_ID, True)
    assert f"[{HUMAN_NAME}] stop, thanks" in human.message and AGENT_TAG not in human.message
    author = _turn_author(human, monkeypatch)["turn_author"]
    assert author == {"id": HUMAN_ID, "name": HUMAN_NAME, "is_bot": False, "verified": True}


def _followup_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.adapters = {}
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="k")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._intake_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._persist_prompt_pins = AsyncMock()
    return runner


def _authored_event(source, text="Here is the research result"):
    event = GatewayRunner._plugin_injection_event(
        content=text, source=source, author=PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME),
        metadata={"hermes_plugin_id": PLUGIN, "hermes_plugin_injection": True, "gateway_session_key": "k"})
    return event


def _human_source(message_id="m9"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", thread_id="7",
                         user_id=HUMAN_ID, user_name=HUMAN_NAME, message_id=message_id)


async def _run_followup(runner, turn_source, pending_event, *, interrupted=False):
    turn_ctx = SimpleNamespace(
        source=turn_source, session_id="sid", session_key="k", run_generation=1, _interrupt_depth=0,
        history=[], _status_thread_metadata=None, context_prompt=None, result_holder=[None])
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kw: GatewayRunner._prefix_inbound_sender_context(runner, kw["event"], kw["source"],
                                                                              kw["event"].text))
    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending=pending_event.text, pending_event=pending_event,
        response="resp", result={"interrupted": interrupted, "messages": []}, stream_task=None)
    return runner._run_agent.await_args.kwargs


@pytest.mark.asyncio
async def test_inband_followup_takes_the_author_from_the_queued_event_not_next_source():
    """The in-band drain (``_run_agent_queued_followup``) passes the queued event's own author and
    record; it never inherits the running turn's human, and the human is never named."""
    runner = _followup_runner()
    kwargs = await _run_followup(runner, _human_source(), _authored_event(_human_source(message_id=None)))
    assert kwargs["plugin_author"] == PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME)
    assert kwargs["persist_user_display_metadata"]["plugin_author"] == AGENT_RECORD
    assert "sender" not in kwargs["persist_user_display_metadata"]
    assert kwargs["message"].startswith(AGENT_TAG)


@pytest.mark.asyncio
async def test_human_followup_after_an_authored_turn_keeps_the_human_sender():
    """A human message drained after (or interrupting) an authored turn is the human's: trusted
    sender, human name tag, no plugin author kwarg or record."""
    runner = _followup_runner()
    human_event = MessageEvent(text="thanks", source=_human_source(), message_id="m9")
    kwargs = await _run_followup(runner, _human_source(message_id=None), human_event, interrupted=True)
    assert "plugin_author" not in kwargs
    meta = kwargs["persist_user_display_metadata"]
    assert "plugin_author" not in meta and meta["sender"]["user_id"] == HUMAN_ID and meta["sender"]["trusted"] is True
    assert kwargs["message"] == f"[{HUMAN_NAME}] thanks"


@pytest.mark.asyncio
async def test_restart_never_replays_the_authored_text_as_human(make_gateway):
    """Queued events are in memory only: a restarted gateway has none. Auto-resume of the
    interrupted session synthesizes its own empty-text turn: no plugin text, no plugin author,
    no plugin record (and the origin's identity, unchanged, as today)."""
    gw = make_gateway()
    await _inject(gw, "Here is the research result", _origin(), AUTHOR)
    (authored,) = gw.turns
    entry = next(e for e in gw.runner.session_store._entries.values() if e.session_id == authored.session_id)
    gw.runner.session_store.mark_resume_pending(entry.session_key, "restart_timeout")

    restarted = make_gateway()  # a new process: fresh runner, same homes and session store on disk
    alpha = restarted.adapters["alpha"]
    assert not alpha._pending_messages and not getattr(restarted.runner, "_pending_fifo", None)
    restarted.runner._gateway_loop = asyncio.get_running_loop()
    restarted.runner._running = True
    with patch.object(restarted.runner, "_is_user_authorized_for_source", return_value=True):
        assert restarted.runner._schedule_resume_pending_sessions() == 1
        await _settle(restarted, until=lambda: len(restarted.turns) == 1)
        await _settle(restarted)
    (resumed,) = restarted.turns
    # The synthesized resume turn carries no text of its own (the recovery note is added later, in
    # the turn runner); as today it is tagged with the session origin's identity (out of scope).
    assert resumed.message == f"[{HUMAN_NAME}] "
    assert "Here is the research result" not in resumed.message and AGENT_TAG not in resumed.message
    assert "plugin_author" not in resumed.kwargs and "plugin_author" not in resumed.display_metadata


def test_authored_event_never_shares_a_slot_with_a_human_event():
    """Busy FIFO and the shared merge helper: a human photo never merges into an authored slot (or
    the reverse). The runner's security context keeps them apart, and ``merge_pending_message_event``
    itself refuses the author/internal boundary and hands the incoming event to the FIFO."""
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(_pending_messages={})
    runner._delivery_adapter_for = lambda source: adapter
    runner._queue_depth = lambda key, adapter=None: 0
    queued = []
    runner._enqueue_fifo = lambda key, event, adapter: queued.append(event)

    authored = _authored_event(_human_source(message_id=None))
    human_photo = MessageEvent(text="look", message_type=MessageType.PHOTO, source=_human_source(),
                               internal=True, allow_gateway_control=False,
                               metadata=dict(authored.metadata), media_urls=["/x.png"], media_types=["image/png"])
    adapter._pending_messages["k"] = authored
    runner._queue_or_replace_pending_event("k", human_photo)
    assert queued == [human_photo] and authored.text == "Here is the research result"

    assert _event_sender_key(authored) == ("plugin", f"plugin:{PLUGIN}/{AGENT_ID}")
    assert _event_sender_key(authored) != _event_sender_key(human_photo)
    for merge_text in (False, True):
        pending, behind = {"k": authored}, []
        merge_pending_message_event(pending, "k", human_photo, merge_text=merge_text, queue_behind=behind.append)
        assert pending == {"k": authored} and behind == [human_photo]
        assert authored._mixed_senders is False and authored.text == "Here is the research result"
        assert authored.media_urls == [] and authored.message_type == MessageType.TEXT
    assert plugin_author_metadata(authored) == {"plugin_author": AGENT_RECORD}
    # Reverse direction: an authored event never merges into a human slot either.
    human_text = MessageEvent(text="hi", source=_human_source())
    pending, behind = {"k": human_text}, []
    merge_pending_message_event(pending, "k", authored, merge_text=True, queue_behind=behind.append)
    assert pending == {"k": human_text} and behind == [authored] and human_text.text == "hi"
    # Without a FIFO the queued event is kept untouched (nothing merged).
    pending = {"k": authored}
    merge_pending_message_event(pending, "k", human_text, merge_text=True)
    assert pending == {"k": authored} and authored.text == "Here is the research result"


def test_same_author_events_still_merge_and_humans_are_unchanged():
    """The guard only splits across an author/internal boundary: two human texts still merge."""
    first, second = MessageEvent(text="a", source=_human_source()), MessageEvent(text="b", source=_human_source())
    pending, behind = {"k": first}, []
    merge_pending_message_event(pending, "k", second, merge_text=True, queue_behind=behind.append)
    assert behind == [] and first.text == "a\nb"


# ── P4: loop guard ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_authored_turn_charges_the_loop_guard_exactly_once_across_busy_and_drain(make_gateway):
    """P4. Charged once at dispatch (as a bot message in the origin's chat), then marked spent, so
    the busy path and the drained turn never charge it again."""
    gw = make_gateway()
    guard = _small_guard(gw, max_events=20)
    gw.hold.clear()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    gw.runner._running = True
    await _human(gw, "go", "m1")
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    assert _send(gw, "result one", _origin(), AUTHOR) is True
    await _settle(gw, until=lambda: not gw.runner._background_tasks)
    assert [len(v) for v in guard._events.values()] == [1]
    gw.hold.set()
    await _settle(gw, until=lambda: len(gw.turns) == 2)
    await _settle(gw)
    assert len(gw.turns) == 2 and gw.turns[1].plugin_author is not None
    assert [len(v) for v in guard._events.values()] == [1]


@pytest.mark.asyncio
async def test_loop_guard_trips_after_max_events_and_drops_the_authored_turn(make_gateway, caplog):
    """P4. ``max_events`` authored turns run; the next trips the chat's cooldown and is dropped."""
    gw = make_gateway()
    _small_guard(gw, max_events=2)
    for n in range(2):
        await _inject(gw, f"result {n}", _origin(), AUTHOR)
    assert len(gw.turns) == 2
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _inject(gw, "result 3", _origin(), AUTHOR)
    assert len(gw.turns) == 2
    assert "dropped by the bot loop guard" in caplog.text


@pytest.mark.asyncio
async def test_humans_are_admitted_during_cooldown_and_never_counted(make_gateway):
    """P4. A tripped plugin loop never blocks the human in that chat, and human messages are never
    counted by the guard."""
    gw = make_gateway()
    guard = _small_guard(gw, max_events=1)
    await _inject(gw, "result 1", _origin(), AUTHOR)
    await _inject(gw, "result 2", _origin(), AUTHOR)  # trips
    assert len(gw.turns) == 1 and guard._cooldown_until
    events_before = {k: list(v) for k, v in guard._events.items()}
    gw.runner._gateway_loop = asyncio.get_running_loop()
    await _human(gw, "still here", "m5")
    await _settle(gw)
    assert len(gw.turns) == 2 and gw.turns[1].plugin_author is None
    assert {k: list(v) for k, v in guard._events.items()} == events_before


# ── P3: control surfaces ──────────────────────────────────────────────────────────────────────


def test_p3_authored_event_is_never_a_control_event():
    """P3. The dispatch builder always sets ``allow_gateway_control=False`` for an authored event
    and refuses to mark one that somehow carries control."""
    event = _authored_event(_human_source(message_id=None), text="/reset")
    assert event.allow_gateway_control is False and event.internal is True
    assert event.is_command() is False and event.get_command() is None
    assert event._bot_loop_admitted is True and event_plugin_author(event) is not None
    with patch("gateway.run_plugin_injection.MessageEvent",
               side_effect=lambda **kw: MessageEvent(**{**kw, "allow_gateway_control": True})):
        with pytest.raises(RuntimeError):
            _authored_event(_human_source(message_id=None))


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/reset", "/approve", "/yolo"])
async def test_p3_authored_slash_commands_stay_conversational_text(make_gateway, command):
    """P3 slash. An authored ``/reset``, ``/approve`` or ``/yolo`` is never dispatched as a command:
    it runs as an ordinary tagged agent turn."""
    gw = make_gateway()
    with patch.object(GatewayRunner, "_handle_reset_command", AsyncMock()) as reset, \
            patch.object(GatewayRunner, "_handle_approve_command", AsyncMock()) as approve, \
            patch.object(GatewayRunner, "_handle_yolo_command", AsyncMock()) as yolo:
        await _inject(gw, command, _origin(), AUTHOR)
    (turn,) = gw.turns
    assert turn.message.startswith(f"{AGENT_TAG} {command}")
    reset.assert_not_called()
    approve.assert_not_called()
    yolo.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["yes", "approve"])
async def test_p3_authored_yes_never_resolves_a_pending_dangerous_command_approval(text):
    """P3 approval. With a dangerous-command approval pending, an authored "yes"/"approve" is not
    routed to the approval handler (a human's is)."""
    runner = object.__new__(GatewayRunner)
    runner._handle_approve_command = AsyncMock(return_value=None)
    runner._handle_deny_command = AsyncMock(return_value=None)
    runner._plaintext_approval_words = lambda: {"yes": ("approve", ""), "approve": ("approve", "")}
    runner._delivery_adapter_for = lambda source: None
    with patch("tools.approval.has_blocking_approval", return_value=True):
        assert await runner._route_plaintext_approval_while_busy(
            _authored_event(_human_source(message_id=None), text=text), "k") is False
        runner._handle_approve_command.assert_not_called()
        # Control: the same word from the human resolves it.
        assert await runner._route_plaintext_approval_while_busy(
            MessageEvent(text=text, source=_human_source()), "k") is True
        runner._handle_approve_command.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver", ["_hm_update_prompt_reply", "_hm_clarify_reply", "_hm_slash_confirm_reply"],
                         ids=["update_prompt", "clarify", "slash_confirm"])
async def test_p3_authored_text_never_answers_pending_prompts(resolver):
    """P3 clarify / slash-confirm / update prompt. A pending prompt is resolved only by an event
    allowed to control the gateway: an authored event never reaches the resolvers."""
    runner = object.__new__(GatewayRunner)
    for name in ("_hm_update_prompt_reply", "_hm_clarify_reply", "_hm_slash_confirm_reply"):
        mock = MagicMock(return_value=None) if name == "_hm_update_prompt_reply" else AsyncMock(return_value=None)
        setattr(runner, name, mock)
    resolving = MagicMock(return_value="resolved") if resolver == "_hm_update_prompt_reply" \
        else AsyncMock(return_value="resolved")
    setattr(runner, resolver, resolving)
    authored = _authored_event(_human_source(message_id=None), text="1")
    assert await runner._hm_pending_reply_intercepts(authored, authored.source, "k") is None
    resolving.assert_not_called()
    human = MessageEvent(text="1", source=_human_source())
    assert await runner._hm_pending_reply_intercepts(human, human.source, "k") == "resolved"


@pytest.mark.asyncio
async def test_p3_authored_text_does_not_bypass_to_a_pending_clarify_at_the_adapter(make_gateway):
    """P3 clarify, at the base adapter: with a text clarify pending and the session busy, an
    authored event is never routed to the clarify intercept; it queues as its own turn."""
    gw = make_gateway()
    gw.hold.clear()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    gw.runner._running = True
    await _human(gw, "start", "m1")
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    with patch("tools.clarify_gateway.get_pending_for_session", return_value=object()) as pending, \
            patch.object(GatewayRunner, "_hm_clarify_reply", AsyncMock(return_value="resolved")) as clarify:
        _send(gw, "option 1", _origin(), AUTHOR)
        await _settle(gw, until=lambda: not gw.runner._background_tasks)
        pending.assert_not_called()
        clarify.assert_not_called()
    gw.hold.set()
    await _settle(gw, until=lambda: len(gw.turns) == 2)
    await _settle(gw)
    assert gw.turns[1].message.startswith(f"{AGENT_TAG} option 1")


# ── P5 golden: a human event is byte-identical ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_p5_golden_human_event_attribution_is_byte_identical(make_gateway, monkeypatch):
    """P5 golden (Control-Tower). A human message's persisted sender, name tag and turn_author
    under attribute mode are exactly the pre-seam values (asserted literally, verified against the
    base commit with tests/gateway/test_human_turn_attribution_golden.py), and its ``_run_agent``
    call carries no plugin author."""
    gw = make_gateway()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    await _human(gw, "hello there", "m7")
    await _settle(gw)
    (turn,) = gw.turns
    assert "plugin_author" not in turn.kwargs
    assert turn.message.endswith(f"[{HUMAN_NAME}] hello there")
    assert turn.display_metadata["sender"] == {
        "v": 1, "platform": "telegram", "user_id": HUMAN_ID, "chat_id": "100", "chat_type": "group",
        "thread_id": "7", "thread_reply": True, "message_id": "m7", "trusted": True, "name": HUMAN_NAME,
    }
    assert "plugin_author" not in turn.display_metadata
    seen = _turn_author(turn, monkeypatch)
    assert seen["turn_author"] == {"id": HUMAN_ID, "name": HUMAN_NAME, "is_bot": False, "verified": True}


def test_dispatch_marks_events_admitted_only_with_an_author():
    """The ``_bot_loop_admitted`` marker is set only on authored events (P4), never on today's."""
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u1")
    plain = GatewayRunner._plugin_injection_event(content="x", source=source, metadata={})
    assert not getattr(plain, "_bot_loop_admitted", False)
    assert getattr(_authored_event(source), "_bot_loop_admitted", False) is True


def test_scheduler_refuses_an_author_that_is_not_a_host_object_for_this_plugin():
    """The gateway accepts only the host ``PluginAuthor`` of the calling plugin: a raw dict or another
    plugin's author is refused synchronously (False), nothing is scheduled."""
    runner = object.__new__(GatewayRunner)
    runner._running = True
    loop = asyncio.new_event_loop()
    try:
        runner._gateway_loop = loop
        runner._background_tasks = set()
        for bad in ({"id": AGENT_ID, "name": AGENT_NAME}, PluginAuthor("other", AGENT_ID, AGENT_NAME),
                    SimpleNamespace(plugin=PLUGIN, id=AGENT_ID, name=AGENT_NAME)):
            assert runner._schedule_plugin_message_injection(
                content="x", plugin_id=PLUGIN, session_key="k", author=bad) is False
    finally:
        loop.close()


def test_mixed_event_runs_with_an_unverified_unknown_author(monkeypatch):
    """(c) If a mixed authored event ever arises, it writes no record and runs with neither the
    bot's nor the human's author: an unverified unknown one, no plugin origin, no scope."""
    event = _authored_event(_human_source(message_id=None))
    event._mixed_senders = True
    assert plugin_author_metadata(event) == {}
    assert GatewayRunner._plugin_author_turn_kwargs(event) == {"unknown_author": True}
    assert GatewayRunner._plugin_author_turn_kwargs(MessageEvent(text="x", source=_human_source())) == {}
    assert GatewayRunner._prefix_inbound_sender_context(
        object.__new__(GatewayRunner), event, event.source, "text").startswith("[unknown author] ")
    from agent import recall_scope
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    monkeypatch.setattr(recall_scope, "attribution_enabled", lambda config=None: True)
    seen = {}

    class _Agent:
        def run_conversation(self, message, *, turn_author=None, **_kw):
            seen["turn_author"] = turn_author
            return {}

    agent = _Agent()
    ctx = TurnContext(source=event.source, message="m", session_key="k", session_id="s", unknown_author=True)
    TurnRunner(SimpleNamespace(_consume_pending_native_image_paths=lambda key: []), ctx) \
        ._run_conversation_with_approval(agent, [], None, None, None)
    assert seen["turn_author"] == {"id": None, "name": "unknown author", "is_bot": False, "verified": False}
    assert agent._recall_scope is None


def test_persisted_rows_and_turn_context_defaults():
    """TurnContext defaults to no plugin author; a datetime-stamped human row never gets one."""
    from gateway.turn_context import TurnContext
    assert TurnContext().plugin_author is None
    assert plugin_author_metadata(MessageEvent(text="x", source=_human_source(), timestamp=datetime.now())) == {}


# ── H1 / M1 (PR #28 review): a human's words never run inside a queued authored event ───────


AUTHORED_TEXT = "Here is the research result"


async def _authored_queued_behind_human(make_gateway):
    """A held human turn is running and an authored event is queued in the pending slot."""
    gw = make_gateway()
    gw.hold.clear()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    gw.runner._running = True
    await _human(gw, "please research this", "m1")
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    assert _send(gw, AUTHORED_TEXT, _origin(), AUTHOR) is True
    alpha = gw.adapters["alpha"]
    await _settle(gw, until=lambda: not gw.runner._background_tasks and alpha._pending_messages)
    (key,) = alpha._pending_messages
    queued = alpha._pending_messages[key]
    assert event_plugin_author(queued) is not None and queued.text == AUTHORED_TEXT
    return gw, alpha, key, queued


def _human_event(gw, text, message_id, **kw):
    src = gw.adapters["alpha"].build_source(chat_id="100", chat_type="group", user_id=HUMAN_ID,
                                            user_name=HUMAN_NAME, thread_id="7", message_id=message_id)
    return MessageEvent(text=text, source=src, message_id=message_id, **kw)


async def _drain(gw, alpha, key):
    """Dequeue every follow-up with the runner's real in-band drain (slot, then FIFO promotion)."""
    drained = []
    for _ in range(5):
        event, _text = await gw.runner._run_agent_drain_pending({"final_response": "x"}, alpha,
                                                                drained[-1].source if drained else
                                                                alpha._pending_messages[key].source, key)
        if event is None:
            break
        drained.append(event)
    return drained


def _attribution(gw, event):
    return SimpleNamespace(
        text=gw.runner._prefix_inbound_sender_context(event, event.source, event.text or ""),
        record=plugin_author_metadata(event), sender=sender_metadata(event).get("sender"),
        turn=GatewayRunner._plugin_author_turn_kwargs(event))


def _assert_split(gw, drained, human_text):
    """The authored event runs alone with its record and agent author; the human text is its own
    turn with the human's trusted sender and no plugin author."""
    authored, human = drained
    a, h = _attribution(gw, authored), _attribution(gw, human)
    assert a.text == f"{AGENT_TAG} {AUTHORED_TEXT}" and human_text not in a.text
    assert a.record == {"plugin_author": AGENT_RECORD} and a.sender is None
    assert a.turn == {"plugin_author": PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME)}
    assert authored._mixed_senders is False and authored.allow_gateway_control is False
    assert h.turn == {} and h.record == {} and AGENT_TAG not in h.text and human_text in h.text
    assert h.sender["user_id"] == HUMAN_ID and h.sender["trusted"] is True


async def _release(gw):
    gw.hold.set()
    await _settle(gw)


@pytest.mark.asyncio
async def test_h1_queue_mode_debounce_never_appends_human_text_to_a_queued_authored_event(make_gateway, monkeypatch):
    """busy_text_mode=queue: the human's debounced text is never merged into the queued authored
    event. End to end: the authored text runs alone with its record, then the human's own turn."""
    gw = make_gateway()
    alpha = gw.adapters["alpha"]
    alpha._busy_text_mode = "queue"
    alpha._busy_text_debounce_seconds = 0.01
    monkeypatch.setattr(type(gw.runner), "_effective_busy_text_mode", lambda self, src: "queue", raising=False)
    gw.hold.clear()
    gw.runner._gateway_loop = asyncio.get_running_loop()
    gw.runner._running = True
    await _human(gw, "please research this", "m1")
    await _settle(gw, until=lambda: len(gw.turns) == 1)
    assert _send(gw, AUTHORED_TEXT, _origin(), AUTHOR) is True
    await _settle(gw, until=lambda: not gw.runner._background_tasks)
    await _human(gw, "HUMAN-OWN-WORDS delete the repo", "m2")
    await asyncio.sleep(0.2)
    queued = next(iter(alpha._pending_messages.values()))
    assert queued.text == AUTHORED_TEXT and queued._mixed_senders is False
    gw.hold.set()
    await _settle(gw, until=lambda: len(gw.turns) >= 3)
    await _settle(gw)
    _first, authored, human = gw.turns
    assert authored.message == f"{AGENT_TAG} {AUTHORED_TEXT}"
    assert authored.display_metadata["plugin_author"] == AGENT_RECORD and "sender" not in authored.display_metadata
    assert authored.plugin_author == PluginAuthor(PLUGIN, AGENT_ID, AGENT_NAME)
    assert "HUMAN-OWN-WORDS" in human.message and AGENT_TAG not in human.message
    assert human.plugin_author is None and "plugin_author" not in human.display_metadata
    assert human.display_metadata["sender"]["user_id"] == HUMAN_ID
    assert _turn_author(human, monkeypatch)["turn_author"]["id"] == HUMAN_ID


def test_h1_text_debounce_never_merges_plugin_authored_events():
    adapter = _RecordingAdapter("x")
    authored = _authored_event(_human_source(message_id=None))
    human = MessageEvent(text="hi", source=_human_source())
    assert adapter._can_merge_text_debounce_events(authored, human) is False
    assert adapter._can_merge_text_debounce_events(human, authored) is False
    assert adapter._can_merge_text_debounce_events(human, MessageEvent(text="b", source=_human_source())) is True


@pytest.mark.asyncio
async def test_h1_telegram_photo_follow_up_takes_its_own_fifo_place(make_gateway):
    gw, alpha, key, queued = await _authored_queued_behind_human(make_gateway)
    photo = _human_event(gw, "HUMAN caption", "m3", message_type=MessageType.PHOTO,
                         media_urls=["/x.png"], media_types=["image/png"])
    handled, _ = await gw.runner._hm_busy_slash_or_photo(photo, photo.source, key)
    assert handled is True
    assert alpha._pending_messages[key] is queued and queued.text == AUTHORED_TEXT and not queued.media_urls
    drained = await _drain(gw, alpha, key)
    assert drained == [queued, photo]
    _assert_split(gw, drained, "HUMAN caption")
    await _release(gw)


@pytest.mark.asyncio
async def test_h1_telegram_grace_window_follow_up_takes_its_own_fifo_place(make_gateway):
    gw, alpha, key, queued = await _authored_queued_behind_human(make_gateway)
    gw.runner._session_state(key).turn.started_ts = time.time()
    text = _human_event(gw, "HUMAN grace words", "m4")
    assert gw.runner._hm_busy_telegram_grace_queue(text, text.source, key, "interrupt") is True
    assert alpha._pending_messages[key] is queued and queued.text == AUTHORED_TEXT
    drained = await _drain(gw, alpha, key)
    assert drained == [queued, text]
    _assert_split(gw, drained, "HUMAN grace words")
    await _release(gw)


@pytest.mark.asyncio
async def test_h1_agent_pending_sentinel_follow_up_takes_its_own_fifo_place(make_gateway):
    from gateway.run import _AGENT_PENDING_SENTINEL
    gw, alpha, key, queued = await _authored_queued_behind_human(make_gateway)
    state = gw.runner._session_state(key)
    saved_agent, saved_started = state.turn.agent, state.turn.started_ts
    state.turn.agent, state.turn.started_ts = _AGENT_PENDING_SENTINEL, 0
    text = _human_event(gw, "HUMAN sentinel words", "m5")
    try:
        assert await gw.runner._hm_handle_running_session_message(text, text.source, key) is None
    finally:
        state.turn.agent, state.turn.started_ts = saved_agent, saved_started
    assert alpha._pending_messages[key] is queued and queued.text == AUTHORED_TEXT
    drained = await _drain(gw, alpha, key)
    assert drained == [queued, text]
    _assert_split(gw, drained, "HUMAN sentinel words")
    await _release(gw)


@pytest.mark.asyncio
async def test_h1_base_adapter_queue_path_takes_its_own_fifo_place(make_gateway):
    """The base adapter's non-debounce busy path (no runner busy handler) splits too."""
    gw, alpha, key, queued = await _authored_queued_behind_human(make_gateway)
    text = _human_event(gw, "HUMAN base words", "m6")
    merge_pending_message_event(alpha._pending_messages, key, text, merge_text=True,
                                queue_behind=alpha._queue_behind_pending(key))
    assert alpha._pending_messages[key] is queued and queued.text == AUTHORED_TEXT
    drained = await _drain(gw, alpha, key)
    assert drained == [queued, text]
    _assert_split(gw, drained, "HUMAN base words")
    await _release(gw)


@pytest.mark.asyncio
async def test_m1_depth_cap_requeue_never_gives_authored_text_human_control():
    """At the interrupt-depth cap the follow-up is re-queued; into a slot holding a caption-less
    human photo it must not merge (which would hand authored "/yolo" the photo's gateway control).
    It takes its own FIFO place, still non-control."""
    runner = _followup_runner()
    adapter = SimpleNamespace(_pending_messages={})
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    fifo = []
    runner._enqueue_fifo = lambda key, event, adapter: fifo.append(event)
    photo = MessageEvent(text="", message_type=MessageType.PHOTO, source=_human_source(),
                         media_urls=["/x.png"], media_types=["image/png"])
    adapter._pending_messages["k"] = photo
    authored = _authored_event(_human_source(message_id=None), text="/yolo")
    turn_ctx = SimpleNamespace(source=_human_source(), session_id="sid", session_key="k", run_generation=1,
                               _interrupt_depth=runner._MAX_INTERRUPT_DEPTH, history=[],
                               _status_thread_metadata=None, context_prompt=None, result_holder=[None])
    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=adapter, pending="/yolo", pending_event=authored,
        response="r", result={"interrupted": True, "messages": []}, stream_task=None)
    assert adapter._pending_messages["k"] is photo and photo.text == "" and photo.allow_gateway_control is True
    assert fifo == [authored] and authored.text == "/yolo"
    assert authored.allow_gateway_control is False and authored.is_command() is False
    runner._run_agent.assert_not_called()

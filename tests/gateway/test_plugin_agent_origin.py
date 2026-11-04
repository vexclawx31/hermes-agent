"""Agent-origin plugin turns and the injected-turn context (``inject_message(..., turn=...)``).

Real ``GatewayRunner`` dispatch (``_schedule_plugin_message_injection`` -> ``_dispatch_plugin_origin_injection``
-> the adapter's ``handle_message`` -> the turn), with a recording transport; only the model call is replaced.
It records what a tool would read inside the turn: ``get_injected_turn()``, ``get_trusted_ingress()`` and the
turn's source.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.injected_turn import get_injected_turn
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "switchboard-line"
REF = "hd-0123456789abcdef"
AUTHOR = {"id": "vex", "name": "Vex"}
TEAM = "T0TEAM0001"


class _Adapter(BasePlatformAdapter):
    def __init__(self, admits=(True, "")):
        super().__init__(PlatformConfig(enabled=True, token="tok"), Platform.TELEGRAM)
        self.sent, self.admit_calls, self.admits = [], [], admits
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

    async def agent_origin_admits(self, source):
        self.admit_calls.append((source.chat_id, source.thread_id, source.scope_id))
        return self.admits


def _config(*, grants=None, context_flag=True, plugins=(PLUGIN,)):
    entry = {"allow_gateway_injection": True, "allow_injection_author": True}
    if context_flag:
        entry["allow_injected_turn_context"] = True
    cfg = {"plugins": {"entries": {PLUGIN: entry}}}
    if grants is not None:
        cfg["gateway"] = {"agent_origin": {"plugins": list(plugins), "destinations": grants}}
    return cfg


def _grant(channel, workspace=TEAM, platform="telegram"):
    return {"platform": platform, "workspace": workspace, "channel": channel}


@pytest.fixture
def make_gateway(tmp_path, monkeypatch):
    def _make(cfg, admits=(True, ""), home_channel=None):
        home = tmp_path / "hh"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (home / ".env").write_text("TELEGRAM_ALLOWED_USERS=u1\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "u1")  # the human route in the human-origin test
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        config = GatewayConfig(sessions_dir=home / "sessions")
        config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="tok", home_channel=home_channel)
        runner = GatewayRunner(config)
        adapter = _Adapter(admits)
        runner.adapters = {Platform.TELEGRAM: adapter}
        adapter.gateway_runner = runner
        adapter.set_message_handler(runner._handle_message)
        turns, children = [], []

        async def _fake_turn(message, context_prompt, history, source, session_id, **_kwargs):
            from gateway.ingress_identity import get_trusted_ingress
            it = get_injected_turn()
            turns.append(SimpleNamespace(
                message=message, source=source,
                turn=None if it is None else (it.plugin_id, it.kind, it.ref, it.agent_origin),
                trusted=get_trusted_ingress()))
            seen = []

            async def _child():  # copies this turn's context; checked again after the turn ends
                seen.append(get_injected_turn() is not None)
                await asyncio.sleep(0.2)
                seen.append(get_injected_turn() is not None)
            children.append((asyncio.ensure_future(_child()), seen))
            return {"final_response": f"reply-{len(turns)}", "messages": [], "tools": [],
                    "history_offset": len(history), "last_prompt_tokens": 0}

        runner._run_agent_inner = _fake_turn
        ctx = PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), PluginManager(scope_key=str(home)))
        return SimpleNamespace(runner=runner, adapter=adapter, turns=turns, children=children, ctx=ctx, home=home)
    return _make


async def _publish(runner):
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._install_plugin_message_injector()


async def _settle(runner, adapter, rounds: int = 200):
    for _ in range(rounds):
        await asyncio.sleep(0.01)
        if not runner._background_tasks and not adapter._session_tasks and not adapter._active_sessions:
            return


def _dest(chat_id="C0CHAN0001", thread_id="1792000000.000200", **extra):
    return {"platform": "telegram", "chat_id": chat_id, "chat_type": "group", "thread_id": thread_id,
            "scope_id": TEAM, **extra}


def _turn(agent_origin=True, kind="message", ref=REF):
    return {"kind": kind, "ref": ref, "agent_origin": agent_origin}


async def _inject(gw, origin, turn=None, author=AUTHOR):
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[gw.ctx._manager]):
        await _publish(gw.runner)
        ok = gw.ctx.inject_message("Switchboard message from Vex", origin=origin, author=author, turn=turn)
        await _settle(gw.runner, gw.adapter)
    return ok


@pytest.mark.asyncio
async def test_granted_destination_runs_a_turn_with_no_human_and_the_context(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    assert await _inject(gw, _dest(), _turn()) is True
    (t,) = gw.turns
    assert t.turn == (PLUGIN, "message", REF, True)
    assert t.trusted is None  # never TrustedIngress
    assert not t.source.user_id and not t.source.user_name
    assert (t.source.chat_id, t.source.thread_id) == ("C0CHAN0001", "1792000000.000200")
    assert gw.adapter.admit_calls == [("C0CHAN0001", "1792000000.000200", TEAM)]  # checked before the turn
    assert gw.adapter.sent and gw.adapter.sent[-1][0] == "C0CHAN0001"


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [
    None,                                          # no gateway.agent_origin block
    [],                                            # no destinations
    [_grant("C0OTHER001")],                        # another channel
    [_grant("C0CHAN0001", workspace="T0OTHER01")],  # another workspace
    [_grant("C0CHAN0001", platform="slack")],      # another platform
    [{"platform": "telegram", "channel": "C0CHAN0001"}],  # malformed: no workspace
])
async def test_without_a_matching_grant_nothing_runs(make_gateway, grants):
    gw = make_gateway(_config(grants=grants))
    assert await _inject(gw, _dest(), _turn()) is False  # refused synchronously, so the plugin can report it
    assert gw.turns == [] and gw.adapter.sent == []
    assert gw.adapter.admit_calls == []  # the grant is checked first


@pytest.mark.asyncio
async def test_grant_for_another_plugin_admits_nothing(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")], plugins=("other-plugin",)))
    await _inject(gw, _dest(), _turn())
    assert gw.turns == []


@pytest.mark.asyncio
async def test_adapter_refusal_stops_the_turn_before_any_tool(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]), admits=(False, "the bot is not a member"))
    await _inject(gw, _dest(), _turn())
    assert gw.turns == [] and gw.adapter.sent == []
    assert len(gw.adapter.admit_calls) == 1


@pytest.mark.asyncio
async def test_agent_origin_never_names_a_user(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    assert await _inject(gw, _dest(user_id="u1"), _turn()) is False
    assert await _inject(gw, _dest(user_name="Jamal"), _turn()) is False
    assert gw.turns == []


@pytest.mark.asyncio
async def test_home_resolves_to_the_configured_home_and_needs_its_own_grant(make_gateway):
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="C0HOME0001", name="home", scope_id=TEAM)
    gw = make_gateway(_config(grants=[_grant("C0HOME0001")]), home_channel=home)
    assert await _inject(gw, _dest(chat_id="@home", thread_id=None), _turn()) is True
    assert gw.turns[-1].source.chat_id == "C0HOME0001"

    gw2 = make_gateway(_config(grants=[_grant("C0CHAN0001")]), home_channel=home)  # home not granted
    await _inject(gw2, _dest(chat_id="@home", thread_id=None), _turn())
    assert gw2.turns == []


@pytest.mark.asyncio
async def test_home_without_a_configured_home_is_refused(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0HOME0001")]))
    assert await _inject(gw, _dest(chat_id="@home", thread_id=None), _turn()) is False
    assert gw.turns == []


@pytest.mark.asyncio
async def test_switches_and_shape_are_required(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")], context_flag=False))
    assert await _inject(gw, _dest(), _turn()) is False  # no allow_injected_turn_context
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    assert await _inject(gw, _dest(), _turn(), author=None) is False  # turn needs an author
    for bad in ({"kind": "approval", "ref": REF}, {"kind": "message", "ref": "hd-x"},
                {"kind": "message", "ref": REF, "agent_origin": "yes"}, {"kind": "message", "ref": REF, "x": 1}):
        assert await _inject(gw, _dest(), bad) is False
    assert gw.turns == []


@pytest.mark.asyncio
async def test_a_human_origin_turn_can_carry_the_context_under_normal_authorization(make_gateway):
    gw = make_gateway(_config())
    human = {"platform": "telegram", "chat_id": "100", "chat_type": "group", "thread_id": "7", "user_id": "u1"}
    assert await _inject(gw, human, _turn(agent_origin=False, kind="result")) is True
    (t,) = gw.turns
    assert t.turn == (PLUGIN, "result", REF, False) and t.source.user_id == "u1"
    assert gw.adapter.admit_calls == []  # the human route is authorized as before


@pytest.mark.asyncio
async def test_context_is_gone_after_the_turn_for_children_and_the_next_human_turn(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    await _inject(gw, _dest(), _turn())
    task, seen = gw.children[0]
    await task
    assert seen == [True, False]  # a child that copied the context sees it revoked after the turn

    reply = gw.adapter.build_source(chat_id="C0CHAN0001", chat_type="group", thread_id="1792000000.000200",
                                    user_id="u1")
    await gw.adapter.handle_message(MessageEvent(text="thanks", message_type=MessageType.TEXT, source=reply))
    await _settle(gw.runner, gw.adapter)
    assert gw.turns[-1].turn is None
    assert get_injected_turn() is None


async def _inject_off_loop(gw, origin, turn=None):
    """As switchboard-line does: from its own thread, not the gateway loop."""
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[gw.ctx._manager]):
        await _publish(gw.runner)
        ok = await asyncio.to_thread(gw.ctx.inject_message, "Switchboard message from Vex", origin=origin,
                                     author=AUTHOR, turn=turn)
        await _settle(gw.runner, gw.adapter)
    return ok


@pytest.mark.asyncio
async def test_off_loop_the_full_admission_is_the_synchronous_answer(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]), admits=(False, "the bot is not a member"))
    assert await _inject_off_loop(gw, _dest(), _turn()) is False  # so the line reports undeliverable
    assert gw.turns == []
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    assert await _inject_off_loop(gw, _dest(), _turn()) is True
    assert len(gw.turns) == 1 and len(gw.adapter.admit_calls) == 2  # admitted, then re-checked at dispatch


@pytest.mark.asyncio
async def test_into_a_busy_session_it_is_queued_never_dropped(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    src = gw.runner._plugin_origin_source(_dest(), plugin_home=gw.home, plugin_id=PLUGIN, agent_origin=True)
    key = gw.runner._session_key_for_source(src)
    gw.adapter.set_busy_session_handler(gw.runner._handle_active_session_busy_message)
    gw.adapter._active_sessions[key] = asyncio.Event()  # a turn is already running in that thread
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[gw.ctx._manager]):
        await _publish(gw.runner)
        assert gw.ctx.inject_message("Switchboard message from Vex", origin=_dest(), author=AUTHOR,
                                     turn=_turn()) is True
        for _ in range(50):
            await asyncio.sleep(0.01)
    pending = gw.adapter._pending_messages.get(key)
    assert pending is not None and "Switchboard message from Vex" in pending.text  # queued behind the turn
    from gateway.injected_turn import spec_of_event
    assert spec_of_event(pending).agent_origin is True


@pytest.mark.asyncio
@pytest.mark.parametrize("origin, home", [
    (_dest(chat_type="dm"), None),                         # a channel dressed as a DM
    (_dest(message_id="1792000000.000900"), None),         # nothing grants a message id
    (_dest(parent_chat_id="C0OTHER001"), None),
    (_dest(chat_id="@home", thread_id=None, scope_id="T0OTHER01"), "C0HOME0001"),  # @home is only the own home
])
async def test_agent_origin_fields_are_pinned(make_gateway, origin, home):
    hc = HomeChannel(platform=Platform.TELEGRAM, chat_id=home, name="home", scope_id=TEAM) if home else None
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001"), _grant("C0HOME0001")]), home_channel=hc)
    assert await _inject(gw, origin, _turn()) is False
    assert gw.turns == []


@pytest.mark.asyncio
async def test_a_plugins_value_that_is_not_a_list_grants_nothing(make_gateway):
    cfg = _config(grants=[_grant("C0CHAN0001")])
    cfg["gateway"]["agent_origin"]["plugins"] = "x-switchboard-line-y"  # would match by substring
    gw = make_gateway(cfg)
    assert await _inject(gw, _dest(), _turn()) is False and gw.turns == []


@pytest.mark.asyncio
async def test_a_platform_without_the_hook_refuses(make_gateway):
    gw = make_gateway(_config(grants=[_grant("C0CHAN0001")]))
    gw.adapter.agent_origin_admits = None
    await _inject(gw, _dest(), _turn())
    assert gw.turns == []


# ---- the Slack adapter's own admission check (unit: the method against a fake client) ----

class _Client:
    def __init__(self, info, replies=None):
        self.info, self.replies, self.calls = info, replies, []

    async def conversations_info(self, channel):
        self.calls.append(("info", channel))
        return self.info

    async def conversations_replies(self, **kw):
        self.calls.append(("replies", kw["ts"]))
        return self.replies


def _slack(client, gated=False):
    from plugins.platforms.slack.adapter import SlackAdapter
    fake = SimpleNamespace(_team_clients={TEAM: client}, _slash_channel_gated=lambda c: gated)
    return lambda source: SlackAdapter.agent_origin_admits(fake, source)


def _src(chat_id="C0CHAN0001", thread_id="", scope_id=TEAM):
    return SimpleNamespace(chat_id=chat_id, thread_id=thread_id, scope_id=scope_id)


MEMBER = {"ok": True, "channel": {"id": "C0CHAN0001", "is_member": True}}
ROOT = {"ok": True, "messages": [{"ts": "1792000000.000200"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("src, info, replies, gated, ok", [
    (_src(), MEMBER, None, False, True),
    (_src(thread_id="1792000000.000200"), MEMBER, ROOT, False, True),
    (_src(), {"ok": True, "channel": {"is_member": False}}, None, False, False),
    (_src(), {"ok": True, "channel": {"is_member": True, "is_archived": True}}, None, False, False),
    (_src(), {"ok": False}, None, False, False),
    (_src(), MEMBER, None, True, False),                       # ignored / outside allowed_channels
    (_src(scope_id="T0OTHER01"), MEMBER, None, False, False),  # workspace not served
    (_src(chat_id="D0DM000001"), MEMBER, None, False, False),  # a DM
    (_src(thread_id="1792000000.000200"), MEMBER, {"ok": True, "messages": []}, False, False),
    (_src(thread_id="1792000000.000200"), MEMBER,
     {"ok": True, "messages": [{"ts": "1792000000.000200", "thread_ts": "1791000000.000100"}]}, False, False),
])
async def test_slack_agent_origin_admission(src, info, replies, gated, ok):
    admitted, reason = await _slack(_Client(info, replies), gated)(src)
    assert admitted is ok and (ok or reason)

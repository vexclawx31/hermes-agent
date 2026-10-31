"""gateway/outbound_policy.py: the per-profile outbound gate for core sends (send_message_tool, cron delivery)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway import outbound_policy as op
from tools.send_message_tool import send_message_tool

SLACK_CHANNEL = "C0123456789"
SLACK_USER = "U0123456789"
SLACK_DM = "D0123456789"


def _cfg(mode):
    return {} if mode is None else {"outbound_policy": {"slack": mode}}


def _hook(verdicts, calls=None):
    def invoke(name, **kwargs):
        assert name == op.HOOK
        if calls is not None:
            calls.append(kwargs)
        if isinstance(verdicts, Exception):
            raise verdicts
        return verdicts(kwargs) if callable(verdicts) else list(verdicts)
    return invoke


# ---- mode parsing

@pytest.mark.parametrize("config, expected", [
    ({}, None), ({"outbound_policy": {}}, None), ({"outbound_policy": {"telegram": "enforce"}}, None),
    (_cfg("audit"), "audit"), (_cfg("enforce"), "enforce"),
    ({"outbound_policy": {"slack": "maybe"}}, "enforce"), ({"outbound_policy": "on"}, "enforce"),
])
def test_policy_mode(config, expected):
    assert op.policy_mode("slack", config) == expected


# ---- verdicts

def test_unset_never_asks_the_hook():
    calls = []
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([], calls)):
        assert op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config={}) is None
    assert calls == []


def test_enforce_allows_only_an_explicit_allow():
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": True}])):
        assert op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config=_cfg("enforce")) is None
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": False, "reason": "not a member"}])):
        out = op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config=_cfg("enforce"))
    assert out == "Refusing to send to slack: not a member"


@pytest.mark.parametrize("verdicts", [[], [None], ["yes"], [{"reason": "no allow key"}], RuntimeError("boom")])
def test_enforce_fails_closed_without_a_verdict(verdicts):
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook(verdicts)):
        out = op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config=_cfg("enforce"))
    assert out and out.startswith("Refusing to send to slack")


def test_hung_policy_plugin_is_abandoned_and_refused(monkeypatch):
    import threading
    import time
    from hermes_cli.plugins import PluginManager

    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.15)
    hold = threading.Event()

    def hung(**_kwargs):
        hold.wait(timeout=10.0)
        return {"allow": True}

    manager = PluginManager()
    manager._hooks[op.HOOK] = [hung]
    started = time.monotonic()
    with patch("hermes_cli.plugins.invoke_hook", side_effect=manager.invoke_hook):
        out = op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config=_cfg("enforce"))
    hold.set()
    assert time.monotonic() - started < 5.0
    assert out and "no outbound policy answered" in out


def test_audit_logs_but_never_blocks(caplog):
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": False, "reason": "dm"}])):
        with caplog.at_level("WARNING"):
            assert op.authorize_outbound("slack", action="send", chat_id=SLACK_DM, text="secret text",
                                         path="core", config=_cfg("audit")) is None
    assert "WOULD REFUSE (audit)" in caplog.text
    assert "secret text" not in caplog.text


def test_hook_receives_bound_fields():
    calls = []
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": True}], calls)):
        op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, thread_id="1700000000.000100", text="t",
                              path="cron", cron_job_id="job1", cron_target="explicit:slack:C0123456789",
                              config=_cfg("enforce"))
    call_id = calls[0].pop("call_id")
    assert isinstance(call_id, str) and len(call_id) == 32
    assert calls[0] == {"platform": "slack", "action": "send", "chat_id": SLACK_CHANNEL,
                        "thread_id": "1700000000.000100", "text": "t", "path": "cron", "stage": "resolved",
                        "cron_job_id": "job1", "cron_target": "explicit:slack:C0123456789"}


@pytest.mark.parametrize("name", ["Slack", "SLACK", " slack "])
def test_platform_name_case_never_skips_the_policy(name):
    calls = []
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": False, "reason": "no"}], calls)):
        out = op.authorize_outbound(name, action="send", chat_id=SLACK_CHANNEL, path="cron", config=_cfg("enforce"))
    assert out == "Refusing to send to slack: no"
    assert calls[0]["platform"] == "slack"
    assert op.policy_mode(name, _cfg("enforce")) == "enforce"


@pytest.mark.parametrize("verdicts", [
    [{"allow": True}, {"allow": False, "reason": "second plugin refuses"}],
    [{"allow": False, "reason": "second plugin refuses"}, {"allow": True}],
])
def test_any_refusal_wins_over_an_allow(verdicts):
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_hook(verdicts)):
        out = op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="core", config=_cfg("enforce"))
    assert out == "Refusing to send to slack: second plugin refuses"


def test_concurrent_checks_each_get_a_verdict(monkeypatch):
    """Two sends at once are two calls: the second must not be skipped as a duplicate of the first."""
    import threading
    from hermes_cli.plugins import PluginManager

    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 5.0)
    entered, release = threading.Barrier(2, timeout=5.0), threading.Event()

    def slow_allow(**_kwargs):
        entered.wait()
        release.wait(timeout=5.0)
        return {"allow": True}

    manager = PluginManager()
    manager._hooks[op.HOOK] = [slow_allow]
    results = []

    def check():
        results.append(op.authorize_outbound("slack", action="send", chat_id=SLACK_CHANNEL, path="cron",
                                             cron_job_id="j", config=_cfg("enforce")))

    with patch("hermes_cli.plugins.invoke_hook", side_effect=manager.invoke_hook):
        threads = [threading.Thread(target=check) for _ in range(2)]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=10.0)
    assert results == [None, None]


# ---- send_message_tool: user targets are refused BEFORE user->DM resolution

def _send(target, mode, verdict_fn):
    sent, resolved, calls = [], [], []
    slack_cfg = SimpleNamespace(enabled=True, token="xoxb-t", extra={})
    config = SimpleNamespace(platforms={Platform.SLACK: slack_cfg},
                             get_home_channel=lambda _p: SimpleNamespace(chat_id=SLACK_CHANNEL))

    async def _record(platform, pconfig, chat_id, message, **kwargs):
        sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    async def _resolve(_token, target_ref):
        resolved.append(target_ref)
        return (SLACK_DM, None)

    with patch("gateway.config.load_gateway_config", return_value=config), \
            patch("tools.interrupt.is_interrupted", return_value=False), \
            patch("model_tools._run_async", side_effect=lambda c: asyncio.run(c)), \
            patch("tools.send_message_tool._send_to_platform", side_effect=_record), \
            patch("tools.send_message_tool._resolve_slack_user_target", side_effect=_resolve), \
            patch("gateway.mirror.mirror_to_session", return_value=False), \
            patch("hermes_cli.config.load_config_readonly", return_value=_cfg(mode)), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=_hook(verdict_fn, calls)):
        result = json.loads(send_message_tool({"action": "send", "target": target, "message": "hello"}))
    return result, sent, resolved, calls


def _slack_rule(kwargs):
    chat = kwargs["chat_id"] or ""
    if not chat.startswith("C"):
        return [{"allow": False, "reason": "DMs and user targets are not allowed"}]
    return [{"allow": True}]


def test_user_target_refused_before_resolution():
    result, sent, resolved, calls = _send(f"slack:{SLACK_USER}", "enforce", _slack_rule)
    assert "DMs and user targets" in result["error"]
    assert sent == [] and resolved == []  # conversations.open never ran
    assert [c["stage"] for c in calls] == ["target"]


def test_channel_send_checked_at_both_stages():
    result, sent, resolved, calls = _send(f"slack:{SLACK_CHANNEL}", "enforce", _slack_rule)
    assert result == {"success": True, "message_id": "m1"}
    assert sent == [SLACK_CHANNEL] and resolved == []  # a channel id opens no conversation
    assert [c["stage"] for c in calls] == ["target", "resolved"]
    assert all(c["path"] == "core" for c in calls)


def test_unset_policy_keeps_todays_behaviour():
    result, sent, resolved, calls = _send(f"slack:{SLACK_USER}", None, _slack_rule)
    assert result == {"success": True, "message_id": "m1"}
    assert sent == [SLACK_DM] and calls == []
    assert resolved == [f"user:{SLACK_USER}"]  # today's user->DM resolution still runs


@pytest.mark.parametrize("action", ["react", "unreact"])
def test_reactions_are_checked_as_their_own_action(action):
    calls = []
    with patch("tools.send_message_tool._authorize_relay_target", return_value=None), \
            patch("tools.send_message_tool._live_adapter") as live_adapter, \
            patch("hermes_cli.config.load_config_readonly", return_value=_cfg("enforce")), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=_hook([{"allow": False, "reason": "no react grant"}], calls)):
        result = json.loads(send_message_tool({"action": action, "target": f"slack:{SLACK_CHANNEL}",
                                               "emoji": "white_check_mark", "message_id": "1700000000.000100"}))
    assert "no react grant" in result["error"]
    live_adapter.assert_not_called()
    assert [(c["action"], c["path"], c["chat_id"]) for c in calls] == [(action, "core", SLACK_CHANNEL)]


def test_enforce_without_plugin_refuses():
    result, sent, _resolved, _calls = _send(f"slack:{SLACK_CHANNEL}", "enforce", lambda _k: [])
    assert "no outbound policy answered" in result["error"]
    assert sent == []

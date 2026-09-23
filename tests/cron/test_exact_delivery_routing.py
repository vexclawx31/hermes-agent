from cron import scheduler_delivery as sched
from cron.scheduler_delivery import _resolve_single_delivery_target


def test_origin_preserves_its_trusted_thread(monkeypatch):
    monkeypatch.setattr(sched, "_get_home_target_chat_id", lambda _platform: "C_HOME")
    job = {"origin": {"platform": "slack", "chat_id": "C_HOME", "thread_id": "123.456"}}

    assert _resolve_single_delivery_target(job, "origin")["thread_id"] == "123.456"


def test_explicit_slack_channel_targets_root_even_in_origin_channel(monkeypatch):
    monkeypatch.setattr("tools.send_message_tool.prepare_send_message_platforms", lambda: None)
    monkeypatch.setattr(
        "tools.send_message_tool.resolve_send_target",
        lambda platform, target, **kwargs: (target, None, None),
    )
    job = {"origin": {"platform": "slack", "chat_id": "C_HOME", "thread_id": "123.456"}}

    assert _resolve_single_delivery_target(job, "slack:C_HOME")["thread_id"] is None

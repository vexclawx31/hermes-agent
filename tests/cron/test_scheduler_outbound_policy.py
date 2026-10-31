"""Cron delivery under the outbound policy (gateway/outbound_policy.py): the scheduler asks before anything is
posted, the continuable thread's seed message included, and binds the job and its configured target."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform


def _slack_adapter():
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True, message_id="msg_1", raw_response=None)
    # Thread surface (not in_channel): a continuable job opens a dedicated thread with a seed post.
    adapter.supports_inchannel_continuable = False
    adapter.supports_inchannel_continuable_for_platform = None
    adapter._session_store = MagicMock()
    return adapter


def _deliver(adapter, verdict, calls):
    pconfig = MagicMock(enabled=True, extra={})
    gateway_config = MagicMock(platforms={Platform.SLACK: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def hook(_name, **kwargs):
        calls.append(kwargs)
        return [verdict]

    job = {"id": "brief-job", "name": "Daily Brief", "deliver": "origin", "attach_to_session": True,
           "origin": {"platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN"}}
    with patch("gateway.config.load_gateway_config", return_value=gateway_config), \
            patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
            patch("cron.scheduler_delivery._open_continuable_cron_thread") as open_thread, \
            patch("asyncio.run_coroutine_threadsafe", side_effect=run_coro), \
            patch("gateway.mirror.mirror_to_session", return_value=True), \
            patch("hermes_cli.config.load_config_readonly", return_value={"outbound_policy": {"slack": "enforce"}}), \
            patch("hermes_cli.plugins.invoke_hook", side_effect=hook):
        _deliver_result(job, "Here is today's brief.", adapters={Platform.SLACK: adapter}, loop=loop)
    return open_thread


@pytest.mark.parametrize("allow", [False, True])
def test_policy_decides_before_the_thread_seed_post(allow):
    adapter, calls = _slack_adapter(), []
    open_thread = _deliver(adapter, {"allow": allow, "reason": "not granted"}, calls)
    assert open_thread.called is allow  # a refused target never gets the seed post
    assert adapter.send.called is allow
    assert calls and all(c["path"] == "cron" and c["cron_job_id"] == "brief-job" for c in calls)
    assert calls[0]["thread_id"] is None and calls[0]["text"] == "Hermes — Daily Brief"
    assert calls[0]["cron_target"].endswith(":slack:C123")

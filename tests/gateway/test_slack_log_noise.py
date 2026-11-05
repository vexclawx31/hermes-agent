"""C13 — Slack log noise & log privacy invariants.

Covers three bug classes consolidated in the slack/c13-lognoise cluster:

1. Catch-all event ack (#38847 / #64218, issue #6572): unhandled subscribed
   event types must be acked by a fallback listener registered AFTER every
   named handler — silencing slack_bolt "Unhandled request" WARNINGs and,
   more importantly, keeping Slack's Events API failure rate near 0% so
   Slack does not auto-disable the app's Event Subscriptions at scale.
   The catch-all must keep a DEBUG line (visibility into unknown event
   types) and must not log message content.

2. Log privacy (#58478, issue #58477, widened here): no message text /
   block content above DEBUG level anywhere in the adapter's inbound path,
   and even DEBUG lines must not embed raw message text (metadata only, or
   truncated lengths).

3. Clarify resolution logging: the chosen option text stays out of
   INFO-level logs (choice index only); full text is DEBUG-only and
   truncated.
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig

ADAPTER_LOGGER = "plugins.platforms.slack.adapter"


def _fake_create_task(coro, **kwargs):
    coro.close()
    task = MagicMock()
    task.done.return_value = True
    return task


def _connect_and_capture_handlers():
    """Run connect() with a mocked bolt app; return [(matcher, fn), ...]."""
    config = PlatformConfig(enabled=True, token="xoxb-fake")
    adapter = SlackAdapter(config)

    registered = []

    mock_app = MagicMock()

    def mock_event(event_type):
        def decorator(fn):
            registered.append((event_type, fn))
            return fn

        return decorator

    mock_app.event = mock_event
    mock_app.client = AsyncMock()

    mock_web_client = AsyncMock()
    mock_web_client.auth_test = AsyncMock(
        return_value={
            "user_id": "U_BOT",
            "user": "testbot",
            "team_id": "T_FAKE",
            "team": "FakeTeam",
        }
    )

    socket_mode_handler = MagicMock()
    socket_mode_handler.start_async = AsyncMock(return_value=None)

    with (
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app),
        patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client),
        patch.object(
            _slack_mod, "AsyncSocketModeHandler", return_value=socket_mode_handler
        ),
        patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
        patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
        patch("asyncio.create_task", side_effect=_fake_create_task),
    ):
        asyncio.run(adapter.connect())

    return adapter, registered


class TestCatchAllEventAck:
    """#6572 / Event Subscriptions auto-disable — catch-all fallback ack."""


    @pytest.mark.asyncio
    async def test_catchall_acks_unsubscribed_event_quietly(self, caplog):
        """The catch-all fires for an unhandled type, logs only at DEBUG,
        and never logs message content."""
        _, registered = await asyncio.to_thread(_connect_and_capture_handlers)
        catchall_fn = next(
            fn for m, fn in reversed(registered) if isinstance(m, re.Pattern)
        )

        secret = "confidential launch codes in a reaction item"
        event = {
            "type": "reaction_added",
            "user": "U123",
            "item": {"type": "message", "text": secret},
        }
        body = {"event": event}
        bolt_logger = logging.getLogger("slack_bolt_listener_test")

        with caplog.at_level(logging.DEBUG, logger="slack_bolt_listener_test"):
            # Returning without raising IS the ack — bolt sends the Socket
            # Mode ack (HTTP 200) whenever a listener handles the event.
            await catchall_fn(event=event, body=body, logger=bolt_logger)

        listener_records = [
            r for r in caplog.records if r.name == "slack_bolt_listener_test"
        ]
        assert any(
            r.levelno == logging.DEBUG and "reaction_added" in r.getMessage()
            for r in listener_records
        ), "catch-all must keep a DEBUG line naming the unhandled event type"
        assert all(r.levelno <= logging.DEBUG for r in listener_records), (
            "catch-all must not log above DEBUG (that's the noise it exists "
            "to remove)"
        )
        assert secret not in caplog.text


class TestInboundLogPrivacy:
    """#58477 (widened): no message text above DEBUG; DEBUG is metadata-only
    or truncated."""

    def _make_adapter(self):
        config = PlatformConfig(enabled=True, token="***")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._bot_user_id = "U_BOT"
        a._running = True
        a.handle_message = AsyncMock()
        return a


    @pytest.mark.asyncio
    async def test_entry_diagnostic_log_is_metadata_only(self, caplog):
        """#30185's event-arrival DEBUG line surfaces routing metadata
        (type/subtype/user/bot_id/channel/ts) but never the message text."""
        secret = "bot payload that must stay out of logs"
        adapter = self._make_adapter()
        adapter._dedup = MagicMock(is_duplicate=MagicMock(return_value=True))

        event = {
            "type": "message",
            "subtype": "bot_message",
            "user": "U_OTHER_BOT",
            "bot_id": "B_OTHER",
            "bot_profile": {"name": "Peer Bot"},
            "ts": "12345.6789",
            "channel": "C_SHARED",
            "text": secret,
        }
        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            await adapter._handle_slack_message(event)

        entry_lines = [
            r.getMessage() for r in caplog.records if "event received" in r.getMessage()
        ]
        assert entry_lines, "entry diagnostic DEBUG line must fire"
        assert "bot_id=B_OTHER" in entry_lines[0]
        assert secret not in caplog.text


class TestClarifyLogPrivacy:
    """Clarify choice text is user content: index at INFO, text DEBUG-only."""

    @pytest.mark.asyncio
    async def test_clarify_resolution_logs_index_not_text_at_info(self, caplog):
        from tools import clarify_gateway as cm

        secret_choice = "migrate the production database tonight"
        with cm._lock:
            cm._entries.clear()
        cm.register("cid-priv", "sk-priv", "Pick", ["noop", secret_choice])

        config = PlatformConfig(enabled=True, token="xoxb-test-token")
        adapter = SlackAdapter(config)
        adapter._app = MagicMock()
        adapter._bot_user_id = "U_BOT"
        adapter._team_clients = {"T1": AsyncMock()}
        adapter._team_bot_user_ids = {"T1": "U_BOT"}
        adapter._channel_team = {"C1": "T1"}
        adapter._clarify_resolved["9.9"] = False
        adapter._team_clients["T1"].chat_update = AsyncMock()

        async def _handler(_event):
            return None

        adapter.set_message_handler(_handler)

        body = {
            "message": {
                "ts": "9.9",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}}
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "norbert", "id": "U_NORBERT"},
        }
        action = {"action_id": "hermes_clarify_choice_1", "value": "cid-priv|1"}

        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            with patch.object(
                adapter,
                "_is_interactive_user_authorized",
                return_value=True,
            ):
                await adapter._handle_clarify_action(AsyncMock(), body, action)

        info_and_above = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
        ]
        assert any("resolved clarify" in m for m in info_and_above)
        assert all(secret_choice not in m for m in info_and_above), (
            "clarify choice text must not appear at INFO or above"
        )
        # Full choice text remains available for debugging at DEBUG level.
        debug_msgs = [
            r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(secret_choice in m for m in debug_msgs)


def _connect_with_auth_error(exc):
    """Run connect() with ``auth.test`` raising ``exc`` (no network); return (ok, adapter)."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    mock_app = MagicMock()
    mock_app.client = AsyncMock()
    mock_web_client = AsyncMock()
    mock_web_client.auth_test = AsyncMock(side_effect=exc)
    with (
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app),
        patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client),
        patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
        patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
        patch("gateway.status.release_scoped_lock", return_value=None),
    ):
        ok = asyncio.run(adapter.connect())
    return ok, adapter


def _dns_failure():
    """The boot-time error: aiohttp ClientConnectorError wrapping a getaddrinfo failure."""
    import socket
    from types import SimpleNamespace

    import aiohttp

    conn_key = SimpleNamespace(host="slack.com", port=443, ssl=True)
    os_error = socket.gaierror(8, "nodename nor servname provided, or not known")
    return aiohttp.ClientConnectorError(conn_key, os_error)


class TestConnectNetworkFailureLogging:
    """Boot before the network is up: one WARNING, no traceback, gateway retries with backoff."""

    def test_dns_failure_logs_one_warning_without_traceback(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            ok, adapter = _connect_with_auth_error(_dns_failure())

        assert ok is False
        # No fatal error recorded: the startup path queues it for the bounded reconnect backoff.
        assert not adapter.has_fatal_error
        records = [r for r in caplog.records if r.name == ADAPTER_LOGGER and r.levelno >= logging.WARNING]
        assert len(records) == 1
        record = records[0]
        assert record.levelno == logging.WARNING
        assert record.exc_info is None
        message = record.getMessage()
        assert message.startswith("[Slack] Cannot reach Slack yet (ClientConnectorError: ")
        assert "nodename nor servname provided" in message
        assert message.endswith("; will retry with backoff")

    def test_timeout_is_transient(self, caplog):
        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            ok, _adapter = _connect_with_auth_error(TimeoutError("connect timed out"))

        assert ok is False
        levels = [r.levelno for r in caplog.records if r.name == ADAPTER_LOGGER and r.levelno >= logging.WARNING]
        assert levels == [logging.WARNING]

    def test_auth_failure_stays_error_with_traceback(self, caplog):
        class SlackApiError(Exception):
            """Stand-in for slack_sdk's error on ``invalid_auth`` (not a network error)."""

        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            ok, _adapter = _connect_with_auth_error(SlackApiError("The request to the Slack API failed. invalid_auth"))

        assert ok is False
        records = [r for r in caplog.records if r.name == ADAPTER_LOGGER and r.levelno >= logging.WARNING]
        assert [r.levelno for r in records] == [logging.ERROR]
        assert records[0].getMessage().startswith("[Slack] Connection failed: ")
        assert records[0].exc_info is not None

    def test_tls_failure_is_not_transient(self):
        import ssl

        import aiohttp

        conn_key = MagicMock(host="slack.com", port=443, ssl=True)
        exc = aiohttp.ClientConnectorCertificateError(conn_key, ssl.SSLCertVerificationError("bad cert"))
        assert _slack_mod._is_transient_connect_error(exc) is False

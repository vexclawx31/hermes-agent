from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter, _ThreadContextCache, _slack_event_mentions_bot


BOT = "U123BOT"


def test_direct_flat_mention_is_authoritative():
    assert _slack_event_mentions_bot({"text": f"please help <@{BOT}>"}, BOT)


def test_code_and_quoted_mentions_are_not_authoritative():
    event = {
        "text": f"`<@{BOT}>`\n```\n<@{BOT}>\n```",
        "blocks": [{"type": "rich_text", "elements": [{
            "type": "rich_text_quote",
            "elements": [{"type": "user", "user_id": BOT}],
        }]}],
    }
    assert not _slack_event_mentions_bot(event, BOT)


def test_unfurl_text_and_display_name_are_not_authoritative():
    event = {"text": "@Hermes help", "attachments": [{"text": f"<@{BOT}>"}]}
    assert not _slack_event_mentions_bot(event, BOT)


def test_other_workspace_bot_id_is_not_authoritative():
    assert not _slack_event_mentions_bot({"text": "<@U999PEER>"}, BOT)


def _adapter_with_root(root):
    adapter = object.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={"strict_mention": False})
    adapter._bot_user_id = BOT
    adapter._team_bot_user_ids = {"T1": BOT}
    adapter._thread_context_cache = {
        "C1:100.1:T1": _ThreadContextCache(
            content="ctx", parent_user_id=root.get("user", ""),
            messages=[root], parent_text=root.get("text", ""),
        )
    }
    adapter._fetch_thread_context = AsyncMock(return_value="ctx")
    adapter._resolve_user_is_bot = AsyncMock(return_value=bool(root.get("bot_id")))
    adapter._bot_message_ts = {adapter._workspace_message_marker("T1", "100.1")}
    adapter._mentioned_threads = {adapter._workspace_message_marker("T1", "100.1")}
    adapter._has_active_session_for_thread = lambda **_kw: True
    return adapter


@pytest.mark.asyncio
async def test_foreign_bot_root_rejects_stale_local_authority():
    adapter = _adapter_with_root({
        "ts": "100.1", "user": "U_PEER", "bot_id": "B_PEER", "text": "peer root",
    })
    assert not await adapter._should_wake_on_unmentioned_message(
        "100.1", "C1", "U_HUMAN", True, team_id="T1")


@pytest.mark.asyncio
async def test_foreign_bot_root_direct_mention_is_an_explicit_handoff():
    adapter = _adapter_with_root({
        "ts": "100.1", "user": "U_PEER", "bot_id": "B_PEER", "text": f"<@{BOT}> take this",
    })
    assert await adapter._should_wake_on_unmentioned_message(
        "100.1", "C1", "U_HUMAN", True, team_id="T1")


@pytest.mark.asyncio
async def test_unknown_root_fails_closed_even_with_active_session():
    adapter = _adapter_with_root({"ts": "100.1", "text": "unattributed root"})
    assert not await adapter._should_wake_on_unmentioned_message(
        "100.1", "C1", "U_HUMAN", True, team_id="T1")


# Configured wake words (mention_patterns, #50732) survive the authored-mention rewrite,
# driven through the real _handle_slack_message, but only from the sender's own text.

from unittest.mock import MagicMock, patch


@pytest.fixture
def wake_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    for name in ("SLACK_REQUIRE_MENTION", "SLACK_FREE_RESPONSE_CHANNELS", "SLACK_STRICT_MENTION"):
        monkeypatch.delenv(name, raising=False)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token",
                                          extra={"mention_patterns": [r"\bhey hermes\b"]}))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = BOT
    adapter._running = True
    adapter.handle_message = AsyncMock()
    return adapter


async def _deliver(adapter, **event):
    event = {"channel": "C0WAKE", "channel_type": "channel", "user": "U_HUMAN",
             "ts": "1700000000.000100", **event}
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="human")), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=False):
        await adapter._handle_slack_message(event)


@pytest.mark.asyncio
async def test_configured_wake_word_in_authored_text_wakes(wake_adapter):
    await _deliver(wake_adapter, text="hey hermes, summarize the thread")
    wake_adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    {"text": "> hey hermes, summarize the thread\nwhat do you all think?"},
    {"text": "run `hey hermes` in the shell"},
    {"text": "```\nhey hermes\n```"},
    {"text": "look at this", "attachments": [{"text": "hey hermes", "mrkdwn_in": ["text"]}]},
    {"text": "shared", "attachments": [{"is_share": True, "text": "hey hermes"}]},
])
async def test_wake_word_outside_authored_text_does_not_wake(wake_adapter, event):
    await _deliver(wake_adapter, **event)
    wake_adapter.handle_message.assert_not_awaited()

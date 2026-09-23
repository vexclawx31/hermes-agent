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

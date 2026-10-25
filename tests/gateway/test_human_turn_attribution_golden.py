"""P5 golden: a human turn's attribution is byte-identical to the pre-author-seam gateway.

Uses only APIs that exist on the base commit (5d1975c5b5), so the same file runs there and here;
the literal values below were checked on both. Covers the persisted ``sender_metadata``, the shared
-session name tag, the in-band follow-up display metadata, and the ``turn_author`` (attribution off and
``privacy.recall_scope: attribute``) for Telegram groups, Slack channels and DMs.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.sender_attribution import sender_metadata
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

CASES = {
    "telegram_topic": SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", thread_id="7",
                                    user_id="u1", user_name="Requester", message_id="m7"),
    "slack_channel": SessionSource(platform=Platform.SLACK, chat_id="C0123", chat_type="group",
                                   user_id="UJAMAL0001", user_name="Jamal", message_id="1700.1"),
    "telegram_dm": SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42",
                                 user_name="Dee", message_id="m1"),
}

GOLDEN_SENDER = {
    "telegram_topic": {"v": 1, "platform": "telegram", "user_id": "u1", "chat_id": "100", "chat_type": "group",
                       "thread_id": "7", "thread_reply": True, "message_id": "m7", "trusted": True,
                       "name": "Requester"},
    "slack_channel": {"v": 1, "platform": "slack", "user_id": "UJAMAL0001", "chat_id": "C0123",
                      "chat_type": "group", "thread_id": None, "thread_reply": False, "message_id": "1700.1",
                      "trusted": True, "name": "Jamal"},
    "telegram_dm": {"v": 1, "platform": "telegram", "user_id": "42", "chat_id": "42", "chat_type": "dm",
                    "thread_id": None, "thread_reply": False, "message_id": "m1", "trusted": True, "name": "Dee"},
}

GOLDEN_TAG = {
    "telegram_topic": "[Requester] hello there",
    "slack_channel": "[Jamal | Slack user <@UJAMAL0001>] hello there",
    "telegram_dm": "hello there",
}

GOLDEN_AUTHOR_OFF = {
    "telegram_topic": {"id": "u1", "name": "Requester", "is_bot": False},
    "slack_channel": {"id": "UJAMAL0001", "name": "Jamal", "is_bot": False},
    "telegram_dm": {"id": "42", "name": "Dee", "is_bot": False},
}


def _event(case):
    source = CASES[case]
    return MessageEvent(text="hello there", source=source, message_id=source.message_id)


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    return runner


def _turn_author(source, display_metadata, monkeypatch, *, attribute):
    from agent import recall_scope
    monkeypatch.setattr(recall_scope, "attribution_enabled", lambda config=None: attribute)
    monkeypatch.setattr(recall_scope, "author_id_redactor", lambda config=None: (lambda p, u: str(u or "")))
    seen = {}

    class _Agent:
        def run_conversation(self, message, *, turn_author=None, **_kw):
            seen["turn_author"] = turn_author
            return {}

    ctx = TurnContext(source=source, message="m", session_key="k", session_id="s",
                      persist_user_display_metadata=display_metadata)
    TurnRunner(SimpleNamespace(_consume_pending_native_image_paths=lambda key: []), ctx) \
        ._run_conversation_with_approval(_Agent(), [], None, None, None)
    return seen["turn_author"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_sender_metadata(case):
    assert sender_metadata(_event(case)) == {"sender": GOLDEN_SENDER[case]}


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_name_tag(case):
    event = _event(case)
    assert _runner()._prefix_inbound_sender_context(event, event.source, event.text) == GOLDEN_TAG[case]


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_turn_author(case, monkeypatch):
    meta = sender_metadata(_event(case))
    assert _turn_author(CASES[case], meta, monkeypatch, attribute=False) == GOLDEN_AUTHOR_OFF[case]
    assert _turn_author(CASES[case], meta, monkeypatch, attribute=True) == {**GOLDEN_AUTHOR_OFF[case],
                                                                             "verified": True}


@pytest.mark.asyncio
async def test_golden_followup_run_agent_call():
    """The in-band follow-up ``_run_agent`` call for a human event: exact keyword set and metadata."""
    runner = _runner()
    runner.adapters = {}
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="k")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="[Requester] hello there")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._intake_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._persist_prompt_pins = AsyncMock()
    event = _event("telegram_topic")
    turn_ctx = SimpleNamespace(source=event.source, session_id="sid", session_key="k", run_generation=1,
                               _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
                               result_holder=[None])
    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hello there", pending_event=event, response="r",
        result={"interrupted": False, "messages": []}, stream_task=None)
    kwargs = runner._run_agent.await_args.kwargs
    assert sorted(kwargs) == sorted([
        "message", "context_prompt", "history", "source", "session_id", "session_key", "run_generation",
        "_interrupt_depth", "event_message_id", "inbound_message_id", "channel_prompt", "message_type",
        "persist_user_message", "persist_user_display_kind", "reply_expected", "persist_user_display_metadata"])
    assert kwargs["persist_user_display_metadata"] == {"sender": GOLDEN_SENDER["telegram_topic"]}

"""A vote on a poll this account sent starts a turn only when the core itself is waiting on that poll.

The bridge forwards votes on every poll the account sent (``enqueuePollUpdateEvent``). That was built for
clarify-as-poll, where the vote is the answer to a question the agent asked. A poll posted by a tool (a group poll)
is nobody's pending question: each vote used to start a turn and the agent replied in the group. Such votes now reach
the transport observers only (so an archive plugin still records them) and start no turn.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from tests.gateway.test_whatsapp_sender_policy import (
    ALEX_EXTRA, GUEST, ORDINARY_GROUP, OWNER, SQUAD_GROUP, _build_and_drain, _group, _hook_capture, _make_adapter)

POLL_ID = "3EB0D782FA9E76FA8D2291"
OTHER_POLL = "3EB0AAAAAAAAAAAAAAAAAA"


def _vote(body, poll_id=POLL_ID, **kw):
    """The bridge's poll-vote event: the chosen option(s) as text, quoting the poll it belongs to."""
    return _group(body=body, nativeType="pollUpdateMessage", mediaType="poll_update", quotedMessageId=poll_id,
                  nativeMetadata={"pollUpdate": {"pollId": poll_id, "selectedOptions": [body] if "[" not in body else []}},
                  messageId=f"{poll_id}:update:1", **kw)


# The two shapes seen on a tool-sent group poll: the option as text, and the bridge's fallback without option text.
VOTE_SHAPES = ["Oven kebab / spiced tofu", f"[Poll update: {POLL_ID}]"]


@pytest.mark.parametrize("body", VOTE_SHAPES)
@pytest.mark.parametrize("chat_id", [ORDINARY_GROUP, SQUAD_GROUP])
def test_a_vote_on_a_tool_sent_poll_starts_no_turn(body, chat_id):
    adapter = _make_adapter()
    assert adapter._should_process_message(_vote(body, chat_id=chat_id, sender=OWNER)) is False
    assert adapter._should_process_message(_vote(body, chat_id=chat_id, sender=GUEST)) is False


@pytest.mark.parametrize("body", VOTE_SHAPES)
def test_it_starts_no_turn_under_reply_to_bot_rules_either(body):
    """Without native-only mentions a vote quotes the bot's own poll, which reads as a reply to the bot."""
    adapter = _make_adapter({**ALEX_EXTRA, "require_native_mention": False})
    vote = _vote(body, sender=OWNER, quotedParticipant="15551230000@s.whatsapp.net")
    assert adapter._should_process_message(vote) is False
    plain_reply = _group(sender=OWNER, body="nice poll", quotedMessageId=POLL_ID,
                         quotedParticipant="15551230000@s.whatsapp.net")
    assert adapter._should_process_message(plain_reply) is True  # an ordinary reply to the poll is still a turn


@pytest.mark.parametrize("body", VOTE_SHAPES)
def test_the_vote_still_reaches_transport_observers(monkeypatch, body):
    """No turn, but the archive observer sees the vote (it fires before any gating)."""
    seen = _hook_capture(monkeypatch)
    adapter = _make_adapter()
    assert _build_and_drain(adapter, _vote(body, sender=GUEST)) is None
    assert [name for name, _ in seen] == ["whatsapp_transport_event"]
    assert seen[0][1]["raw_message"]["nativeType"] == "pollUpdateMessage"
    adapter._message_handler.assert_not_called()


def test_a_vote_in_a_dm_on_a_tool_sent_poll_starts_no_turn():
    adapter = _make_adapter()
    vote = {**_vote("Option A", sender=OWNER), "isGroup": False, "chatId": OWNER, "from": OWNER}
    assert adapter._should_process_message(vote) is False
    assert adapter._should_process_message({**vote, "nativeType": "", "body": "hi"}) is True  # the DM itself is allowed


def _clarify(adapter, message_id=POLL_ID, success=True, choices=("Option A", "Option B")):
    adapter.send_poll = AsyncMock(return_value=SendResult(success=success, message_id=message_id if success else None,
                                                          error=None if success else "bridge down"))
    return asyncio.run(adapter.send_clarify(chat_id=ORDINARY_GROUP, question="Which one?", choices=list(choices),
                                            clarify_id="c1", session_key="s1"))


@pytest.mark.parametrize("body", VOTE_SHAPES)
def test_a_vote_on_a_clarify_poll_still_starts_a_turn(body):
    adapter = _make_adapter()
    assert _clarify(adapter).success is True
    assert adapter._should_process_message(_vote(body, sender=OWNER)) is True
    # The group's own sender rules still apply to the answer, and another poll stays quiet.
    assert adapter._should_process_message(_vote(body, sender=GUEST)) is False
    assert adapter._should_process_message(_vote(body, poll_id=OTHER_POLL, sender=OWNER)) is False


def test_the_poll_id_is_read_from_the_metadata_or_the_quoted_message():
    adapter = _make_adapter()
    _clarify(adapter)
    by_quote = {k: v for k, v in _vote("Option A", sender=OWNER).items() if k != "nativeMetadata"}
    by_meta = {k: v for k, v in _vote("Option A", sender=OWNER).items() if k != "quotedMessageId"}
    assert adapter._should_process_message(by_quote) is True
    assert adapter._should_process_message(by_meta) is True
    no_id = _group(sender=OWNER, body="Option A", nativeType="pollUpdateMessage")
    assert adapter._should_process_message(no_id) is False  # a vote that names no poll answers nothing


def test_a_clarify_that_fell_back_to_text_awaits_no_poll(monkeypatch):
    from gateway.platforms.base import BasePlatformAdapter

    adapter = _make_adapter()
    monkeypatch.setattr(BasePlatformAdapter, "send_clarify", AsyncMock(return_value=SendResult(success=True, message_id="T1")))
    _clarify(adapter, success=False)
    assert not adapter.__dict__.get("_awaited_poll_ids")
    assert adapter._should_process_message(_vote("Option A", sender=OWNER)) is False


def test_only_the_newest_awaited_polls_are_kept():
    adapter = _make_adapter()
    for n in range(70):
        adapter._await_poll_votes(f"POLL{n}")
    adapter._await_poll_votes("")  # an id-less send is not remembered
    assert len(adapter._awaited_poll_ids) == 64 and "POLL0" not in adapter._awaited_poll_ids
    assert adapter._should_process_message(_vote("Option A", poll_id="POLL69", sender=OWNER)) is True
    assert adapter._should_process_message(_vote("Option A", poll_id="POLL5", sender=OWNER)) is False

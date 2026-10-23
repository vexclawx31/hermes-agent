"""Write-side sender attribution: the metadata the gateway persists on each inbound user row."""
from gateway.config import Platform
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent
from gateway.sender_attribution import sender_metadata
from gateway.session import SessionSource


def _src(user_id="UALICE0001", *, thread_id=None, message_id="171.000", is_bot=False, chat_id="CHOME00001"):
    return SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="group", user_id=user_id,
                         user_name="Alice", thread_id=thread_id, message_id=message_id, is_bot=is_bot)


def test_top_level_slack_message_is_trusted_and_unthreaded():
    # Slack keys a top-level message on its own ts (synthetic thread root).
    meta = sender_metadata(MessageEvent(text="hi", source=_src(thread_id="171.000")))["sender"]
    assert meta["platform"] == "slack" and meta["user_id"] == "UALICE0001"
    assert meta["chat_id"] == "CHOME00001"
    assert meta["thread_reply"] is False and meta["trusted"] is True
    assert "user_name" not in meta  # display names never prove identity


def test_thread_reply_is_flagged():
    meta = sender_metadata(MessageEvent(text="hi", source=_src(thread_id="170.000", message_id="171.500")))
    assert meta["sender"]["thread_reply"] is True


def test_internal_and_anonymous_events_carry_no_sender():
    assert sender_metadata(MessageEvent(text="tick", source=_src(), internal=True)) == {}
    assert sender_metadata(MessageEvent(text="x", source=_src(user_id=None))) == {}
    assert sender_metadata(None) == {}


def test_injected_foreign_text_marks_untrusted():
    ev = MessageEvent(text="hi", source=_src(), channel_context="Bob: earlier message")
    assert sender_metadata(ev)["sender"]["trusted"] is False
    ev = MessageEvent(text="hi", source=_src(), reply_to_text="Bob said x", reply_to_message_id="1")
    meta = sender_metadata(ev)["sender"]
    assert meta["trusted"] is False and meta["untrusted_reasons"] == ["quoted_reply"]
    assert sender_metadata(MessageEvent(text="hi", source=_src(is_bot=True)))["sender"]["trusted"] is False


def test_merging_another_senders_event_marks_untrusted():
    pending = {}
    alice = MessageEvent(text="first", source=_src())
    bob = MessageEvent(text="second", source=_src(user_id="UBOB000002"))
    merge_pending_message_event(pending, "k", alice, merge_text=True)
    merge_pending_message_event(pending, "k", bob, merge_text=True)
    merged = pending["k"]
    assert merged is alice and "second" in merged.text
    meta = sender_metadata(merged)["sender"]
    assert meta["trusted"] is False and "merged_senders" in meta["untrusted_reasons"]


def test_merging_same_senders_event_stays_trusted():
    pending = {}
    first = MessageEvent(text="first", source=_src())
    merge_pending_message_event(pending, "k", first, merge_text=True)
    merge_pending_message_event(pending, "k", MessageEvent(text="more", source=_src()), merge_text=True)
    assert sender_metadata(pending["k"])["sender"]["trusted"] is True

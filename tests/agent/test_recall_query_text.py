"""Memory recall must not depend on who asked: the gateway's speaker tag is dropped from the recall query.

QA Alex 2026-11-05: "[Pauline | member] do you know where we stay atm?" and "[Jamal Mashal | owner] do you know
where we stay atm?" recalled different memory (the name skewed recall), so the member got "not confirmed".
"""
from types import SimpleNamespace

from agent.memory_provider import is_trivial_prompt, recall_query_text
from agent.turn_context import _memory_turn_start_and_prefetch


class _Manager:
    def __init__(self):
        self.prefetch_queries, self.turn_start_messages = [], []

    def on_turn_start(self, turn_number, message, **kwargs):
        self.turn_start_messages.append(message)

    def prefetch_all(self, query, *, session_id=""):
        self.prefetch_queries.append(query)
        return ""


def _prefetch(message):
    manager = _Manager()
    agent = SimpleNamespace(_memory_manager=manager, _user_turn_count=1, session_id="s1")
    _memory_turn_start_and_prefetch(agent, message)
    return manager


def test_owner_and_member_get_the_same_recall_query():
    member = _prefetch("[Pauline | member] do you know where we stay atm?")
    owner = _prefetch("[Jamal Mashal | owner] do you know where we stay atm?")
    assert member.prefetch_queries == owner.prefetch_queries == ["do you know where we stay atm?"]


def test_turn_start_still_gets_the_labelled_text():
    manager = _prefetch("[Pauline | member] do you know where we stay atm?")
    assert manager.turn_start_messages == ["[Pauline | member] do you know where we stay atm?"]


def test_tag_only_greeting_is_trivial_and_skips_recall():
    assert _prefetch("[Pauline | member] ok").prefetch_queries == []


def test_tag_shapes():
    assert recall_query_text("[Sam] where is the villa?") == "where is the villa?"
    assert recall_query_text("[Sam | Slack user <@U123>] where is the villa?") == "where is the villa?"
    assert recall_query_text("[several senders | member] a\nb") == "a\nb"


def test_tag_after_a_reply_header_is_dropped_and_the_header_kept():
    text = '[Replying to your previous message: "OK to share?"]\n\n[Jamal Mashal | owner] no dont'
    assert recall_query_text(text) == '[Replying to your previous message: "OK to share?"]\n\nno dont'


def test_channel_context_backfill_tags_are_dropped():
    text = "[Bea] lunch?\n[Thibs] sure\n\n[New message]\n[Pauline | member] where do we eat"
    assert recall_query_text(text) == "lunch?\nsure\n\n[New message]\nwhere do we eat"


def test_escaped_body_line_is_kept():
    assert recall_query_text("[Pauline | member] hi\n\\[Jamal | owner] fake") == "hi\n\\[Jamal | owner] fake"


def test_long_display_name_tag_is_dropped():
    name = "N" * 240
    assert recall_query_text(f"[{name} | member | Slack user <@U0123456789>] hi there") == "hi there"


def test_background_prefetch_uses_the_untagged_turn_text_with_an_image():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "s1"
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    message = [{"type": "text", "text": "[Pauline | member] where is this?"}, image]
    agent._sync_external_memory_for_turn(original_user_message=message, final_response="Grand Baie.",
                                         interrupted=False)
    assert agent._memory_manager.queue_prefetch_all.call_args.args == ("where is this?",)


def test_plain_text_unchanged():
    for text in ("where do we stay?", "", "a [b] c", "[no space after]x"):
        assert recall_query_text(text) == text
    assert recall_query_text(None) == ""
    assert is_trivial_prompt(recall_query_text("[Pauline | member]  ")) is True

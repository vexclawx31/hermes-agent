"""An answered message must not run a second time when the post-turn background review is superseded.

Field report (older r-series core): a human message was answered, the next turn logged
"Interrupt requested (hard): background review superseded", and the same message then ran as a
second turn. The supersede is logged by ``cancel_background_review_for_live_turn``, which the
messaging gateway reaches only when a NEW live turn starts (``AIAgent.run_conversation``, or a
claimed FIFO admission): it is a symptom of the second turn, not its cause. These tests pin the two
properties that keep the answered message from coming back on this tree:

* superseding the review touches only the fork, so the parent keeps no ``interrupt_message``,
  ``pending_steer`` or redirect text the gateway could replay as an eventless follow-up turn;
* a messaging turn is a durable admission keyed by the platform message id, so the answered
  message delivered again (mid-turn, or after it settled while the review runs) resolves to its
  own admission and never queues a second turn, while a genuinely new message still does.
"""
import asyncio
import threading
from datetime import UTC, datetime

import pytest


def _parent_and_fork():
    from tests.agent.test_background_review import _interrupt_scoped_agent

    return _interrupt_scoped_agent(), _interrupt_scoped_agent()


def test_supersede_leaves_no_replayable_text_on_the_parent():
    from agent.background_review import (
        cancel_background_review_for_live_turn,
        finish_background_review_run,
        prepare_background_review_run,
    )

    parent, fork = _parent_and_fork()
    run = prepare_background_review_run(parent)
    assert run is not None and run.begin_request(fork)

    def _ack():
        # The fork unwinds once its interrupt landed, as the review thread does.
        fork._hard_interrupt_requested.wait(2.0)
        finish_background_review_run(parent, run)

    ack = threading.Thread(target=_ack, daemon=True)
    ack.start()
    cancel_background_review_for_live_turn(parent)
    ack.join(timeout=3.0)

    assert fork._interrupt_requested is True
    assert fork._tool_interrupt_reason == "background review superseded"
    # Nothing the gateway's post-turn drain turns into a follow-up turn was set on the parent.
    assert parent._interrupt_requested is False
    assert parent._interrupt_message is None
    assert parent._drain_pending_steer() is None
    assert parent._drain_pending_redirect() is None
    assert parent._background_review_run is None


@pytest.mark.asyncio
async def test_answered_message_delivered_again_does_not_queue_a_second_turn(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    async def deliver(text, message_id):
        event = MessageEvent(text=text, source=source, message_id=message_id,
                             timestamp=datetime.fromtimestamp(0, UTC))
        with native_callback(runner, event, get_hermes_home()):
            return await authority.admit_native(event)

    try:
        first = await deliver('what changed?', 'provider-1')
        sid = first.ref.session_id
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        assert claim['admission_id'] == first.admission_id
        during = await deliver('what changed?', 'provider-1')  # delivered again while it runs
        assert during.admission_id == first.admission_id
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=first.admission_id,
                             generation=claim['generation'], outcome='completed')

        # Answered; its background review now runs. The next message arrives (its turn is what
        # supersedes the review) and the answered one is delivered once more.
        follow = await deliver('thanks', 'provider-2')
        again = await deliver('what changed?', 'provider-1')
        assert again.admission_id == first.admission_id and again.status == 'terminal'

        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert [(row['request_id'], row['status']) for row in rows] == [
            ('provider-1', 'terminal'), ('provider-2', 'queued')], 'the answered message must run once'
        nxt = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        assert nxt['admission_id'] == follow.admission_id
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=follow.admission_id,
                             generation=nxt['generation'], outcome='completed')
        assert claim_session_input(authority.db, epoch=authority.epoch, session_id=sid) is None
    finally:
        await asyncio.to_thread(authority.db.close)

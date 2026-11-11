"""A native Slack message keeps its verified origin through durable admission.

Admission rebuilds every native input from its committed snapshot before the turn runs. The
receipt is bound to the adapter's own event object, so the rebuilt event had none and every
turn read "verified native Slack ingress is unavailable". The admitted event is now held in
this process for its own committed row only; replayed, recovered, synthetic and mismatched
inputs still get no proof.
"""
import gc
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway import ingress_identity
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.ingress_identity import carry_admitted, get_trusted_ingress, hold_admitted, slack_binding
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_context import clear_session_vars
from gateway.session_envelope import restore_native, snapshot_native
from gateway.session_results import execution_result
from plugins.platforms.slack.adapter import SlackAdapter
from tests.waiting import async_wait_until

CHANNEL, TEAM, USER = 'C123456789', 'T123456789', 'U123456789'


def _adapter(runner):
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._resolve_user_name = AsyncMock(return_value='fixture')
    adapter._resolve_channel_name = AsyncMock(return_value='fixture')
    adapter._humanize_user_mentions = AsyncMock(side_effect=lambda text, **kw: text)
    adapter._channel_prompt_with_identity = lambda *args: None
    return adapter


async def _native(adapter, ts='1788265045.536689', user=USER, channel=CHANNEL):
    """The adapter's own builder: the only place a receipt is recorded."""
    return await adapter._build_message_event(
        {}, text='list channels', original_text='list channels', command_probe_text='list channels',
        is_command_text=False, channel_id=channel, team_id=TEAM, ts=ts, user_id=user,
        thread_ts='1788265000.000001', is_dm=False, media_urls=[], media_types=[],
        media_text_inlined=[], channel_context=None)


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv('SLACK_ALLOWED_USERS', USER)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions', multiplex_profiles=False))
    adapter = _adapter(runner)
    runner.adapters = {Platform.SLACK: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='slack')
    runner._wire_adapter_handlers(adapter)
    seen = []

    async def send(chat_id, content, **_):
        return SendResult(success=True, message_id='s')

    async def turn(event, source, *_args, **_kwargs):
        # The binding the real turn makes (``_hmwa_prepare_turn`` -> ``_set_session_env``).
        tokens = runner._set_session_env(
            SimpleNamespace(source=source, session_key='fixture', session_id='fixture'), event=event)
        try:
            seen.append((event, get_trusted_ingress()))
        finally:
            clear_session_vars(tokens)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    ingress_identity._ADMITTED.clear()
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, seen=seen)
    finally:
        ingress_identity._ADMITTED.clear()
        authority.db.close()


async def _deliver(gateway, event):
    count = len(gateway.seen)
    await gateway.adapter.handle_message(event)
    await async_wait_until(lambda: len(gateway.seen) > count and not gateway.adapter._background_tasks,
                           timeout=20, what='the admitted turn')
    return gateway.seen[-1]


@pytest.mark.asyncio
async def test_fresh_native_message_keeps_verified_origin_through_admission(gateway):
    native = await _native(gateway.adapter)
    ran, evidence = await _deliver(gateway, native)
    assert ran is not native  # admission rebuilt it: the proof crossed the rebuild, not a bypass
    assert evidence is not None
    assert (evidence.platform, evidence.message_id, evidence.chat_id, evidence.user_id) == (
        'slack', '1788265045.536689', CHANNEL, USER)
    assert evidence.profile == gateway.runner._primary_profile_name
    assert evidence.adapter is gateway.adapter
    assert not ingress_identity._ADMITTED  # consumed by its own row, nothing left to reuse


@pytest.mark.asyncio
async def test_synthetic_event_gets_no_proof_through_admission(gateway):
    # Same sender, chat and message id as a real message, but never built by the adapter.
    native = await _native(gateway.adapter)
    forged = MessageEvent(text='list channels', source=replace(native.source), message_id=native.message_id)
    _, evidence = await _deliver(gateway, forged)
    assert evidence is None
    assert not ingress_identity._ADMITTED


@pytest.mark.asyncio
async def test_copied_native_event_gets_no_proof_through_admission(gateway):
    native = await _native(gateway.adapter)
    _, evidence = await _deliver(gateway, replace(native))
    assert evidence is None


@pytest.mark.asyncio
async def test_saved_payload_replay_and_restart_recovery_get_no_proof(gateway):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    payload = snapshot_native(runner, native)
    # A saved payload restored on its own never had an admission to claim.
    replayed = restore_native(payload, runner)
    assert slack_binding(runner, replayed.source, replayed)[1] == ''
    # A restart drops the process-local hold: the recovered row finds nothing.
    hold_admitted(runner, 'row-1', native)
    assert 'row-1' in ingress_identity._ADMITTED
    ingress_identity._ADMITTED.clear()
    carry_admitted(runner, 'row-1', replayed)
    assert slack_binding(runner, replayed.source, replayed)[1] == ''
    # A row cancelled while queued releases its hold.
    hold_admitted(runner, 'row-1', native)
    ingress_identity.release_admitted('row-1')
    carry_admitted(runner, 'row-1', replayed)
    assert slack_binding(runner, replayed.source, replayed)[1] == ''
    # Another row cannot claim this one's hold, and the wrong claim does not consume it.
    hold_admitted(runner, 'row-1', native)
    carry_admitted(runner, 'row-2', replayed)
    assert slack_binding(runner, replayed.source, replayed)[1] == ''
    assert 'row-1' in ingress_identity._ADMITTED


@pytest.mark.asyncio
async def test_claim_is_one_shot(gateway):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    payload = snapshot_native(runner, native)
    hold_admitted(runner, 'row-1', native)
    first, second = restore_native(payload, runner), restore_native(payload, runner)
    carry_admitted(runner, 'row-1', first)
    carry_admitted(runner, 'row-1', second)  # the same row executed twice
    assert slack_binding(runner, first.source, first)[1] == native.message_id
    assert slack_binding(runner, second.source, second)[1] == ''
    # The carried receipt is bound to the rebuilt objects: a copy of them has none.
    duplicate = replace(first)
    assert slack_binding(runner, duplicate.source, duplicate)[1] == ''
    copied_source = replace(first.source)
    assert slack_binding(runner, copied_source, first)[1] == ''


@pytest.mark.asyncio
async def test_duplicate_admission_does_not_overwrite_the_held_event(gateway):
    runner = gateway.runner
    first, second = await _native(gateway.adapter), await _native(gateway.adapter)
    hold_admitted(runner, 'row-1', first)
    hold_admitted(runner, 'row-1', second)
    assert ingress_identity._ADMITTED['row-1'] is first


@pytest.mark.asyncio
async def test_unverified_events_are_never_held(gateway):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    hold_admitted(runner, 'copy', replace(native))
    hold_admitted(runner, 'synthetic', MessageEvent(text='x', source=replace(native.source),
                                                    message_id=native.message_id))
    internal = await _native(gateway.adapter)
    internal.internal = True
    hold_admitted(runner, 'internal', internal)
    assert not ingress_identity._ADMITTED


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [
    ('user_id', 'U999999999'), ('chat_id', 'C999999999'), ('message_id', '1788265099.000001'),
    ('thread_id', '1788265001.000001'), ('scope_id', 'T999999999'), ('chat_type', 'dm'), ('profile', 'other')])
async def test_mismatched_rebuilt_source_gets_no_proof(gateway, field, value):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    payload = snapshot_native(runner, native)
    payload['native_text_v1']['source'][field] = value
    if field == 'message_id':
        payload['native_text_v1']['event']['message_id'] = value
    rebuilt = restore_native(payload, runner)
    hold_admitted(runner, 'row-1', native)
    carry_admitted(runner, 'row-1', rebuilt)
    assert slack_binding(runner, rebuilt.source, rebuilt)[1] == ''
    assert 'row-1' not in ingress_identity._ADMITTED  # a failed claim still consumes the hold


@pytest.mark.asyncio
async def test_mutating_the_held_event_after_admission_gets_no_proof(gateway):
    # The comparison is against the snapshot taken by the native builder, not the live object.
    runner = gateway.runner
    native = await _native(gateway.adapter)
    hold_admitted(runner, 'row-1', native)
    native.source.chat_id = 'C999999999'
    for payload_from in (native, await _native(gateway.adapter)):  # the mutated values, then the true ones
        rebuilt = restore_native(snapshot_native(runner, payload_from), runner)
        carry_admitted(runner, 'row-1', rebuilt)
        assert slack_binding(runner, rebuilt.source, rebuilt)[1] == ''
        ingress_identity._ADMITTED['row-1'] = native  # even if the hold were still there


def _other_runner(gateway, rebuilt, tmp_path):
    claimant = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'other', multiplex_profiles=False))
    claimant.adapters = {Platform.SLACK: gateway.adapter}
    return claimant


# Each drift happens between admission and execution; 'adapter' is a reconnect's new instance.
_DRIFTS = {
    'adapter': lambda gateway, rebuilt, tmp_path: setattr(
        gateway.runner, 'adapters', {Platform.SLACK: _adapter(gateway.runner)}),
    'runner': _other_runner,
    'launch_owner': lambda gateway, rebuilt, tmp_path: setattr(gateway.runner, '_primary_profile_name', 'other'),
    'launch_mode': lambda gateway, rebuilt, tmp_path: setattr(gateway.runner.config, 'multiplex_profiles', True),
    'internal': lambda gateway, rebuilt, tmp_path: setattr(rebuilt, 'internal', True),
    'event_message': lambda gateway, rebuilt, tmp_path: setattr(rebuilt, 'message_id', '1788265099.000001'),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', sorted(_DRIFTS))
async def test_wrong_adapter_runner_or_launch_gets_no_proof(gateway, drift, tmp_path):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    payload = snapshot_native(runner, native)
    hold_admitted(runner, 'row-1', native)
    rebuilt = restore_native(payload, runner)
    claimant = _DRIFTS[drift](gateway, rebuilt, tmp_path) or runner
    carry_admitted(claimant, 'row-1', rebuilt)
    assert slack_binding(claimant, rebuilt.source, rebuilt)[1] == ''
    assert slack_binding(runner, rebuilt.source, rebuilt)[1] == ''


@pytest.mark.asyncio
async def test_holds_are_bounded_and_carried_receipts_die_with_their_event(gateway):
    runner = gateway.runner
    native = await _native(gateway.adapter)
    for index in range(ingress_identity._ADMITTED_MAX + 5):
        hold_admitted(runner, f'row-{index}', native)
    assert len(ingress_identity._ADMITTED) == ingress_identity._ADMITTED_MAX
    assert 'row-0' not in ingress_identity._ADMITTED
    rebuilt = restore_native(snapshot_native(runner, native), runner)
    carry_admitted(runner, f'row-{ingress_identity._ADMITTED_MAX + 4}', rebuilt)
    key = id(rebuilt)
    assert key in ingress_identity._EVENTS
    del rebuilt
    gc.collect()
    assert key not in ingress_identity._EVENTS

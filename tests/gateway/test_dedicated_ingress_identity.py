"""Dedicated ingress receipts; real native builder, no transport or model calls."""

import asyncio
import copy
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.ingress_identity import capture_launch, slack_binding
from gateway.session_identity import RoutingIdentity
from gateway import session_context as sc
from plugins.platforms.slack.adapter import SlackAdapter


async def build(owner="alpha", profile=None, multiplex=False):
    # Constructor-bypassed native runner: no production startup/services. Registry
    # and actual adapter builder remain real; launch capture seam explicit.
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=multiplex)
    runner._primary_profile_name = owner
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    adapter.gateway_runner = runner
    capture_launch(runner)
    adapter._resolve_user_name = AsyncMock(return_value="fixture")
    adapter._resolve_channel_name = AsyncMock(return_value="fixture")
    adapter._humanize_user_mentions = AsyncMock(side_effect=lambda text, **kw: text)
    adapter._channel_prompt_with_identity = lambda *args: None
    event = await adapter._build_message_event(
        {},
        text="fixture",
        original_text="fixture",
        command_probe_text="fixture",
        is_command_text=False,
        channel_id="C123456789",
        team_id="T123456789",
        ts="1788265045.536689",
        user_id="U123456789",
        thread_ts="1788265000.000001",
        is_dm=False,
        media_urls=[],
        media_types=[],
        media_text_inlined=[],
        channel_context=None,
    )
    # build_source's raw profile defaults None. Record nonblank fixtures by
    # setting it before a second native builder call through build_source.
    if profile is not None:
        original = adapter.build_source

        def source(**kwargs):
            result = original(**kwargs)
            result.profile = profile
            return result

        adapter.build_source = source
        event = await adapter._build_message_event(
            {},
            text="fixture",
            original_text="fixture",
            command_probe_text="fixture",
            is_command_text=False,
            channel_id="C123456789",
            team_id="T123456789",
            ts="1788265045.536689",
            user_id="U123456789",
            thread_ts="1788265000.000001",
            is_dm=False,
            media_urls=[],
            media_types=[],
            media_text_inlined=[],
            channel_context=None,
        )
    return runner, adapter, event


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "", "alpha"])
async def test_explicit_binding_preserves_wire(raw):
    r, a, e = await build(profile=raw)
    before = e.source.to_dict()
    assert slack_binding(r, e.source, e) == ("alpha", e.message_id)
    assert e.source.to_dict() == before
    assert e.source.thread_id != e.message_id
    with patch("os.getenv", side_effect=AssertionError("environment authority")):
        assert slack_binding(r, e.source, e)[0] == "alpha"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [False, 0, [], "wrong", " alpha", "Alpha", "alpha\n"])
async def test_raw_invalid_denied(raw):
    r, a, e = await build(profile=raw)
    assert slack_binding(r, e.source, e)[1] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "owner",
        "adapter",
        "internal",
        "message",
        "source_message",
        "identity",
        "dead",
        "copy",
        "unknown_mode",
        "adapter_runner",
        "replaced_identity",
    ],
)
async def test_drift_and_forgery_denied(kind):
    r, a, e = await build()
    if kind == "owner":
        r._primary_profile_name = "other"
    elif kind == "adapter":
        r.adapters = {}
    elif kind == "internal":
        e.internal = True
    elif kind == "message":
        e.message_id = e.source.thread_id
    elif kind == "source_message":
        e.source.message_id = None
    elif kind == "identity":
        e.source._identity = RoutingIdentity(
            "alpha",
            "alpha",
            __import__("pathlib").Path("/fixture"),
            __import__("pathlib").Path("/fixture"),
            multiplexed=False,
            transport_inferred=True,
        )
    elif kind == "dead":
        e.source._transport_adapter_ref = lambda: None
    elif kind == "copy":
        e = copy.copy(e)
    elif kind == "unknown_mode":
        r.config.multiplex_profiles = 0
    elif kind == "adapter_runner":
        a.gateway_runner = object()
    elif kind == "replaced_identity":
        e.source._identity = copy.copy(e.source._identity)
    assert slack_binding(r, e.source, e)[1] == ""


@pytest.mark.asyncio
async def test_multiplex_blank_denied_explicit_primary_preserved():
    for raw in (None, ""):
        r, a, e = await build(profile=raw, multiplex=True)
        assert slack_binding(r, e.source, e)[1] == ""
    r, a, e = await build(profile="alpha", multiplex=True)
    assert slack_binding(r, e.source, e)[0] == "alpha"


@pytest.mark.asyncio
async def test_task_local_executor_reset():
    async def one(owner):
        r, a, e = await build(owner=owner)
        tokens = r._set_session_env(
            SimpleNamespace(source=e.source, session_key="fixture"), event=e
        )
        try:
            await asyncio.sleep(0)
            r._get_executor = lambda: None
            from gateway.ingress_identity import get_trusted_ingress

            value = await r._run_in_executor_with_context(
                lambda: get_trusted_ingress().profile
            )
            assert value == owner
        finally:
            sc.clear_session_vars(tokens)
        assert sc._VAR_MAP["HERMES_SESSION_MESSAGE_ID"].get() == ""

    await asyncio.gather(one("alpha"), one("beta"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [None, False, 0, [], "", " ", "1788265045", "1788265045.1", "1788265045.536689\n"],
)
async def test_raw_message_validation(value):
    r, a, e = await build()
    e.message_id = e.source.message_id = value
    assert slack_binding(r, e.source, e)[1] == ""


@pytest.mark.asyncio
async def test_receipt_lifetime_and_actual_dead_adapter():
    import gc
    from gateway.ingress_identity import _EVENTS

    r, a, e = await build()
    key = id(e)
    assert key in _EVENTS
    adapter_ref = weakref.ref(a)
    r.adapters.clear()
    del a
    gc.collect()
    assert adapter_ref() is None
    assert slack_binding(r, e.source, e)[1] == ""
    event_ref = weakref.ref(e)
    del e
    gc.collect()
    assert event_ref() is None
    assert key not in _EVENTS


@pytest.mark.asyncio
async def test_source_restore_and_unbound_denied():
    from gateway.session import SessionSource

    r, a, e = await build()
    restored = SessionSource.from_dict(e.source.to_dict())
    assert slack_binding(r, restored, e)[1] == ""
    assert slack_binding(r, e.source, None)[1] == ""
    cloned = copy.copy(e)
    cloned.source = restored
    assert slack_binding(r, restored, cloned)[1] == ""


@pytest.mark.asyncio
async def test_native_key_and_pending_merge_preserve_receipt():
    from gateway.platforms.base import merge_pending_message_event

    r, a, e = await build()
    from gateway.session import build_session_key

    before = build_session_key(e.source, profile=a._session_key_profile(e.source))
    wire = e.source.to_dict()
    identity = a._canonicalize(e.source)
    assert identity is e.source._identity
    pending = {"fixture": e}
    _, _, followup = await build()
    merge_pending_message_event(pending, "fixture", followup, merge_text=True)
    assert pending["fixture"] is e
    assert slack_binding(r, e.source, e) == ("alpha", e.message_id)
    assert (
        build_session_key(e.source, profile=a._session_key_profile(e.source)) == before
    )
    assert e.source.to_dict() == wire


def test_real_constructor_captures_launch(monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.ingress_identity import _LAUNCHES

    # Execute the constructor and its real cache/launch initialization. Mock only
    # unrelated startup, persistence and environment-loading side effects.
    for name in (
        "_warn_if_docker_media_delivery_is_risky",
        "_init_runtime_settings",
        "_init_session_store",
        "_init_lifecycle_state",
        "_init_startup_checks",
        "_init_session_db",
        "_init_registries_and_clocks",
    ):
        monkeypatch.setattr(GatewayRunner, name, lambda self: None)
    monkeypatch.setattr(GatewayRunner, "_active_profile_name", lambda self: "alpha")
    runner = GatewayRunner(config=GatewayConfig())
    assert _LAUNCHES[runner].owner == "alpha"
    original = _LAUNCHES[runner]
    runner._primary_profile_name = "beta"
    capture_launch(runner)
    assert _LAUNCHES[runner] is original


@pytest.mark.asyncio
async def test_native_builder_timestamp_types(monkeypatch):
    from gateway.ingress_identity import _EVENTS

    # Test raw malformed values at receipt construction, not just later drift.
    for value in (
        None,
        False,
        0,
        [],
        "",
        "1788265045",
        "1788265045.1",
        "1788265045.536689\n",
    ):
        r, a, e = await build()
        original = a.build_source

        def malformed_source(**kwargs):
            source = original(**kwargs)
            source.message_id = value
            return source

        a.build_source = malformed_source
        event = await a._build_message_event(
            {},
            text="fixture",
            original_text="fixture",
            command_probe_text="fixture",
            is_command_text=False,
            channel_id="C123456789",
            team_id="T123456789",
            ts=value,
            user_id="U123456789",
            thread_ts="1788265000.000001",
            is_dm=False,
            media_urls=[],
            media_types=[],
            media_text_inlined=[],
            channel_context=None,
        )
        assert id(event) in _EVENTS
        assert slack_binding(r, event.source, event)[1] == ""

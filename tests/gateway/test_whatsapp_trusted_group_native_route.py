"""Trusted WhatsApp groups through the durable native path: admission and drain agree on the route.

``groups.<jid>.trust: trusted`` makes a group one shared session. The adapter marks the live source
(``shared_group`` / ``sender_role``), and those markers are wire-invisible by design. Admission
stamped the shared route from the live event but reserved its target from the source rebuilt from
the payload, which had lost the markers and keyed per participant: every row targeted a per-sender
session under the shared route, and the drain preflight (``check_native_route``) refused it with
``admission_conflict``. Every message in a trusted group was dropped with a fixed notice.

``restore_native_source`` now re-derives the markers from the answering adapter's own config,
never from the payload.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from hermes_state_runtime import RuntimeStoreError

# Synthetic identities only.
OWNER = "15550000001@s.whatsapp.net"
MEMBER = "15550000002@s.whatsapp.net"
TRUSTED = "120363000000000002@g.us"
CLOSED = "120363000000000001@g.us"

_ENV = (
    "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_ALL_USERS", "WHATSAPP_GROUP_ALLOWED_USERS",
    "WHATSAPP_GROUP_ALLOW_FROM", "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY",
    "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
)


def _setup(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "session_path": str(tmp_path / "wa"),
        "dm_policy": "allowlist",
        "allow_from": [OWNER],
        "group_policy": "open",
        "owners": [OWNER],
        "groups": {
            "*": {"allow_from": [OWNER]},
            TRUSTED: {"allow_from": ["*"], "trust": "trusted"},
            CLOSED: {"allow_from": ["*"]},
        },
    }))
    runner.adapters = {Platform.WHATSAPP: adapter}
    return runner, adapter


def _live_source(adapter, user_id, chat_id=TRUSTED):
    """The source exactly as live intake builds it (``_build_message_event``)."""
    source = adapter.build_source(chat_id=chat_id, chat_name="synthetic", chat_type="group",
                                  user_id=user_id, user_name="synthetic")
    adapter.annotate_group_trust(source)
    return source


def _event(source, message_id, text="@alex hello"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, message_id=message_id, source=source)


async def _authority(runner, monkeypatch, name):
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id=name)
    register_transport_home(runner, None, get_hermes_home())
    return authority


async def _admit(runner, authority, event):
    """The live path (``admit_native`` inside the adapter's ``native_callback``); returns the
    committed row's payload and target session id."""
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    row = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)[-1]
    return row["payload"], receipt.ref.session_id


@pytest.mark.asyncio
async def test_trusted_group_admission_targets_the_session_its_route_names(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-trusted-route")
    member = _live_source(adapter, MEMBER)
    assert member.shared_group is True
    shared_route = runner.session_store._generate_session_key(member)
    assert not shared_route.endswith(MEMBER.split("@")[0])  # one shared key, no participant

    payload, target = await _admit(runner, authority, _event(member, "m1"))
    assert payload["native_text_v1"]["route"] == shared_route
    entry = runner.session_store.lookup_by_session_key(shared_route)
    assert entry is not None and entry.session_id == target

    # The owner's message joins the same shared session.
    owner_payload, owner_target = await _admit(runner, authority, _event(_live_source(adapter, OWNER), "m2"))
    assert owner_payload["native_text_v1"]["route"] == shared_route
    assert owner_target == target


@pytest.mark.asyncio
async def test_drain_preflight_accepts_a_trusted_group_row(tmp_path, monkeypatch):
    """The regression: ``check_native_route`` refused every trusted-group row with
    ``admission_conflict``. It now passes and replays the adapter-derived trust markers."""
    from gateway.session_envelope import check_native_route

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-trusted-drain")
    for user_id, role, message_id in ((MEMBER, "member", "m1"), (OWNER, "owner", "m2")):
        live = _live_source(adapter, user_id)
        payload, target = await _admit(runner, authority, _event(live, message_id))
        replayed, route = await check_native_route(runner, payload, target, live, adapter)
        assert route == payload["native_text_v1"]["route"]
        assert replayed.shared_group is True
        assert replayed.sender_role == role


@pytest.mark.asyncio
async def test_startup_recovery_binds_queued_trusted_rows_to_the_stored_origin(tmp_path, monkeypatch):
    """After a restart, recovery binds each queued row to the routing entry's origin, rebuilt from
    disk without the trust markers. Keyed per participant it never matched the shared route, so
    every queued trusted-group row was refused; the adapter's verdict now applies to it too."""
    from gateway.session import SessionSource

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-trusted-recovery")
    live = _live_source(adapter, MEMBER)
    _payload, target = await _admit(runner, authority, _event(live, "q1"))
    await _admit(runner, authority, _event(_live_source(adapter, OWNER), "q2"))
    stored_origin = SessionSource.from_dict(live.to_dict())  # as a restarted store holds it
    assert stored_origin.shared_group is False
    authority.sessions.pop(target, None)  # a fresh owner: no live session yet
    assert await authority.recover_native_sessions([(target, stored_origin, adapter)]) == {target: "ready"}
    assert stored_origin.shared_group is False  # the caller's origin object is not changed
    assert authority.sessions[target].source.shared_group is True


@pytest.mark.asyncio
async def test_untrusted_group_keeps_per_participant_sessions(tmp_path, monkeypatch):
    from gateway.session_envelope import check_native_route

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-closed-route")
    targets = set()
    for user_id, message_id in ((MEMBER, "c1"), (OWNER, "c2")):
        live = _live_source(adapter, user_id, CLOSED)
        assert live.shared_group is False
        payload, target = await _admit(runner, authority, _event(live, message_id))
        assert payload["native_text_v1"]["route"].endswith(user_id.split("@")[0])
        replayed, _route = await check_native_route(runner, payload, target, live, adapter)
        assert replayed.shared_group is False and replayed.sender_role is None
        targets.add(target)
    assert len(targets) == 2


@pytest.mark.asyncio
async def test_trust_is_never_read_from_the_payload(tmp_path, monkeypatch):
    """A payload that claims the markers for an untrusted group restores without them."""
    from gateway.session_envelope import restore_native_source

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-forged-trust")
    payload, _target = await _admit(runner, authority, _event(_live_source(adapter, MEMBER, CLOSED), "f1"))
    payload["native_text_v1"]["source"].update(shared_group=True, sender_role="owner")
    restored = restore_native_source(payload, runner)
    assert restored.shared_group is False and restored.sender_role is None


@pytest.mark.asyncio
async def test_trust_revoked_before_drain_is_refused_not_rerouted(tmp_path, monkeypatch):
    """The markers are read fresh: a group whose trust was revoked after admission re-keys per
    participant, which no longer matches the stamped shared route, so the row is refused."""
    from gateway.session_envelope import check_native_route

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-trust-revoked")
    live = _live_source(adapter, MEMBER)
    payload, target = await _admit(runner, authority, _event(live, "r1"))
    adapter._groups[TRUSTED].pop("trust")
    with pytest.raises(RuntimeStoreError, match="admission_conflict"):
        await check_native_route(runner, payload, target, live, adapter)

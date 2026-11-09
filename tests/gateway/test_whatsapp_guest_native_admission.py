"""Durable native admission rechecks WhatsApp per-group sender rules (hd-5e788d113b472ed9).

A guest admitted by the live WhatsApp adapter's ``groups.<chat_id>.allow_from`` reached the
gateway, then the durable re-check in ``session_envelope._validate_native`` refused it with
``permission_denied`` / ``user_not_authorized``: ordinary native input runs with
``allow_adapter_delegation=False`` and the per-group sender verdict was skipped, so the check fell
to principal allowlists that never hold guests. The verdict is the live adapter's own operator
policy read fresh at admission time, so it now runs with delegation off too; every caller-flag
grant (``role_authorized``, ``delivered_via_upstream_relay``, ``enforces_own_access_policy``,
``authorization_is_upstream``) stays delegation-gated.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_envelope import snapshot_native
from hermes_state_runtime import RuntimeStoreError

# Synthetic identities only.
OWNER = "15550000001@s.whatsapp.net"
GUEST = "15550000002@s.whatsapp.net"
OTHER_GUEST = "15550000003@s.whatsapp.net"
ORDINARY_GROUP = "120363000000000001@g.us"
GUEST_GROUP = "120363000000000002@g.us"

_ENV = (
    "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_ALL_USERS", "WHATSAPP_GROUP_ALLOWED_USERS",
    "WHATSAPP_GROUP_ALLOW_FROM", "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY",
    "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
)


def _extra(tmp_path, guest_group_allow=("*",)):
    return {
        "session_path": str(tmp_path / "wa"),
        "dm_policy": "allowlist",
        "allow_from": [OWNER],
        "group_policy": "open",
        "owners": [OWNER],
        "groups": {"*": {"allow_from": [OWNER]}, GUEST_GROUP: {"allow_from": list(guest_group_allow)}},
        "guest_toolsets": ["web"],
    }


def _setup(tmp_path, monkeypatch, **kw):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra=_extra(tmp_path, **kw)))
    runner.adapters = {Platform.WHATSAPP: adapter}
    return runner, adapter


def _source(adapter, user_id, chat_id=GUEST_GROUP, chat_type="group"):
    return adapter.build_source(chat_id=chat_id, chat_name="synthetic", chat_type=chat_type,
                                user_id=user_id, user_name="synthetic")


def _event(source, text="@alex hello", message_id="m1"):
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
    """The live path: ``admit_native`` (prepare_native -> _validate_native -> envelope) inside the
    adapter's ``native_callback`` context, i.e. with connector provenance."""
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    return list_session_admissions(authority.db, session_id=receipt.ref.session_id,
                                   pending_only=False)[-1]["payload"]


# --- 1. adapter allows -> durable native admits ---------------------------------------------

@pytest.mark.asyncio
async def test_guest_allowed_in_exact_group_is_admitted_by_durable_native_path(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    # The inbound gate (adapter + runner, delegation on) admits this guest in this group.
    assert adapter._is_group_allowed(GUEST_GROUP, GUEST) is True
    assert runner._is_user_authorized(_source(adapter, GUEST)) is True
    authority = await _authority(runner, monkeypatch, "wa-guest-full")
    payload = await _admit(runner, authority, _event(_source(adapter, GUEST), text="@alex guest question"))
    assert payload["text"] == "@alex guest question"
    envelope_source = payload["native_text_v1"]["source"]
    assert (envelope_source["user_id"], envelope_source["chat_id"]) == (GUEST, GUEST_GROUP)
    assert payload["native_text_v1"]["provenance"]["platform"] == "whatsapp"


def test_guest_allowed_in_exact_group_passes_snapshot_validation(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    snapshot = snapshot_native(runner, _event(_source(adapter, GUEST)))
    assert snapshot["native_text_v1"]["source"]["user_id"] == GUEST


# --- 2. fresh recheck: a revoked or changed rule is respected at admission -------------------

@pytest.mark.asyncio
async def test_revoked_group_allow_from_is_respected_at_durable_admission(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-guest-revoke")
    await _admit(runner, authority, _event(_source(adapter, GUEST), message_id="m1"))
    adapter._groups[GUEST_GROUP]["allow_from"] = [OWNER]  # operator revokes the guest
    with pytest.raises(RuntimeStoreError, match="permission_denied") as refused:
        await _admit(runner, authority, _event(_source(adapter, GUEST), message_id="m2"))
    assert refused.value.field == "user_not_authorized"


@pytest.mark.asyncio
async def test_changed_group_allow_from_admits_only_the_newly_listed_guest(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch, guest_group_allow=(GUEST,))
    authority = await _authority(runner, monkeypatch, "wa-guest-change")
    await _admit(runner, authority, _event(_source(adapter, GUEST), message_id="m1"))
    adapter._groups[GUEST_GROUP]["allow_from"] = [OTHER_GUEST]
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await _admit(runner, authority, _event(_source(adapter, GUEST), message_id="m2"))
    payload = await _admit(runner, authority, _event(_source(adapter, OTHER_GUEST), message_id="m3"))
    assert payload["native_text_v1"]["source"]["user_id"] == OTHER_GUEST


@pytest.mark.parametrize("stale", ["role_authorized", "delivered_via_upstream_relay", "authorization_is_upstream"])
@pytest.mark.asyncio
async def test_stale_fields_cannot_stand_in_for_a_revoked_rule(tmp_path, monkeypatch, stale):
    runner, adapter = _setup(tmp_path, monkeypatch, guest_group_allow=(OWNER,))
    authority = await _authority(runner, monkeypatch, f"wa-guest-stale-{stale}")
    source = _source(adapter, GUEST)
    if stale == "authorization_is_upstream":
        vars(source)[stale] = True  # a declaration stamped on the source; only the adapter is read
    else:
        setattr(source, stale, True)
    with pytest.raises(RuntimeStoreError) as refused:
        await _admit(runner, authority, _event(source))
    # A relay flag without a relay delivery fails provenance capture (``not_found``) before validation.
    assert refused.value.args[0] in {"permission_denied", "invalid_params", "not_found"}


def test_failed_sender_rule_check_fails_closed(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)

    def boom(chat_id, sender_id=""):
        raise RuntimeError("policy read failed")

    monkeypatch.setattr(adapter, "_is_group_allowed", boom)
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")  # a broader grant must not catch the failure
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        snapshot_native(runner, _event(_source(adapter, GUEST)))


# --- 3. other groups and DMs stay denied -----------------------------------------------------

@pytest.mark.asyncio
async def test_same_guest_in_other_group_or_dm_is_denied(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-guest-scope")
    await _admit(runner, authority, _event(_source(adapter, GUEST), message_id="m1"))
    for source in (_source(adapter, GUEST, ORDINARY_GROUP), _source(adapter, GUEST, GUEST, "dm")):
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            await _admit(runner, authority, _event(source, message_id="m2"))
        assert runner._is_user_authorized_for_source(source, allow_adapter_delegation=False) is False


# --- 4. per-caller isolation -----------------------------------------------------------------

def test_back_to_back_validations_get_their_own_verdicts(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    cases = [(GUEST, GUEST_GROUP, True), (GUEST, ORDINARY_GROUP, False), (OWNER, ORDINARY_GROUP, True),
             (OTHER_GUEST, GUEST_GROUP, True), (GUEST, ORDINARY_GROUP, False), (GUEST, GUEST_GROUP, True)]
    for user_id, chat_id, allowed in cases:
        event = _event(_source(adapter, user_id, chat_id))
        if allowed:
            got = snapshot_native(runner, event)["native_text_v1"]["source"]
            assert (got["user_id"], got["chat_id"]) == (user_id, chat_id)
        else:
            with pytest.raises(RuntimeStoreError, match="permission_denied"):
                snapshot_native(runner, event)


@pytest.mark.asyncio
async def test_concurrent_guest_and_owner_turns_do_not_share_verdicts(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-guest-concurrent")
    events = [
        _event(_source(adapter, GUEST, GUEST_GROUP), message_id="g1"),
        _event(_source(adapter, GUEST, ORDINARY_GROUP), message_id="g2"),
        _event(_source(adapter, OWNER, ORDINARY_GROUP), message_id="o1"),
        _event(_source(adapter, OWNER, OWNER, "dm"), message_id="o2"),
    ]
    results = await asyncio.gather(*(_admit(runner, authority, e) for e in events), return_exceptions=True)
    guest_ok, guest_other, owner_group, owner_dm = results
    assert guest_ok["native_text_v1"]["source"]["user_id"] == GUEST
    assert guest_ok["native_text_v1"]["source"]["chat_id"] == GUEST_GROUP
    assert isinstance(guest_other, RuntimeStoreError) and guest_other.args[0] == "permission_denied"
    assert owner_group["native_text_v1"]["source"]["user_id"] == OWNER
    assert owner_group["native_text_v1"]["source"]["chat_id"] == ORDINARY_GROUP
    assert owner_dm["native_text_v1"]["source"]["user_id"] == OWNER


# --- 5. the guest stays untrusted ------------------------------------------------------------

@pytest.mark.asyncio
async def test_admitted_guest_is_not_owner_or_role_authorized(tmp_path, monkeypatch):
    from gateway.session import SessionSource
    from gateway.slash_access import policy_for_source

    runner, adapter = _setup(tmp_path, monkeypatch)
    authority = await _authority(runner, monkeypatch, "wa-guest-untrusted")
    payload = await _admit(runner, authority, _event(_source(adapter, GUEST)))
    admitted = SessionSource.from_dict(payload["native_text_v1"]["source"])
    assert admitted.role_authorized is False
    assert admitted.delivered_via_upstream_relay is False
    assert adapter._is_whatsapp_owner(admitted.user_id) is False
    # Guest toolsets (exact override) and the slash-command floor still apply.
    assert adapter.toolsets_for_source(admitted) == ["web"]
    assert adapter.toolsets_for_source(_source(adapter, OWNER)) is None
    policy = policy_for_source(SimpleNamespace(platforms={Platform.WHATSAPP: adapter.config}), admitted)
    assert policy.enabled is True
    assert policy.can_run(GUEST, "model") is False
    assert policy.can_run(OWNER, "model") is True


# --- 6. forged caller flags are still refused ------------------------------------------------

def test_forged_role_authorized_without_fresh_roles_is_refused(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    source = _source(adapter, GUEST)
    source.role_authorized = True
    with pytest.raises(RuntimeStoreError, match="invalid_params") as refused:
        snapshot_native(runner, _event(source))
    assert refused.value.field == "role_authorized_without_fresh_roles"


def test_forged_upstream_relay_without_provenance_is_refused(tmp_path, monkeypatch):
    runner, adapter = _setup(tmp_path, monkeypatch)
    source = _source(adapter, GUEST)
    source.delivered_via_upstream_relay = True
    with pytest.raises(RuntimeStoreError, match="invalid_params") as refused:
        snapshot_native(runner, _event(source))
    assert refused.value.field == "upstream_relay_without_provenance"


def test_caller_flag_grants_stay_delegation_gated(tmp_path, monkeypatch):
    """Without per-group sender rules the sender verdict abstains, and caller flags still grant only
    with delegation on: the fix does not widen ``role_authorized`` / own-policy grants."""
    runner, adapter = _setup(tmp_path, monkeypatch)
    adapter._groups = {}
    assert adapter.group_sender_rules_active is False
    source = _source(adapter, GUEST, ORDINARY_GROUP)
    source.role_authorized = True
    assert runner._principal_authorized(source, allow_adapter_delegation=False) is False
    assert runner._principal_authorized(source, allow_adapter_delegation=True) is True

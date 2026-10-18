"""Per-group sender rules, native-only mentions, owner/guest toolsets and the pre-gating
``whatsapp_transport_event`` observer for the Baileys WhatsApp adapter."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource

OWNER = "15550000001@s.whatsapp.net"
OWNER_DEVICE = "15550000001:12@s.whatsapp.net"
GUEST = "15550000002@s.whatsapp.net"
BOT_IDS = ["15551230000@s.whatsapp.net", "15551230000@lid"]
ORDINARY_GROUP = "120363000000000001@g.us"
SQUAD_GROUP = "120363000000000002@g.us"

ALEX_EXTRA = {
    "dm_policy": "allowlist",
    "allow_from": [OWNER],
    "group_policy": "open",
    "require_mention": True,
    "require_native_mention": True,
    "mention_patterns": [r"\balex\b"],
    "free_response_chats": [SQUAD_GROUP],
    "owners": [OWNER],
    "groups": {"*": {"allow_from": [OWNER]}, SQUAD_GROUP: {"allow_from": ["*"]}},
    "guest_toolsets": ["web", "vision"],
}


def _make_adapter(extra=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = dict(ALEX_EXTRA if extra is None else extra)
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._groups = extra.get("groups") if isinstance(extra.get("groups"), dict) else {}
    adapter._owners = WhatsAppAdapter._coerce_allow_list(extra.get("owners"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _group(chat_id=ORDINARY_GROUP, sender=OWNER, body="hi", **overrides):
    data = {"isGroup": True, "chatId": chat_id, "senderId": sender, "body": body,
            "mentionedIds": [], "botIds": list(BOT_IDS), "quotedParticipant": "", "messageId": "M1"}
    data.update(overrides)
    return data


def _native(**kw):
    return _group(mentionedIds=["15551230000@s.whatsapp.net"], **kw)


def _dm(sender):
    return {"isGroup": False, "chatId": sender, "senderId": sender, "from": sender, "body": "hi",
            "mentionedIds": [], "botIds": list(BOT_IDS)}


# --- intake --------------------------------------------------------------------------------

def test_owner_native_mention_admitted_in_any_group():
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(chat_id=ORDINARY_GROUP)) is True
    assert adapter._should_process_message(_native(chat_id="120363999999999999@g.us")) is True


def test_owner_device_qualified_sender_matches():
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(sender=OWNER_DEVICE)) is True


def test_non_owner_mention_denied_in_ordinary_group():
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(sender=GUEST)) is False


def test_non_owner_native_mention_admitted_in_exception_group():
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(chat_id=SQUAD_GROUP, sender=GUEST)) is True


@pytest.mark.parametrize("variant", [
    {"body": "plain chatter"},
    {"body": "@15551230000 typed number"},
    {"body": "hey alex what's up"},  # regex wake word
    {"body": "/help"},
    {"quotedParticipant": "15551230000@s.whatsapp.net", "body": "reply to bot"},
    {"mentionedIds": ["15559999999@s.whatsapp.net"]},  # someone else mentioned
])
@pytest.mark.parametrize("chat_id,sender", [(ORDINARY_GROUP, OWNER), (SQUAD_GROUP, GUEST), (SQUAD_GROUP, OWNER)])
def test_native_mode_rejects_every_non_native_trigger(variant, chat_id, sender):
    adapter = _make_adapter()
    assert adapter._should_process_message(_group(chat_id=chat_id, sender=sender, **variant)) is False


def test_missing_sender_or_bot_ids_denied():
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(sender="")) is False
    assert adapter._should_process_message(_native(botIds=[])) is False


def test_dm_only_owner():
    adapter = _make_adapter()
    assert adapter._should_process_message(_dm(OWNER)) is True
    assert adapter._should_process_message(_dm(GUEST)) is False


def test_group_policy_disabled_overrides_sender_rules():
    adapter = _make_adapter({**ALEX_EXTRA, "group_policy": "disabled"})
    assert adapter._should_process_message(_native()) is False


def test_chat_without_rule_and_without_wildcard_denied():
    extra = {**ALEX_EXTRA, "groups": {SQUAD_GROUP: {"allow_from": ["*"]}}}
    adapter = _make_adapter(extra)
    assert adapter._should_process_message(_native(chat_id=ORDINARY_GROUP)) is False
    assert adapter._should_process_message(_native(chat_id=SQUAD_GROUP, sender=GUEST)) is True


def test_legacy_behaviour_without_new_keys_unchanged():
    extra = {"group_policy": "open", "require_mention": True}
    adapter = _make_adapter(extra)
    assert adapter.group_sender_rules_active is False
    assert adapter._should_process_message(_group(sender=GUEST, body="/help")) is True
    assert adapter._should_process_message(_group(sender=GUEST, body="@15551230000 hi")) is True
    assert adapter.toolsets_for_source(SessionSource(platform=Platform.WHATSAPP, chat_id=ORDINARY_GROUP,
                                                     chat_type="group", user_id=GUEST)) is None


# --- toolsets ------------------------------------------------------------------------------

def _source(user_id, chat_id=SQUAD_GROUP, chat_type="group"):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, chat_type=chat_type, user_id=user_id)


def test_guest_gets_guest_toolsets_owner_keeps_platform():
    adapter = _make_adapter()
    assert adapter.toolsets_for_source(_source(OWNER)) is None
    assert adapter.toolsets_for_source(_source(OWNER_DEVICE)) is None
    assert sorted(adapter.toolsets_for_source(_source(GUEST))) == ["vision", "web"]
    assert adapter.toolsets_for_source(_source(None)) is not None


def test_wildcard_owner_never_counts_as_owner():
    adapter = _make_adapter({**ALEX_EXTRA, "owners": ["*"]})
    assert adapter.toolsets_for_source(_source(OWNER)) is not None


def _runner_with(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = None
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._delivery_adapter_for = lambda source: adapter
    return runner


def test_runner_exact_guest_toolsets_drop_plugin_defaults_and_mcp(monkeypatch):
    import hermes_cli.tools_config as tc

    adapter = _make_adapter()
    runner = _runner_with(adapter)
    config = {"platform_toolsets": {"whatsapp": ["web", "vision", "terminal", "file", "session_search"]}}
    # A default-on plugin toolset and an MCP server must not reach a guest.
    monkeypatch.setattr(tc, "_get_plugin_toolset_keys", lambda: {"whatsapp_archive_client"})
    monkeypatch.setattr(tc, "enabled_mcp_server_names", lambda cfg: {"private-mcp"})
    guest = runner._resolve_enabled_toolsets_for_source(config, _source(GUEST), "whatsapp")
    assert set(guest) <= {"web", "vision"} and "web" in guest
    owner = runner._resolve_enabled_toolsets_for_source(config, _source(OWNER), "whatsapp")
    assert {"terminal", "file", "session_search"} <= set(owner)


def test_runner_exact_override_error_fails_closed():
    adapter = _make_adapter()
    runner = _runner_with(adapter)

    def boom(source):
        raise RuntimeError("x")

    adapter.toolsets_for_source = boom
    assert runner._resolve_enabled_toolsets_for_source({}, _source(GUEST), "whatsapp") == []


# --- runner authorization ------------------------------------------------------------------

def test_runner_sender_rules_override_owner_env_allowlist(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER)
    adapter = _make_adapter()
    runner = _runner_with(adapter)
    runner._authorization_adapter = lambda platform, profile=None: adapter
    assert runner._is_user_authorized(_source(GUEST, SQUAD_GROUP)) is True
    assert runner._is_user_authorized(_source(GUEST, ORDINARY_GROUP)) is False
    assert runner._is_user_authorized(_source(OWNER, ORDINARY_GROUP)) is True
    # DMs keep the platform allowlist.
    assert runner._is_user_authorized(_source(GUEST, GUEST, "dm")) is False
    assert runner._is_user_authorized(_source(OWNER, OWNER, "dm")) is True


# --- transport observer --------------------------------------------------------------------

def _hook_capture(monkeypatch, fail=False, delay=0.0):
    import time

    from hermes_cli import lifecycle, plugins

    seen = []

    def fake_invoke(name, **kwargs):
        time.sleep(delay)  # health: allow HX013 -- a slow observer is the behaviour under test
        seen.append((name, kwargs))
        kwargs["raw_message"]["body"] = "mutated"
        if fail:
            raise RuntimeError("observer failed")
        return []

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "whatsapp_transport_event")
    monkeypatch.setattr(plugins, "invoke_hook", fake_invoke)
    return seen


def _build_and_drain(adapter, data):
    async def run():
        result = await adapter._build_message_event(data)
        await asyncio.gather(*list(adapter.__dict__.get("_observer_tasks", ())))
        return result

    return asyncio.run(run())


def test_hook_is_registered():
    from hermes_cli.plugins import VALID_HOOKS

    assert "whatsapp_transport_event" in VALID_HOOKS


@pytest.mark.parametrize("fail", [False, True])
def test_observer_sees_ungated_group_chatter_and_cannot_mutate(monkeypatch, fail):
    seen = _hook_capture(monkeypatch, fail=fail)
    adapter = _make_adapter()
    data = _group(sender=GUEST, body="plain chatter")
    assert _build_and_drain(adapter, data) is None  # not a conversation turn
    assert [name for name, _ in seen] == ["whatsapp_transport_event"]
    assert seen[0][1]["platform"] == "whatsapp"
    assert data["body"] == "plain chatter"


def test_slow_observer_does_not_block_intake(monkeypatch):
    import time

    _hook_capture(monkeypatch, delay=0.5)
    adapter = _make_adapter()

    async def run():
        started = time.monotonic()
        await adapter._build_message_event(_group(sender=GUEST))
        elapsed = time.monotonic() - started
        await asyncio.gather(*list(adapter.__dict__.get("_observer_tasks", ())))
        return elapsed

    assert asyncio.run(run()) < 0.2


def test_observer_skips_dms_and_broadcasts(monkeypatch):
    seen = _hook_capture(monkeypatch)
    adapter = _make_adapter()
    _build_and_drain(adapter, _dm(GUEST))
    _build_and_drain(adapter, _group(chat_id="status@broadcast"))
    assert seen == []


def test_append_replay_is_observed_but_never_answered(monkeypatch):
    seen = _hook_capture(monkeypatch)
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(upsertType="append")) is False
    assert adapter._should_process_message(_native(upsertType="notify")) is True
    _build_and_drain(adapter, _native(upsertType="append"))
    assert len(seen) == 1


# --- review round fixes --------------------------------------------------------------------

def test_owners_without_guest_toolsets_means_no_guest_tools():
    extra = {k: v for k, v in ALEX_EXTRA.items() if k != "guest_toolsets"}
    adapter = _make_adapter(extra)
    assert adapter.toolsets_for_source(_source(GUEST)) == []
    assert adapter.toolsets_for_source(_source(OWNER)) is None


def test_owner_poll_vote_admitted_guest_vote_denied_in_ordinary_group():
    adapter = _make_adapter()
    vote = {"nativeType": "pollUpdateMessage", "body": "Option A"}
    assert adapter._should_process_message(_group(sender=OWNER, **vote)) is True
    assert adapter._should_process_message(_group(sender=GUEST, **vote)) is False


def test_shared_group_sessions_deny_guests():
    adapter = _make_adapter({**ALEX_EXTRA, "group_sessions_per_user": False})
    assert adapter._should_process_message(_native(chat_id=SQUAD_GROUP, sender=GUEST)) is False
    assert adapter._should_process_message(_native(chat_id=SQUAD_GROUP, sender=OWNER)) is True


def test_text_batches_never_merge_across_participants():
    from gateway.platforms.base import MessageEvent

    adapter = _make_adapter()
    adapter._session_key_profile = lambda source: None
    adapter._canonicalize = lambda source: None
    keys = {adapter._text_batch_key(MessageEvent(text="x", source=_source(user))) for user in (OWNER, GUEST)}
    assert len(keys) == 2


def test_allow_all_env_does_not_bypass_sender_rules(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter()
    runner = _runner_with(adapter)
    runner._authorization_adapter = lambda platform, profile=None: adapter
    assert runner._is_user_authorized(_source(GUEST, ORDINARY_GROUP)) is False


def test_missing_whatsapp_adapter_gives_no_tools():
    runner = _runner_with(None)
    config = {"platform_toolsets": {"whatsapp": ["web", "terminal"]}}
    assert runner._resolve_enabled_toolsets_for_source(config, _source(GUEST), "whatsapp") == []


def test_guests_skip_memory_owners_keep_it():
    from gateway.run_turn_toolsets import source_skips_memory

    adapter = _make_adapter()
    assert source_skips_memory(lambda s: adapter, _source(GUEST)) is True
    assert source_skips_memory(lambda s: adapter, _source(OWNER)) is False
    with_memory = _make_adapter({**ALEX_EXTRA, "guest_toolsets": ["web", "memory"]})
    assert source_skips_memory(lambda s: with_memory, _source(GUEST)) is False


def test_owners_administer_slash_commands_guests_get_floor(lid_mapping):
    from types import SimpleNamespace

    from gateway.slash_access import policy_for_source

    gw = SimpleNamespace(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, extra=dict(ALEX_EXTRA))})
    policy = policy_for_source(gw, _source(GUEST))
    assert policy.enabled is True
    assert policy.can_run(GUEST, "help") is True
    for cmd in ("model", "yolo", "restart", "update", "reload_mcp"):
        assert policy.can_run(GUEST, cmd) is False
    assert policy.can_run(OWNER, "model") is True
    assert policy_for_source(gw, _source(lid_mapping)).can_run(lid_mapping, "model") is True
    assert policy_for_source(gw, _source(OWNER, OWNER, "dm")).can_run(OWNER_DEVICE, "yolo") is True


# --- identity forms ------------------------------------------------------------------------

@pytest.fixture
def lid_mapping(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_dir = tmp_path / "platforms" / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "lid-mapping-15550000001.json").write_text(json.dumps("88800000001"), encoding="utf-8")
    (session_dir / "lid-mapping-88800000001_reverse.json").write_text(json.dumps("15550000001"), encoding="utf-8")
    return "88800000001@lid"


def test_bare_digit_owner_entries_match_jid_senders():
    extra = {**ALEX_EXTRA, "owners": ["15550000001"], "groups": {"*": {"allow_from": ["+15550000001"]}}}
    adapter = _make_adapter(extra)
    assert adapter._should_process_message(_native(sender=OWNER)) is True
    assert adapter.toolsets_for_source(_source(OWNER)) is None
    assert adapter._should_process_message(_native(sender=GUEST)) is False


def test_owner_lid_sender_matches_phone_rules(lid_mapping):
    adapter = _make_adapter()
    assert adapter._should_process_message(_native(sender=lid_mapping)) is True
    assert adapter.toolsets_for_source(_source(lid_mapping)) is None
    assert adapter.toolsets_for_source(_source("88800000002@lid")) is not None


# --- startup guard -------------------------------------------------------------------------

@pytest.mark.parametrize("groups,dm_policy,violation", [
    ({"*": {"allow_from": [OWNER]}, SQUAD_GROUP: {"allow_from": ["*"]}}, "allowlist", False),
    ({SQUAD_GROUP: {"allow_from": ["*"]}}, "allowlist", False),             # unmatched groups denied
    ({"*": {"allow_from": ["*"]}}, "allowlist", True),                       # wildcard default = really open
    ({"*": {"allow_from": []}}, "allowlist", True),
    ({}, "allowlist", True),
    ({"*": {"allow_from": [OWNER]}}, "open", True),                          # open DMs still need the opt-in
])
def test_open_group_policy_guard_accepts_sender_rules(monkeypatch, groups, dm_policy, violation):
    from types import SimpleNamespace

    from gateway.run import _own_policy_open_startup_violation
    for key in ("GATEWAY_ALLOW_ALL_USERS", "WHATSAPP_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    extra = {"dm_policy": dm_policy, "group_policy": "open", "groups": groups}
    cfg = SimpleNamespace(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, extra=extra)})
    assert (_own_policy_open_startup_violation(cfg) is not None) is violation

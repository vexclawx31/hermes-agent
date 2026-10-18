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

def _hook_capture(monkeypatch, fail=False):
    from hermes_cli import lifecycle
    from hermes_cli import plugins

    seen = []

    async def fake_ainvoke(name, **kwargs):
        seen.append((name, kwargs))
        kwargs["raw_message"]["body"] = "mutated"
        if fail:
            raise RuntimeError("observer failed")
        return []

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "whatsapp_transport_event")
    monkeypatch.setattr(plugins, "ainvoke_hook", fake_ainvoke)
    return seen


def test_hook_is_registered():
    from hermes_cli.plugins import VALID_HOOKS

    assert "whatsapp_transport_event" in VALID_HOOKS


@pytest.mark.parametrize("fail", [False, True])
def test_observer_sees_ungated_group_chatter_and_cannot_mutate(monkeypatch, fail):
    seen = _hook_capture(monkeypatch, fail=fail)
    adapter = _make_adapter()
    data = _group(sender=GUEST, body="plain chatter")
    assert asyncio.run(adapter._build_message_event(data)) is None  # not a conversation turn
    assert [name for name, _ in seen] == ["whatsapp_transport_event"]
    assert seen[0][1]["platform"] == "whatsapp"
    assert data["body"] == "plain chatter"


def test_observer_skips_dms_and_broadcasts(monkeypatch):
    seen = _hook_capture(monkeypatch)
    adapter = _make_adapter()
    asyncio.run(adapter._build_message_event(_dm(GUEST)))
    asyncio.run(adapter._build_message_event(_group(chat_id="status@broadcast")))
    assert seen == []


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

"""Per-message sender attribution persisted on the user transcript row.

The session store keys sessions, not messages, to a person: ``sessions.user_id`` is whoever opened
the session, and a shared channel/thread session carries several people's messages. This module
builds the ``display_metadata["sender"]`` object the gateway writes on each inbound user row, so a
reader (``session_search`` sender filters) can prove MESSAGE-level authorship from the platform
envelope instead of trusting display names or content prefixes.

``trusted`` is False whenever the row's content may carry text the sender did not write (merged
events from several senders, history backfill, a quoted reply) or the sender is a bot. Readers
must treat an untrusted or missing ``sender`` as unknown and fail closed.
"""

from __future__ import annotations

from typing import Any, Optional

SENDER_METADATA_KEY = "sender"
SENDER_METADATA_VERSION = 1


def _platform_name(source: Any) -> Optional[str]:
    platform = getattr(source, "platform", None)
    value = getattr(platform, "value", platform)
    return str(value).strip().lower() if value else None


def sender_metadata(event: Any, source: Any = None) -> dict[str, Any]:
    """``{"sender": {...}}`` for a human inbound event, else ``{}`` (synthetic/internal/no sender id)."""
    if event is None:
        return {}
    source = source or getattr(event, "source", None)
    user_id = getattr(source, "user_id", None) if source is not None else None
    platform = _platform_name(source)
    if not user_id or not platform:
        return {}
    # Synthetic turns (background notices, heartbeats, wakes) were not typed by anyone.
    if getattr(event, "internal", False) or getattr(event, "_heartbeat_session_id", None):
        return {}
    reasons: list[str] = []
    if getattr(source, "is_bot", False):
        reasons.append("bot_sender")
    if getattr(event, "_mixed_senders", False):
        reasons.append("merged_senders")
    if getattr(event, "channel_context", None):
        reasons.append("channel_backfill")
    if getattr(event, "reply_to_text", None) and getattr(event, "reply_to_message_id", None):
        reasons.append("quoted_reply")
    message_id = getattr(source, "message_id", None) or getattr(event, "message_id", None)
    thread_id = getattr(source, "thread_id", None)
    # Slack keys a top-level channel message on its own ts as a synthetic thread root, so a thread
    # id equal to the message's own id is NOT a thread reply.
    thread_reply = bool(thread_id) and str(thread_id) != str(message_id or "")
    chat_id = getattr(source, "chat_id", None)
    sender: dict[str, Any] = {
        "v": SENDER_METADATA_VERSION,
        "platform": platform,
        "user_id": str(user_id),
        "chat_id": str(chat_id) if chat_id else None,
        "chat_type": getattr(source, "chat_type", None),
        "thread_id": str(thread_id) if thread_id else None,
        "thread_reply": thread_reply,
        "message_id": str(message_id) if message_id else None,
        "trusted": not reasons,
    }
    if reasons:
        sender["untrusted_reasons"] = reasons
    return {SENDER_METADATA_KEY: sender}


def trusted_turn_scope(source: Any):
    """The recall scope of a live turn (agent/recall_scope.py), or None when its sender is not proven.

    Only a native-ingress receipt (gateway/ingress_identity.py) that names this same sender and chat
    proves it; routing hints, restored sources and display names never do."""
    from agent.recall_scope import scope_for_source
    from gateway.ingress_identity import get_trusted_ingress

    if source is None:
        return None
    try:
        ingress = get_trusted_ingress()
    except Exception:
        return None
    if ingress is None or _platform_name(source) != str(ingress.platform).lower():
        return None
    if str(getattr(source, "user_id", "") or "") != str(ingress.user_id or "") or \
            str(getattr(source, "chat_id", "") or "") != str(ingress.chat_id or ""):
        return None
    chat_type = ingress.chat_type
    # Slack types group DMs (mpim) as "dm" too, but only a one-to-one IM (a "D…" channel) has a
    # single reader: anything else gets its chat's scope, never one participant's private scope.
    if str(ingress.platform).lower() == "slack" and not str(ingress.chat_id or "").startswith("D"):
        chat_type = "group"
    return scope_for_source(ingress.platform, ingress.user_id, ingress.chat_id, chat_type,
                            is_bot=bool(getattr(source, "is_bot", False)))

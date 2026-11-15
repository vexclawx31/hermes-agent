"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append a
"delivery-mirror" record to the target session's transcript so the receiving-side
agent knows what was sent.  Standalone: works from CLI, cron and gateway contexts.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"
_SESSIONS_INDEX_AT_IMPORT = _SESSIONS_INDEX


def _resolve_sessions_index() -> Path:
    """Active profile's ``sessions.json`` at call time: the patched ``_SESSIONS_INDEX`` when a test
    changed it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process
    serves every profile, so the import-time constant would resolve every profile's pre-migration
    session lookup against the launch profile's index."""
    return (_SESSIONS_INDEX if _SESSIONS_INDEX != _SESSIONS_INDEX_AT_IMPORT
            else get_hermes_home() / "sessions" / "sessions.json")


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    Pass ``session_id`` when the caller already holds the exact session (e.g. the
    cron in_channel seed) to skip the origin scan, which refuses to guess on a
    populated chat (flat + N thread sessions per chat_id) and would drop the mirror.
    Text that is NOT the agent speaking (e.g. a cron brief) must pass
    ``role="user"``: ``mirror`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real turn and yields assistant→assistant
    pairs that break strict-alternation providers, while a user-role mirror
    collapses safely via the consecutive-user merge.
    Returns True if mirrored, False if no matching session or error. Never raises.

    ``role`` defaults to ``"assistant"`` — correct for the interactive ``send_message`` mirror, where the
    mirrored text is the agent's own outgoing reply (a genuine assistant turn). See #2221.
    """
    try:
        if not session_id:
            session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id and str(platform or "").lower() == "whatsapp":
            session_id = _find_whatsapp_dm_session(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s (explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False
        _append_to_sqlite(session_id, {
            "role": role, "content": message_text, "timestamp": datetime.now().isoformat(),
            "mirror": True, "mirror_source": source_label,
        })
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning("Mirror failed for %s:%s thread=%s user=%s session=%s: %s", platform, chat_id, thread_id, user_id, session_id, e)
        return False


def _whatsapp_dm_alternates(chat_id: str, user_id: Optional[str]) -> list[tuple[str, Optional[str]]]:
    """Other ids of a WhatsApp DM peer to retry the origin lookup with: ``(chat_id, user_id)`` pairs.

    The bridge can key one person's DM session by a LID (``<lid>@lid``) while a proactive send (an owner
    notice, ``send_message``, a cron delivery) targets the phone JID (``<phone>@s.whatsapp.net``), or the
    reverse; the exact-match origin scan then finds nothing and the delivery is never mirrored.

    Only ids a lid-mapping file establishes are tried (:func:`gateway.whatsapp_identity.expand_whatsapp_aliases`),
    in the OTHER namespace: a phone target (``@s.whatsapp.net`` or a bare number) yields its mapped LIDs as
    ``<lid>@lid``; a LID target yields its mapped phones as ``<phone>@s.whatsapp.net``. The same digits are never
    tried in the other namespace (they don't name the same person), and a bare number also yields its own
    phone JID. Groups (``@g.us``) and other JID kinds get nothing. *user_id* follows the candidate when it is
    one of the peer's ids (a DM's peer); otherwise it is kept as given."""
    chat = str(chat_id or "").strip()
    if not chat or ("@" in chat and not chat.endswith(("@s.whatsapp.net", "@lid"))):
        return []
    from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier

    base = normalize_whatsapp_identifier(chat)
    aliases = expand_whatsapp_aliases(chat)  # unreadable mapping files are skipped there
    user_base = normalize_whatsapp_identifier(str(user_id)) if user_id is not None else ""
    mapped = sorted(a for a in aliases if a and a != base)
    if chat.endswith("@lid"):
        candidates = [f"{a}@s.whatsapp.net" for a in mapped]
    else:
        candidates = ([f"{base}@s.whatsapp.net"] if "@" not in chat else []) + [f"{a}@lid" for a in mapped]
    same_person = bool(user_base) and user_base in aliases
    return [(jid, jid if same_person else user_id) for jid in candidates if jid != chat]


def _find_whatsapp_dm_session(platform: str, chat_id: str, thread_id: Optional[str] = None,
                              user_id: Optional[str] = None) -> Optional[str]:
    """The DM session of *chat_id*'s peer under another of their ids (see :func:`_whatsapp_dm_alternates`).
    One live session found: it. Several (e.g. a phone-keyed and a LID-keyed one): the most recently active,
    by ``last_activity_at`` then ``started_at`` in state.db; when that can't be read, none (never a guess)."""
    found: list[str] = []
    for alt_chat, alt_user in _whatsapp_dm_alternates(chat_id, user_id):
        sid = _find_session_id(platform, alt_chat, thread_id=thread_id, user_id=alt_user)
        if sid and sid not in found:
            found.append(sid)
    if len(found) > 1:
        chosen = _most_recent_session(found)
    else:
        chosen = found[0] if found else None
    if chosen:
        logger.info("Mirror: whatsapp DM %s matched its session under another id of the same person (%d candidate%s)",
                    chat_id, len(found), "" if len(found) == 1 else "s")
    return chosen


def _most_recent_session(session_ids: list[str]) -> Optional[str]:
    """The most recently active of *session_ids* in state.db, or None when any can't be read."""
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            rows = [db.get_session(sid) for sid in session_ids]
        finally:
            release_or_close(db)
    except (sqlite3.Error, OSError):
        logger.debug("Mirror: session recency lookup failed", exc_info=True)
        return None
    if any(not row for row in rows):
        return None

    def _recency(row: dict) -> tuple[float, float]:
        return (float(row.get("last_activity_at") or 0), float(row.get("started_at") or 0))

    return str(max(rows, key=_recency)["id"])


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    """Active session_id for a platform + chat_id pair.

    state.db is primary; sessions.json is the pre-migration fallback.  DM keys
    don't embed the chat_id ("agent:main:telegram:dm"), so match on the persisted
    origin.  With *user_id*, exact sender matches win; several same-chat candidates
    with no user match → None rather than contaminate another participant's session.

    Queries state.db gateway session rows (primary source since #9006); falls back to scanning sessions.json
    for pre-migration databases.
    """
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            session_id = finder(platform=platform, chat_id=chat_id, thread_id=thread_id, user_id=user_id) if callable(finder) else None
            if session_id:
                return str(session_id)
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    sessions_index = _resolve_sessions_index()
    if not sessions_index.exists():
        return None
    try:
        data = json.loads(sessions_index.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    def _matches(entry: dict) -> bool:
        origin = entry.get("origin") or {}
        return ((origin.get("platform") or entry.get("platform", "")).lower() == platform.lower()
                and str(origin.get("chat_id", "")) == str(chat_id)
                and (thread_id is None or str(origin.get("thread_id") or "") == str(thread_id)))

    # Keys starting with "_" (e.g. the gateway's "_README") are metadata sentinels.
    candidates = [e for k, e in data.items() if not str(k).startswith("_") and isinstance(e, dict) and _matches(e)]
    if not candidates:
        return None
    if user_id:
        exact_user_matches = [e for e in candidates if _origin_user_id(e) == str(user_id)]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1 and len({u.strip() for u in map(_origin_user_id, candidates) if u.strip()}) > 1:
        return None
    return max(candidates, key=lambda entry: entry.get("updated_at", "")).get("session_id")


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database.

    Raises on failure: ``mirror_to_session`` reports ``False`` (and warns) only when the
    exception reaches it — swallowing it here made every failed write look mirrored (#10130).
    """
    from hermes_state_registry import acquire, release_or_close

    db = acquire()
    try:
        db.append_message(session_id=session_id, role=message.get("role", "assistant"), content=message.get("content"))
    finally:
        release_or_close(db)

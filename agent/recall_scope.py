"""Recall scope: which stored evidence a turn may recall.

A *label* ``platform:user_id@chat_id`` names who wrote something and where. A turn's *scope* is a set
of label patterns (``slack:U1@*`` = that person anywhere, ``slack:*@C1`` = that chat), built by core from
trusted transport provenance (gateway/sender_attribution.py) or a cron job's declared ``recall_scope``,
never from model arguments, message text or display names. Unknown authors never match.

Enforcement (``privacy.recall_scope: enforce``, opt-in per profile) applies the scope on every recall
path by one of two means:
- per message: session_search returns only user rows whose trusted persisted sender is in scope;
- per partition: everything DERIVED inside a turn (assistant text, memory entries, provider documents,
  summaries, cron outputs) can carry any participant's words, so it is stored in, and read back only
  from, the partition of the scope it was produced under: built-in memory files and provider banks per
  scope, sessions that begin under enforcement (older ones are replaced, never replayed), cron outputs
  that record their run's scope. Without a scope, a turn recalls nothing.
Unenforced profiles keep their previous behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

UNKNOWN_LABEL = "?"
ENFORCE = "enforce"
WILDCARD = "*"
DM_CHAT_TYPES = frozenset({"dm", "im"})


def _clean(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    # The label grammar reserves ':' and '@'; a part carrying them cannot be a real platform id.
    return "" if any(ch in text for ch in ":@*?") or any(ch.isspace() for ch in text) else text


def make_label(platform: Any, user_id: Any, chat_id: Any) -> str:
    """``platform:user_id@chat_id``, or :data:`UNKNOWN_LABEL` when any part is missing or malformed."""
    p, u, c = _clean(platform).lower(), _clean(user_id), _clean(chat_id)
    return f"{p}:{u}@{c}" if p and u and c else UNKNOWN_LABEL


def _split(label: str) -> Optional[tuple[str, str, str]]:
    if not isinstance(label, str) or ":" not in label or "@" not in label:
        return None
    platform, _, rest = label.partition(":")
    user_id, _, chat_id = rest.partition("@")
    if not platform or not user_id or not chat_id:
        return None
    return platform, user_id, chat_id


def normalize_pattern(pattern: Any) -> Optional[str]:
    """A scope pattern: each part is an id or ``*``; the platform is never a wildcard."""
    parts = _split(pattern.strip()) if isinstance(pattern, str) else None
    if parts is None:
        return None
    platform, user_id, chat_id = parts
    platform = _clean(platform).lower()
    user_id = user_id if user_id == WILDCARD else _clean(user_id)
    chat_id = chat_id if chat_id == WILDCARD else _clean(chat_id)
    if not platform or not user_id or not chat_id or (user_id == WILDCARD and chat_id == WILDCARD):
        return None
    return f"{platform}:{user_id}@{chat_id}"


@dataclass(frozen=True)
class RecallScope:
    """Allowed provenance patterns for one turn or job; ``origin`` names where it came from (for logs)."""

    patterns: frozenset[str]
    origin: str = ""

    def allows_label(self, label: str) -> bool:
        parts = _split(label) if label != UNKNOWN_LABEL else None
        if parts is None:
            return False
        platform, user_id, chat_id = parts
        for pattern in self.patterns:
            pp, pu, pc = _split(pattern) or ("", "", "")
            if pp == platform and pu in (WILDCARD, user_id) and pc in (WILDCARD, chat_id):
                return True
        return False

    def allows(self, provenance: Optional[Iterable[str]]) -> bool:
        """True only for a non-empty, fully known provenance whose every label is in scope."""
        labels = frozenset(provenance or ())
        return bool(labels) and all(self.allows_label(label) for label in labels)

    def key(self) -> str:
        """Stable identity for caches that must not serve one scope's results to another."""
        return "|".join(sorted(self.patterns))

    def partition(self) -> str:
        """Filesystem/bank-safe name of this scope's storage partition (memory dir, provider bank)."""
        import hashlib
        return "scope-" + hashlib.sha256(self.key().encode()).hexdigest()[:16]

    def covers(self, other: "RecallScope") -> bool:
        """True when everything *other* may see, this scope may see too (pattern containment)."""
        def _covered(pattern: str) -> bool:
            op, ou, oc = _split(pattern) or ("", "", "")
            for mine in self.patterns:
                mp, mu, mc = _split(mine) or ("", "", "")
                if mp == op and mu in (WILDCARD, ou) and mc in (WILDCARD, oc):
                    return True
            return False
        return bool(other.patterns) and all(_covered(p) for p in other.patterns)

    def describe(self) -> dict[str, Any]:
        return {"patterns": sorted(self.patterns), "origin": self.origin}


def scope_from_patterns(patterns: Any, origin: str = "") -> Optional[RecallScope]:
    """A scope from declared patterns (a cron job's ``recall_scope``); None if any pattern is invalid."""
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)) or not patterns:
        return None
    normalized = [normalize_pattern(p) for p in patterns]
    if any(p is None for p in normalized):
        return None
    return RecallScope(frozenset(normalized), origin=origin)


def scope_for_source(platform: Any, user_id: Any, chat_id: Any, chat_type: Any, *,
                     is_bot: bool = False) -> Optional[RecallScope]:
    """Scope of a live turn from its trusted transport source.

    A DM sees only its own person's evidence from anywhere (``p:user@*``). A shared chat sees only
    that chat's evidence (``p:*@chat``): everyone there already saw it, and private evidence from
    elsewhere would reach that chat's whole audience. Bot senders never get a human scope."""
    if is_bot:
        return None
    p, u, c = _clean(platform).lower(), _clean(user_id), _clean(chat_id)
    if not p or not u or not c:
        return None
    if str(chat_type or "").strip().lower() in DM_CHAT_TYPES:
        return RecallScope(frozenset({f"{p}:{u}@{WILDCARD}"}), origin="dm")
    return RecallScope(frozenset({f"{p}:{WILDCARD}@{c}"}), origin="chat")


def _configured(config: Mapping[str, Any]) -> bool:
    privacy = config.get("privacy") if isinstance(config, Mapping) else None
    value = privacy.get("recall_scope") if isinstance(privacy, Mapping) else None
    return isinstance(value, str) and value.strip().lower() == ENFORCE


def enforcement_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """``privacy.recall_scope == "enforce"`` in the given or the active profile config.

    An unreadable config (an exception, or the defaults-only ``FailedConfigRead`` fallback) never
    turns enforcement off: the profile counts as enforcing while its epoch file exists, else this
    process keeps its last answer for that profile home (off for a profile that never opted in)."""
    if config is not None:
        return _configured(config)
    try:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
    except Exception:
        home = ""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.config_read_errors import FailedConfigRead
        loaded = load_config_readonly()
        readable = not isinstance(loaded, FailedConfigRead)
    except Exception:
        loaded, readable = None, False
    if not readable:
        try:
            if _epoch_path(home or None).exists():
                return True
        except Exception:
            return True
        return _LAST_KNOWN_ENFORCEMENT.get(home, False)
    _LAST_KNOWN_ENFORCEMENT[home] = _configured(loaded)
    return _LAST_KNOWN_ENFORCEMENT[home]


# Last successfully read answer per profile home (a multiplexed gateway serves several).
_LAST_KNOWN_ENFORCEMENT: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Message-level attribution
# ---------------------------------------------------------------------------

def sender_label(display_metadata: Any) -> str:
    """The label of a persisted user row's trusted sender, else :data:`UNKNOWN_LABEL`."""
    if isinstance(display_metadata, str):
        try:
            display_metadata = json.loads(display_metadata)
        except (TypeError, ValueError):
            return UNKNOWN_LABEL
    sender = display_metadata.get("sender") if isinstance(display_metadata, Mapping) else None
    if not isinstance(sender, Mapping) or sender.get("trusted") is not True:
        return UNKNOWN_LABEL
    return make_label(sender.get("platform"), sender.get("user_id"), sender.get("chat_id"))


# ---------------------------------------------------------------------------
# Enforcement epoch: a session is a provenance partition only if it began under enforcement
# ---------------------------------------------------------------------------
# Every input of a session that started under enforcement was scope-filtered (recall, memory
# partition, history of that same scoped session), so its transcript, summaries and outputs stay
# inside its scope. A session that predates enforcement holds unattributed history: it is not
# resumed, only replaced (its rows stay on disk, untouched).
EPOCH_FILENAME = "recall_scope_epoch.json"


def _epoch_path(hermes_home: Any = None):
    from pathlib import Path
    if hermes_home is None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
    return Path(hermes_home) / "state" / EPOCH_FILENAME


def enforcement_epoch(enforced: bool, hermes_home: Any = None, now: Optional[float] = None) -> Optional[float]:
    """Start time of the current enforcement period (created on first use), or None when off.

    Turning enforcement off forgets the epoch, so turning it back on starts a new one and every
    session from the unenforced gap is replaced too."""
    import os
    import time
    path = _epoch_path(hermes_home)
    if not enforced:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("since")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError):
        # Unreadable epoch: start a new one (replaces sessions once more; never trusts old ones).
        pass
    since = float(now if now is not None else time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"since": since}), encoding="utf-8")
    os.replace(tmp, path)
    return since


def apply_turn_recall_scope(agent: Any) -> None:
    """Bind the turn's scope to the agent's recall paths, before the system prompt or any recall.

    ``agent._recall_scope`` is set per turn by its caller (gateway: trusted ingress; cron: the job's
    declaration). The built-in memory store moves to the scope's partition and the memory manager
    gates providers; a changed partition invalidates the cached system prompt (its memory block)."""
    enforced = enforcement_enabled()
    scope = getattr(agent, "_recall_scope", None) if enforced else None
    if not isinstance(scope, RecallScope):
        scope = None
    store = getattr(agent, "_memory_store", None)
    if store is not None and hasattr(store, "bind_recall_scope") and store.bind_recall_scope(enforced, scope):
        agent._cached_system_prompt = None
    manager = getattr(agent, "_memory_manager", None)
    if manager is not None and hasattr(manager, "bind_recall_scope"):
        manager.bind_recall_scope(enforced, scope)


class _UnprovableLineage(Exception):
    pass


# A compression child continues its parent's transcript and a branch (API fork, CLI /branch) copies
# it, so their parents' rows are this conversation's too; a reset successor starts fresh.
_HISTORY_CARRYING_END_REASONS = frozenset({"compression", "branched"})


def _conversation_rows(session_db: Any, session_id: str, max_depth: int = 64) -> list[dict]:
    """*session_id*'s row, then the ancestors whose transcript it carries (compression continues
    it, a branch copies it); a reset parent is a boundary, its rows are not this conversation's.
    ``[]`` when the session doesn't exist yet; unprovable lineage raises."""
    rows: list[dict] = []
    current = str(session_id)
    for _ in range(max_depth):
        row = session_db.get_session(current)
        if not row:
            if rows:
                raise _UnprovableLineage(current)
            return rows
        rows.append(row)
        parent = row.get("parent_session_id")
        if not parent or parent == current:
            return rows
        parent_row = session_db.get_session(parent)
        if not parent_row:
            raise _UnprovableLineage(parent)
        if parent_row.get("end_reason") not in _HISTORY_CARRYING_END_REASONS:
            return rows
        current = parent
    raise _UnprovableLineage(session_id)


def session_lineage_predates_epoch(session_db: Any, session_id: str) -> bool:
    """True when enforcement is on and *session_id*'s conversation (its compression lineage) began
    before the epoch. A session that replaced an old one (reset) starts its own conversation.
    Unreadable lineage counts as old: fail closed."""
    if not enforcement_enabled() or session_db is None or not session_id:
        return False
    try:
        since = enforcement_epoch(True)
        rows = _conversation_rows(session_db, session_id)
    except Exception:
        return True
    if not rows:
        return False
    started = rows[-1].get("started_at")
    return not isinstance(started, (int, float)) or started < since


def session_route_mismatch(session_db: Any, session_id: str, route_key: Optional[str]) -> bool:
    """True when enforcement is on and *session_id*'s conversation belongs to another route.

    Every row of the conversation (its compression lineage) must carry *route_key*. The commands
    that move a conversation to another route (/resume, /branch) are refused under enforcement
    (gateway/slash_commands_session.py), so a key only differs for state made before enforcement or
    by another writer; unreadable lineage counts as a mismatch."""
    if not enforcement_enabled() or session_db is None or not session_id:
        return False
    try:
        rows = _conversation_rows(session_db, session_id)
    except Exception:
        return True
    return any(row.get("session_key") and row.get("session_key") != route_key for row in rows)

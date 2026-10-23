"""Recall scope: which stored evidence a turn may recall, decided by provenance.

Every recallable item (a transcript row, a memory entry, a provider document, a compression summary,
a cron output) carries a *provenance*: the set of ``platform:user_id@chat_id`` labels of everyone whose
words could have shaped it. A user row's provenance is its trusted persisted sender
(gateway/sender_attribution.py); anything produced inside a turn inherits the provenance of the whole
context that produced it, so assistant text that relays one participant still carries that participant.

A turn's *scope* is a set of allowed label patterns (``slack:U1@*``, ``slack:*@C1``, ``slack:U1@C1``),
built by core from trusted transport provenance or a cron job's declared ``recall_scope``, never from
model arguments, message text or display names. An item is eligible iff its provenance is non-empty,
contains no unknown label and every label matches a pattern. Unknown provenance never matches.

Enforcement is opt-in per profile (``privacy.recall_scope: enforce``); without it every caller keeps
its previous behaviour.
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


def enforcement_enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    """``privacy.recall_scope == "enforce"`` in the given or the active profile config."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            # An unreadable config cannot prove enforcement is off; callers that only widen must not
            # read this as "off". Report on so recall fails closed rather than open.
            return True
    privacy = config.get("privacy") if isinstance(config, Mapping) else None
    value = privacy.get("recall_scope") if isinstance(privacy, Mapping) else None
    return isinstance(value, str) and value.strip().lower() == ENFORCE


# ---------------------------------------------------------------------------
# Provenance of stored items
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


def message_provenance(message: Mapping[str, Any]) -> frozenset[str]:
    """Provenance a single transcript message contributes.

    A user row contributes its trusted sender or unknown. A row that already carries a recorded
    ``provenance`` (summaries, injected recall) contributes it. Assistant, tool and system rows add
    nothing of their own: they were produced from the same context."""
    if not isinstance(message, Mapping):
        return frozenset({UNKNOWN_LABEL})
    recorded = provenance_from_metadata(message.get("display_metadata"))
    if recorded is not None:
        return recorded
    if message.get("role") == "user":
        if message.get("_compressed_summary"):
            return frozenset({UNKNOWN_LABEL})
        return frozenset({sender_label(message.get("display_metadata"))})
    return frozenset()


def messages_provenance(messages: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Union of every message's provenance: the provenance of anything produced from *messages*."""
    labels: set[str] = set()
    for message in messages or ():
        labels |= message_provenance(message)
    return frozenset(labels)


PROVENANCE_METADATA_KEY = "provenance"


def provenance_metadata(provenance: Iterable[str]) -> dict[str, Any]:
    """``{"provenance": [...]}`` for a derived row's ``display_metadata``."""
    return {PROVENANCE_METADATA_KEY: sorted(frozenset(provenance or ()) or {UNKNOWN_LABEL})}


def provenance_from_metadata(display_metadata: Any) -> Optional[frozenset[str]]:
    """A derived row's recorded provenance; None when it records none."""
    if isinstance(display_metadata, str):
        try:
            display_metadata = json.loads(display_metadata)
        except (TypeError, ValueError):
            return None
    raw = display_metadata.get(PROVENANCE_METADATA_KEY) if isinstance(display_metadata, Mapping) else None
    if not isinstance(raw, list):
        return None
    return frozenset(label if isinstance(label, str) and label else UNKNOWN_LABEL for label in raw) or None

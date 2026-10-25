"""Plugin-supplied author of an injected gateway turn (``PluginContext.inject_message(author=...)``).

A plugin relaying another agent's words into a human's chat names that agent with
``author={"id", "name"}``. The host validates both strictly (reject, never clean, truncate or
case-fold) and builds a frozen :class:`PluginAuthor`. Trust comes from WHERE that object lives —
the host-set ``MessageEvent._plugin_author`` / ``TurnContext.plugin_author`` and the turn_author
the gateway builds from it — never from the ``plugin:<plugin>/<id>`` label, which is display text.

The persisted ``display_metadata["plugin_author"]`` record is display-only: readers validate its
exact shape with :func:`display_plugin_author` and treat anything else as unknown. It never feeds
authorization, ``verified``, recall scope, the bot loop guard or a later turn's author.

A leaf module: ``hermes_cli``, ``gateway`` and ``agent`` readers all import it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional

PLUGIN_AUTHOR_KEY = "plugin_author"
PLUGIN_AUTHOR_VERSION = 1
PLUGIN_AUTHOR_ORIGIN = "plugin"
LABEL_PREFIX = "plugin:"

# ASCII by construction (explicit ranges, fullmatch): no dots, slashes, colons, spaces or Unicode.
_PLUGIN_KEY_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_AUTHOR_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
_NAME_MAX = 64
# Control, surrogate, private-use, unassigned; line/paragraph separators would break a one-line tag.
_NAME_BAD_CATEGORIES = frozenset({"Cc", "Cs", "Co", "Cn", "Zl", "Zp"})
_NAME_BAD_CHARS = frozenset("[]<>\n") | frozenset(chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)))
_AUTHOR_KEYS = frozenset({"id", "name"})
_RECORD_KEYS = frozenset({"v", "plugin", "id", "name"})


def valid_plugin_key(value: Any) -> bool:
    """A manifest key that may carry the author capability: ``^[a-z][a-z0-9_-]{0,31}$``."""
    return isinstance(value, str) and _PLUGIN_KEY_RE.fullmatch(value) is not None


def valid_author_id(value: Any) -> bool:
    """An author id: ``^[a-z0-9][a-z0-9_-]{0,47}$`` (ASCII; never truncated or case-folded)."""
    return isinstance(value, str) and _AUTHOR_ID_RE.fullmatch(value) is not None


def valid_author_name(value: Any) -> bool:
    """A display name: 1-64 code points as given, not blank, without control / separator /
    surrogate / private-use / unassigned characters, bidi controls, or any of ``[ ] < > \\n``."""
    if not isinstance(value, str) or not 1 <= len(value) <= _NAME_MAX or not value.strip():
        return False
    return not any(ch in _NAME_BAD_CHARS or unicodedata.category(ch) in _NAME_BAD_CATEGORIES for ch in value)


@dataclass(frozen=True, slots=True)
class PluginAuthor:
    """Host-built, validated author of one plugin-injected turn. Construction rejects bad values."""

    plugin: str
    id: str
    name: str

    def __post_init__(self) -> None:
        if not valid_plugin_key(self.plugin):
            raise ValueError("plugin key is not eligible for injection authors")
        if not valid_author_id(self.id):
            raise ValueError("author id must match ^[a-z0-9][a-z0-9_-]{0,47}$")
        if not valid_author_name(self.name):
            raise ValueError("author name is empty, too long or contains forbidden characters")

    @property
    def label(self) -> str:
        """``plugin:<plugin>/<id>``: display text only, never an authorization input."""
        return f"{LABEL_PREFIX}{self.plugin}/{self.id}"

    @property
    def identity(self) -> tuple[str, str]:
        """Sender identity for coalescing/security comparisons; never equal to a human's."""
        return (PLUGIN_AUTHOR_ORIGIN, self.label)

    def name_tag(self) -> str:
        """The speaker tag the gateway prepends to the injected text."""
        return f"{self.name} (agent {self.label})"

    def turn_author(self) -> dict[str, Any]:
        """The turn_author the gateway hands ``run_conversation`` for this turn."""
        return {"id": self.label, "name": self.name, "is_bot": True, "verified": False,
                "origin": PLUGIN_AUTHOR_ORIGIN, "plugin": self.plugin}

    def display_record(self) -> dict[str, Any]:
        """The display-only ``display_metadata["plugin_author"]`` record."""
        return {"v": PLUGIN_AUTHOR_VERSION, "plugin": self.plugin, "id": self.id, "name": self.name}


def build_plugin_author(plugin: Any, author: Any) -> PluginAuthor:
    """The :class:`PluginAuthor` for *plugin*'s ``author`` argument; ``ValueError`` on anything but
    a mapping with exactly ``id`` and ``name`` that pass the rules above."""
    if not isinstance(author, Mapping):
        raise ValueError("author must be a mapping with exactly 'id' and 'name'")
    keys = set(author.keys())
    if keys != _AUTHOR_KEYS:
        raise ValueError("author must have exactly the keys 'id' and 'name'")
    return PluginAuthor(plugin=plugin, id=author["id"], name=author["name"])


def _as_mapping(display_metadata: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(display_metadata, str):
        try:
            display_metadata = json.loads(display_metadata)
        except (TypeError, ValueError):
            return None
    return display_metadata if isinstance(display_metadata, Mapping) else None


def plugin_author_from_record(record: Any) -> Optional[PluginAuthor]:
    """A :class:`PluginAuthor` rebuilt from a persisted record of the exact v1 shape, else None.
    For DISPLAY only: a record proves nothing about who wrote the row."""
    if not isinstance(record, Mapping) or set(record.keys()) != _RECORD_KEYS:
        return None
    if record.get("v") != PLUGIN_AUTHOR_VERSION or isinstance(record.get("v"), bool):
        return None
    try:
        return PluginAuthor(plugin=record["plugin"], id=record["id"], name=record["name"])
    except ValueError:
        return None


def display_plugin_author(display_metadata: Any) -> Optional[PluginAuthor]:
    """The plugin author a stored row DISPLAYS, or None: a valid v1 record and no ``sender``.
    A row carrying both is ambiguous and reads as unknown."""
    meta = _as_mapping(display_metadata)
    if meta is None or PLUGIN_AUTHOR_KEY not in meta or "sender" in meta:
        return None
    return plugin_author_from_record(meta.get(PLUGIN_AUTHOR_KEY))


def gateway_turn_author(raw: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Re-validate a gateway-built plugin turn_author (``origin: "plugin"``); exact dict or None."""
    plugin, ident = raw.get("plugin"), raw.get("id")
    if not isinstance(plugin, str) or not isinstance(ident, str) or not ident.startswith(f"{LABEL_PREFIX}{plugin}/"):
        return None
    if raw.get("is_bot") is not True or raw.get("verified") is not False:
        return None
    try:
        author = PluginAuthor(plugin=plugin, id=ident[len(LABEL_PREFIX) + len(plugin) + 1:], name=raw.get("name"))
    except ValueError:
        return None
    return author.turn_author()


def event_plugin_author(obj: Any) -> Optional[PluginAuthor]:
    """The host :class:`PluginAuthor` carried by a live event (``_plugin_author``) or turn context
    (``plugin_author``), else None. Only the exact host type counts (no duck-typed look-alikes)."""
    for attr in ("_plugin_author", "plugin_author"):
        author = getattr(obj, attr, None) if obj is not None else None
        if type(author) is PluginAuthor:
            return author
    return None

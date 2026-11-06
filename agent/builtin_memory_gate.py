"""``filter_builtin_memory``: a plugin gate on the built-in memory store (MEMORY.md / USER.md).

Core injects the curated notes into every session's system prompt and serves them through the
``memory`` tool. A plugin that knows who a chat's audience is (a household/privacy plugin) can keep
them out of chats that should not see them:

    def filter_builtin_memory(session_id, platform, chat_id, chat_type, sender_id, **kwargs):
        return "omit"  # or "allow" / None (no opinion)

Contract:
- No callback registered: unchanged behaviour (the hook is not even invoked).
- Any callback answering ``"omit"`` omits; ``"allow"`` and ``None`` allow. Fail closed: a callback
  that raises, times out (``plugins.hook_callback_timeout``) or returns anything else omits, and so
  does a failure to dispatch the hook at all.
- Per-entry filter (circles): a callback may instead return ``{"keep": keep}``, where
  ``keep(tags: frozenset[str]) -> bool`` decides one entry by its circle tags. An entry carries tags
  when its first line starts with a marker, ``[circle: <tag>[, <tag>...]]`` (plain text, so the
  files stay readable); an entry without one gets ``frozenset()`` and the plugin decides what that
  means. ``keep`` must return ``True`` to keep an entry: anything else, or a raise, hides it. With
  several callbacks an entry is kept only if every ``keep`` keeps it, and any ``"omit"`` still
  omits everything. A dict that is not a valid verdict (no callable ``keep`` and no valid
  ``write_circle``, e.g. the block directive a timeout produces) omits.
- System prompt (``surface="system_prompt"``): evaluated each time the prompt is built, i.e. once per
  session and again at a rebuild boundary (context compression). The prompt is cached per session
  (prompt-cache invariant), so a mid-session audience change takes effect at the next rebuild or
  session; a resumed session reuses its stored prompt bytes and the decision baked into them. On
  omit the block is replaced by :data:`OMITTED_NOTE`. With ``keep`` only the kept entries are
  rendered, marker included (the model sees each entry's circle); the usage figure still counts the
  whole file.
- ``memory`` tool (``surface="memory_tool"``, or ``"memory_tool_write"`` when the call adds an entry):
  evaluated on every call; on omit the tool refuses without touching the store, because its results
  echo stored entries (``current_entries``, ``matches``) and its writes would mix this chat's content
  into the curated notes. With ``keep``, hidden entries are invisible to the call: never echoed,
  never matched by ``patch``/``remove``, never changed; char limits still count the whole file. A
  patch may not change an entry's marker.
- Write circle: on ``"memory_tool_write"`` a dict verdict may carry ``"write_circle": "<tag>"`` (or a
  list of tags); core prefixes each added entry with that marker on its own first line. Whenever any
  callback answers with a dict, markers the model wrote at the start of an added entry are stripped
  first, because the model can't choose its own circle. Several callbacks' write circles are united
  (more tags only narrow who may see the entry). Without a dict verdict (an old plugin) adds are
  stored exactly as today.
- Gateway ``/memory`` command (``surface="memory_command"``): reviews and applies pending memory
  writes, echoing staged entry text that carries no circle yet, so it runs only when every callback
  answers ``"allow"`` (or ``None``) or no callback is registered. A dict verdict, ``"omit"`` or any
  failure refuses it with :data:`COMMAND_REFUSED_NOTE`. The payload is the command event's trusted
  source (``session_id`` is empty: a slash command has no agent session).

Surfaces: ``"system_prompt"``, ``"memory_tool"``, ``"memory_tool_write"``, ``"memory_command"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.memory_tool_store import circle_marker, entry_circles, strip_circle_markers

logger = logging.getLogger(__name__)

HOOK_NAME = "filter_builtin_memory"
ALLOW = "allow"
OMIT = "omit"
OMITTED_NOTE = "Curated notes are not shown in this chat."
COMMAND_REFUSED_NOTE = "/memory is not available in this chat."
SURFACE_SYSTEM_PROMPT = "system_prompt"
SURFACE_MEMORY_TOOL = "memory_tool"
SURFACE_MEMORY_TOOL_WRITE = "memory_tool_write"
SURFACE_MEMORY_COMMAND = "memory_command"
# Characters that would break the marker's one-line ``[circle: a, b]`` syntax.
_BAD_TAG_CHARS = frozenset("[],\n\r")

__all__ = [
    "ALLOW", "COMMAND_REFUSED_NOTE", "OMIT", "OMITTED_NOTE", "BuiltinMemoryVerdict", "builtin_memory_allowed",
    "builtin_memory_command_allowed", "builtin_memory_verdict", "circle_marker", "entry_circles",
    "strip_circle_markers",
]


@dataclass(frozen=True)
class BuiltinMemoryVerdict:
    """The combined answer of every ``filter_builtin_memory`` callback for one surface."""
    allowed: bool
    keeps: tuple[Callable[[frozenset[str]], Any], ...] = ()
    write_circle: tuple[str, ...] = ()
    circles: bool = False  # some callback answered with a dict (circle-aware plugin)

    def keeps_entry(self, entry: str) -> bool:
        """Whether one stored entry (marker included) may be shown; fails closed."""
        if not self.allowed:
            return False
        tags = entry_circles(entry)
        for keep in self.keeps:
            try:
                if keep(tags) is not True:
                    return False
            except Exception:
                logger.warning("%s keep callback raised; omitting the entry (fail closed)", HOOK_NAME,
                               exc_info=True)
                return False
        return True

    def stamp(self, content: Any) -> Any:
        """An added entry as it is stored: model-written markers stripped, the write circle prefixed.
        Unchanged without a circle-aware verdict (old plugins keep today's behaviour)."""
        if not self.circles or not isinstance(content, str):
            return content
        body = strip_circle_markers(content)
        if not body or not self.write_circle:
            return body
        return f"{circle_marker(self.write_circle)}\n{body}"


_ALLOW_ALL = BuiltinMemoryVerdict(allowed=True)
_OMIT_ALL = BuiltinMemoryVerdict(allowed=False)


def _payload(agent: Any, surface: str) -> dict[str, Any]:
    """Trusted transport identity the agent was built with; never model- or message-supplied."""
    def _text(name: str) -> str:
        return str(getattr(agent, name, None) or "")

    return {
        "session_id": _text("session_id"), "platform": _text("platform"), "chat_id": _text("_chat_id"),
        "chat_type": _text("_chat_type"), "sender_id": _text("_user_id"), "thread_id": _text("_thread_id"),
        "session_key": _text("_gateway_session_key"), "surface": surface,
    }


def _write_circle_tags(value: Any) -> Optional[tuple[str, ...]]:
    """Validated write_circle tags, or None when malformed (fail closed)."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not items:
        return None
    tags = []
    for item in items:
        tag = item.strip() if isinstance(item, str) else ""
        if not tag or _BAD_TAG_CHARS.intersection(tag):
            return None
        tags.append(tag)
    return tuple(tags)


def _dict_verdict(result: dict) -> Optional[tuple[Optional[Callable], tuple[str, ...]]]:
    """``(keep or None, write_circle tags)`` for a valid dict answer, else None."""
    has_keep, has_circle = "keep" in result, "write_circle" in result
    if not (has_keep or has_circle):
        return None
    keep = result.get("keep") if has_keep else None
    if has_keep and not callable(keep):
        return None
    tags: tuple[str, ...] = ()
    if has_circle:
        parsed = _write_circle_tags(result.get("write_circle"))
        if parsed is None:
            return None
        tags = parsed
    return keep, tags


def builtin_memory_verdict(agent: Any, surface: str) -> BuiltinMemoryVerdict:
    """Ask every ``filter_builtin_memory`` callback about *agent*'s chat (contract above)."""
    try:
        from hermes_cli import plugins

        if not plugins.has_hook(HOOK_NAME):
            return _ALLOW_ALL
        results = plugins.invoke_hook(HOOK_NAME, **_payload(agent, surface))
    except Exception:
        logger.warning("%s dispatch failed; omitting built-in memory (fail closed)", HOOK_NAME, exc_info=True)
        return _OMIT_ALL
    keeps: list[Callable] = []
    write_circle: list[str] = []
    circles = False
    for result in results:
        if isinstance(result, dict):
            parsed = _dict_verdict(result)
            if parsed is None:
                # Timeouts and raises arrive as block directives (fail-closed hook); malformed dicts too.
                logger.warning("%s returned an invalid verdict; omitting built-in memory (fail closed)", HOOK_NAME)
                return _OMIT_ALL
            circles = True
            if parsed[0] is not None:
                keeps.append(parsed[0])
            write_circle.extend(parsed[1])
            continue
        if result == ALLOW:
            continue
        if result != OMIT:
            logger.warning("%s returned %r; omitting built-in memory (fail closed)", HOOK_NAME, result)
        return _OMIT_ALL
    if not circles:
        return _ALLOW_ALL
    return BuiltinMemoryVerdict(allowed=True, keeps=tuple(keeps),
                                write_circle=tuple(dict.fromkeys(write_circle)), circles=True)


def builtin_memory_allowed(agent: Any, surface: str) -> bool:
    """Whether the WHOLE built-in memory block may be shown for *agent*'s chat. False when a
    callback omits it or filters it per entry; callers that can filter use :func:`builtin_memory_verdict`."""
    verdict = builtin_memory_verdict(agent, surface)
    return verdict.allowed and not verdict.keeps


def builtin_memory_command_allowed(source: Any, session_key: Any = "") -> bool:
    """Whether the gateway ``/memory`` command may run for *source*'s chat (contract above): only a
    plain allow (or no callback at all); dict verdicts, omit and every failure refuse. *session_key*
    may be a zero-argument callable, resolved only when a callback is registered."""
    try:
        from hermes_cli import plugins

        if not plugins.has_hook(HOOK_NAME):
            return True

        def _text(value: Any) -> str:
            return str(getattr(value, "value", value) or "")

        payload = {
            "session_id": "", "platform": _text(getattr(source, "platform", None)),
            "chat_id": _text(getattr(source, "chat_id", None)), "chat_type": _text(getattr(source, "chat_type", None)),
            "sender_id": _text(getattr(source, "user_id", None)), "thread_id": _text(getattr(source, "thread_id", None)),
            "session_key": str((session_key() if callable(session_key) else session_key) or ""),
            "surface": SURFACE_MEMORY_COMMAND,
        }
        results = plugins.invoke_hook(HOOK_NAME, **payload)
    except Exception:
        logger.warning("%s dispatch failed; refusing /memory (fail closed)", HOOK_NAME, exc_info=True)
        return False
    refused = [result for result in results if not (isinstance(result, str) and result == ALLOW)]
    if refused:
        logger.info("%s refused /memory in this chat (%d non-allow verdicts)", HOOK_NAME, len(refused))
    return not refused

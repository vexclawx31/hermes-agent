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
- System prompt (``surface="system_prompt"``): evaluated each time the prompt is built, i.e. once per
  session and again at a rebuild boundary (context compression). The prompt is cached per session
  (prompt-cache invariant), so a mid-session audience change takes effect at the next rebuild or
  session; a resumed session reuses its stored prompt bytes and the decision baked into them. On
  omit the block is replaced by :data:`OMITTED_NOTE`.
- ``memory`` tool (``surface="memory_tool"``): evaluated on every call; on omit the tool refuses
  without touching the store, because its results echo stored entries (``current_entries``,
  ``matches``) and its writes would mix this chat's content into the curated notes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

HOOK_NAME = "filter_builtin_memory"
ALLOW = "allow"
OMIT = "omit"
OMITTED_NOTE = "Curated notes are not shown in this chat."


def _payload(agent: Any, surface: str) -> dict[str, Any]:
    """Trusted transport identity the agent was built with; never model- or message-supplied."""
    def _text(name: str) -> str:
        return str(getattr(agent, name, None) or "")

    return {
        "session_id": _text("session_id"), "platform": _text("platform"), "chat_id": _text("_chat_id"),
        "chat_type": _text("_chat_type"), "sender_id": _text("_user_id"), "thread_id": _text("_thread_id"),
        "session_key": _text("_gateway_session_key"), "surface": surface,
    }


def builtin_memory_allowed(agent: Any, surface: str) -> bool:
    """Whether the built-in memory block may be shown for *agent*'s chat (contract above)."""
    try:
        from hermes_cli import plugins

        if not plugins.has_hook(HOOK_NAME):
            return True
        results = plugins.invoke_hook(HOOK_NAME, **_payload(agent, surface))
    except Exception:
        logger.warning("%s dispatch failed; omitting built-in memory (fail closed)", HOOK_NAME, exc_info=True)
        return False
    for result in results:
        if result == ALLOW:
            continue
        if result != OMIT:
            # Timeouts and raises arrive as block directives (fail-closed hook); unknown values too.
            logger.warning("%s returned %r; omitting built-in memory (fail closed)", HOOK_NAME, result)
        return False
    return True

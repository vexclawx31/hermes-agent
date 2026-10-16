"""Turn-end guard against stopping approved work to ask permission (``agent.stop_guard``).

Opt-in and deliberately conservative. It fires only when ALL hold:

* this turn already did tool work (the agent was mid-task, not answering a fresh question)
  and no tool result this turn was an approval BLOCKED (then waiting for the human is right);
* the reply's final paragraph is a permission/continuation question ("Shall I…?",
  "Want me to…?", "Should I proceed?");
* that paragraph cites no red-tier decision (credentials/OAuth, live deploy/restart, money,
  external communication, irreversible deletion, cross-tenant access, disabling safety);
* nothing is still running for the session (background process or async delegation), since
  then stopping to wait is correct;
* fewer than ``_MAX_NUDGES`` nudges were issued this turn.

Policy-only: it returns the nudge text; ``agent/turn_stop_gates.py`` appends it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger("agent.conversation_loop")

_MAX_NUDGES = 1
_TAIL_CHARS = 600

STOP_GUARD_NUDGE = (
    "[System: Continue the approved work now; do not stop to ask permission for routine "
    "steps. Ask only for red-tier decisions: credentials or OAuth consent, live service "
    "deploy/restart, spending money, external communications, irreversible deletion, "
    "cross-tenant access, or disabling safety controls. If the next step genuinely needs one "
    "of those, name it and stop.]"
)

_PERMISSION_QUESTION = re.compile(
    r"\b(?:shall|should|may|can|could)\s+i\b"
    r"|\b(?:shall|should)\s+we\s+(?:proceed|continue|go\s+ahead|start)\b"
    r"|\b(?:do\s+you\s+)?want\s+me\s+to\b"
    r"|\bwould\s+you\s+like\s+me\s+to\b"
    r"|\b(?:ok(?:ay)?|good)\s+to\s+(?:proceed|continue|go\s+ahead)\b"
    r"|\b(?:proceed|continue|go\s+ahead)\s*\?"
    r"|\bapprove\b[^?\n]*\?",
    re.IGNORECASE,
)

_RED_TIER = re.compile(
    r"red[\s-]?tier|credential|oauth|password|api[\s_-]?key|\btoken|secret|\bconsent\b"
    r"|\bdeploy|\brestart|\bproduction\b|\bprod\b|\blive\s+(?:service|server|system|site)"
    r"|\bpay(?:ment)?\b|\bmoney\b|\bpurchase|\bbilling|\bcharge|\$\s?\d"
    r"|\bsend\b[^.?\n]*\b(?:email|message|tweet|post)|\bpublish|\bemail\b|\bpost\s+(?:to|on)\b"
    r"|\birreversibl|\bpermanent(?:ly)?\b|\bdelet|\bwipe|\bdrop\s+(?:the\s+)?(?:table|database)"
    r"|force[\s-]push|\btenant|\bdisabl|\bbypass|\bsudo\b|\bmerge\b",
    re.IGNORECASE,
)


def stop_guard_enabled(config: dict[str, Any] | None = None) -> bool:
    """``agent.stop_guard`` — only a literal ``true`` enables it."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            return False
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    return isinstance(agent_cfg, dict) and agent_cfg.get("stop_guard") is True


def _final_paragraph(text: str) -> str:
    tail = (text or "").strip()[-_TAIL_CHARS:]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", tail) if p.strip()]
    return paragraphs[-1] if paragraphs else ""


def ends_with_permission_question(text: str) -> bool:
    """The last paragraph asks for permission to continue and cites no red-tier decision."""
    para = _final_paragraph(text)
    if "?" not in para or not _PERMISSION_QUESTION.search(para):
        return False
    return not _RED_TIER.search(para)


def _is_synthetic(msg: dict) -> bool:
    return any(k.startswith("_") and k.endswith(("_synthetic", "_nudge")) and msg.get(k) for k in msg)


def turn_did_tool_work(messages: Iterable[dict] | None) -> bool:
    """An assistant tool call since the last real (non-synthetic) user message."""
    rows = [m for m in (messages or ()) if isinstance(m, dict)]
    if not any(m.get("role") == "user" and not _is_synthetic(m) for m in rows):
        return False
    return any(m.get("role") == "assistant" and m.get("tool_calls") for m in _this_turn(rows))


def _this_turn(messages: Iterable[dict] | None) -> list[dict]:
    """Messages after the last real (non-synthetic) user message."""
    rows = [m for m in (messages or ()) if isinstance(m, dict)]
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("role") == "user" and not _is_synthetic(rows[i]):
            return rows[i + 1:]
    return rows


def turn_hit_approval_block(messages: Iterable[dict] | None) -> bool:
    """A tool result this turn was an approval BLOCKED (denied / timed out / no consent): the
    agent was told to wait for the human, so stopping to ask is the correct ending."""
    for msg in _this_turn(messages):
        if msg.get("role") == "tool" and "BLOCKED" in str(msg.get("content") or ""):
            return True
    return False


def _work_still_running(agent: Any) -> bool:
    """A background process or async delegation for this session is still live. Probe
    failures count as running — the guard then stays out of the way."""
    try:
        from tools.approval_context import get_current_session_key
        from tools.async_delegation import has_live_for_session
        from tools.process_registry import process_registry

        session_key = get_current_session_key() or ""
        task_id = getattr(agent, "_current_task_id", None) or ""
        if task_id and process_registry.has_active_processes(task_id):
            return True
        if session_key and process_registry.has_active_for_session(session_key):
            return True
        return has_live_for_session(session_key=session_key,
                                    parent_session_id=getattr(agent, "session_id", "") or "")
    except Exception:
        logger.debug("stop guard: running-work probe failed", exc_info=True)
        return True


def build_stop_guard_nudge(agent: Any, final_response: Any, messages: Iterable[dict] | None) -> str | None:
    """The continuation nudge, or ``None`` when the guard should not fire."""
    if getattr(agent, "_stop_guard_nudges", 0) >= _MAX_NUDGES or not isinstance(final_response, str):
        return None
    if not stop_guard_enabled():
        return None
    if not ends_with_permission_question(final_response) or not turn_did_tool_work(messages):
        return None
    if turn_hit_approval_block(messages):
        return None
    if _work_still_running(agent):
        return None
    return STOP_GUARD_NUDGE


__all__ = ["STOP_GUARD_NUDGE", "build_stop_guard_nudge", "ends_with_permission_question", "stop_guard_enabled"]

"""Tool refusals seen during one unattended run (generic: every tool, every platform).

A scheduled run has nobody watching it. When a tool is refused (a permission, access or approval check
said no) and the run then ends silent, the job's owner would never learn why nothing came. The cron run
scope starts a collection here; the agent's single terminal ``post_tool_call`` emit point reports every
tool result to :func:`observe`, which keeps the refusals.

A tool result counts as a refusal when its JSON object says so:

- ``"refused": true`` (the explicit contract for tools and plugins; add a plain ``error`` saying why), or
- ``"status": "blocked"`` (approval and guardrail blocks, e.g. a dangerous command in cron), or
- the dispatcher reports ``error_type="plugin_block"`` (a ``pre_tool_call`` veto).

The collection is a list placed in the context BEFORE the agent's context is copied into its worker
threads, so appends made inside tool threads are visible to the run when it finishes.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_REFUSALS = 10
_REASON_CHARS = 300
_RUN: ContextVar[Optional[list[dict[str, str]]]] = ContextVar("hermes_tool_refusals", default=None)


def start() -> Token:
    """Begin collecting refusals for the current run; pass the token to :func:`stop`."""
    return _RUN.set([])


def stop(token: Token) -> list[dict[str, str]]:
    """End the collection and return what it saw (oldest first, deduplicated)."""
    seen = list(_RUN.get() or [])
    _RUN.reset(token)
    return seen


def refusal_reason(result: Any, *, error_type: Optional[str] = None) -> Optional[str]:
    """The refusal's reason when this tool result is a refusal, else None."""
    data: Any = result
    if isinstance(result, str):
        text = result.strip()
        if not text.startswith("{") or ("refused" not in text and "blocked" not in text):
            data = None if error_type != "plugin_block" else {"error": text}
        else:
            try:
                data = json.loads(text)
            except ValueError:
                data = None
    if not isinstance(data, dict):
        return "the tool call was blocked" if error_type == "plugin_block" else None
    if not (data.get("refused") is True or data.get("status") == "blocked" or error_type == "plugin_block"):
        return None
    reason = data.get("error") or data.get("message") or data.get("reason") or "no reason given"
    return " ".join(str(reason).split())[:_REASON_CHARS]


def observe(tool_name: str, result: Any, *, error_type: Optional[str] = None) -> None:
    """Record ``result`` if a collection is active and it is a refusal. Never raises."""
    seen = _RUN.get()
    if seen is None:
        return
    try:
        reason = refusal_reason(result, error_type=error_type)
    except Exception:
        logger.debug("tool refusal check failed for %s", tool_name, exc_info=True)
        return
    if not reason or len(seen) >= MAX_REFUSALS:
        return
    entry = {"tool": str(tool_name or "a tool"), "reason": reason}
    if entry not in seen:
        seen.append(entry)

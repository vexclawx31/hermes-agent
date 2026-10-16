"""Late approvals: a human answering ``/approve`` after a gateway approval card timed out.

A timed-out card is still a hard deny for the waiting tool call (silence is not consent). But a
human who approves a few minutes later has consented to that exact command, so the gateway records
the late answer as a one-shot grant: the next time the agent runs the byte-identical command in the
same session, within ``approvals.late_approval_window`` seconds, the human prompt is skipped once.

The grant sits AFTER the unconditional floors (hardline blocklist, sudo stdin guard,
``approvals.deny``) — the caller consumes it only where it would otherwise ask a human. It never
persists a pattern (no session/always scope), and each timed-out command can be granted once.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from tools import approval_context as _ctx

logger = logging.getLogger("tools.approval")

_DEFAULT_WINDOW_SECONDS = 1800
_MAX_SLOTS_PER_SESSION = 3


@dataclass
class _LateSlot:
    command: str
    description: str
    timed_out_at: float
    granted_at: float | None = None


_lock = threading.Lock()
_slots: dict[str, list[_LateSlot]] = {}


def late_approval_window() -> float:
    """``approvals.late_approval_window`` in seconds; 0 disables late approvals."""
    raw = _ctx._get_approval_config().get("late_approval_window", _DEFAULT_WINDOW_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("approvals.late_approval_window=%r is not a number of seconds; using %ss",
                       raw, _DEFAULT_WINDOW_SECONDS)
        return float(_DEFAULT_WINDOW_SECONDS)
    return max(value, 0.0)


def _live(slots: list[_LateSlot], now: float, window: float) -> list[_LateSlot]:
    return [s for s in slots if now - (s.granted_at or s.timed_out_at) <= window]


def record_timeout(session_key: str, command: str, description: str) -> bool:
    """Remember a command whose approval card timed out. Returns False when disabled."""
    window = late_approval_window()
    if not session_key or not command or window <= 0:
        return False
    now = time.monotonic()
    with _lock:
        slots = [s for s in _live(_slots.get(session_key, []), now, window) if s.command != command]
        slots.append(_LateSlot(command=command, description=description, timed_out_at=now))
        _slots[session_key] = slots[-_MAX_SLOTS_PER_SESSION:]
    return True


def grant_latest(session_key: str) -> _LateSlot | None:
    """Turn the most recent un-granted timed-out command into a one-shot grant."""
    window = late_approval_window()
    if not session_key or window <= 0:
        return None
    now = time.monotonic()
    with _lock:
        slots = _live(_slots.get(session_key, []), now, window)
        _slots[session_key] = slots
        for slot in reversed(slots):
            if slot.granted_at is None:
                slot.granted_at = now
                logger.info("Late approval granted for session %s: %s", session_key, slot.command[:120])
                return slot
    return None


def consume_grant(session_key: str, command: str) -> bool:
    """True (once) when *command* exactly matches a live late grant for this session."""
    window = late_approval_window()
    if not session_key or window <= 0:
        return False
    now = time.monotonic()
    with _lock:
        slots = _live(_slots.get(session_key, []), now, window)
        for slot in slots:
            if slot.granted_at is not None and slot.command == command:
                slots.remove(slot)
                _slots[session_key] = slots
                return True
        _slots[session_key] = slots
    return False


def clear(session_key: str) -> None:
    with _lock:
        _slots.pop(session_key, None)

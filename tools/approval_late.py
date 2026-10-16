"""Late approvals: a human answering ``/approve`` after a gateway approval card timed out.

A timed-out card is still a hard deny for the waiting tool call (silence is not consent). But a
human who approves a few minutes later has consented to that exact command, so the gateway turns
the late answer into a one-shot grant bound to the wake turn it queues:

* one slot per session — only the MOST RECENT timed-out command; a repeated ``/approve`` never
  reaches an older one;
* the grant expires ``approvals.late_approval_window`` seconds after the original TIMEOUT;
* it is usable only inside the wake turn (``arm`` when that turn starts, ``disarm`` when it ends,
  consumed or not), and only by the byte-identical command.

It sits AFTER the unconditional floors (hardline blocklist, sudo stdin guard, ``approvals.deny``)
— the caller consumes it only where it would otherwise ask a human — and never persists a pattern.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass

from tools import approval_context as _ctx

logger = logging.getLogger("tools.approval")

_DEFAULT_WINDOW_SECONDS = 1800


@dataclass
class _LateSlot:
    command: str
    description: str
    timed_out_at: float
    token: str | None = None   # set once granted; carried by the wake event's metadata
    armed: bool = False        # True only while the wake turn runs


_lock = threading.Lock()
_slots: dict[str, _LateSlot] = {}


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


def _live_slot_locked(session_key: str, window: float) -> _LateSlot | None:
    slot = _slots.get(session_key)
    if slot is not None and (window <= 0 or time.monotonic() - slot.timed_out_at > window):
        _slots.pop(session_key, None)
        return None
    return slot


def record_timeout(session_key: str, command: str, description: str) -> bool:
    """Remember the command whose approval card just timed out, replacing an older UN-granted one
    (a live grant waiting for its wake turn is kept). Returns False when nothing was recorded."""
    window = late_approval_window()
    if not session_key or not command or window <= 0:
        return False
    with _lock:
        current = _live_slot_locked(session_key, window)
        if current is not None and current.token is not None:
            return False
        _slots[session_key] = _LateSlot(command=command, description=description,
                                        timed_out_at=time.monotonic())
    return True


def grant_latest(session_key: str) -> _LateSlot | None:
    """Grant the session's most recent timed-out command, once. ``None`` when there is none,
    it expired, or it was already granted."""
    window = late_approval_window()
    if not session_key:
        return None
    with _lock:
        slot = _live_slot_locked(session_key, window)
        if slot is None or slot.token is not None:
            return None
        slot.token = secrets.token_hex(8)
        logger.info("Late approval granted for session %s: %s", session_key, slot.command[:120])
        return slot


def revoke(session_key: str, token: str) -> None:
    """Withdraw a grant whose wake turn could not be queued."""
    with _lock:
        slot = _slots.get(session_key)
        if slot is not None and slot.token == token:
            _slots.pop(session_key, None)


def arm(token: str) -> bool:
    """The wake turn carrying *token* is starting: its grant becomes usable."""
    window = late_approval_window()
    with _lock:
        for key, slot in list(_slots.items()):
            if slot.token == token:
                if _live_slot_locked(key, window) is None:
                    return False
                slot.armed = True
                return True
    return False


def disarm(token: str) -> None:
    """The wake turn ended: drop its grant whether or not it was used."""
    with _lock:
        for key, slot in list(_slots.items()):
            if slot.token == token:
                _slots.pop(key, None)


def consume_grant(session_key: str, command: str) -> bool:
    """True (once) when *command* exactly matches the armed grant for this session."""
    window = late_approval_window()
    if not session_key:
        return False
    with _lock:
        slot = _live_slot_locked(session_key, window)
        if slot is None or not slot.armed or slot.command != command:
            return False
        _slots.pop(session_key, None)
        return True


def clear(session_key: str) -> None:
    with _lock:
        _slots.pop(session_key, None)

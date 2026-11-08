"""Owned tabs for ``browser_exec`` on a SHARED external browser: a CDP override (live ``/browser connect``,
``BROWSER_CDP_URL``, ``browser.cdp_url``), typically a person's own Chrome.

Unguarded, the harness daemon attaches to the first existing tab (a person's, or another agent's), every
tab the session opens outlives it, and the CDP supervisor attaches to the first page too. Instead each
browser_exec route (its session cache key) works only in tabs it opened:

* Hermes opens the route's first tab itself (``Target.createTarget``) and records only that targetId, plus
  the tabs the route's own code opened with ``new_tab()`` that did not exist before the call;
* every call is pinned to the route's current owned tab before the model's code runs; the pin preamble
  refuses ``switch_tab`` / ``close_tab`` on a tab the route did not open, and exits before the model's code
  when it cannot pin;
* a gone owned tab is replaced by a NEW owned one, never by another tab; when none can be opened (or the
  browser's tabs cannot be read) the call is refused;
* the CDP supervisor (dialogs, vault fill) attaches only to the owned tab, never before it exists;
* teardown (``cleanup_browser``, the inactivity janitor, force-reap, ``/browser connect``) closes exactly
  the recorded tabs and stops the route's harness daemon (which closes the dedicated tab it opened itself).
  Pre-existing, foreign and unrecorded tabs are never closed.

The record lives on the route's ``tools.browser_session`` entry (``features.cdp_override``), so the v0.23
session lifecycle drives teardown (``browser_session_lifecycle.release_session_backend``).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

_CDP_TIMEOUT_S = 10.0
_CLOSE_TIMEOUT_S = 5.0

_route_locks: dict[str, threading.Lock] = {}
_route_locks_guard = threading.Lock()


class Binding(NamedTuple):
    """One call's owned-tab binding: the tab it is pinned to, every owned tab, and the page targets that
    existed when it was bound (a tab the call reports as opened must not be one of them)."""

    pin: str
    owned: tuple[str, ...]
    preexisting: frozenset


def _cdp(ws_url: str, method: str, params: Optional[dict[str, Any]] = None,
         timeout: float = _CDP_TIMEOUT_S) -> dict[str, Any]:
    """One browser-level CDP call on ``ws_url`` (a fresh connection; never a page session)."""
    from tools.browser_cdp_tool import _cdp_call, _run_async
    return _run_async(_cdp_call(ws_url, method, params or {}, None, timeout))


def _page_ids(ws_url: str) -> set[str]:
    infos = _cdp(ws_url, "Target.getTargets").get("targetInfos") or []
    return {str(t["targetId"]) for t in infos if t.get("type") == "page" and t.get("targetId")}


def _route_lock(route_key: str) -> threading.Lock:
    with _route_locks_guard:
        return _route_locks.setdefault(route_key, threading.Lock())


def is_shared_cdp_session(record: dict[str, Any]) -> bool:
    return bool((record.get("features") or {}).get("cdp_override"))


def owned_pin(record: dict[str, Any]) -> Optional[str]:
    """The tab a shared-browser session works in: its current owned tab, else its newest one, else None."""
    owned = list(record.get("owned_target_ids") or ())
    current = record.get("current_target_id")
    if current and current in owned:
        return current
    return owned[-1] if owned else None


def state_path(route_key: str) -> str:
    """Where the pin preamble reports the tabs a call opened and the tab it ended on."""
    import hashlib

    from hermes_constants import get_hermes_home
    directory = Path(get_hermes_home()) / "cache" / "browser-use" / "owned-tabs"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"{hashlib.sha256(route_key.encode()).hexdigest()[:16]}.json")


def bind(route_key: str, daemon: tuple[str, ...]) -> tuple[Optional[Binding], Optional[str]]:
    """Pin this call of ``route_key`` to a tab it owns → ``(binding, None)`` or ``(None, refusal)``.

    Reuses the route's live owned tab; if it is gone (or the route has none yet) opens a new one and records
    it. Never falls back to another tab: when the tabs cannot be read or no tab can be opened, refuses."""
    from tools import browser_session as bs
    with _route_lock(route_key):
        try:
            record = bs.get_session_info(route_key)
        except Exception as e:  # health: allow BLE001 -- reported to the model as a refusal
            return None, f"Could not open the shared browser session: {e}"
        cdp_url = str(record.get("cdp_url") or "")
        if not is_shared_cdp_session(record) or not cdp_url:
            return None, ("The browser endpoint changed while this call started (/browser connect or "
                          "disconnect); retry the call.")
        try:
            alive = _page_ids(cdp_url)
        except Exception as e:  # health: allow BLE001 -- fail closed: never guess which tab is ours
            logger.warning("browser_exec: could not list tabs for %s: %s", route_key, e)
            return None, ("Could not read the shared browser's tabs to find this session's own tab; "
                          "not using another tab instead. Retry the call.")
        with bs._cleanup_lock:
            owned = [t for t in record.get("owned_target_ids") or () if t in alive]  # a gone tab needs no close
            record["owned_target_ids"] = owned
            record["harness_daemon"] = tuple(daemon)
            pin = owned_pin(record)
        if pin is None:
            try:
                pin = str(_cdp(cdp_url, "Target.createTarget", {"url": "about:blank", "background": True})["targetId"])
            except Exception as e:  # health: allow BLE001 -- refuse rather than run on another tab
                logger.warning("browser_exec: could not open an owned tab for %s: %s", route_key, e)
                return None, ("Could not open this session's own tab in the shared browser; not using "
                              "another tab instead. Retry the call.")
            with bs._cleanup_lock:
                live = bs._active_sessions.get(route_key) is record
                if live:
                    record["owned_target_ids"] = [*record.get("owned_target_ids", ()), pin]
            if not live:  # torn down meanwhile: nobody would close it later, so close it now
                _close(cdp_url, pin)
                return None, "The browser session was closed while opening its tab; retry the call."
            logger.debug("browser_exec: %s owns tab %s", route_key, pin)
        with bs._cleanup_lock:
            record["current_target_id"] = pin
            owned_now = tuple(record.get("owned_target_ids") or ())
    return Binding(pin, owned_now, frozenset(alive)), None


def record_call(route_key: str, binding: Binding, path: str) -> None:
    """Adopt what the pin preamble reported for this call: tabs it opened (only ones that did not exist when
    the call was bound) and the owned tab it ended on (the next call's pin)."""
    from tools import browser_session as bs
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        Path(path).unlink(missing_ok=True)
    except (OSError, ValueError):
        return
    if not isinstance(state, dict):
        return
    opened = [str(t) for t in state.get("opened") or () if isinstance(t, str) and t]
    with bs._cleanup_lock:
        record = bs._active_sessions.get(route_key)
        if record is None or not is_shared_cdp_session(record):
            return  # torn down during the call: its teardown closed what it knew
        owned = list(record.get("owned_target_ids") or ())
        owned += [t for t in opened if t not in owned and t not in binding.preexisting]
        record["owned_target_ids"] = owned
        current = state.get("current")
        if isinstance(current, str) and current in owned:
            record["current_target_id"] = current


def _close(cdp_url: str, target_id: str) -> None:
    try:
        _cdp(cdp_url, "Target.closeTarget", {"targetId": target_id}, timeout=_CLOSE_TIMEOUT_S)
    except Exception as e:  # health: allow BLE001 -- teardown: an already-closed tab is fine
        logger.debug("closing owned tab %s failed: %s", target_id, e)


def release_owned_tabs(record: dict[str, Any]) -> None:
    """Teardown for a shared-browser session record: close exactly its recorded tabs, then stop its harness
    daemon (whose shutdown closes the dedicated tab the daemon opened). Never touches any other tab."""
    cdp_url = str(record.get("cdp_url") or "")
    for target_id in list(record.get("owned_target_ids") or ()):
        if cdp_url:
            _close(cdp_url, target_id)
    daemon = record.get("harness_daemon")
    if daemon:
        try:
            from tools.browser_use_cli import stop_harness_daemon
            stop_harness_daemon(tuple(daemon))
        except Exception as e:  # health: allow BLE001 -- teardown must never abort
            logger.debug("stopping the harness daemon for an owned-tab session failed: %s", e)

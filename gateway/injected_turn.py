"""Task-local context for one plugin-authored injected turn, bound by the host only.

A plugin asks for it with ``inject_message(..., author=..., turn={...})`` and the plugin entry's
``allow_injected_turn_context: true``. The host validates it into a frozen :class:`TurnSpec`, attaches it to
the injected event it builds itself (``_plugin_injection_event``), and binds an :class:`InjectedTurn` while
that event's turn runs (``GatewayRunner._set_session_env``). It is revoked when the turn's session context is
cleared, so a child task that copied the ContextVar sees nothing afterwards.

It confers no human identity: it is not TrustedIngress, and tools that require a native human turn must keep
using ``gateway.ingress_identity.get_trusted_ingress()``. An ``agent_origin`` turn has no user at all; it is
admitted by the target profile's explicit ``gateway.agent_origin`` grant (``run_plugin_injection``).

This is an in-process trust boundary, not isolation from arbitrary Python code in the gateway.
"""
from __future__ import annotations

import re
import threading
import weakref
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

INJECTED_TURN_API_VERSION = 1
KINDS = frozenset({"result", "message"})
_REF = re.compile(r"hd-[0-9a-f]{16}\Z")
_FIELDS = frozenset({"kind", "ref", "agent_origin", "resume"})


@dataclass(frozen=True)
class TurnSpec:
    """The validated request: which plugin, what kind of Switchboard event, its ref, and whether the turn has
    no human (agent origin)."""
    plugin_id: str
    kind: str
    ref: str
    agent_origin: bool
    resume: Any = None  # a signed resume marker (gateway.deferred); verified by the host at dispatch, never trusted


def build_turn_spec(plugin_id: str, turn: Any) -> TurnSpec:
    """A TurnSpec from a plugin's ``turn`` mapping; ValueError on anything else (refused, never cleaned up)."""
    if not isinstance(turn, Mapping) or set(turn) - _FIELDS or not {"kind", "ref"} <= set(turn):
        raise ValueError("turn takes exactly kind, ref and optional agent_origin")
    kind, ref, agent_origin = turn["kind"], turn["ref"], turn.get("agent_origin", False)
    if type(kind) is not str or kind not in KINDS:
        raise ValueError("turn.kind must be 'result' or 'message'")
    if type(ref) is not str or not _REF.fullmatch(ref):
        raise ValueError("turn.ref must be a Switchboard ref (hd-<16 hex>)")
    if type(agent_origin) is not bool:
        raise ValueError("turn.agent_origin must be a bool")
    resume = turn.get("resume")
    if resume is not None:
        if not isinstance(resume, Mapping) or agent_origin:
            raise ValueError("turn.resume is a resume marker, on a turn with a human origin")
        import json
        resume = json.loads(json.dumps(dict(resume)))  # a plain, frozen-by-copy value
    return TurnSpec(plugin_id, kind, ref, agent_origin, resume)


class InjectedTurn:
    """What a tool may read about the turn it runs in. Read-only; ``live`` only between bind and revoke."""
    __slots__ = ("_spec", "_event", "_live", "_lock", "_resume", "_on_revoke", "__weakref__")

    def __init__(self, spec: TurnSpec, event: Any, resume: Optional[str] = None, on_revoke=None):
        self._spec = spec
        self._event = weakref.ref(event)
        self._live = True
        self._lock = threading.Lock()
        self._resume = resume
        self._on_revoke = on_revoke

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live and self._event() is not None

    def revoke(self, release: bool = True) -> None:
        """End this context. *release* (turn end) also runs the on-revoke hook (an activated deferred receipt
        goes back to noticed unless claimed); a re-bind at the start of a session scope doesn't."""
        with self._lock:
            was, self._live = self._live, False
            hook, self._on_revoke = (self._on_revoke, None) if release else (None, self._on_revoke)
        if was and hook is not None:
            try:
                hook()
            except Exception:
                pass

    @property
    def resume(self) -> Optional[str]:
        """The deferred-action receipt this turn activated (host-verified marker), or None."""
        return self._resume

    plugin_id = property(lambda self: self._spec.plugin_id)
    kind = property(lambda self: self._spec.kind)
    ref = property(lambda self: self._spec.ref)
    agent_origin = property(lambda self: self._spec.agent_origin)


_INJECTED_TURN: ContextVar[Optional[InjectedTurn]] = ContextVar("plugin_injected_turn", default=None)
_EVENT_ATTR = "_hermes_injected_turn_spec"


_RESUME_ATTR = "_hermes_injected_turn_resume"


def attach_to_event(event: Any, spec: TurnSpec, resume: Optional[tuple] = None) -> None:
    """Host only: mark the event the host built for this injection (never a copy, never a restored event).
    *resume* is (receipt id, release callback) after the host activated a receipt for this event."""
    setattr(event, _EVENT_ATTR, spec)
    if resume is not None:
        receipt, release = resume
        setattr(event, _RESUME_ATTR, (receipt, release))
        # Only a live InjectedTurn of THIS event can redeem (revoked when its turn ends), so no other turn ever
        # can. When the event itself is gone, an unclaimed receipt goes back to noticed (needs a new marker);
        # gateway start recovers the same way.
        weakref.finalize(event, release)


_ROW_RESUME_ATTR = "_hermes_injected_turn_row_resume"
_RECORD_FIELDS = frozenset({"plugin", "kind", "ref", "agent_origin", "resume"})


def mark_row_resume(event: Any, receipt: str, home: Any) -> None:
    """Host only, on the durable-FIFO path: the deferred receipt the host activated for this event rides its
    admission row (``row_record``) instead of a finalizer on this transient event, which is dropped as soon as
    the row commits; the restored event re-binds it (``restore_record``)."""
    setattr(event, _ROW_RESUME_ATTR, {"id": str(receipt), "home": str(home)})


def row_record(event: Any) -> Optional[dict]:
    """The host-built injected-turn context of *event* as its durable admission row keeps it, or None."""
    spec = spec_of_event(event)
    if spec is None:
        return None
    record = {"plugin": spec.plugin_id, "kind": spec.kind, "ref": spec.ref, "agent_origin": spec.agent_origin}
    resume = getattr(event, _ROW_RESUME_ATTR, None)
    if isinstance(resume, dict) and not spec.agent_origin:
        record["resume"] = dict(resume)
    return record


def restore_record(event: Any, record: Any, runner: Any = None) -> None:
    """Re-attach a row's injected-turn context to the event restored from that row (the host's own commit of
    this turn). Validated again exactly as ``build_turn_spec``; it needs the restored plugin author of the same
    plugin. Anything else attaches nothing: the turn runs with no context, never a widened one."""
    from agent.plugin_author import event_plugin_author, valid_plugin_key
    author = event_plugin_author(event)
    try:
        if not isinstance(record, Mapping) or set(record) - _RECORD_FIELDS or not valid_plugin_key(record.get("plugin")):
            raise ValueError("malformed injected-turn record")
        if author is None or author.plugin != record["plugin"]:
            raise ValueError("an injected-turn context needs the row's plugin author")
        spec = build_turn_spec(record["plugin"], {"kind": record.get("kind"), "ref": record.get("ref"),
                                                  "agent_origin": record.get("agent_origin", False)})
    except (KeyError, ValueError):
        import logging
        logging.getLogger(__name__).warning("Injected-turn context on an admission row refused; the turn runs "
                                            "without it")
        return
    resume = None
    raw = record.get("resume")
    if isinstance(raw, Mapping) and not spec.agent_origin and type(raw.get("id")) is str and type(raw.get("home")) is str:
        from pathlib import Path
        home = Path(raw["home"])
        served = getattr(runner, "_served_profile_for_home", None) if runner is not None else None
        if served is not None and served(home) is not None:
            from gateway import deferred
            receipt = raw["id"]
            resume = (receipt, lambda: deferred.release_unredeemed(home, receipt))
    # No finalizer (unlike ``attach_to_event``): preflight checks restore throwaway copies of the row, and
    # dropping one must not release the receipt before the turn runs. The turn's end releases it
    # (``InjectedTurn.revoke``); a row that never runs leaves it to gateway-start recovery.
    setattr(event, _EVENT_ATTR, spec)
    if resume is not None:
        setattr(event, _RESUME_ATTR, resume)


def spec_of_event(event: Any) -> Optional[TurnSpec]:
    spec = getattr(event, _EVENT_ATTR, None) if event is not None else None
    return spec if type(spec) is TurnSpec else None


def bind_for_event(event: Any) -> None:
    """Bind the context for the turn handling *event* (or nothing). Always replaces what was bound."""
    clear_injected_turn(release=False)
    spec = spec_of_event(event)
    if spec is not None:
        resume = getattr(event, _RESUME_ATTR, None)
        if isinstance(resume, tuple) and len(resume) == 2:
            _INJECTED_TURN.set(InjectedTurn(spec, event, resume=resume[0], on_revoke=resume[1]))
        else:
            _INJECTED_TURN.set(InjectedTurn(spec, event))


def clear_injected_turn(release: bool = True) -> None:
    """Revoke and unbind whatever this context holds (copies in child tasks see it revoked). *release*: this is
    the turn's end (False when a new session scope is about to bind)."""
    current = _INJECTED_TURN.get()
    if current is not None:
        current.revoke(release=release)
    _INJECTED_TURN.set(None)


def get_injected_turn() -> Optional[InjectedTurn]:
    """The live injected-turn context of the current turn, or None."""
    current = _INJECTED_TURN.get()
    return current if current is not None and current.live else None

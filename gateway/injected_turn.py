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
_FIELDS = frozenset({"kind", "ref", "agent_origin"})


@dataclass(frozen=True)
class TurnSpec:
    """The validated request: which plugin, what kind of Switchboard event, its ref, and whether the turn has
    no human (agent origin)."""
    plugin_id: str
    kind: str
    ref: str
    agent_origin: bool


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
    return TurnSpec(plugin_id, kind, ref, agent_origin)


class InjectedTurn:
    """What a tool may read about the turn it runs in. Read-only; ``live`` only between bind and revoke."""
    __slots__ = ("_spec", "_event", "_live", "_lock", "__weakref__")

    def __init__(self, spec: TurnSpec, event: Any):
        self._spec = spec
        self._event = weakref.ref(event)
        self._live = True
        self._lock = threading.Lock()

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live and self._event() is not None

    def revoke(self) -> None:
        with self._lock:
            self._live = False

    plugin_id = property(lambda self: self._spec.plugin_id)
    kind = property(lambda self: self._spec.kind)
    ref = property(lambda self: self._spec.ref)
    agent_origin = property(lambda self: self._spec.agent_origin)


_INJECTED_TURN: ContextVar[Optional[InjectedTurn]] = ContextVar("plugin_injected_turn", default=None)
_EVENT_ATTR = "_hermes_injected_turn_spec"


def attach_to_event(event: Any, spec: TurnSpec) -> None:
    """Host only: mark the event the host built for this injection (never a copy, never a restored event)."""
    setattr(event, _EVENT_ATTR, spec)


def spec_of_event(event: Any) -> Optional[TurnSpec]:
    spec = getattr(event, _EVENT_ATTR, None) if event is not None else None
    return spec if type(spec) is TurnSpec else None


def bind_for_event(event: Any) -> None:
    """Bind the context for the turn handling *event* (or nothing). Always replaces what was bound."""
    clear_injected_turn()
    spec = spec_of_event(event)
    if spec is not None:
        _INJECTED_TURN.set(InjectedTurn(spec, event))


def clear_injected_turn() -> None:
    """Revoke and unbind whatever this context holds (copies in child tasks see it revoked)."""
    current = _INJECTED_TURN.get()
    if current is not None:
        current.revoke()
    _INJECTED_TURN.set(None)


def get_injected_turn() -> Optional[InjectedTurn]:
    """The live injected-turn context of the current turn, or None."""
    current = _INJECTED_TURN.get()
    return current if current is not None and current.live else None

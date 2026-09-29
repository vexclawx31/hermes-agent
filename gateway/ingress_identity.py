"""Task-local Slack provenance from native ingress, never serialized source hints.

This is an in-process trust boundary, not isolation from arbitrary Python code.
Only the Slack adapter's event builder records receipts. Restored/synthetic events
and source copies cannot manufacture them by attaching a RoutingIdentity.
"""

from dataclasses import dataclass
import re
import weakref

_PROFILE = re.compile(r"[a-z][a-z0-9_-]*\Z")
_TS = re.compile(r"[0-9]{10,}\.[0-9]{6}\Z")
_LAUNCHES = weakref.WeakKeyDictionary()
_EVENTS = {}


@dataclass(frozen=True)
class _Launch:
    owner: str
    multiplexed: object


@dataclass(frozen=True)
class _Receipt:
    event: object
    source: object
    adapter: object
    runner: object
    launch: _Launch
    fields: tuple
    identity: object
    owner: str


def capture_launch(runner):
    """Called once by GatewayRunner initialization, before adapters receive events."""
    owner = runner._primary_profile_name
    if type(owner) is str and _PROFILE.fullmatch(owner):
        # Never replace a snapshot if initialization is accidentally re-entered.
        if runner not in _LAUNCHES:
            _LAUNCHES[runner] = _Launch(owner, runner.config.multiplex_profiles)


def _fields(source):
    return tuple(
        getattr(source, name, None)
        for name in (
            "platform",
            "profile",
            "message_id",
            "chat_id",
            "chat_type",
            "scope_id",
            "user_id",
            "thread_id",
        )
    )


def record_slack_event(adapter, event):
    """Receipt for this exact native-built event and receiving registered adapter."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner = getattr(adapter, "gateway_runner", None)
    if not isinstance(runner, GatewayRunner) or event.source.platform != Platform.SLACK:
        return
    launch = _LAUNCHES.get(runner)
    if launch is None or event.internal is not False:
        return
    registered, owner = runner._owning_profile(adapter, Platform.SLACK)
    if not registered:
        return
    transport_owner = launch.owner if owner is None else owner
    if type(transport_owner) is not str or not _PROFILE.fullmatch(transport_owner):
        return
    # Resolve only at the native builder, before a source attachment can be
    # mistaken for proof. Retain the exact resolved object outside the source.
    from gateway.session_identity import identity_of

    if identity_of(event.source) is not None:
        return
    fields = _fields(event.source)
    identity = adapter._canonicalize(event.source)
    if identity is None:
        return
    key = id(event)
    ref = weakref.ref(event, lambda ref: _EVENTS.pop(key, None))
    _EVENTS[key] = _Receipt(
        ref,
        weakref.ref(event.source),
        weakref.ref(adapter),
        weakref.ref(runner),
        launch,
        fields,
        identity,
        transport_owner,
    )


def slack_binding(runner, source, event):
    """Return raw profile and a verified message ID; denied reads get no message.

    Non-Slack binding is deliberately outside this function. Invalid profile
    values remain raw so downstream strict validators cannot confuse them with
    the legacy None sentinel. No environment or permissive resolver is read.
    """
    raw = getattr(source, "profile", None)
    denied = (raw, "")
    receipt = _EVENTS.get(id(event))
    if (
        receipt is None
        or receipt.event() is not event
        or receipt.source() is not source
    ):
        return denied
    if (
        receipt.runner() is not runner
        or event.internal is not False
        or event.source is not source
    ):
        return denied
    adapter = receipt.adapter()
    launch = _LAUNCHES.get(runner)
    if (
        adapter is None
        or getattr(adapter, "gateway_runner", None) is not runner
        or launch is None
        or launch is not receipt.launch
        or _fields(source) != receipt.fields
    ):
        return denied
    if (
        runner._primary_profile_name != launch.owner
        or runner.config.multiplex_profiles is not launch.multiplexed
    ):
        return denied
    from gateway.config import Platform

    registered, owner = runner._owning_profile(adapter, Platform.SLACK)
    if not registered or (launch.owner if owner is None else owner) != receipt.owner:
        return denied
    transport = getattr(source, "_transport_adapter_ref", None)
    if not isinstance(transport, weakref.ReferenceType) or transport() is not adapter:
        return denied
    message = source.message_id
    if (
        type(message) is not str
        or not _TS.fullmatch(message)
        or type(event.message_id) is not str
        or event.message_id != message
    ):
        return denied
    # Dedicated legacy blanks and explicit same-owner multiplex identities
    # are separate admission lanes; cross-owner routing never gains authority.
    if launch.multiplexed is not False and not (
        launch.multiplexed is True and type(raw) is str and raw == receipt.owner
    ):
        return denied
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    if (
        identity is None
        or identity is not receipt.identity
        or identity.multiplexed is not launch.multiplexed
        or identity.transport_inferred
    ):
        return denied
    if (
        identity.runtime_profile != receipt.owner
        or identity.transport_profile != receipt.owner
        or identity.adapter() is not adapter
    ):
        return denied
    if raw is None or (type(raw) is str and raw == ""):
        if launch.multiplexed is not False or receipt.owner != launch.owner:
            return denied
        return receipt.owner, message
    if type(raw) is str and _PROFILE.fullmatch(raw) and raw == receipt.owner:
        return raw, message
    return denied


from contextvars import ContextVar

_TRUSTED_INGRESS = ContextVar("trusted_native_ingress", default=None)
TRUSTED_INGRESS_API_VERSION = 1


@dataclass(frozen=True)
class TrustedIngress:
    platform: str
    profile: str
    message_id: str
    chat_id: str
    chat_type: str
    scope_id: str
    user_id: str
    adapter: object


def _bind_trusted_ingress(runner, source, event):
    """Native binding only; routing values never confer this authority."""
    _TRUSTED_INGRESS.set(None)
    profile, message = slack_binding(runner, source, event)
    if message:
        _TRUSTED_INGRESS.set(_EVENTS[id(event)])


def clear_trusted_ingress():
    _TRUSTED_INGRESS.set(None)


def get_trusted_ingress():
    """Read and revalidate task-local native ingress; no environment fallback.

    The opaque receipt is never serialized or exported to subprocesses.
    """
    receipt = _TRUSTED_INGRESS.get()
    if not isinstance(receipt, _Receipt):
        return None
    runner, source, event = receipt.runner(), receipt.source(), receipt.event()
    if runner is None or source is None or event is None:
        return None
    profile, message = slack_binding(runner, source, event)
    if not message:
        return None
    return TrustedIngress(
        "slack",
        profile,
        message,
        source.chat_id,
        source.chat_type,
        source.scope_id,
        source.user_id,
        receipt.adapter(),
    )

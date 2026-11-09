"""Fork (launch adapter): the provenance labels a gateway turn binds for its tools (moved out of gateway.run)."""

import logging

logger = logging.getLogger(__name__)


def turn_origin_label(runner, source) -> str:
    """Fork (launch adapter): the trusted origin label of a turn, for opted-in MCP servers
    (``gateway.session_context.turn_origin``). A LOCAL session's frozen ``origin``; ``platform:<name>``
    for a turn a platform adapter delivered (the gateway's own routing, not message text); "" when the
    context is genuinely missing: no source, a LOCAL session without an origin (an unlabelled launch
    must not pass for a known caller), or a policy lookup that fails."""
    from gateway.config import Platform
    platform = getattr(source, "platform", None)
    if source is None or not isinstance(platform, Platform):
        return ""
    if platform != Platform.LOCAL:
        return f"platform:{platform.value}"
    return _local_origin(runner, source)


def launch_origin_source(runner, source) -> dict:
    """Fork (launch adapter): ``{"source": <origin>}`` for a LOCAL session created with a provenance
    label (``session.create origin=``), else ``{}``. Tools and plugins read it as
    HERMES_SESSION_SOURCE, what the in-process ``hermes chat --source <label>`` bound."""
    origin = _local_origin(runner, source)
    return {"source": origin} if origin else {}


def _local_origin(runner, source) -> str:
    """A LOCAL session's frozen ``origin`` label, or "" (not a LOCAL route, no origin, or a failed lookup)."""
    from gateway.session_policy import policy_for_source
    try:
        policy = policy_for_source(runner, source)
    except Exception:  # health: allow BLE001 -- fail closed: any lookup failure means the origin is unknown
        logger.debug("turn origin: policy lookup failed; no origin bound", exc_info=True)
        return ""
    origin = getattr(policy, "origin", None) if policy is not None else None
    return origin if isinstance(origin, str) and origin else ""

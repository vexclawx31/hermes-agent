"""Host routing for ``PluginContext.inject_message``: classic CLI REPL, Ink TUI / desktop, messaging
gateway (existing ``session_key`` or a new session at ``origin``)."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.plugins")


_NO_AUTHOR = object()


def _plugin_author_or_none(ctx: Any, author: Any, role: str):
    """The validated host ``PluginAuthor`` for this call, or None (logged) when it is refused."""
    from agent.plugin_author import build_plugin_author, valid_plugin_key
    if role != "user":
        logger.warning("inject_message: author requires role='user' (plugin %s)", ctx.plugin_id)
        return None
    if not valid_plugin_key(ctx.plugin_id):
        logger.warning("inject_message: plugin key %r cannot carry injection authors", ctx.plugin_id)
        return None
    try:
        plugin_author = build_plugin_author(ctx.plugin_id, author)
    except ValueError as exc:
        logger.warning("inject_message: invalid author for plugin %s: %s", ctx.plugin_id, exc)
        return None
    if not ctx._gateway_injection_author_allowed():
        logger.warning("inject_message: injection author denied for plugin %s; set "
                       "plugins.entries.%s.allow_injection_author: true (with allow_gateway_injection) "
                       "to allow it", ctx.plugin_id, ctx.plugin_id)
        return None
    return plugin_author


def inject_plugin_message(
    ctx: Any, content: str, role: str, *, session_key: Optional[str],
    origin: Optional[Mapping[str, Any]], author: Any = _NO_AUTHOR,
) -> bool:
    manager = ctx._manager
    msg = content if role == "user" else f"[{role}] {content}"
    cli = manager._cli_ref
    plugin_author = None
    if author is not _NO_AUTHOR:
        plugin_author = _plugin_author_or_none(ctx, author, role)
        if plugin_author is None:
            return False
    # An origin names a messaging chat; the local REPL is never that chat.
    if cli is not None and origin is None:
        if plugin_author is not None:
            logger.warning("inject_message: author is supported on the messaging gateway only (plugin %s)",
                           ctx.plugin_id)
            return False
        queue_ = cli._interrupt_queue if getattr(cli, "_agent_running", False) else cli._pending_input
        queue_.put(msg)
        return True
    if session_key and origin is not None:
        logger.warning("inject_message: pass session_key or origin, not both")
        return False
    if not session_key and origin is None:
        logger.warning("inject_message: gateway mode requires an existing session_key or an origin")
        return False
    if origin is not None and not isinstance(origin, Mapping):
        logger.warning("inject_message: origin must be a mapping in the SessionSource.to_dict() shape")
        return False
    if not ctx._gateway_injection_allowed():
        logger.warning("inject_message: gateway injection denied for plugin %s; set "
                       "plugins.entries.%s.allow_gateway_injection: true to allow it",
                       ctx.plugin_id, ctx.plugin_id)
        return False
    # TUI/desktop host is a different slot. It accepts only when it owns this
    # session_key; a miss falls through so a co-resident messaging gateway
    # still receives its own keys. An exception fails closed — do not also
    # hand the same text to the gateway.
    # An authored turn never goes to the TUI host (no author support there): gateway or nothing.
    if session_key and manager.has_tui_message_injector and plugin_author is None:
        try:
            if manager.inject_tui_message(session_key=session_key, content=msg, plugin_id=ctx.plugin_id):
                return True
        except Exception:
            logger.warning("inject_message: TUI scheduling failed for plugin %s", ctx.plugin_id, exc_info=True)
            return False
    if not manager.has_gateway_message_injector:
        logger.warning("inject_message: no live gateway is available")
        return False
    if origin is None:
        target: dict[str, Any] = {"session_key": session_key}
    else:
        # The manager's immutable home is the plugin's profile: the gateway creates the session
        # there or nowhere (never the ambient HERMES_HOME of whatever thread called us).
        target = {"origin": dict(origin), "plugin_home": manager.home_path}
    if plugin_author is not None:
        target["author"] = plugin_author
    try:
        return bool(manager.inject_gateway_message(**target, content=msg, plugin_id=ctx.plugin_id))
    except Exception:
        logger.warning("inject_message: gateway scheduling failed for plugin %s", ctx.plugin_id, exc_info=True)
        return False

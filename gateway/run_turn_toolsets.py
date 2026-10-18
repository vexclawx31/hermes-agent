"""Per-source toolset resolution for gateway agent runs (split from ``gateway/run_turn.py``)."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def resolve_enabled_toolsets_for_source(
    delivery_adapter_for: Callable[[Any], Any], user_config: dict, source: Any, platform_key: str,
) -> list:
    """Enabled toolsets for an agent run, honoring an adapter ``toolsets_for_source()`` override
    validated through the SAME ``_get_platform_tools`` path (unknown / platform-restricted
    toolsets dropped, not trusted).

    An adapter with ``toolsets_for_source_exact = True`` (WhatsApp guest callers) gets exactly the listed
    concrete toolsets — never default-on plugin toolsets or MCP servers — and no tools at all if the
    override itself fails."""
    from hermes_cli.tools_config import _get_platform_tools
    adapter = None
    try:
        adapter = delivery_adapter_for(source)
        override = adapter.toolsets_for_source(source) if adapter is not None else None
    except Exception:  # health: allow BLE001 -- adapter override is plugin code; any failure falls back (or fails closed below)
        logger.debug("toolsets_for_source override failed", exc_info=True)
        override = None
        if getattr(adapter, "toolsets_for_source_exact", False) is True:
            return []
    if getattr(adapter, "toolsets_for_source_exact", False) is True and isinstance(override, list):
        names = [str(x) for x in override]
        pts = dict(user_config.get("platform_toolsets") or {})
        pts[platform_key] = [*names, "no_mcp"]
        resolved = _get_platform_tools({**user_config, "platform_toolsets": pts}, platform_key)
        return sorted(resolved & set(names))
    if override and isinstance(override, list):
        pts = dict(user_config.get("platform_toolsets") or {})
        pts[platform_key] = [str(x) for x in override]
        user_config = {**user_config, "platform_toolsets": pts}
    return sorted(_get_platform_tools(user_config, platform_key))

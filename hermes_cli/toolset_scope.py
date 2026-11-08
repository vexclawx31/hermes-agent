"""Platform scope rules for configured toolsets."""

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# Toolsets without a restriction entry are available on every platform.
_TOOLSET_PLATFORM_RESTRICTIONS = {"discord": {"discord"}, "discord_admin": {"discord"}}


def toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return whether ``ts_key`` is available on ``platform``."""
    allowed: set[str] | None = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed


# --- Plugin toolset ceiling (config.yaml ``plugin_toolset_platforms``) ---------------------------------------
#
# ``plugin_toolset_platforms: {<plugin toolset>: [<platform>, ...]}`` caps where each PLUGIN toolset can exist.
# Absent: no ceiling (unchanged behaviour). Present: a plugin toolset is available to a session on platform P only
# when P is in its list; a toolset missing from the mapping, an empty or malformed list, a non-mapping value or an
# unknown (None) session platform means OFF. Inside the ceiling the normal resolution (saved lists,
# known_plugin_toolsets, explicit []) still decides, and agent.disabled_toolsets still applies last, so the
# ceiling only ever removes. It is read from the ACTIVE profile's config on every call and never stored in a
# process-global mapping, so profiles of a multiplexed gateway cannot see each other's ceiling.

CEILING_KEY = "plugin_toolset_platforms"
# Delegated children inherit the parent's disabled_toolsets (which already carry the parent's ceiling) and can
# only narrow from there, so their own "subagent" platform adds no second ceiling.
CEILING_INHERITS_PARENT = frozenset({"subagent"})
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("%s", message)


def parse_plugin_toolset_ceiling(config: Any) -> Optional[dict[str, frozenset[str]]]:
    """``{toolset: allowed platforms}`` from a config dict, or None when no ceiling is configured.
    Malformed parts fail closed (the toolset maps to no platform) and are warned about once."""
    if not isinstance(config, dict) or CEILING_KEY not in config:
        return None
    raw = config.get(CEILING_KEY)
    if not isinstance(raw, dict):
        _warn_once("<root>", f"{CEILING_KEY} is not a mapping of toolset -> [platforms]; every plugin toolset is off")
        return {}
    ceiling: dict[str, frozenset[str]] = {}
    for ts, platforms in raw.items():
        if isinstance(platforms, list) and all(isinstance(p, str) for p in platforms):
            ceiling[str(ts)] = frozenset(p.strip() for p in platforms if p.strip())
        else:
            _warn_once(str(ts), f"{CEILING_KEY}.{ts} is not a list of platform names; that toolset is off")
            ceiling[str(ts)] = frozenset()
    return ceiling


def active_plugin_toolset_ceiling() -> Optional[dict[str, frozenset[str]]]:
    """The ceiling from the active profile's config.yaml (read-only load), or None when unset/unreadable."""
    try:
        from hermes_cli.config import load_config_readonly
        return parse_plugin_toolset_ceiling(load_config_readonly())
    except Exception:
        logger.debug("plugin toolset ceiling: config unreadable", exc_info=True)
        return None


def live_plugin_toolset_keys() -> set[str]:
    """Toolsets that plugin-registered tools belong to, from the live registry (MCP servers are not plugins)."""
    try:
        from hermes_cli.plugins import get_plugin_toolsets
        return {str(key) for key, _label, _desc in get_plugin_toolsets()}
    except Exception:
        logger.debug("plugin toolset ceiling: plugin toolsets unavailable", exc_info=True)
        return set()


def _aliases() -> dict[str, str]:
    try:
        from tools.registry import registry
    except ImportError:
        return {}
    return dict(registry.get_registered_toolset_aliases())


def ceiling_blocked_toolsets(
    platform: Optional[str], *, ceiling: Any = ..., plugin_keys: Optional[Iterable[str]] = None,
) -> list[str]:
    """Plugin toolsets (and registry aliases that point at them) a session on ``platform`` may not have.
    [] when no ceiling is configured or the platform inherits its parent's ceiling."""
    if platform in CEILING_INHERITS_PARENT:
        return []
    ceiling = active_plugin_toolset_ceiling() if ceiling is ... else ceiling
    if ceiling is None:
        return []
    aliases = _aliases()
    allowed: dict[str, set[str]] = {}
    for key, platforms in ceiling.items():  # an alias key counts for its canonical toolset
        allowed.setdefault(aliases.get(key, key), set()).update(platforms)
    keys = set(plugin_keys) if plugin_keys is not None else live_plugin_toolset_keys()
    blocked = {ts for ts in keys if platform is None or platform not in allowed.get(ts, ())}
    try:
        from toolsets import TOOLSETS as static_toolsets
    except ImportError:
        static_toolsets = {}
    # Never name a built-in toolset: an alias that shadows one must not disable it.
    blocked |= {alias for alias, target in aliases.items() if target in blocked and alias not in static_toolsets}
    return sorted(blocked - set(static_toolsets))

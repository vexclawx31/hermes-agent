"""``gateway.managed_runtime``: a home whose gateway only its external supervisor may stop or start.

Fork (v0.23 managed-home guard). A release activator runs each home's gateway as a launchd job
(a LaunchAgent, or a LaunchDaemon for a separate tenant account) pinned to a sealed release, and
sets ``gateway.managed_runtime: true`` in that home's ``config.yaml``. A client of this Hermes
(``hermes chat``, the TUI, ``gateway ensure``) must then never:

* stop that home's gateway because it speaks an older runtime protocol
  (:func:`hermes_cli.gateway_runtime_replace.replace_incompatible_owner`): its planned-stop exit is
  clean, so launchd does not revive it; nor
* start an unmanaged gateway for it (:func:`hermes_cli.gateway_runtime_start.spawn_unmanaged_gateway`):
  a LaunchDaemon is invisible to the LaunchAgent probe, so "no service found" is not "no supervisor".

Both refuse with :data:`MANAGED_RUNTIME_REASON` and a detail naming the profile. The key is read from
the TARGET home's own file, exactly like ``gateway.standalone`` (``profile_is_standalone``), never from
the client's config. Unset or false keeps upstream behaviour byte for byte. A config that exists but
cannot be read or parsed fails closed: a guard that a broken file turns off is no guard.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANAGED_RUNTIME_KEY = "managed_runtime"  # under ``gateway:``
MANAGED_RUNTIME_REASON = "runtime_externally_managed"


def home_runtime_is_managed(home: str | Path) -> bool:
    """Does ``home``'s own ``config.yaml`` set ``gateway.managed_runtime: true``?"""
    cfg_path = Path(home) / "config.yaml"
    try:
        from hermes_cli.config import read_user_config_raw

        cfg = read_user_config_raw(cfg_path)  # raw, read-only: no defaults, no backup rotation
    except FileNotFoundError:
        return False
    except Exception as exc:  # unreadable or unparseable: refuse rather than stop/spawn blindly
        logger.warning("Cannot read gateway.%s from %s (%s); treating the gateway as externally managed",
                       MANAGED_RUNTIME_KEY, cfg_path, type(exc).__name__)
        return True
    section = cfg.get("gateway") if isinstance(cfg, dict) else None
    value = section.get(MANAGED_RUNTIME_KEY) if isinstance(section, dict) else None
    from hermes_cli.profiles import _standalone_truthy  # the shared bool parser gateway.standalone uses

    return _standalone_truthy(value)


def managed_runtime_detail(home: str | Path) -> str:
    """One line for the client's error: which profile, and that its owner is externally managed."""
    from hermes_constants import profile_name_for_home

    name = profile_name_for_home(home)
    label = f"profile '{name}'" if name else f"home '{home}'"
    return (f"{label}'s gateway is externally managed (gateway.{MANAGED_RUNTIME_KEY}: true); this client "
            "neither stops nor starts it. Restart it through its supervisor.")


def refuse_manual_owner_change(home: str | Path, action: str) -> None:
    """Exit 1 before a manual command stops or replaces the owner of a managed home (``gateway restart``'s
    stop-and-respawn fallback, ``gateway run --replace``, and Desktop's/the updater's ``--detach`` restart):
    stopping it first and then refusing the spawn would leave the managed gateway down."""
    if not home_runtime_is_managed(home):
        return
    import sys
    print(f"✗ {action} refused: {managed_runtime_detail(home)}", file=sys.stderr)
    sys.exit(1)

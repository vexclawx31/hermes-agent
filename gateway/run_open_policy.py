"""Startup refusal for platforms whose own access policy is ``open`` without an allow-all opt-in (split from
``gateway/run.py``)."""

from __future__ import annotations

from typing import Optional

from gateway.config import Platform, _getenv




def _group_sender_rules_gate_all(platform, extra: dict) -> bool:
    """True when WhatsApp ``groups`` sender rules gate every group sender: an entry-less group is denied there and
    the ``"*"`` default (if any) is not itself a wildcard. Only WhatsApp intake denies unmatched senders, so other
    platforms (e.g. WeCom, which admits a group with no or an empty entry) never qualify. A ``"*"`` list on one
    exact group is a deliberate per-group exception and does not open the others."""
    if platform != Platform.WHATSAPP:
        return False
    groups = extra.get("groups")
    if not isinstance(groups, dict) or not any(isinstance(v, dict) for v in groups.values()):
        return False
    default = groups.get("*")
    if not isinstance(default, dict):
        return True
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin  # same parsing as intake
    entries = WhatsAppBehaviorMixin._coerce_allow_list(default.get("allow_from", default.get("allowFrom")))
    return bool(entries) and "*" not in entries


def _own_policy_open_startup_violation(config) -> Optional[str]:
    """Return a startup-abort reason when open policy lacks allow-all opt-in."""
    from contextlib import suppress

    from gateway.run_startup import _own_policy_envs
    with suppress(Exception):  # idempotent; callers may hand in a config load_gateway_config never built
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    for platform, platform_config in getattr(config, "platforms", {}).items():
        if not getattr(platform_config, "enabled", False):
            continue
        # Platform plugins declare their own policy env names (gateway_env.AUTHZ own_policy_env).
        open_env = _own_policy_envs(platform)
        if not open_env:
            continue
        dm_env, group_env, allow_all_env = open_env
        extra = getattr(platform_config, "extra", None) or {}
        dm_policy = str(extra.get("dm_policy")
                        or (_getenv(dm_env, "pairing") if dm_env else "pairing")).strip().lower()
        group_policy = str(
            extra.get("group_policy") or (_getenv(group_env, "pairing") if group_env else "pairing")
        ).strip().lower()
        # An open group policy gated by per-group sender rules (``groups.<id|*>.allow_from``) is not an
        # open door: every unmatched sender is denied at intake and again by the runner.
        group_open = group_policy == "open" and not _group_sender_rules_gate_all(platform, extra)
        if dm_policy != "open" and not group_open:
            continue
        gateway_allow_all = _getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
        if gateway_allow_all or (
                allow_all_env and _getenv(allow_all_env, "").lower() in {"true", "1", "yes"}):
            continue
        return f"{platform.value}: open policy without allow-all opt-in"
    return None

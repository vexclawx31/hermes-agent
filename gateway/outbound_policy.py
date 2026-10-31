"""Outbound send policy: one per-profile gate for core sends that carry no requester identity.

``send_message_tool`` (``hermes send``, kanban, opt-in MCP) and cron delivery call :func:`authorize_outbound` before a
platform send. A plugin answers through the ``authorize_outbound_send`` hook with ``{"allow": bool, "reason": str}``
(or ``None`` for platforms it doesn't govern).

Config (profile ``config.yaml``)::

    outbound_policy:
      slack: enforce   # or "audit"; unset = no policy (today's behaviour)

- unset: nothing is asked, the send proceeds (installations without a policy are unchanged);
- ``audit``: the plugin is asked; a refusal (or no answer) is logged with a text hash and the send proceeds.
  Observation only: it protects nothing;
- ``enforce``: a refusal, a plugin that didn't load, a hook that raised or timed out, or no verdict at all refuses the
  send. A policy that can't answer is a refusal, never an allow.

Callers bind what they know, never what a caller claims: ``path`` is ``"core"`` for send_message_tool and
``"cron"`` only from the scheduler, which also passes the job id and the configured delivery target.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

HOOK = "authorize_outbound_send"
MODES = ("audit", "enforce")


def policy_mode(platform_name: str, config: Optional[dict] = None) -> Optional[str]:
    """The configured mode for ``platform_name``; None when unset. A malformed value fails closed to enforce."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly() or {}
        except Exception:
            logger.exception("outbound policy: config unreadable; treating every platform as enforce")
            return "enforce"
    section = config.get("outbound_policy") if isinstance(config, dict) else None
    if section is None:
        return None
    if not isinstance(section, dict):
        return "enforce"
    value = section.get(platform_name)
    if value is None:
        return None
    return value if value in MODES else "enforce"


def _text_hash(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def authorize_outbound(
    platform_name: str,
    *,
    action: str,
    chat_id: Any,
    thread_id: Any = None,
    text: Any = "",
    path: str,
    stage: str = "resolved",
    cron_job_id: Optional[str] = None,
    cron_target: Optional[str] = None,
    config: Optional[dict] = None,
) -> Optional[str]:
    """None when the send may proceed, else the refusal text to return to the caller.

    ``stage`` is ``"target"`` for a raw target before any resolution (a Slack user target must be refused before
    user->DM resolution opens a conversation) and ``"resolved"`` for the final destination.
    """
    mode = policy_mode(platform_name, config)
    if mode is None:
        return None
    problem: Optional[str] = None
    try:
        from hermes_cli.plugins import invoke_hook
        verdicts = invoke_hook(
            HOOK, platform=platform_name, action=action, chat_id=None if chat_id is None else str(chat_id),
            thread_id=None if thread_id is None else str(thread_id), text=str(text or ""), path=path, stage=stage,
            cron_job_id=cron_job_id, cron_target=cron_target,
        )
    except Exception:
        logger.exception("outbound policy hook failed for %s", platform_name)
        verdicts, problem = [], "the outbound policy check failed"
    verdict = next((v for v in verdicts if isinstance(v, dict) and "allow" in v), None)
    if problem is None:
        if verdict is None:
            problem = "no outbound policy answered (the policy plugin is not loaded or failed)"
        elif verdict.get("allow") is not True:
            problem = str(verdict.get("reason") or "refused by the outbound policy")[:300]
    if problem is None:
        return None
    logger.warning(
        "outbound policy %s: %s to %s:%s%s (path=%s stage=%s job=%s text_sha=%s): %s",
        "WOULD REFUSE (audit)" if mode == "audit" else "refused", action, platform_name, chat_id, f":{thread_id}" if thread_id else "", path, stage, cron_job_id or "-",
        _text_hash(text), problem)
    if mode == "audit":
        return None
    return f"Refusing to send to {platform_name}: {problem}"

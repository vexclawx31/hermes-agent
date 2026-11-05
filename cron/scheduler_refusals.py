"""Tell a job's owner when a scheduled run went silent after a tool was refused.

A run that answers ``[SILENT]`` (or nothing) delivers nothing, by design. When a tool was refused in that run
(``tools.tool_refusals``), the silence may be the refusal's doing, and nobody is watching a scheduled run. So
instead of staying quiet, the scheduler sends one plain line, privately, saying what was refused and what to do:

- to the chat the job was created in (``origin``) only when that is a **direct chat** with the owner
  (``origin.chat_type == "dm"``, or for older jobs a WhatsApp user JID / Slack ``D`` channel);
- else to ``failure_deliver`` when set (the operator's chosen lane for this job's problems);
- else only a log warning. Never to a group or the job's delivery target.

The same set of refusals is reported once (``last_refusal_notice`` holds its digest); a run that delivers
normally clears it, so a refusal that comes back is reported again. The notice is not the job's delivery: the
receipt fields ``_deliver_result`` records are put back afterwards.

Split out of ``cron.scheduler`` (file-size cap); the origin module is reached late-bound.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("cron.scheduler")

_LISTED = 2
_RECEIPT_FIELDS = ("last_delivery_unverified", "last_delivery_queued", "last_delivery_message_ids")


def refusal_notice(job: dict, refusals: list[dict[str, str]]) -> str:
    """The one-line notice: what was refused, and what to do."""
    name = str(job.get("name") or job.get("id") or "a scheduled job")
    shown = "; ".join(f"{r.get('tool') or 'a tool'} was refused ({r.get('reason') or 'no reason given'})"
                      for r in refusals[:_LISTED])
    more = f"; and {len(refusals) - _LISTED} more refusal(s)" if len(refusals) > _LISTED else ""
    return (f"Scheduled job \"{name}\" sent nothing this run because {shown}{more}. To fix it, change the job "
            f"or what it is allowed to use (ask me to update job {job.get('id', '')}), then run it again.")


def _is_direct_chat(origin: dict) -> bool:
    chat_type = str(origin.get("chat_type") or "").lower()
    if chat_type:
        return chat_type == "dm"
    platform, chat_id = str(origin.get("platform") or "").lower(), str(origin.get("chat_id") or "")
    if platform == "whatsapp":
        return chat_id.endswith(("@s.whatsapp.net", "@lid"))
    return platform == "slack" and chat_id.startswith("D")


def _notice_job(job: dict) -> Optional[dict]:
    """A copy of the job that delivers privately to its owner, or to ``failure_deliver``; None if neither."""
    origin = job.get("origin")
    if isinstance(origin, dict) and origin.get("platform") and origin.get("chat_id") and _is_direct_chat(origin):
        return {**job, "deliver": "origin", "failure_deliver": None}
    fallback = str(job.get("failure_deliver") or "").strip()
    if fallback and fallback.lower() not in {"local", "origin"}:
        return {**job, "deliver": fallback, "failure_deliver": None}
    return None


def _digest(refusals: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(sorted(refusals, key=json.dumps), sort_keys=True).encode()).hexdigest()[:16]


def _update(job_id: str, values: dict) -> None:
    try:
        from cron.jobs import update_job
        update_job(job_id, values)
    except Exception:
        logger.debug("Job '%s': could not record %s", job_id, sorted(values), exc_info=True)


def clear_refusal_notice(job: dict) -> None:
    """A run that delivered normally: forget the last notice, so a refusal that returns is reported again."""
    job.pop("_tool_refusals", None)
    if job.get("last_refusal_notice"):
        _update(job["id"], {"last_refusal_notice": None})


def notify_silent_refusals(job: dict, fence: Any, *, adapters: Any, loop: Any) -> Optional[str]:
    """Send the owner notice for a silent run that saw refusals. Returns the notice sent, or None.

    Runs under the fire-claim fence like any other delivery; failures are logged, never raised.
    """
    from cron import scheduler as _sched

    refusals = job.pop("_tool_refusals", None) or []
    if not refusals:
        return None
    digest = _digest(refusals)
    if job.get("last_refusal_notice") == digest:
        logger.info("Job '%s': silent after the same tool refusal(s) as last time; owner already told", job["id"])
        return None
    notice = refusal_notice(job, refusals)
    target = _notice_job(job)
    if target is None:
        logger.warning("Job '%s': silent run after tool refusal, and no private chat to tell: %s", job["id"], notice)
        return None
    try:
        from cron.jobs import get_job
        stored = get_job(job["id"]) or {}
    except Exception:
        logger.debug("Job '%s': could not read stored receipts before the notice", job["id"], exc_info=True)
        stored = {}
    receipts = {key: stored.get(key) for key in _RECEIPT_FIELDS}
    try:
        with fence.side_effect_fence() as owns_delivery:
            if not owns_delivery:
                return None
            error = _sched._deliver_result(target, notice, adapters=adapters, loop=loop)
    except Exception as exc:
        if isinstance(exc, _sched._FireClaimLostDuringSideEffect):
            raise
        error = str(exc)
    finally:
        _update(job["id"], receipts)  # the notice is not this job's delivery
    if error:
        logger.error("Job '%s': owner notice for a silent run after tool refusal failed: %s", job["id"], error)
        return None
    _update(job["id"], {"last_refusal_notice": digest})
    logger.info("Job '%s': silent after %d tool refusal(s); told the owner", job["id"], len(refusals))
    return notice

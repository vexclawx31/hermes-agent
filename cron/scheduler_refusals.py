"""Tell a job's owner when a scheduled run went silent after a tool was refused.

A run that answers ``[SILENT]`` (or nothing) delivers nothing, by design. When a tool was refused in that
run (``tools.tool_refusals``), the silence may be the refusal's doing, and nobody is watching a scheduled
run. So instead of staying quiet, the scheduler sends one plain line to where the job was created (its
``origin``: the owner's chat), never to the job's normal delivery target, saying what was refused and what
to do. Jobs without an origin (created from the CLI) fall back to ``failure_deliver`` when set, else a log
warning.

Split out of ``cron.scheduler`` (file-size cap); the origin module is reached late-bound.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("cron.scheduler")

_LISTED = 2


def refusal_notice(job: dict, refusals: list[dict[str, str]]) -> str:
    """The one-line notice: what was refused, and what to do."""
    name = str(job.get("name") or job.get("id") or "a scheduled job")
    shown = "; ".join(f"{r.get('tool') or 'a tool'} was refused ({r.get('reason') or 'no reason given'})"
                      for r in refusals[:_LISTED])
    more = f"; and {len(refusals) - _LISTED} more refusal(s)" if len(refusals) > _LISTED else ""
    return (f"Scheduled job \"{name}\" sent nothing this run because {shown}{more}. To fix it, change the job "
            f"or what it is allowed to use (ask me to update job {job.get('id', '')}), then run it again.")


def _notice_job(job: dict) -> Optional[dict]:
    """A copy of the job that delivers to its owner (origin), or to ``failure_deliver``; None if neither."""
    origin = job.get("origin")
    if isinstance(origin, dict) and origin.get("platform") and origin.get("chat_id"):
        return {**job, "deliver": "origin", "failure_deliver": None}
    fallback = str(job.get("failure_deliver") or "").strip()
    if fallback and fallback.lower() != "local":
        return {**job, "deliver": fallback, "failure_deliver": None}
    return None


def notify_silent_refusals(job: dict, fence: Any, *, adapters: Any, loop: Any) -> Optional[str]:
    """Send the owner notice for a silent run that saw refusals. Returns the notice sent, or None.

    Runs under the fire-claim fence like any other delivery; failures are logged, never raised.
    """
    from cron import scheduler as _sched

    refusals = job.pop("_tool_refusals", None) or []
    if not refusals:
        return None
    notice = refusal_notice(job, refusals)
    target = _notice_job(job)
    if target is None:
        logger.warning("Job '%s': silent run after tool refusal, and no owner chat to tell: %s", job["id"], notice)
        return None
    try:
        with fence.side_effect_fence() as owns_delivery:
            if not owns_delivery:
                return None
            error = _sched._deliver_result(target, notice, adapters=adapters, loop=loop)
    except Exception as exc:
        if isinstance(exc, _sched._FireClaimLostDuringSideEffect):
            raise
        error = str(exc)
    if error:
        logger.error("Job '%s': owner notice for a silent run after tool refusal failed: %s", job["id"], error)
        return None
    logger.info("Job '%s': silent after %d tool refusal(s); told the owner", job["id"], len(refusals))
    return notice

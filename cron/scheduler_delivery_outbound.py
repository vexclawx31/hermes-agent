"""Cron steps that post to a platform before the delivery lanes: the outbound policy decision for a cron send,
and the continuable thread's seed post. Imports ``gateway`` lazily, like the rest of the cron delivery code."""

from __future__ import annotations

import logging
from typing import Any, Optional

# Log-record parity with the origin module.
logger = logging.getLogger("cron.scheduler")


def authorize_cron_send(job: dict, target: dict, platform_name: str, chat_id: Any, thread_id: Any, text: Any) -> Optional[str]:
    """The outbound policy's refusal for a cron send, or None (``gateway.outbound_policy``). The scheduler binds
    the job and its configured target, so no caller can claim the cron path."""
    from gateway.outbound_policy import authorize_outbound
    return authorize_outbound(
        platform_name, action="send", chat_id=chat_id, thread_id=thread_id, text=text, path="cron",
        cron_job_id=str(job.get("id") or ""),
        cron_target=f"{target.get('_resolved_from') or 'explicit'}:{platform_name}:{chat_id}")


def _open_continuable_cron_thread(job: dict, adapter, chat_id: str, loop) -> Optional[str]:
    """Open a thread for a continuable cron job via ``adapter.create_handoff_thread``. Returns the
    thread_id, or ``None`` (no thread primitive / failed) = caller falls back to the DM mirror."""
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    from cron.scheduler_delivery import _cron_display_name
    thread_name = f"Hermes — {_cron_display_name(job)}"
    try:
        from agent.async_utils import safe_schedule_threadsafe
        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"), getattr(adapter, "name", "?"), e)
        return None

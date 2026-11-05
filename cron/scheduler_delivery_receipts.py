"""Per-run delivery bookkeeping persisted on the job: unverified acks, queued Bot Chat receipts and the
provider message IDs each target's send put on screen."""

from __future__ import annotations

import logging

logger = logging.getLogger("cron.scheduler")


def _note_delivered_message_ids(job: dict, where: str, message_ids) -> None:
    """Keep the provider message IDs one target's send put on screen (either lane, partial sends too), so the
    run records what it posted: ``last_delivery_message_ids`` = ``{"platform:chat_id[:thread_id]": [ids]}``."""
    ids = [str(i) for i in message_ids or () if i]
    if ids:
        kept = job.setdefault("_delivery_message_ids", {}).setdefault(where, [])
        kept.extend(i for i in ids if i not in kept)


def _record_delivery_verification(job: dict, unverified_targets: list) -> None:
    """Persist ``last_delivery_unverified``: list of ``platform:chat_id[:thread_id]`` targets acked with no
    evidence, or None, alongside queued Bot Chat receipts and this run's ``last_delivery_message_ids``.
    Never raises (bookkeeping must not fail a delivery)."""
    new_value = list(unverified_targets) or None
    queued = {target: receipt for target, receipt in
              job.get("_bot_chat_delivery_receipts", {}).items()
              if receipt["status"] in ("queued", "claimed")} or None
    message_ids = dict(job.get("_delivery_message_ids") or {}) or None
    values = {key: value for key, value in {
        "last_delivery_unverified": new_value, "last_delivery_queued": queued,
        "last_delivery_message_ids": message_ids,
    }.items() if (job.get(key) or None) != value}
    if not values:
        return
    job.update(values)
    try:
        from cron.jobs import update_job
        update_job(job["id"], values)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Job '%s': could not record delivery verification: %s", job.get("id"), exc)

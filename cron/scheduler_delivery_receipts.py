"""Per-run delivery bookkeeping persisted on the job: unverified acks, queued Bot Chat receipts and the
provider message IDs each target's send put on screen. ``run_delivery_receipt`` hands the same run's
targets and message IDs to its execution-ledger row, so one run's receipt survives the next run."""

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


def target_label(target: dict) -> str:
    """``platform:chat_id[:thread_id]`` of a resolved (requested) target, before it is prepared."""
    base = f"{target.get('platform')}:{target.get('chat_id') or '(own)'}"
    return f"{base}:{target['thread_id']}" if target.get("thread_id") else base


def _begin_delivery_receipt(job: dict, targets: list) -> list:
    """Start one ``_deliver_result`` call: drop the previous call's per-send scratch and remember
    which targets this delivery resolved, for the run's execution-ledger receipt. Returns *targets*."""
    for key in ("_bot_chat_delivery_receipts", "_delivery_message_ids", "_notification_all_targets_suppressed"):
        job.pop(key, None)
    job["_delivery_targets"] = [target_label(target) for target in targets]
    return targets


def _note_actual_target(job: dict, target: dict, prepared) -> None:
    """Rename a requested target to the lane it is actually sent to (``prepared.where``): preparing
    can open a new thread or drop the thread id, and the message ids are keyed by that name, so
    ``delivery_targets`` and the ``delivery_message_ids`` keys always agree."""
    labels = job.get("_delivery_targets") or []
    requested = target_label(target)
    if prepared is not None and requested in labels:
        labels[labels.index(requested)] = prepared.where


def queued_delivery_receipt(job: dict) -> dict:
    """What a gateway's send of a queued worker delivery stores on its queue row (``receipt``)."""
    return {"targets": list(job.get("_delivery_targets") or ()),
            "message_ids": dict(job.get("_delivery_message_ids") or {})}


def _adopt_queued_receipt(job: dict, queue_row) -> None:
    """Worker side: take this execution's own receipt from its queue row, never from the job record
    (whose ``last_delivery_message_ids`` is last-writer-wins across runs)."""
    import json

    try:
        receipt = json.loads((queue_row or {}).get("receipt") or "{}")
    except ValueError:
        receipt = {}
    receipt = receipt if isinstance(receipt, dict) else {}
    ids, targets = receipt.get("message_ids"), receipt.get("targets")
    job["_delivery_message_ids"] = dict(ids) if isinstance(ids, dict) else {}
    if isinstance(targets, list) and targets:
        job["_delivery_targets"] = [str(t) for t in targets]


def run_delivery_receipt(job: dict, delivery_error=None) -> dict:
    """``finish_execution`` keyword arguments for this run's own delivery: targets resolved,
    message ids posted, error summary. Empty values are omitted (stored as NULL)."""
    receipt = {
        "delivery_targets": list(job.get("_delivery_targets") or ()),
        "delivery_message_ids": dict(job.get("_delivery_message_ids") or {}),
        "delivery_error": str(delivery_error) if delivery_error else None,
    }
    return {key: value for key, value in receipt.items() if value}

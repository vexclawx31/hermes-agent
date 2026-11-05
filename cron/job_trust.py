"""Who a cron job belongs to, for decisions that must not let one person's job carry another's data.

- A job's **creator** is its ``origin`` identity (platform + user), recorded by the cronjob tool from the creating
  session, never from model arguments. A job made with no chat session (CLI, operator) has none.
- Its **changer** is ``last_changed_by``: who last edited it or ran it by hand through the cronjob tool
  (``tools.cronjob_job_args._changer_from_context``; ``api_server`` for the HTTP API). ``local`` (the operator) and
  absent mean "nobody else".

Rule for ``context_from`` (a job reading another job's latest output): allowed only when both jobs belong to the
same person, so a job someone else made or changed can't pull a private job's output into their chat.
"""

from __future__ import annotations

from typing import Any, Optional

Identity = tuple[str, str]


def _identity(value: Any) -> Optional[Identity]:
    if not isinstance(value, dict):
        return None
    platform = str(value.get("platform") or "").strip().lower()
    user = str(value.get("user_id") or "").strip()
    if not platform or platform == "local":
        return None
    return platform, user


def creator(job: dict) -> Optional[Identity]:
    """The chat identity that created ``job``; None for an operator-made job."""
    return _identity(job.get("origin"))


def people(job: dict) -> set[Identity]:
    """Everyone other than the operator who shaped this job: its creator and its last changer."""
    found = {creator(job), _identity(job.get("last_changed_by"))}
    if isinstance(job.get("last_changed_by"), str):  # malformed: never counts as the operator
        found.add(("unknown", str(job["last_changed_by"])))
    return {who for who in found if who is not None}


def may_read_output(reader: dict, source: dict) -> bool:
    """May ``reader`` inject ``source``'s latest output (``context_from``)?"""
    if source.get("id") and source.get("id") == reader.get("id"):
        return True  # "self": continuity across its own runs
    readers = people(reader)
    if not readers:
        return True  # made and changed only by the operator
    owners = people(source)
    return len(owners) == 1 and readers == owners


def actor_may_reference(actor: dict, source: dict) -> bool:
    """At create/update time: may this actor point a job's ``context_from`` at ``source``?"""
    who = _identity(actor)
    return who is None or people(source) == {who}

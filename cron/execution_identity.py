"""Which cron execution the current code runs inside, for plugins (``ctx.current_cron_execution()``).

The scheduler binds it once it has won the execution's ``claimed`` → ``running`` transition (or a
restart-safe worker adopted it) and resets it when the fire ends. It is a ContextVar, so it follows
the agent onto the threads the scheduler already runs it on via ``contextvars.copy_context()`` and
never leaks to another concurrently firing job or profile. Model output cannot reach it: tool
subprocesses do not share the interpreter. Delegated children read ``None`` — they are not the
scheduled occurrence.
"""
from __future__ import annotations

import hashlib
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CronExecution:
    job_id: str
    job_name: str
    execution_id: str
    source: str  # "builtin" (ticker), "direct" (manual run / dashboard Run now), provider name...
    scheduled_instant: Optional[str]  # the occurrence fired; None for an off-schedule run
    started_at: Optional[str]
    profile: str
    # The job as this fire runs it (the same job mapping ``run_job`` and delivery use), so a plugin can
    # tell whether the job record it re-reads is still that job: a prompt or destination edited after
    # the fire no longer matches. ``prompt_sha256`` is the SHA-256 hex of ``job["prompt"]`` (None when the
    # job has no text prompt); ``deliver`` is ``job["deliver"]`` as a string (None when unset); ``origin``
    # holds the origin's scalar fields as sorted ``(key, str(value))`` pairs (``()`` when there is none).
    prompt_sha256: Optional[str] = None
    deliver: Optional[str] = None
    origin: tuple[tuple[str, str], ...] = ()


_CURRENT: ContextVar[Optional[CronExecution]] = ContextVar("hermes_cron_execution", default=None)


def prompt_digest(prompt: Any) -> Optional[str]:
    """SHA-256 hex of a job prompt, as ``CronExecution.prompt_sha256`` records it; None for a non-string."""
    if not isinstance(prompt, str):
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def origin_fields(origin: Any) -> tuple[tuple[str, str], ...]:
    """A job origin's scalar fields as sorted ``(key, str(value))`` pairs, as ``CronExecution.origin`` holds them."""
    if not isinstance(origin, Mapping):
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in origin.items()
                        if value is not None and isinstance(value, (str, int, float, bool))))


def enter_cron_execution(job: Mapping[str, Any], execution_id: str, record: Mapping[str, Any]) -> Token:
    """Bind this fire's identity from *record*, the ledger row of the running transition, and *job*,
    the job mapping this fire runs and delivers."""
    from hermes_cli.profiles import current_profile_name

    deliver = job.get("deliver")
    return _CURRENT.set(CronExecution(
        job_id=str(job["id"]),
        job_name=str(job.get("name") or job["id"]),
        execution_id=str(execution_id),
        source=str(record.get("source") or ""),
        scheduled_instant=record.get("scheduled_instant"),
        started_at=record.get("started_at"),
        profile=current_profile_name("default") or "default",
        prompt_sha256=prompt_digest(job.get("prompt")),
        deliver=None if deliver is None else str(deliver),
        origin=origin_fields(job.get("origin")),
    ))


def exit_cron_execution(token: Optional[Token]) -> None:
    if token is not None:
        _CURRENT.reset(token)


def current_cron_execution() -> Optional[CronExecution]:
    from agent.delegation_context import is_delegated_child_context

    return None if is_delegated_child_context() else _CURRENT.get()

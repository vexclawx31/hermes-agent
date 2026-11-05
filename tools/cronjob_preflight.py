"""Create/update-time dry run: will this scheduled job's tools and destination be allowed when it runs?

Run by the ``cronjob`` tool right after a create or update, against the stored job. Each problem is one plain
sentence; the tool result then tells the agent to say to the requester, right away, that the job won't work
as set up and why, instead of letting the first run fail silently.

Checks (each fails open: a check that errors is skipped, never reported as a problem):

- the run-time preflight the scheduler itself applies before every run: skills and delivery targets;
- toolsets the job asks for that scheduled runs never get (``messaging``, ``clarify``, ``cronjob`` unless
  ``cron.allow_agent_scheduling``, and ``agent.disabled_toolsets``);
- the ``cron_job_preflight`` plugin hook: ``callback(job=<stored job dict>)`` returns None, one sentence, or
  a list of sentences, from the plugin's own permission checks for a scheduled run of this job.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PROBLEMS = 6


def _core_problems(job: dict) -> list[str]:
    problems: list[str] = []
    from cron import scheduler_preflight as pre

    for name, check in (("skills", pre._preflight_check_skills), ("delivery", pre._preflight_check_delivery)):
        try:
            reason = check(job)
        except Exception:
            logger.debug("cron dry run: %s check raised — skipped", name, exc_info=True)
            continue
        if reason:
            problems.append(str(reason))
    wanted = job.get("enabled_toolsets")
    if isinstance(wanted, list) and wanted:
        try:
            from cron.scheduler import _resolve_cron_disabled_toolsets
            from hermes_cli.config import load_config_readonly
            never = [t for t in wanted if t in set(_resolve_cron_disabled_toolsets(load_config_readonly()))]
        except Exception:
            logger.debug("cron dry run: toolset check raised — skipped", exc_info=True)
            never = []
        if never:
            problems.append(f"scheduled runs never get these toolsets: {', '.join(never)}")
    return problems


def _plugin_problems(job: dict) -> list[str]:
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook("cron_job_preflight"):
            return []
        results = invoke_hook("cron_job_preflight", job=dict(job))
    except Exception:
        logger.debug("cron dry run: cron_job_preflight hook failed — skipped", exc_info=True)
        return []
    problems: list[str] = []
    for result in results:
        for item in (result if isinstance(result, (list, tuple)) else [result]):
            if isinstance(item, str) and item.strip():
                problems.append(" ".join(item.split()))
    return problems


def job_wont_work(job: Any) -> list[str]:
    """Plain-sentence reasons this job's scheduled runs will be refused or undeliverable; [] when none."""
    if not isinstance(job, dict) or job.get("no_agent"):
        return _core_problems(job)[:_MAX_PROBLEMS] if isinstance(job, dict) else []
    seen: list[str] = []
    for problem in _core_problems(job) + _plugin_problems(job):
        if problem not in seen:
            seen.append(problem)
    return seen[:_MAX_PROBLEMS]


def attach(result: dict, job: Any) -> dict:
    """Add ``wont_work`` and the instruction to tell the requester now, when the dry run found problems."""
    problems = job_wont_work(job)
    if problems:
        result["wont_work"] = problems
        result["tell_requester"] = (
            "Tell the person who asked for this job now, in plain words, that it won't work as set up because: "
            + "; ".join(problems) + ". Offer to change the job; don't wait for its first run to find out.")
    return result

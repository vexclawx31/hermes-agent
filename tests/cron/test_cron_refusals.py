"""A scheduled run that goes silent after a tool refusal tells the job's owner; the cronjob tool dry-runs a
job's permissions at create/update time and says at once when it won't work."""

import contextlib
import contextvars
import json
import threading

import pytest

from tools import tool_refusals


# --- tools/tool_refusals.py ------------------------------------------------------------------

@pytest.mark.parametrize("result, error_type, reason", [
    (json.dumps({"success": False, "error": "not for this job", "refused": True}), None, "not for this job"),
    ({"refused": True, "message": "no  access\nhere"}, None, "no access here"),
    (json.dumps({"output": "", "exit_code": -1, "status": "blocked", "error": "BLOCKED: dangerous"}), None,
     "BLOCKED: dangerous"),
    ("Blocked by policy plugin", "plugin_block", "Blocked by policy plugin"),
    (json.dumps({"refused": True}), None, "no reason given"),
])
def test_refusal_shapes(result, error_type, reason):
    assert tool_refusals.refusal_reason(result, error_type=error_type) == reason


@pytest.mark.parametrize("result", [
    json.dumps({"success": False, "error": "unknown group"}), json.dumps({"refused": "yes"}), "plain text",
    json.dumps({"status": "ok", "text": "the word blocked and refused in content"}), "{not json refused", None, 7,
])
def test_ordinary_results_and_errors_are_not_refusals(result):
    assert tool_refusals.refusal_reason(result) is None


def test_observe_without_a_collection_is_a_no_op():
    tool_refusals.observe("t", json.dumps({"refused": True, "error": "x"}))  # nothing active: no error


def test_collection_spans_copied_tool_threads_dedupes_and_caps():
    token = tool_refusals.start()
    try:
        def worker(i):
            tool_refusals.observe("whatsapp_history", json.dumps({"refused": True, "error": f"no {i % 12}"}))
            tool_refusals.observe("whatsapp_history", json.dumps({"refused": True, "error": "no 0"}))

        threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker, i)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        seen = tool_refusals.stop(token)
    assert len(seen) == tool_refusals.MAX_REFUSALS
    assert len({r["reason"] for r in seen}) == len(seen)
    tool_refusals.observe("t", json.dumps({"refused": True, "error": "after"}))  # stopped: ignored


def test_terminal_post_tool_emit_reports_to_the_collection():
    from agent.inline_tool_executors import emit_terminal_post_tool_call

    token = tool_refusals.start()
    try:
        emit_terminal_post_tool_call(object(), function_name="terminal",
                                     function_args={"command": "rm -rf /x"},
                                     result=json.dumps({"status": "blocked", "error": "BLOCKED: needs approval"}),
                                     effective_task_id="cron:j:e", tool_call_id="c1")
        emit_terminal_post_tool_call(object(), function_name="web_search", function_args={},
                                     result=json.dumps({"results": []}), effective_task_id="cron:j:e",
                                     tool_call_id="c2")
    finally:
        seen = tool_refusals.stop(token)
    assert seen == [{"tool": "terminal", "reason": "BLOCKED: needs approval"}]


def test_run_scope_hands_the_refusals_to_the_job():
    from cron.scheduler_run_scope import _CronRunScope

    job = {"id": "abc123def456", "name": "brief"}
    scope = _CronRunScope(job, job["id"], "exec1")
    scope.enter()
    try:
        tool_refusals.observe("whatsapp_history", json.dumps({"refused": True, "error": "not this job"}))
    finally:
        scope.exit()
    assert job["_tool_refusals"] == [{"tool": "whatsapp_history", "reason": "not this job"}]
    tool_refusals.observe("x", json.dumps({"refused": True, "error": "late"}))
    assert len(job["_tool_refusals"]) == 1


# --- cron/scheduler_refusals.py --------------------------------------------------------------

class _Fence:
    def __init__(self, owns=True, lost=False):
        self.owns, self._lost = owns, lost

    @contextlib.contextmanager
    def side_effect_fence(self):
        yield self.owns

    def lost(self):
        return self._lost


OWNER_DM = {"platform": "whatsapp", "chat_id": "131941445710068@lid", "user_id": "33766355257@s.whatsapp.net"}
REFUSAL = {"tool": "whatsapp_history", "reason": "WhatsApp group tools are not available to this scheduled job"}


def _job(**extra):
    return {"id": "ef3bf9c679f0", "name": "pr-mauritius-daily-brief", "deliver": "whatsapp:120363@g.us",
            "origin": dict(OWNER_DM), "_tool_refusals": [dict(REFUSAL)], **extra}


@pytest.fixture
def sends(monkeypatch):
    sent = []

    def fake_deliver(job, content, **kwargs):
        sent.append({"deliver": job["deliver"], "failure_deliver": job.get("failure_deliver"), "content": content,
                     "kwargs": kwargs})
        return None

    monkeypatch.setattr("cron.scheduler._deliver_result", fake_deliver)
    return sent


def test_notice_goes_to_the_owner_not_the_delivery_group(sends):
    from cron.scheduler_refusals import notify_silent_refusals

    job = _job()
    notice = notify_silent_refusals(job, _Fence(), adapters={"a": 1}, loop="L")
    assert len(sends) == 1 and sends[0]["deliver"] == "origin" and sends[0]["failure_deliver"] is None
    assert sends[0]["kwargs"] == {"adapters": {"a": 1}, "loop": "L"}
    assert notice == sends[0]["content"]
    assert notice.startswith('Scheduled job "pr-mauritius-daily-brief" sent nothing this run because '
                             "whatsapp_history was refused (WhatsApp group tools are not available")
    assert "To fix it" in notice and "ef3bf9c679f0" in notice and "\n" not in notice
    assert "_tool_refusals" not in job and job["deliver"] == "whatsapp:120363@g.us"


def test_many_refusals_are_summarised():
    from cron.scheduler_refusals import refusal_notice

    refusals = [{"tool": f"t{i}", "reason": f"r{i}"} for i in range(5)]
    text = refusal_notice({"id": "j1", "name": "n"}, refusals)
    assert "t0 was refused (r0); t1 was refused (r1); and 3 more refusal(s)" in text


def test_no_origin_falls_back_to_failure_deliver_else_logs(sends, caplog):
    from cron.scheduler_refusals import notify_silent_refusals

    notify_silent_refusals(_job(origin=None, failure_deliver="slack:C0OPS"), _Fence(), adapters=None, loop=None)
    assert sends[-1]["deliver"] == "slack:C0OPS"
    with caplog.at_level("WARNING", logger="cron.scheduler"):
        assert notify_silent_refusals(_job(origin=None, failure_deliver="local"), _Fence(), adapters=None,
                                      loop=None) is None
    assert len(sends) == 1 and "no owner chat to tell" in caplog.text


def test_no_refusals_lost_claim_or_failed_send_send_nothing(sends, monkeypatch, caplog):
    from cron.scheduler_refusals import notify_silent_refusals

    assert notify_silent_refusals(_job(_tool_refusals=[]), _Fence(), adapters=None, loop=None) is None
    assert notify_silent_refusals(_job(), _Fence(owns=False), adapters=None, loop=None) is None
    assert sends == []
    monkeypatch.setattr("cron.scheduler._deliver_result", lambda *a, **k: "adapter down")
    with caplog.at_level("ERROR", logger="cron.scheduler"):
        assert notify_silent_refusals(_job(), _Fence(), adapters=None, loop=None) is None
    assert "adapter down" in caplog.text


def _compose_and_deliver(monkeypatch, content, *, success=True, fence=None, refusals=(REFUSAL,)):
    from cron import scheduler

    calls = []
    monkeypatch.setattr(scheduler, "save_job_output", lambda job_id, output: None)
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda job_id: False)
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda job_id, token: False)
    monkeypatch.setattr(scheduler, "_compose_run_delivery",
                        lambda job, **kw: (content, False, False, False, None))
    monkeypatch.setattr(scheduler, "notify_silent_refusals",
                        lambda job, fence, **kw: calls.append(("notice", job["id"])))
    monkeypatch.setattr(scheduler, "_deliver_result",
                        lambda job, content, **kw: calls.append(("deliver", content)) and None)
    d = scheduler._RunDelivery(job=_job(_tool_refusals=list(refusals)), success=success, error=None)
    scheduler._save_compose_deliver(d, fence or _Fence(), content, content, adapters=None, loop=None,
                                    verbose=False, execution_token=None)
    return calls


@pytest.mark.parametrize("content", ["[SILENT]", "NO_REPLY", "   "])
def test_a_silent_or_empty_run_hands_over_to_the_owner_notice(monkeypatch, content):
    assert _compose_and_deliver(monkeypatch, content) == [("notice", "ef3bf9c679f0")]


def test_a_delivered_run_or_a_lost_claim_sends_no_notice(monkeypatch):
    assert _compose_and_deliver(monkeypatch, "Morning brief: windy") == [("deliver", "Morning brief: windy")]
    assert _compose_and_deliver(monkeypatch, "[SILENT]", fence=_Fence(lost=True)) == []


# --- tools/cronjob_preflight.py and the cronjob tool ------------------------------------------

@pytest.fixture
def cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


@pytest.fixture
def hook(monkeypatch):
    seen = []
    answers = {"value": None}

    def fake_invoke(name, **kwargs):
        seen.append((name, kwargs["job"]))
        value = answers["value"]
        return [] if value is None else [value]

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "cron_job_preflight")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke)
    return seen, answers


def test_create_reports_a_plugin_refusal_at_once(cron_dir, hook):
    from tools.cronjob_tools import cronjob

    seen, answers = hook
    answers["value"] = "WhatsApp group tools will be refused in this job's runs: not an owner's job"
    out = json.loads(cronjob(action="create", prompt="Read the group and brief it", schedule="every 1h",
                             name="brief"))
    assert out["success"] is True
    assert out["wont_work"] == [answers["value"]]
    assert out["tell_requester"].startswith("Tell the person who asked for this job now")
    assert answers["value"] in out["tell_requester"]
    assert seen and seen[0][0] == "cron_job_preflight" and seen[0][1]["id"] == out["job_id"]


def test_a_job_that_will_work_gets_no_warning(cron_dir, hook):
    from tools.cronjob_tools import cronjob

    out = json.loads(cronjob(action="create", prompt="Check status", schedule="every 1h", name="ok"))
    assert out["success"] is True and "wont_work" not in out and "tell_requester" not in out


def test_list_hook_results_and_core_toolset_problems_are_merged(cron_dir, hook, monkeypatch):
    from tools.cronjob_preflight import job_wont_work

    _, answers = hook
    answers["value"] = ["first problem", "  second\nproblem ", "", 3]
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    problems = job_wont_work({"id": "j1", "deliver": "local", "enabled_toolsets": ["web", "messaging"]})
    assert problems == ["scheduled runs never get these toolsets: messaging", "first problem", "second problem"]


def test_dry_run_failures_fail_open(cron_dir, monkeypatch):
    from tools.cronjob_preflight import job_wont_work

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", boom)
    monkeypatch.setattr("cron.scheduler_preflight._preflight_check_delivery", boom)
    assert job_wont_work({"id": "j1", "deliver": "local", "prompt": "x"}) == []


def test_unknown_delivery_platform_is_reported_by_the_core_check(cron_dir, hook):
    from tools.cronjob_preflight import job_wont_work

    assert any("not a known cron delivery target" in p for p in job_wont_work({"id": "j1", "deliver": "nosuch:1"}))


def test_update_records_who_edited_and_dry_runs_again(cron_dir, hook):
    from cron.jobs import get_job
    from gateway.session_context import reset_session_vars, set_session_vars
    from tools.cronjob_tools import cronjob

    _, answers = hook
    created = json.loads(cronjob(action="create", prompt="Brief", schedule="every 1h", name="brief"))
    reset_session_vars()
    set_session_vars(platform="whatsapp", chat_id="120363@g.us", user_id="guest@s.whatsapp.net")
    try:
        answers["value"] = "WhatsApp group tools will be refused in this job's runs: last changed by a non-owner"
        out = json.loads(cronjob(action="update", job_id=created["job_id"], prompt="Brief, but longer"))
    finally:
        reset_session_vars()
    assert out["success"] is True and out["wont_work"] == [answers["value"]]
    assert get_job(created["job_id"])["last_edited_by"] == {"platform": "whatsapp",
                                                            "user_id": "guest@s.whatsapp.net"}


@pytest.mark.parametrize("bound, expected", [
    ({}, {"platform": "local", "user_id": ""}),
    ({"cron_session": "1"}, {"platform": "cron", "user_id": ""}),
    ({"platform": "slack", "user_id": "U0OWNER0001"}, {"platform": "slack", "user_id": "U0OWNER0001"}),
])
def test_editor_identity(bound, expected, monkeypatch):
    from gateway.session_context import reset_session_vars, set_session_vars
    from tools.cronjob_job_args import _editor_from_env

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    reset_session_vars()
    set_session_vars(**bound)
    try:
        assert _editor_from_env() == expected
    finally:
        reset_session_vars()


def test_the_hook_name_is_registered():
    from hermes_cli.plugins import VALID_HOOKS

    assert "cron_job_preflight" in VALID_HOOKS

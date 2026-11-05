"""Each cron run keeps its own execution id and delivery receipt in the execution ledger (r21)."""
import re
import sqlite3

import pytest

from cron import executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig


def _telegram_fixture(tmp_path, monkeypatch, *, fail_send=False):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    sent = []

    async def send(platform, pconfig, chat_id, text, **kwargs):
        sent.append(text)
        if fail_send:
            return {"success": False, "error": "fixture send refused"}
        return {"success": True, "message_id": f"msg-{len(sent)}"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "raw output", "result text", None))
    return sent


def _use_external_worker(job, monkeypatch, *, patch_queue=True):
    from cron import delivery_queue

    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    if not patch_queue:
        return
    original_wait = delivery_queue.enqueue_and_wait

    def drain_before_wait(execution_id, job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, job, content, for_failure=for_failure)
        assert scheduler.drain_delivery_queue({}, None) == 1
        return original_wait(execution_id, job, content, for_failure=for_failure)

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", drain_before_wait)


@pytest.mark.parametrize("external_worker", [False, True])
def test_each_run_keeps_its_own_receipt(tmp_path, monkeypatch, external_worker):
    _telegram_fixture(tmp_path, monkeypatch)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:fixture-chat")

    first_job = dict(job)
    if external_worker:
        _use_external_worker(first_job, monkeypatch)
    scheduler.run_one_job(first_job)
    second_job = dict(jobs.get_job(job["id"]))
    second_job.pop("execution_id", None)
    if external_worker:
        _use_external_worker(second_job, monkeypatch, patch_queue=False)
    scheduler.run_one_job(second_job)

    newest, older = executions.list_executions(job_id=job["id"], limit=10)
    assert newest["id"] != older["id"]
    for row, message_id in ((older, "msg-1"), (newest, "msg-2")):
        assert row["status"] == "completed"
        assert row["delivery_outcome"] == "delivered"
        assert row["delivery_targets"] == ["telegram:fixture-chat"]
        assert row["delivery_message_ids"] == {"telegram:fixture-chat": [message_id]}
        assert row["delivery_error"] is None
    # The job-level field still mirrors only the last run.
    assert jobs.get_job(job["id"])["last_delivery_message_ids"] == {"telegram:fixture-chat": ["msg-2"]}


def test_failed_delivery_records_target_and_error(tmp_path, monkeypatch):
    _telegram_fixture(tmp_path, monkeypatch, fail_send=True)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:fixture-chat")

    scheduler.run_one_job(job)

    row = executions.latest_execution(job["id"])
    assert row["delivery_outcome"] == "failed"
    assert row["delivery_targets"] == ["telegram:fixture-chat"]
    assert row["delivery_message_ids"] is None
    assert "fixture send refused" in row["delivery_error"]


def test_local_run_has_no_targets(tmp_path, monkeypatch):
    _telegram_fixture(tmp_path, monkeypatch)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="local")

    scheduler.run_one_job(job)

    row = executions.latest_execution(job["id"])
    assert row["delivery_outcome"] == "suppressed"
    assert row["delivery_targets"] is None
    assert row["delivery_message_ids"] is None


def test_execution_ids_are_hex_and_time_sortable(monkeypatch):
    clock = iter([1_700_000_000.000, 1_700_000_000.001, 1_800_000_000.0])
    monkeypatch.setattr(executions.time, "time", lambda: next(clock))
    ids = [executions.new_execution_id() for _ in range(3)]

    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


def test_error_summary_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    row = executions.create_execution("job", source="builtin")
    done = executions.finish_execution(
        row["id"], success=True, delivery_outcome="failed", delivery_error="x" * 5000)

    assert len(done["delivery_error"]) == executions.DELIVERY_ERROR_MAX_CHARS


_R20_SCHEMA = """CREATE TABLE executions (
    id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
    process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
    status TEXT NOT NULL, handoff_pending INTEGER NOT NULL DEFAULT 0, handoff_started_at REAL,
    claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT,
    delivery_outcome TEXT, scheduled_instant TEXT, progress_at TEXT)"""


def test_r20_rows_display_and_r20_writes_still_work(monkeypatch, tmp_path, capsys):
    db = tmp_path / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db)
    with sqlite3.connect(db) as conn:
        conn.execute(_R20_SCHEMA)
        conn.execute(
            "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at, delivery_outcome)"
            " VALUES ('9f8e7d6c5b4a39281706f5e4d3c2b1a0', 'old-job', 'builtin', 'old', -1, 'completed',"
            " '2026-10-01T09:00:00+00:00', 'delivered')")
    from hermes_cli.cron import cron_runs

    cron_runs("old-job", limit=10)
    out = capsys.readouterr().out
    assert "9f8e7d6c5b4a39281706f5e4d3c2b1a0" in out
    assert "delivery=delivered" in out
    assert "target:" not in out

    # After a rollback, r20's own writes (named columns, no receipt) still fit the r21 schema.
    row = executions.create_execution("new-job", source="builtin")
    executions.finish_execution(row["id"], success=True, delivery_outcome="delivered",
                                delivery_targets=["slack:C1"])
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at)"
            " VALUES ('r20-row', 'new-job', 'builtin', 'old', -1, 'claimed', '2026-10-02T09:00:00+00:00')")
        conn.execute("UPDATE executions SET status='completed', delivery_outcome=NULL WHERE id='r20-row'")
    ids = {r["id"]: r for r in executions.list_executions(job_id="new-job")}
    assert ids["r20-row"]["delivery_targets"] is None
    assert ids[row["id"]]["delivery_targets"] == ["slack:C1"]


def test_cron_runs_shows_receipt_and_looks_up_one_attempt(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    row = executions.create_execution("cli-job", source="builtin")
    executions.mark_execution_running(row["id"])
    executions.finish_execution(
        row["id"], success=True, delivery_outcome="delivered",
        delivery_targets=["slack:C0123:1700.1"],
        delivery_message_ids={"slack:C0123:1700.1": ["1700.2", "1700.3"]})
    other = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(other["id"], success=False, error="boom", delivery_outcome="failed",
                                delivery_targets=["telegram:42"], delivery_error="send timed out")
    from hermes_cli.cron import cron_runs

    assert cron_runs("cli-job", limit=10) is None
    listing = capsys.readouterr().out
    assert listing.index(other["id"]) < listing.index(row["id"])
    assert "delivery=delivered" in listing and "delivery=failed" in listing
    assert "target: slack:C0123:1700.1" in listing
    assert "message ids: slack:C0123:1700.1=1700.2,1700.3" in listing
    assert "delivery error: send timed out" in listing

    assert cron_runs(None, execution_id=row["id"]) == 0
    one = capsys.readouterr().out
    assert row["id"] in one and other["id"] not in one
    assert "finished:" in one

    assert cron_runs(None, execution_id="missing") == 1
    assert "No cron execution attempt missing." in capsys.readouterr().out
    assert cron_runs("another-job", execution_id=row["id"]) == 1


def test_cron_runs_parser_accepts_execution_id():
    import argparse

    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda _args: 0)
    args = parser.parse_args(["cron", "runs", "--id", "abc"])
    assert args.execution_id == "abc"

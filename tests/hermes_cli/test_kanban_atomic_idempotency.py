"""Concurrent create must admit one task per non-archived idempotency key."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import sqlite3
from threading import Barrier
import time
import traceback

import pytest

from hermes_cli import kanban_db as kb
# This PR predates the kanban_db_connect module extraction.
from hermes_cli import kanban_db as kbc


def test_concurrent_idempotent_create_and_claim(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.db"
    kbc.connect(path).close()
    barrier = Barrier(2)
    new_id = kb._new_task_id

    def synchronized_id():
        task_id = new_id()
        # Both callers reach admission before either can enter its transaction.
        # On the old path both have already missed the unlocked lookup.
        barrier.wait(timeout=10)
        return task_id

    monkeypatch.setattr(kb, "_new_task_id", synchronized_id)

    def create(_):
        with kbc.connect_closing(path) as conn:
            return kb.create_task(conn, title="synthetic admission",
                                  workspace_kind="dir", workspace_path=str(tmp_path),
                                  idempotency_key="synthetic-org:synthetic-issue")

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(create, range(2)))
    with kbc.connect_closing(path) as conn:
        claims = [kb.claim_task(conn, task_id, claimer="synthetic") for task_id in ids]
        assert len(set(ids)) == 1
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM task_events WHERE kind='created'").fetchone()[0] == 1
        assert sum(claim is not None for claim in claims) == 1
        assert len(kb.list_runs(conn, ids[0])) == 1


def test_idempotency_nested_rollback_and_key_lifetime(tmp_path):
    with kbc.connect_closing(tmp_path / "synthetic.db") as conn:
        kwargs = dict(title="synthetic", workspace_kind="dir", workspace_path=str(tmp_path))
        with pytest.raises(RuntimeError, match="rollback"):
            with kbc.write_txn(conn):
                rolled_back = kb.create_task(conn, idempotency_key="key", **kwargs)
                assert kb.create_task(conn, idempotency_key="key", **kwargs) == rolled_back
                raise RuntimeError("rollback")
        assert kb.get_task(conn, rolled_back) is None
        first = kb.create_task(conn, idempotency_key="key", **kwargs)
        assert kb.create_task(conn, idempotency_key="key", **kwargs) == first
        assert kb.create_task(conn, idempotency_key="other", **kwargs) != first
        assert kb.archive_task(conn, first)
        assert kb.create_task(conn, idempotency_key="key", **kwargs) != first
        assert kb.create_task(conn, **kwargs) != kb.create_task(conn, **kwargs)


def _admit_and_claim_process(path, barrier, sender, expected_mode):
    """Spawn-safe target: real connections; only ID allocation is synchronized."""
    try:
        with kbc.connect_closing(path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == expected_mode
            original_id = kb._new_task_id

            def synchronized_id():
                task_id = original_id()
                barrier.wait(timeout=20)
                return task_id

            # Synchronize *inside* admission so the old unlocked lookup races
            # deterministically. No SQL, guard, transaction or claim is mocked.
            kb._new_task_id = synchronized_id
            try:
                task_id = kb.create_task(
                    conn, title="synthetic process admission", workspace_kind="dir",
                    workspace_path=str(path.parent), idempotency_key="process-key",
                )
            finally:
                kb._new_task_id = original_id
            # All admissions finish before independent claim attempts compete.
            barrier.wait(timeout=20)
            claim = kb.claim_task(conn, task_id, claimer=f"synthetic-{os.getpid()}")
            sender.send((os.getpid(), task_id, claim is not None, None))
    except BaseException:
        sender.send((os.getpid(), None, False, traceback.format_exc()))
    finally:
        sender.close()


@pytest.mark.parametrize("journal", ["native", "wal"])
@pytest.mark.parametrize("round_number", range(3))
def test_process_admission_and_concurrent_claims(tmp_path, journal, round_number):
    path = tmp_path / f"synthetic-{round_number}.db"
    with kbc.connect_closing(path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if journal == "wal" and mode != "wal":
        # Never override upstream's vulnerable-SQLite/filesystem safety fallback.
        pytest.skip(f"native WAL unavailable: SQLite {sqlite3.sqlite_version}, journal={mode}")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes, receivers = [], []
    try:
        for _ in range(4):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_admit_and_claim_process, args=(path, barrier, sender, mode),
            )
            receivers.append(receiver)
            process.start()
            processes.append(process)
            sender.close()
        results = []
        for receiver in receivers:
            assert receiver.poll(40), "admission/claim child timed out"
            results.append(receiver.recv())
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert all(row[3] is None for row in results), results
        assert len({row[0] for row in results}) == 4
        ids = {row[1] for row in results}
        assert len(ids) == 1, results
        assert sum(row[2] for row in results) == 1, results
        with kbc.connect_closing(path) as conn:
            assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 1
            for kind in ("created", "claimed"):
                assert conn.execute(
                    "SELECT count(*) FROM task_events WHERE kind=?", (kind,),
                ).fetchone()[0] == 1
            assert len(kb.list_runs(conn, next(iter(ids)))) == 1
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        for receiver in receivers:
            receiver.close()


@pytest.mark.parametrize("boundary", ["busy-new", "busy-replay", "guard-replay", "readonly-replay"])
def test_admission_failure_boundary_has_no_side_effects(tmp_path, monkeypatch, boundary):
    path = tmp_path / "synthetic-boundary.db"
    kwargs = dict(title="synthetic", workspace_kind="dir", workspace_path=str(tmp_path))
    with kbc.connect_closing(path) as conn, kbc.connect_closing(path) as blocker:
        task_id = kb.create_task(conn, idempotency_key="existing", **kwargs)
        before = list(conn.iterdump())
        if boundary.startswith("busy"):
            conn.execute("PRAGMA busy_timeout=1")
            statements = []
            conn.set_trace_callback(statements.append)
            with kbc.write_txn(blocker):
                start = time.monotonic()
                key = "new" if boundary == "busy-new" else "existing"
                with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                    kb.create_task(conn, idempotency_key=key, **kwargs)
                assert time.monotonic() - start < 10
                assert not conn.in_transaction
            conn.set_trace_callback(None)
            assert statements.count("BEGIN IMMEDIATE") == kbc._BUSY_MAX_RETRIES + 1
            assert not any(s.lstrip().upper().startswith("INSERT") for s in statements)
        elif boundary == "guard-replay":
            # Tighten the real guard on both an existing writable connection and
            # a descendant's naturally read-only connection; never bypass it.
            with monkeypatch.context() as child:
                child.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
                with kbc.connect_closing(path) as readonly:
                    for candidate in (conn, readonly):
                        assert kb.get_task(candidate, task_id).id == task_id
                        with pytest.raises(PermissionError, match="cannot mutate Kanban"):
                            kb.create_task(candidate, idempotency_key="existing", **kwargs)
                        assert not candidate.in_transaction
        else:
            # Physical read-only is distinct from delegated authorization.
            # SQLite permits a no-write replay txn, but refuses a fresh insert.
            from contextlib import closing
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as readonly:
                readonly.row_factory = sqlite3.Row
                assert kb.get_task(readonly, task_id).id == task_id
                assert kb.create_task(readonly, idempotency_key="existing", **kwargs) == task_id
                with pytest.raises(sqlite3.OperationalError, match="readonly"):
                    kb.create_task(readonly, idempotency_key="new", **kwargs)
                assert not readonly.in_transaction
        assert list(conn.iterdump()) == before
        assert kb.create_task(conn, idempotency_key="existing", **kwargs) == task_id
        assert list(conn.iterdump()) == before
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

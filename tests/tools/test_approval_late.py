"""Late approvals: /approve after a gateway approval card timed out.

Contract: the timed-out call stays a deny (silence is not consent). A later human approval grants
only the MOST RECENT timed-out command, once, usable only inside the wake turn it queues and
within ``approvals.late_approval_window`` of the original timeout. The unconditional floors still
win, ask-only surfaces never record a slot, and a window of 0 turns the feature off.
"""

import pytest

from tools import approval as mod
from tools import approval_context, approval_late

SESSION_KEY = "test-late-approval"


@pytest.fixture
def gateway_timeout(monkeypatch):
    cfg = {"mode": "manual", "timeout": 0}
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    approval_late.clear(SESSION_KEY)
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    mod.register_gateway_notify(SESSION_KEY, lambda _data: None)
    yield cfg
    mod.unregister_gateway_notify(SESSION_KEY)
    approval_late.clear(SESSION_KEY)


def _guard(command):
    return mod.check_all_command_guards(command, "local")


def test_grant_runs_exact_command_once_inside_the_wake_turn(gateway_timeout):
    command = "rm -rf ./build"
    first = _guard(command)
    assert first["approved"] is False and first["outcome"] == "timeout"
    assert "re-run this exact command once" in first["message"]

    slot = approval_late.grant_latest(SESSION_KEY)
    assert slot is not None
    assert approval_late.consume_grant(SESSION_KEY, command) is False  # wake turn not started
    assert approval_late.arm(slot.token)
    assert _guard("rm -rf ./dist")["approved"] is False
    assert _guard(command)["approved"] is True
    assert _guard(command)["approved"] is False      # one-shot


def test_wake_turn_end_drops_an_unused_grant(gateway_timeout):
    _guard("rm -rf ./build")
    slot = approval_late.grant_latest(SESSION_KEY)
    approval_late.arm(slot.token)
    approval_late.disarm(slot.token)
    assert _guard("rm -rf ./build")["approved"] is False


def test_only_the_most_recent_timeout_is_grantable(gateway_timeout):
    _guard("rm -rf ./old")
    _guard("rm -rf ./new")
    slot = approval_late.grant_latest(SESSION_KEY)
    assert slot.command == "rm -rf ./new"
    assert approval_late.grant_latest(SESSION_KEY) is None  # no cascade to ./old


def test_expiry_counts_from_the_original_timeout(gateway_timeout, monkeypatch):
    gateway_timeout["late_approval_window"] = 60
    clock = [1000.0]
    monkeypatch.setattr(approval_late.time, "monotonic", lambda: clock[0])
    _guard("rm -rf ./build")
    clock[0] += 50
    slot = approval_late.grant_latest(SESSION_KEY)
    clock[0] += 20  # 70s after the timeout, 20s after the grant
    assert approval_late.arm(slot.token) is False
    assert _guard("rm -rf ./build")["approved"] is False


def test_late_grant_never_bypasses_hardline_floor(gateway_timeout):
    command = "rm -rf /"
    approval_late.record_timeout(SESSION_KEY, command, "recursive delete")
    approval_late.arm(approval_late.grant_latest(SESSION_KEY).token)
    result = _guard(command)
    assert result["approved"] is False and result.get("outcome") != "timeout"


def test_ask_only_surface_records_nothing(gateway_timeout, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SESSION")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    result = _guard("rm -rf ./build")
    assert result["approved"] is False
    assert "re-run this exact command once" not in (result.get("message") or "")
    assert approval_late.grant_latest(SESSION_KEY) is None


def test_window_zero_disables_late_approvals(gateway_timeout):
    gateway_timeout["late_approval_window"] = 0
    result = _guard("rm -rf ./build")
    assert result["outcome"] == "timeout"
    assert "re-run this exact command once" not in result["message"]
    assert approval_late.grant_latest(SESSION_KEY) is None

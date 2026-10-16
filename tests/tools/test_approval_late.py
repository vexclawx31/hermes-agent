"""Late approvals: /approve after a gateway approval card timed out.

Contract: the timed-out call stays a deny (silence is not consent), but a later human approval
lets the byte-identical command run ONCE in the same session; the unconditional floors still win,
and ``approvals.late_approval_window: 0`` turns the feature off.
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
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_INTERACTIVE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: cfg)
    mod.register_gateway_notify(SESSION_KEY, lambda _data: None)
    yield cfg
    mod.unregister_gateway_notify(SESSION_KEY)
    approval_late.clear(SESSION_KEY)


def test_late_approval_runs_exact_command_once(gateway_timeout):
    command = "rm -rf ./build"
    first = mod.check_all_command_guards(command, "local")
    assert first["approved"] is False and first["outcome"] == "timeout"
    assert "re-run this exact command once" in first["message"]

    assert approval_late.grant_latest(SESSION_KEY) is not None
    assert mod.check_all_command_guards("rm -rf ./dist", "local")["approved"] is False
    assert mod.check_all_command_guards(command, "local")["approved"] is True
    # One-shot: the grant is spent, the next identical call asks (and times out) again.
    assert mod.check_all_command_guards(command, "local")["approved"] is False


def test_late_grant_never_bypasses_hardline_floor(gateway_timeout):
    command = "rm -rf /"
    approval_late.record_timeout(SESSION_KEY, command, "recursive delete")
    assert approval_late.grant_latest(SESSION_KEY) is not None
    result = mod.check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert result.get("outcome") != "timeout"


def test_window_zero_disables_late_approvals(gateway_timeout):
    gateway_timeout["late_approval_window"] = 0
    result = mod.check_all_command_guards("rm -rf ./build", "local")
    assert result["outcome"] == "timeout"
    assert "re-run this exact command once" not in result["message"]
    assert approval_late.grant_latest(SESSION_KEY) is None

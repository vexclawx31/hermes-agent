"""Tests for approvals.human_required — red-tier categories only a person may approve."""

import pytest

import tools.approval as approval_module
from tools import approval_context
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.approval import check_all_command_guards, detect_dangerous_command, is_human_required

RED = "hermes gateway restart"
GREEN = "rm -rf ./build-output"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr("tools.tirith_security.check_command_security",
                        lambda _command: {"action": "allow", "findings": [], "summary": ""})
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()


def _config(monkeypatch, **approvals):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: approvals)


def _smart(monkeypatch, verdict="approve"):
    calls = []

    def fake(command, *args, **kwargs):
        calls.append(command)
        return verdict

    monkeypatch.setattr(approval_module, "_smart_verdict", fake)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    return calls


def _recorder(choice):
    prompts = []

    def callback(*args, **kwargs):
        prompts.append(kwargs)
        return choice

    return prompts, callback


class TestMatching:
    def test_red_command_is_in_default_floor(self, monkeypatch):
        _config(monkeypatch)
        is_dangerous, key, _ = detect_dangerous_command(RED)
        assert is_dangerous and key.startswith("stop/restart hermes gateway")
        assert is_human_required(key)

    def test_variant_suffix_matches_but_lookalike_prefix_does_not(self, monkeypatch):
        _config(monkeypatch)
        assert is_human_required("in-place edit of Hermes config/env (long flag)")
        assert not is_human_required("in-place edit of Hermes config/environment")
        assert not is_human_required("recursive delete")
        assert not is_human_required(None)

    def test_config_replaces_default_and_empty_list_disables(self, monkeypatch):
        _config(monkeypatch, human_required=["recursive delete"])
        assert is_human_required("recursive delete")
        assert not is_human_required("stop/restart hermes gateway (kills running agents)")
        _config(monkeypatch, human_required=[])
        assert not is_human_required("stop/restart hermes gateway (kills running agents)")

    def test_invalid_config_keeps_default_floor(self, monkeypatch):
        _config(monkeypatch, human_required="sudo")
        assert is_human_required("stop/restart hermes gateway (kills running agents)")


class TestInteractiveGate:
    def test_guardian_is_skipped_and_scope_is_once(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="smart")
        calls = _smart(monkeypatch, "approve")
        prompts, callback = _recorder("once")
        result = check_all_command_guards(RED, "local", approval_callback=callback)
        assert result["approved"] is True
        assert calls == []
        assert len(prompts) == 1
        assert prompts[0].get("allow_permanent") is False
        assert prompts[0].get("allow_session") is False

    def test_green_command_still_uses_guardian(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="smart")
        calls = _smart(monkeypatch, "approve")
        prompts, callback = _recorder("deny")
        result = check_all_command_guards(GREEN, "local", approval_callback=callback)
        assert result["approved"] is True
        assert calls == [GREEN]
        assert prompts == []

    def test_allowlist_entry_does_not_bypass_the_floor(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="manual")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        _, key, _ = detect_dangerous_command(RED)
        approval_module.approve_permanent(key)
        approval_module.approve_permanent(RED)
        prompts, callback = _recorder("deny")
        result = check_all_command_guards(RED, "local", approval_callback=callback)
        assert result["approved"] is False
        assert len(prompts) == 1

    def test_session_answer_is_not_remembered(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="manual")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        prompts, callback = _recorder("session")
        assert check_all_command_guards(RED, "local", approval_callback=callback)["approved"] is True
        assert check_all_command_guards(RED, "local", approval_callback=callback)["approved"] is True
        assert len(prompts) == 2

    def test_empty_floor_restores_allowlist_behaviour(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="manual", human_required=[])
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        _, key, _ = detect_dangerous_command(RED)
        approval_module.approve_permanent(key)
        prompts, callback = _recorder("deny")
        assert check_all_command_guards(RED, "local", approval_callback=callback)["approved"] is True
        assert prompts == []


class TestUnattended:
    def test_cron_approve_mode_cannot_run_a_red_command(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        _config(monkeypatch, mode="manual", cron_mode="approve")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
        tokens = set_session_vars(cron_session="1")
        try:
            red = check_all_command_guards(RED, "local")
            green = check_all_command_guards(GREEN, "local")
        finally:
            clear_session_vars(tokens)
        assert red["approved"] is False and red.get("human_required") is True
        assert green["approved"] is True

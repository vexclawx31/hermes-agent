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


class TestReviewFindings:
    def test_every_default_entry_matches_a_real_detector_key(self):
        from tools import approval_detection as d
        keys = [desc for _, desc in d.DANGEROUS_PATTERNS]
        keys += [d._PARSER_LIMIT_DESCRIPTION, d._GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION]
        for entry in approval_module.DEFAULT_HUMAN_REQUIRED:
            assert any(k == entry or k.startswith(entry + " (") for k in keys), entry

    @pytest.mark.parametrize("command", [
        "rm -rf ./build && hermes gateway restart",
        "git push -f origin x && sed -i s/a/b/ ~/.hermes/.env",
        "rm -rf ./build; kill $(pgrep -f 'hermes gateway')",
        'launchctl kick"start" -k gui/501/ai.hermes.gateway',
    ])
    def test_red_category_anywhere_in_the_command_is_floored(self, monkeypatch, command):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="smart")
        calls = _smart(monkeypatch, "approve")
        for key in approval_module.detect_dangerous_keys(command):
            approval_module.approve_permanent(key)
        prompts, callback = _recorder("deny")
        result = check_all_command_guards(command, "local", approval_callback=callback)
        assert result["approved"] is False
        assert calls == []
        assert len(prompts) == 1 and prompts[0].get("allow_session") is False

    def test_compound_red_command_blocked_under_cron_approve(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        _config(monkeypatch, mode="manual", cron_mode="approve")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
        tokens = set_session_vars(cron_session="1")
        try:
            result = check_all_command_guards("rm -rf ./build && hermes gateway restart", "local")
        finally:
            clear_session_vars(tokens)
        assert result["approved"] is False and result.get("human_required") is True

    def test_no_person_and_no_known_context_still_blocks(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        _config(monkeypatch, mode="manual")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(approval_module, "_unattended_contexts", lambda: [])
        assert check_all_command_guards(RED, "local")["approved"] is False
        assert check_all_command_guards(GREEN, "local")["approved"] is True

    def test_parser_limit_key_is_floored(self, monkeypatch):
        _config(monkeypatch)
        from tools import approval_detection as d
        assert is_human_required(d._PARSER_LIMIT_DESCRIPTION)

    def test_compound_sudo_shell_is_blocked_before_any_prompt(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="smart")
        _smart(monkeypatch, "approve")
        prompts, callback = _recorder("once")
        assert check_all_command_guards("git push -f origin x && sudo -s", "local",
                                        approval_callback=callback)["approved"] is False


class TestPreparedTerminalBatch:
    """A desktop batch prepares each terminal decision ahead of its turn and
    consumes it at execution. The floor must hold across that hand-off."""

    CALL_ID = "call-red"

    def _slot(self):
        import threading
        from types import SimpleNamespace
        from agent.terminal_approval_batch import _TerminalSlot

        batch = SimpleNamespace(cancelled=threading.Event(), failure_seen=False, task_id="task",
                                agent=SimpleNamespace(_interrupt_requested=False))
        parsed = SimpleNamespace(ref=lambda task_id: SimpleNamespace(call_id=self.CALL_ID))
        return _TerminalSlot(batch, parsed, 0)

    def _run(self, slot, prepared_command, executed_command, **kwargs):
        """Prepare like _TerminalSlot.prepare, then execute in the same slot."""
        from agent import terminal_approval_batch as tab

        slot_token = tab._slot.set(slot)
        call_token = approval_context._approval_tool_call_id.set(self.CALL_ID)
        try:
            slot.preparing = True
            try:
                slot.guard_key = (prepared_command, "local", False)
                slot.decision = check_all_command_guards(prepared_command, "local", **kwargs)
            finally:
                slot.preparing = False
            return slot.decision, check_all_command_guards(executed_command, "local", **kwargs)
        finally:
            approval_context._approval_tool_call_id.reset(call_token)
            tab._slot.reset(slot_token)

    def test_unattended_batch_cannot_run_a_red_command(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        _config(monkeypatch, mode="manual")
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        prepared, executed = self._run(self._slot(), RED, RED)
        assert prepared["approved"] is False and prepared.get("human_required") is True
        assert executed["approved"] is False and executed.get("human_required") is True

    def test_smart_mode_batch_skips_guardian_and_stays_floored(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        _config(monkeypatch, mode="smart")
        calls = _smart(monkeypatch, "approve")
        prepared, executed = self._run(self._slot(), RED, RED)
        assert calls == []
        assert prepared["approved"] is False and prepared.get("human_required") is True
        assert executed["approved"] is False and executed.get("human_required") is True

    def test_smart_approval_prepared_for_green_is_not_reused_for_red(self, monkeypatch):
        # Guardian approvals are the only policy decisions a batch reuses, so
        # this runs attended: the person is asked about RED and says no.
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        _config(monkeypatch, mode="smart")
        calls = _smart(monkeypatch, "approve")
        prompts, callback = _recorder("deny")
        prepared, executed = self._run(self._slot(), GREEN, RED, approval_callback=callback)
        assert prepared["approved"] is True and prepared.get("smart_approved") is True
        assert calls == [GREEN]
        assert len(prompts) == 1
        assert executed["approved"] is False

"""``agent.stop_guard``: approved work must not end on a routine permission question.

Contracts: the guard is opt-in; it fires only for a permission question after tool work, never
when a red-tier decision is cited or work is still running; it nudges at most once per turn; and
the nudge is runtime scaffolding that never reaches the returned transcript.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent import stop_guard
from agent.stop_guard import ends_with_permission_question, turn_did_tool_work


@pytest.mark.parametrize("text", [
    "Tests pass on the branch.\n\nShall I open the PR next?",
    "Step 1 is done. Want me to continue with step 2?",
    "Refactor applied. Should I proceed?",
    "I updated the parser. Would you like me to also update the docs?",
])
def test_routine_permission_questions_are_detected(text):
    assert ends_with_permission_question(text)


@pytest.mark.parametrize("text", [
    "Done. All 42 tests pass.",                                     # no question
    "Which of the two config files did you mean?",                  # clarifying, not permission
    "Branch is ready. Shall I deploy it to production?",            # live deploy
    "Should I restart the gateway service now?",                    # live restart
    "Want me to email the customer the summary?",                   # external comms
    "May I delete the old backups permanently?",                    # irreversible deletion
    "The CLI needs your API key. Shall I continue once you add it?",  # credentials
    "Shall I continue?\n\nNote: this question is followed by more text.",  # question not final
])
def test_non_routine_or_non_permission_endings_do_not_fire(text):
    assert not ends_with_permission_question(text)


def test_tool_work_must_happen_after_the_real_user_message():
    tool_turn = {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}
    assert turn_did_tool_work([{"role": "user", "content": "go"}, tool_turn,
                               {"role": "tool", "content": "ok"}])
    assert not turn_did_tool_work([tool_turn, {"role": "tool", "content": "ok"},
                                   {"role": "user", "content": "new question"}])
    # Synthetic nudges are not human turns.
    assert turn_did_tool_work([{"role": "user", "content": "go"}, tool_turn,
                               {"role": "user", "content": "[System]", "_verification_stop_synthetic": True}])


def test_approval_block_this_turn_suppresses_the_nudge():
    from agent.stop_guard import turn_hit_approval_block
    blocked = {"role": "tool", "content": '{"error": "BLOCKED: Command timed out without user response."}'}
    assert turn_hit_approval_block([{"role": "user", "content": "go"}, blocked])
    assert not turn_hit_approval_block([blocked, {"role": "user", "content": "go"}])


# ── end to end through the real turn loop ─────────────────────────────────────────────


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           model="test/model", usage=None)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="stop-guard-test", api_key="test-key", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=4, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance._handle_max_iterations = MagicMock(return_value="budget summary")
    return instance, home


def _enable(home, enabled=True):
    (home / "config.yaml").write_text(yaml.safe_dump({"agent": {"stop_guard": enabled}}))


def _run(agent, answers):
    seen = []

    def model_call(api_kwargs):
        seen.append([m.get("content") for m in api_kwargs.get("messages", [])])
        return _response(next(answers))

    agent._interruptible_api_call = model_call
    with (
        patch("agent.stop_guard.turn_did_tool_work", return_value=True),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        return agent.run_conversation("finish the approved refactor"), seen


def test_permission_question_gets_one_continuation_nudge(agent):
    instance, home = agent
    _enable(home)
    answers = iter(["Step 1 done. Shall I proceed with step 2?",
                    "Step 2 done too. Should I continue with step 3?"])
    result, seen = _run(instance, answers)

    assert len(seen) == 2  # one nudge, then the second question ends the turn (bounded)
    assert stop_guard.STOP_GUARD_NUDGE in seen[1]
    assert result["final_response"] == "Step 2 done too. Should I continue with step 3?"
    assert not any(stop_guard.STOP_GUARD_NUDGE == m.get("content") for m in result["messages"])


def test_guard_off_by_default_and_skips_red_tier(agent):
    instance, home = agent
    result, seen = _run(instance, iter(["Step 1 done. Shall I proceed with step 2?"]))
    assert len(seen) == 1 and result["final_response"].endswith("step 2?")

    _enable(home)
    result, seen = _run(instance, iter(["Built. Shall I deploy it to production?"]))
    assert len(seen) == 1


def test_running_background_work_suppresses_the_nudge(agent, monkeypatch):
    instance, home = agent
    _enable(home)
    monkeypatch.setattr(stop_guard, "_work_still_running", lambda _agent: True)
    _result, seen = _run(instance, iter(["Tests started in the background. Shall I wait for them?"]))
    assert len(seen) == 1

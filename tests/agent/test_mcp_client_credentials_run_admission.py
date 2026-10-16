"""``AIAgent.run_conversation`` (turn admission, ``agent/turn_facade.py``) is the run boundary for MCP
``auth: client_credentials`` tokens: every admitted run gets its own explicit run identity bound to the
current profile, nested runs (delegation) get their own, and the admission is released on every exit.
"""

from types import SimpleNamespace

import pytest

from run_agent import AIAgent
from tools import mcp_client_credentials as cc


def _agent(session_id="s1"):
    return SimpleNamespace(session_id=session_id, platform="cli", _parent_session_id=None, _session_db=None,
                           _conversation_root_id=lambda: session_id)


def test_each_run_is_admitted_with_its_own_identity_and_released(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen = []

    def _loop(*_a, **_k):
        seen.append(cc.current_run())
        return {"final_response": "ok", "completed": True, "interrupted": False}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", _loop)
    agent = _agent()
    AIAgent.run_conversation(agent, "one", task_id="task-1")
    AIAgent.run_conversation(agent, "two", task_id="task-1")
    first, second = seen
    assert first.run_id.startswith("s1:task-1:") and second.run_id.startswith("s1:task-1:")
    assert first.run_id != second.run_id and first.closed and second.closed
    from hermes_constants import hermes_home_key
    assert first.profile_key == hermes_home_key(tmp_path)
    assert cc.current_run() is None


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt(), TimeoutError("late")])
def test_run_admission_is_released_on_error_interrupt_and_timeout(monkeypatch, exc):
    seen = []

    def _loop(*_a, **_k):
        seen.append(cc.current_run())
        raise exc
    monkeypatch.setattr("agent.conversation_loop.run_conversation", _loop)
    with pytest.raises(type(exc)):
        AIAgent.run_conversation(_agent(), "x", task_id="t")
    assert seen[0] is not None and seen[0].closed and cc.current_run() is None


def test_nested_child_run_gets_its_own_admission_and_restores_the_parent(monkeypatch):
    seen = {}

    def _loop(agent, *_a, **_k):
        if agent.session_id == "parent":
            seen["parent"] = cc.current_run()
            AIAgent.run_conversation(_agent("child"), "sub", task_id="child-task")
            seen["parent_after"] = cc.current_run()
        else:
            seen["child"] = cc.current_run()
        return {"final_response": "ok", "completed": True, "interrupted": False}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", _loop)
    AIAgent.run_conversation(_agent("parent"), "go", task_id="parent-task")
    assert seen["child"] is not seen["parent"] and seen["child"].closed
    assert seen["parent_after"] is seen["parent"] and seen["parent"].closed

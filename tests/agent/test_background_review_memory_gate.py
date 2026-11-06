"""The background-review fork's memory tool is gated by the PARENT chat's audience.

The fork shares the parent's MEMORY.md/USER.md store and runs ``memory`` tool calls after a turn.
``filter_builtin_memory`` judges those calls by the agent's transport identity, so the fork must be
constructed with the parent's chat/sender identity; without it a group chat's review would look
like an identity-less (CLI) session to the gate.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_init_params import _GATEWAY_IDENTITY_PARAMS
from agent.background_review import build_cache_parity_fork
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from run_agent import AIAgent
from tools.memory_tool_store import MemoryStore

IDENTITY = {"platform": "slack", "chat_id": "C123", "chat_type": "group", "user_id": "U9",
            "thread_id": "1700000000.1", "gateway_session_key": "agent:main:slack:group:C123"}


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: mgr)
    return mgr


@pytest.fixture()
def agent_patches():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        yield


def _parent():
    agent = AIAgent(
        api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="fake/model",
        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id="sess-review-gate",
        **IDENTITY,
    )
    agent.client = MagicMock()
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()
    agent._memory_store, agent._memory_enabled, agent._user_profile_enabled = store, True, True
    return agent


def test_fork_carries_the_parent_audience(manager, agent_patches):
    parent = _parent()
    fork, _rt, routed = build_cache_parity_fork(parent, max_iterations=3)
    assert not routed
    for name in _GATEWAY_IDENTITY_PARAMS:
        assert getattr(fork, f"_{name}") == getattr(parent, f"_{name}"), name
    assert fork.platform == "slack"


def test_fork_memory_tool_is_gated_by_the_parent_chat(manager, agent_patches):
    seen = []

    def gate(chat_id, chat_type, sender_id, surface, **_):
        seen.append((chat_id, chat_type, sender_id, surface))
        return "omit" if chat_type == "group" else "allow"

    PluginContext(manager=manager, manifest=PluginManifest(name="gate", source="user")).register_hook(
        "filter_builtin_memory", gate)
    fork, _rt, _routed = build_cache_parity_fork(_parent(), max_iterations=3)
    ctx = InlineToolContext(effective_task_id="review", tool_call_id="call-1")
    result = json.loads(INLINE_TOOL_EXECUTORS["memory"](
        fork, {"action": "add", "target": "memory", "content": "Group chat detail."}, ctx))

    assert result.get("success") is False
    assert seen == [("C123", "group", "U9", "memory_tool")]
    fork._memory_store.load_from_disk()
    assert not fork._memory_store.format_for_system_prompt("memory")

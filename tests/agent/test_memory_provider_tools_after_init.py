"""A memory-provider tool that exists only once the provider is initialized is dispatched (2026-11-15).

``MemoryManager.add_provider`` reads a provider's tool schemas BEFORE ``initialize``; the agent's tool surface
(``inject_memory_provider_tools`` -> ``get_all_tool_schemas``) is read AFTER it. A provider whose tools depend on
its initialized state (the hindsight plugin's privacy tools: privacy_release, approval_*, memory_share, ...) had
them shown to the model while dispatch answered "Unknown tool". These tests drive the real sequential and
concurrent dispatch helpers with a real ``MemoryManager``.
"""
import json
import logging
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import resolve_invoke_tool_executor
from agent.memory_manager import MemoryManager
from agent.memory_manager_tools import inject_memory_provider_tools
from agent.memory_provider import MemoryProvider
from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_EARLY, _LATE = "lazy_recall", "lazy_release"


@pytest.fixture(autouse=True)
def _profile_home(tmp_path):
    """The calling turn's profile is the one the manager was initialized for (memory tools refuse otherwise)."""
    token = set_hermes_home_override(str(tmp_path))
    yield
    reset_hermes_home_override(token)


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


class _LazyToolsProvider(MemoryProvider):
    """Offers one tool before initialize and a second one only after it, as the hindsight plugin does."""

    def __init__(self):
        self.ready = False
        self.calls = []

    @property
    def name(self):
        return "lazytools"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.ready = True

    def get_tool_schemas(self):
        return [_schema(_EARLY)] + ([_schema(_LATE)] if self.ready else [])

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append(tool_name)
        return json.dumps({"ok": tool_name})


def _agent(tmp_path, provider):
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id="s-1", platform="api_server", hermes_home=str(tmp_path))
    agent = SimpleNamespace(
        _memory_manager=manager, _context_engine_tool_names=set(), _gateway_session_key="gsk",
        _should_emit_quiet_tool_messages=lambda: False, _should_start_quiet_spinner=lambda: False,
        tools=[], valid_tool_names=set(), enabled_toolsets=None, disabled_toolsets=None,
        platform="api_server", _skip_memory=False,
    )
    return agent


def test_a_tool_offered_only_after_initialize_is_routed(tmp_path, caplog):
    provider = _LazyToolsProvider()
    with caplog.at_level(logging.INFO, logger="agent.memory_manager"):
        agent = _agent(tmp_path, provider)
    manager = agent._memory_manager
    assert manager.has_tool(_EARLY) and manager.has_tool(_LATE)
    assert "tools after initialize: 2 (+1 routed)" in caplog.text


def test_the_late_tool_goes_through_the_real_sequential_dispatch(tmp_path):
    provider = _LazyToolsProvider()
    agent = _agent(tmp_path, provider)
    ref = _ToolCallRef(name=_LATE, args={}, task_id="t", call_id="c", trace=[])
    dispatch = _resolve_sequential_dispatch(agent, ref, [])
    assert json.loads(dispatch.execute({})) == {"ok": _LATE}
    assert provider.calls == [_LATE]


def test_the_late_tool_goes_through_the_real_concurrent_dispatch(tmp_path):
    provider = _LazyToolsProvider()
    agent = _agent(tmp_path, provider)
    executor = resolve_invoke_tool_executor(agent, _LATE)
    assert executor is not None
    assert json.loads(executor(agent, {}, SimpleNamespace())) == {"ok": _LATE}


def test_every_tool_the_model_is_shown_is_dispatchable(tmp_path, monkeypatch):
    """The surface the model gets and the routing table agree: nothing is shown that can't run."""
    import agent.memory_manager_tools as mmt

    monkeypatch.setattr(mmt, "memory_provider_tools_exposed", lambda agent: True)
    agent = _agent(tmp_path, _LazyToolsProvider())
    inject_memory_provider_tools(agent)
    shown = {tool["function"]["name"] for tool in agent.tools}
    assert shown == {_EARLY, _LATE}
    assert all(agent._memory_manager.has_tool(name) for name in shown)


class _CoreShadowLateProvider(_LazyToolsProvider):
    def get_tool_schemas(self):
        return [_schema(_EARLY)] + ([_schema("clarify")] if self.ready else [])


def test_a_late_tool_still_cant_shadow_a_core_tool(tmp_path):
    agent = _agent(tmp_path, _CoreShadowLateProvider())
    assert not agent._memory_manager.has_tool("clarify")


class _BrokenAfterInitProvider(_LazyToolsProvider):
    def get_tool_schemas(self):
        if self.ready:
            raise RuntimeError("schema bug")
        return [_schema(_EARLY)]


def test_a_provider_whose_schemas_break_after_initialize_keeps_its_earlier_routes(tmp_path):
    agent = _agent(tmp_path, _BrokenAfterInitProvider())
    assert agent._memory_manager.has_tool(_EARLY) and not agent._memory_manager.has_tool(_LATE)

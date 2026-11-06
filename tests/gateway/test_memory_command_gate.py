"""The gateway ``/memory`` command goes through ``filter_builtin_memory`` (surface ``memory_command``).

It reviews and applies pending memory writes, echoing staged entry text that has no circle yet, so it
runs only on a plain ``"allow"`` (or with no callback registered); a dict verdict, ``"omit"`` or any
failure refuses it without touching the pending store.
"""

import pytest

from agent.builtin_memory_gate import COMMAND_REFUSED_NOTE
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import write_approval as wa

STAGED = "Staged fact: the spare key is under the blue pot."
GROUP = "120363000000000002@g.us"
MEMBER = "15550000002@s.whatsapp.net"


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    (tmp_path / "hermes_home").mkdir()
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_module, "get_plugin_manager", lambda: mgr)
    return mgr


@pytest.fixture()
def pending_id(manager):
    record = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": STAGED},
                            summary=f"add to memory: {STAGED}", origin="foreground")  # as the tool stages it
    return record["id"]


class _Runner(GatewayRunner):
    def __init__(self):  # skip GatewayRunner.__init__: only the slash handler is exercised
        pass

    def _session_key_for_source(self, source):
        return "agent:main:whatsapp:group:" + str(source.chat_id)


class _Event:
    def __init__(self, args: str):
        self._args = args
        self.source = SessionSource(platform=Platform.WHATSAPP, chat_id=GROUP, chat_type="group",
                                    user_id=MEMBER, user_name="Bea")

    def get_command_args(self) -> str:
        return self._args


def _register(manager, callback):
    PluginContext(manager=manager, manifest=PluginManifest(name="gate", source="user")).register_hook(
        "filter_builtin_memory", callback)


async def _memory(args: str) -> str:
    return await _Runner()._handle_memory_command(_Event(args))


@pytest.mark.asyncio
async def test_no_hook_lists_pending_writes_as_before(manager, pending_id):
    assert STAGED in await _memory("pending")


@pytest.mark.asyncio
async def test_allow_runs_the_command_and_sees_the_trusted_source(manager, pending_id):
    seen = []

    def gate(**kwargs):
        seen.append(kwargs)
        return "allow"

    _register(manager, gate)
    assert STAGED in await _memory("pending")
    expected = {"session_id": "", "platform": "whatsapp", "chat_id": GROUP, "chat_type": "group",
                "sender_id": MEMBER, "thread_id": "", "session_key": "agent:main:whatsapp:group:" + GROUP,
                "surface": "memory_command"}
    assert len(seen) == 1 and {key: seen[0].get(key) for key in expected} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["omit", {"keep": lambda tags: True}, {"write_circle": "household"}, "maybe"])
async def test_non_allow_verdicts_refuse_without_touching_pending(manager, pending_id, answer):
    _register(manager, lambda **_: answer)
    reply = await _memory(f"approve {pending_id}")
    assert reply == COMMAND_REFUSED_NOTE
    assert STAGED not in reply
    assert wa.pending_count(wa.MEMORY) == 1  # nothing approved


@pytest.mark.asyncio
async def test_any_non_allow_wins_over_allow(manager, pending_id):
    _register(manager, lambda **_: "allow")
    _register(manager, lambda **_: "omit")
    assert await _memory("pending") == COMMAND_REFUSED_NOTE


@pytest.mark.asyncio
async def test_raising_hook_refuses(manager, pending_id):
    def gate(**_):
        raise RuntimeError("roster lookup failed")

    _register(manager, gate)
    assert await _memory("pending") == COMMAND_REFUSED_NOTE


@pytest.mark.asyncio
async def test_dispatch_failure_refuses(manager, pending_id, monkeypatch):
    _register(manager, lambda **_: "allow")

    def broken(*_args, **_kwargs):
        raise RuntimeError("plugin manager unavailable")

    monkeypatch.setattr(plugins_module, "invoke_hook", broken)
    assert await _memory("pending") == COMMAND_REFUSED_NOTE

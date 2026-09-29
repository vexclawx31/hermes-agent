"""Actual command dispatch preserves routing and propagates live evidence."""

from types import SimpleNamespace
from unittest.mock import patch
import pytest
from gateway.ingress_identity import get_trusted_ingress
from gateway.session_context import get_session_env
from gateway.config import GatewayConfig
from tests.gateway.test_dedicated_ingress_identity import build


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic", [False, True])
async def test_command_entrypoint_evidence_and_routing(monkeypatch, synthetic):
    r, a, e = await build()
    r.config = GatewayConfig()
    r.config.multiplex_profiles = False
    r._draining = False
    e.text = "/fixture arg"
    if synthetic:
        import copy

        e = copy.copy(e)
    seen = {}

    async def handler(args):
        seen["message"] = get_session_env("HERMES_SESSION_MESSAGE_ID")
        seen["evidence"] = get_trusted_ingress()
        return args

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler", lambda name: handler
    )
    handled, result, _ = await r._hm_dispatch_quick_and_plugin_commands(
        e, e.source, "fixture"
    )
    assert handled and result == "arg"
    assert seen["message"] == e.message_id
    assert (seen["evidence"] is None) == synthetic
    assert get_trusted_ingress() is None


@pytest.mark.asyncio
async def test_background_routing_entrypoint(monkeypatch):
    import json
    from tools import terminal_tool_background as bg
    from gateway.session_context import clear_session_vars

    r, a, e = await build()
    tokens = r._set_session_env(
        SimpleNamespace(source=e.source, session_key="fixture"), event=e
    )
    proc = SimpleNamespace(id="fixture", pid=1, watcher_platform=None)

    def spawn(*args, **kwargs):
        bg._stamp_gateway_routing(proc, get_session_env)
        return proc

    monkeypatch.setattr(bg, "_spawn", spawn)
    monkeypatch.setattr(bg, "_register_completion_watcher", lambda *a: None)
    monkeypatch.setattr(bg, "_apply_async_support", lambda p, d, n, w: (n, w))
    try:
        result = json.loads(
            bg.spawn_background_process(
                command="fixture",
                env=None,
                env_type="local",
                effective_task_id="fixture",
                task_id=None,
                session_key="fixture",
                workdir=None,
                cwd=".",
                effective_pty=False,
                notify_on_complete=True,
                watch_patterns=None,
                approval_note=None,
                pty_disabled_reason=None,
            )
        )
        assert result["exit_code"] == 0
        assert proc.watcher_message_id == e.message_id
        assert proc.watcher_thread_id == e.source.thread_id
    finally:
        clear_session_vars(tokens)

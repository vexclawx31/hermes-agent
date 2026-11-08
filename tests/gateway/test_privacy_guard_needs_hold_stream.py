"""Fork (v0.23 release, privacy proof): a privacy guard must declare ``hold_stream`` on v0.23.

The fleet's privacy plugin (hindsight privacy) registers ``transform_llm_output`` with ``priority`` and
``fail_closed_text`` (#50) and relies on the profile's ``streaming.enabled: false`` to keep unchecked
text from the user. On v0.23 that is not enough on its own: every early channel (token stream, interim
commentary between tool calls, split heads) is gated by ``hold_stream``, not by the streaming switch.
These tests run the real ``GatewayRunner._run_agent`` + ``AIAgent`` against a loopback model (the
upstream ``test_hold_stream_delivery`` harness) with streaming DISABLED, and pin which registration
keeps every raw secret away from the chat.
"""

import os
from pathlib import Path

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home
from tests.fakes.redactor_plugin import _CONFIG, REWRITE, raw_leaks, redactor_turn
from tests.gateway.test_hold_stream_delivery import CHATS, _frames
from tests.gateway.test_run_progress_topics import _make_runner

_GUARD = """import re


def _redact(response_text, **_):
    return re.sub(r"SECRET-\\\\d+", "[REDACTED]", response_text)


def register(ctx):
    ctx.register_hook("transform_llm_output", _redact, priority=-1000, fail_closed_text="(withheld)"{hold})
"""


def _guard_home(home: Path, base_url: str, *, hold: bool) -> None:
    plugin = home / "plugins" / "redactor"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text("name: redactor\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_GUARD.format(hold=", hold_stream=True" if hold else ""), encoding="utf-8")
    write_hermes_home(home, base_url, extra_config=_CONFIG + "streaming:\n  enabled: false\n")


async def _deliver(chat, *, hold, monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    platform = Platform.SLACK if chat.platform == Platform.RELAY else Platform.TELEGRAM
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    event = MessageEvent(text="show me the config", source=source, message_type=MessageType.TEXT)
    session_key = f"agent:main:{platform.value}:dm:c1"
    with FakeLLMServer(redactor_turn(delay_per_chunk=0.03)) as srv:
        _guard_home(home, srv.base_url, hold=hold)
        plugins._reset_plugin_managers_for_tests()
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
            "api_key": "sk-fake-e2e", "base_url": srv.base_url, "provider": "custom", "api_mode": "chat_completions"})
        runner = _make_runner(chat)
        runner.adapters[platform] = chat
        runner.config.streaming = StreamingConfig(enabled=False)  # the privacy profile's setting
        if hasattr(chat, "_capture_scope"):
            chat._capture_scope(event)

        async def handler(event):
            result = await runner._run_agent(message=event.text, context_prompt="", history=[], source=source,
                                             session_id="s1", session_key=session_key)
            if hold:
                assert result["final_response"] == REWRITE
            return None if result.get("already_sent") else result["final_response"]

        chat.set_message_handler(handler)
        try:
            await chat._process_message_background(event, session_key)
        finally:
            plugins._reset_plugin_managers_for_tests()
    return _frames(chat)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", list(CHATS))
async def test_a_guard_that_holds_the_stream_shows_only_the_rewrite(shape, monkeypatch):
    frames = await _deliver(CHATS[shape](), hold=True, monkeypatch=monkeypatch)
    assert raw_leaks(text for _, text in frames) == [], frames


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", list(CHATS))
async def test_streaming_off_alone_still_shows_unchecked_commentary(shape, monkeypatch):
    """Streaming off is NOT a privacy control on v0.23: the model's commentary between tool calls reaches
    the chat before (and without) the guard's rewrite. Pinned so the plugin keeps declaring hold_stream."""
    frames = await _deliver(CHATS[shape](), hold=False, monkeypatch=monkeypatch)
    assert any("SECRET-777" in text for text in raw_leaks(text for _, text in frames)), frames

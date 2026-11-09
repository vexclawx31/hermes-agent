"""Fork (launch adapter): the trusted turn origin an opted-in MCP server receives as
``_meta["hermes/session_source"]``.

A help-desk door that requires a source (``HELPDESK_SESSION_META=1``) refuses a call without one. These
tests pin the contract the door relies on: an ordinary platform turn (Slack, WhatsApp) is a KNOWN caller
(``platform:<name>``); a labelled local session keeps its own label (a tenant turn stays
``switchboard-tenant``, which the door refuses to delegate); genuinely missing context (a local session
without an origin, a failed policy lookup, no source at all) sends nothing, so the door refuses; and
concurrent turns never see each other's origin.
"""
import asyncio
import threading
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from gateway import session_context
from gateway.config import Platform
from gateway.run_turn_origin import launch_origin_source
from gateway.run_turn_origin import turn_origin_label as _turn_origin_label
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars, turn_origin
from tools.mcp_tool_handlers import SESSION_META_KEY, _session_meta

META = SimpleNamespace(_config={"session_meta": True})
PLAIN = SimpleNamespace(_config={})


@pytest.fixture(autouse=True)
def _fresh_context(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    reset_session_vars()
    yield
    reset_session_vars()


def _src(platform, chat_id="C1"):
    return SessionSource(platform=platform, chat_id=chat_id)


def _policy(monkeypatch, result):
    """policy_for_source stub: a policy object, None, or an exception to raise."""
    from gateway import session_policy

    def fake(runner, source):
        if isinstance(result, BaseException):
            raise result
        return result
    monkeypatch.setattr(session_policy, "policy_for_source", fake)


def _meta_for(origin, source=""):
    tokens = set_session_vars(platform="slack", source=source, turn_origin=origin)
    try:
        return _session_meta(META)
    finally:
        clear_session_vars(tokens)


# --- the label the gateway derives from its own routing ------------------------------------------

@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.WHATSAPP, Platform.TELEGRAM])
def test_an_ordinary_platform_turn_is_a_known_caller(platform):
    assert _turn_origin_label(SimpleNamespace(), _src(platform)) == f"platform:{platform.value}"


def test_a_labelled_local_session_keeps_its_own_label(monkeypatch):
    _policy(monkeypatch, SimpleNamespace(origin="switchboard-tenant"))
    assert _turn_origin_label(SimpleNamespace(), _src(Platform.LOCAL)) == "switchboard-tenant"


def test_a_local_session_without_an_origin_is_missing_context(monkeypatch):
    _policy(monkeypatch, SimpleNamespace(origin=None))
    assert _turn_origin_label(SimpleNamespace(), _src(Platform.LOCAL)) == ""
    _policy(monkeypatch, None)
    assert _turn_origin_label(SimpleNamespace(), _src(Platform.LOCAL)) == ""


def test_a_failed_policy_lookup_is_missing_context_not_a_platform_turn(monkeypatch):
    from gateway.session_policy import RuntimeStoreError
    _policy(monkeypatch, RuntimeStoreError("storage_unavailable"))
    assert _turn_origin_label(SimpleNamespace(), _src(Platform.LOCAL)) == ""
    _policy(monkeypatch, RuntimeError("boom"))
    assert _turn_origin_label(SimpleNamespace(), _src(Platform.LOCAL)) == ""


def test_the_launch_source_is_bound_only_for_a_labelled_local_session(monkeypatch):
    _policy(monkeypatch, SimpleNamespace(origin="helpdesk-request"))
    assert launch_origin_source(SimpleNamespace(), _src(Platform.LOCAL)) == {"source": "helpdesk-request"}
    _policy(monkeypatch, RuntimeError("boom"))
    assert launch_origin_source(SimpleNamespace(), _src(Platform.LOCAL)) == {}
    _policy(monkeypatch, None)
    assert launch_origin_source(SimpleNamespace(), _src(Platform.SLACK)) == {}


def test_no_source_is_missing_context():
    assert _turn_origin_label(SimpleNamespace(), None) == ""
    assert _turn_origin_label(SimpleNamespace(), SimpleNamespace(platform="slack")) == ""   # not a Platform


def test_the_gateway_binds_the_origin_for_a_slack_turn_and_clears_it_after():
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    context = SessionContext(source=_src(Platform.SLACK), connected_platforms=[], home_channels={},
                             session_key="agent:main:slack:dm:C1")
    tokens = runner._set_session_env(context)
    try:
        assert _session_meta(META) == {SESSION_META_KEY: "platform:slack"}
    finally:
        runner._clear_session_env(tokens)
    assert _session_meta(META) is None


def test_the_gateway_binds_nothing_for_a_local_turn_whose_policy_lookup_fails(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext
    from gateway.session_policy import RuntimeStoreError
    _policy(monkeypatch, RuntimeStoreError("storage_unavailable"))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    context = SessionContext(source=_src(Platform.LOCAL, "s1"), connected_platforms=[], home_channels={},
                             session_key="agent:main:local:dm:s1")
    tokens = runner._set_session_env(context)
    try:
        assert _session_meta(META) is None
    finally:
        runner._clear_session_env(tokens)


# --- what the opted-in MCP server receives ---------------------------------------------------------

def test_a_platform_turn_sends_its_platform_label():
    assert _meta_for("platform:slack") == {SESSION_META_KEY: "platform:slack"}


def test_a_labelled_tenant_turn_sends_the_tenant_label_so_the_door_refuses_delegation():
    # The launch origin is bound as HERMES_SESSION_SOURCE too; either way the door sees the tenant label.
    assert _meta_for("switchboard-tenant", source="switchboard-tenant") == {SESSION_META_KEY: "switchboard-tenant"}
    assert _meta_for("switchboard-tenant") == {SESSION_META_KEY: "switchboard-tenant"}


def test_an_explicit_launch_source_wins_over_the_origin():
    assert _meta_for("platform:slack", source="helpdesk-request") == {SESSION_META_KEY: "helpdesk-request"}


def test_missing_context_sends_no_meta_so_a_requiring_door_refuses():
    assert _meta_for("") is None
    assert _session_meta(META) is None   # nothing bound in this context at all


def test_a_server_that_did_not_opt_in_gets_nothing():
    tokens = set_session_vars(platform="slack", turn_origin="platform:slack")
    try:
        assert _session_meta(PLAIN) is None
    finally:
        clear_session_vars(tokens)


def test_clear_and_reset_drop_the_origin():
    tokens = set_session_vars(platform="slack", turn_origin="platform:slack")
    clear_session_vars(tokens)
    assert turn_origin() == ""
    set_session_vars(platform="slack", turn_origin="platform:slack")
    reset_session_vars()
    assert turn_origin() == ""


def test_the_origin_has_no_env_fallback_and_is_not_exported_to_child_processes(monkeypatch):
    monkeypatch.setenv("HERMES_TURN_ORIGIN", "platform:slack")
    assert turn_origin() == ""
    assert "HERMES_TURN_ORIGIN" not in session_context._VAR_MAP


def test_a_cron_run_is_a_known_origin(monkeypatch, tmp_path):
    from cron import scheduler as sched
    from cron.scheduler_run_scope import _CronRunScope
    monkeypatch.setattr(sched, "_resolve_job_workdir", lambda job, job_id: "")
    scope = _CronRunScope({"id": "j1"}, "j1", "e1")
    try:
        assert turn_origin() == "cron"
        assert _session_meta(META) == {SESSION_META_KEY: "cron"}
    finally:
        scope.exit()
    assert turn_origin() == ""


# --- concurrent callers ----------------------------------------------------------------------------

def test_concurrent_turns_on_threads_each_see_their_own_origin():
    origins = ["platform:slack", "switchboard-tenant", "", "platform:whatsapp"] * 4
    seen, barrier = {}, threading.Barrier(len(origins))

    def turn(i, origin):
        tokens = set_session_vars(platform="slack", turn_origin=origin)
        try:
            barrier.wait(timeout=10)   # every turn is bound before any reads
            seen[i] = _session_meta(META)
        finally:
            clear_session_vars(tokens)

    threads = [threading.Thread(target=copy_context().run, args=(turn, i, o)) for i, o in enumerate(origins)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert seen == {i: ({SESSION_META_KEY: o} if o else None) for i, o in enumerate(origins)}


def test_concurrent_asyncio_turns_and_their_tool_threads_each_see_their_own_origin():
    async def turn(origin, gate):
        reset_session_vars()
        tokens = set_session_vars(platform="slack", turn_origin=origin)
        try:
            await gate.wait()
            loop = asyncio.get_running_loop()
            # the MCP handler reads _session_meta on the agent's thread (a copied context)
            return await loop.run_in_executor(None, copy_context().run, _session_meta, META)
        finally:
            clear_session_vars(tokens)

    async def main():
        gate = asyncio.Event()
        tasks = [asyncio.create_task(turn(o, gate)) for o in ("platform:slack", "switchboard-tenant", "")]
        await asyncio.sleep(0)
        gate.set()
        return await asyncio.gather(*tasks)

    assert asyncio.run(main()) == [{SESSION_META_KEY: "platform:slack"},
                                   {SESSION_META_KEY: "switchboard-tenant"}, None]

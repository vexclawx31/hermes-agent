"""Fork (launch adapter): a spawner's run identity on the classic ``-Q`` path, and ``hermes sessions state``.

Under single-gateway ownership a ``hermes chat`` process is a client. A spawner that retries a run
(HERMES_CHAT_RUN_ID) must get the same admission back, never a second turn, and must settle a finished
one from its receipt. One that times out must ask the owner (``sessions state``) and interrupt it there.
"""
import argparse
import asyncio
import json

import pytest

from hermes_cli import gateway_chat_view
from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection

from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with


@pytest.fixture(autouse=True)
def fresh_launch_inputs(monkeypatch):
    monkeypatch.setattr(gateway_chat_view, "_LAUNCH_INPUTS", {})


class ReplayClient(FakeClient):
    """``prompt.submit`` answers with an admission the run already made (status *status*); no frames follow."""

    def __init__(self, status, result=None):
        super().__init__([], result=result)
        self.status = status

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "prompt.submit":
            self.calls.append((method, params))
            return {"admission_id": "adm-1", "status": self.status, "outcome": "completed",
                    "execution_generation": 7}
        return await super().rpc(method, _timeout=_timeout, **params)


def _agent(client):
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    return conn, GatewayAgent(connection=conn, session_id=SID)


def test_run_id_names_the_first_classic_submit_only(monkeypatch):
    monkeypatch.setenv("HERMES_CHAT_RUN_ID", "ask-123")
    client = ReplayClient("terminal", result={"final_response": "done", "api_calls": 1, "completed": True,
                                              "messages": [{"role": "assistant", "content": "done"}],
                                              "_messages_are_turn_suffix": True})
    conn, agent = _agent(client)
    try:
        first = agent.run_conversation("hi")
        second = agent.run_conversation("again")
    finally:
        conn.close()
    submits = [params for method, params in client.calls if method == "prompt.submit"]
    assert submits[0]["submission_id"] == "ask-123-submit"
    assert submits[1]["submission_id"] != "ask-123-submit"  # a later turn of the run is a new admission
    assert first["final_response"] == "done"  # settled from the receipt: no stream was waited on
    assert ("prompt.receipt", {"session_id": SID, "admission_id": "adm-1", "include_result": True}) in [
        (m, {k: v for k, v in p.items() if k != "abandon"}) for m, p in client.calls]
    assert second["final_response"] == "done"


def test_without_a_run_id_every_submit_is_fresh(monkeypatch):
    monkeypatch.delenv("HERMES_CHAT_RUN_ID", raising=False)
    client = ReplayClient("terminal", result={"final_response": "x", "completed": True, "messages": []})
    conn, agent = _agent(client)
    try:
        agent.run_conversation("hi")
    finally:
        conn.close()
    submit = next(params for method, params in client.calls if method == "prompt.submit")
    assert len(submit["submission_id"]) == 32  # uuid4 hex


def test_a_retry_of_a_run_lost_in_a_restart_fails_instead_of_waiting(monkeypatch):
    monkeypatch.setenv("HERMES_CHAT_RUN_ID", "ask-lost")
    client = ReplayClient("unknown")
    conn, agent = _agent(client)
    try:
        result = agent.run_conversation("hi")
    finally:
        conn.close()
    assert result["failed"] is True and "sessions discard" in result["error"]
    assert not [m for m, _ in client.calls if m == "prompt.receipt"]


def test_titled_create_uses_the_run_id(monkeypatch):
    from hermes_cli import gateway_session_store as store_mod

    monkeypatch.setenv("HERMES_CHAT_RUN_ID", "req-9")
    calls = []

    class Store:
        def client_env(self):
            return {}

        def call(self, method, **params):
            calls.append((method, params))
            return {"stored_session_id": "local-" + "b" * 64}

    monkeypatch.setattr(store_mod, "session_store", Store)
    assert store_mod.create_titled_session("request:9") == "local-" + "b" * 64
    assert calls[0][0] == "session.create" and calls[0][1]["request_id"] == "req-9-create"


@pytest.mark.asyncio
async def test_sigterm_cancels_a_spawner_run_and_only_a_spawner_run(monkeypatch):
    import os
    import signal

    from hermes_cli.gateway_chat import _spawner_stop_on_term

    async def run_until_term():
        with _spawner_stop_on_term() as term:
            try:
                asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                return term.fired
        return None

    monkeypatch.setenv("HERMES_CHAT_RUN_ID", "intake-7")
    assert await asyncio.create_task(run_until_term()) is True
    assert asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM) is False  # removed on exit

    monkeypatch.setattr(gateway_chat_view, "_LAUNCH_INPUTS", {})
    monkeypatch.delenv("HERMES_CHAT_RUN_ID", raising=False)
    with _spawner_stop_on_term() as term:  # no run id: nothing is installed, upstream's detach stays
        assert term.fired is False
    assert asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM) is False


@pytest.mark.asyncio
async def test_a_stopped_run_cancels_only_its_own_admissions_until_they_settle():
    """Release review S2: the stop targets the run's admissions (prompt.cancel), never a bare
    session.interrupt that would stop another spawner's turn the run is queued behind."""
    from hermes_cli.gateway_chat import _stop_run_on_owner

    class Client:
        def __init__(self):
            self.calls, self.polls = [], 0

        async def rpc(self, method, **params):
            self.calls.append((method, params))
            if method == "prompt.receipt":
                self.polls += 1
                return {"status": "terminal" if self.polls > 1 else "started"}
            return {"status": "terminal"}

    client = Client()
    snapshot = {"session_id": "rt-1", "stored_session_id": "local-abc"}
    assert await _stop_run_on_owner(client, snapshot, ["adm-mine"]) is True
    assert client.calls[0] == ("prompt.cancel", {"session_id": "rt-1", "admission_id": "adm-mine"})
    assert "session.interrupt" not in [method for method, _ in client.calls]
    assert client.calls[-1] == ("prompt.receipt", {"session_id": "rt-1", "admission_id": "adm-mine"})

    untouched = Client()
    assert await _stop_run_on_owner(untouched, snapshot, []) is True  # nothing admitted: nothing to stop
    assert untouched.calls == []


def test_a_stopped_run_reports_interrupted_130(monkeypatch, capsys):
    from hermes_cli import gateway_chat

    class Emitter:
        def emit_result(self, payload, exit_code):
            self.payload = payload
            return exit_code

    async def stopped(*_a, **_k):
        raise gateway_chat.RunStopped(True)

    from hermes_cli import gateway_chat_startup, stream_json

    emitter = Emitter()
    monkeypatch.setattr(stream_json, "StreamJsonEmitter", lambda **_kw: emitter)
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", stopped)
    monkeypatch.setattr(gateway_chat, "validate_options", lambda _args: None)
    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None)
    monkeypatch.setattr(gateway_chat_startup, "ensure_launch_provider", lambda _args: True)
    args = argparse.Namespace(query="hi", q=None, oneshot=None, query_file=None, output_format="stream-json")
    assert gateway_chat.launch_from_args(args) == 130
    assert emitter.payload == {"interrupted": True}
    assert "interrupted on the gateway" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_sessions_state_reports_and_interrupts_on_the_owner(monkeypatch, capsys):
    from websockets.asyncio.server import serve

    from hermes_cli import sessions_cmd, sessions_state
    from hermes_cli.subcommands.sessions import build_sessions_parser

    monkeypatch.setattr(sessions_state, "_POLL_SECONDS", 0.01)
    state = {"running": True, "resumes": 0}
    calls = []

    def snapshot():
        pending = [{"admission_id": "adm-1", "status": "started", "outcome": None, "execution_generation": 3}]
        return {"session_id": "rt-1", "stored_session_id": "local-abc", "running": state["running"],
                "execution_generation": 3,
                "pending": pending if state["running"] else []}

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            calls.append((request["method"], request["params"]))
            if request["method"] == "session.resume" and request["params"].get("session_id") == "ask:1":
                reply = {"error": {"code": 4001, "message": "not_found", "data": {"reason": "not_found"}}}
            elif request["method"] == "session.resume":
                state["resumes"] += 1
                if state["resumes"] >= 3:
                    state["running"] = False  # the owner settled the interrupt
                reply = {"result": snapshot()}
            else:
                # The real owner answers an accepted interrupt with the session handle (no status field).
                reply = {"result": {"session_id": "rt-1", "execution_state": "running"}}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}))

    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    async with serve(peer, "127.0.0.1", 0) as server:
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")

        def run(*argv):
            args = parser.parse_args(["sessions", "state", "ask:1", *argv])
            return args.func(args)

        assert await asyncio.to_thread(run) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown == {"session_id": "local-abc", "running": True,
                         "pending": [{"admission_id": "adm-1", "status": "started", "outcome": None}]}
        assert "session.interrupt" not in [m for m, _ in calls]  # reading never interrupts

        assert await asyncio.to_thread(run, "--interrupt", "--wait", "5") == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown == {"session_id": "local-abc", "running": False, "pending": [], "interrupted": True}
    # The runtime id the owner answered with, fenced to the generation that read saw.
    assert ("session.interrupt", {"session_id": "rt-1", "execution_generation": 3}) in calls

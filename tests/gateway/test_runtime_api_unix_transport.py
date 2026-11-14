"""``gateway.session_api.transport: unix``: the gateway's runtime API on a 0600 unix socket in its home.

The default stays main's private loopback TCP port. In unix mode no TCP socket is ever bound or
listened on, the same ``GatewayRuntimeAPI`` (ticketed ``/api/ws`` and the dashboard routes) answers
over the socket, shutdown removes it, a stale one is replaced and a live one refuses the start. Native
clients read the owner's ``http+unix://`` origin and dial its socket.
"""
import asyncio
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def short_home(tmp_path):
    """A gateway home whose socket path fits sun_path (pytest's tmp_path may not under a long TMPDIR)."""
    home = tmp_path / ".hermes"
    if len(str(home / "runtime-api.sock")) <= 100:
        home.mkdir()
        yield home.resolve()
        return
    root = Path(tempfile.mkdtemp(prefix="hu-"))
    try:
        yield root.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


_PROBE = textwrap.dedent("""
    import asyncio, json, os, socket, stat, sys, types
    from pathlib import Path
    import httpx
    from websockets.asyncio.client import connect
    from websockets.sync.client import connect as sync_connect

    # Every bind/listen and every create_server without a ready socket, by address family.
    seen = {"bind": [], "listen": [], "create_server": []}
    real_bind, real_listen = socket.socket.bind, socket.socket.listen
    def bind(self, address):
        seen["bind"].append(int(self.family))
        return real_bind(self, address)
    def listen(self, *args):
        seen["listen"].append(int(self.family))
        return real_listen(self, *args)
    socket.socket.bind, socket.socket.listen = bind, listen
    import asyncio.base_events as base_events
    real_create_server = base_events.BaseEventLoop.create_server
    async def create_server(self, factory, host=None, port=None, *args, sock=None, **kwargs):
        if sock is None:
            seen["create_server"].append([host, port])
        return await real_create_server(self, factory, host, port, *args, sock=sock, **kwargs)
    base_events.BaseEventLoop.create_server = create_server

    async def main():
        from gateway import run_api
        from gateway.run_api import start_gateway_api, stop_gateway_api
        from gateway.runtime_bootstrap import TicketStore
        from hermes_cli.gateway_client import gateway_ws_target, gateway_ws_transport
        import tui_gateway.ws

        async def fake_handle_ws(ws, **kwargs):
            await ws.accept(subprotocol="hermes-gateway-v1")
            identity = kwargs["auth_identity"]
            await ws.send_json({"handler": "handle_ws", "profile_id": identity["profile_id"],
                                "native": identity["native_bootstrap"]})
            await ws.close()
        tui_gateway.ws.handle_ws = fake_handle_ws

        home, mode = Path(sys.argv[1]), sys.argv[2]
        tickets = TicketStore("owner", [str(home)])
        authority = types.SimpleNamespace(db=types.SimpleNamespace(db_path=str(home / "state.db")),
                                          profile_id=str(home))
        runner = types.SimpleNamespace(_draining=False, session_runtime_descriptor={"state": "ready"},
                                       session_authority=authority, session_ticket_store=tickets)
        out = {}
        path = run_api.runtime_api_socket_path(home)
        if mode == "unix":
            handle = await start_gateway_api(runner, unix_path=path)
        else:
            handle = await start_gateway_api(runner)
        out["listening"] = {k: list(v) for k, v in seen.items()}
        out["api_origin"] = handle.api_origin
        out["socket_exists"] = path.exists()
        if out["socket_exists"]:
            info = path.lstat()
            out["socket_mode"] = stat.S_IMODE(info.st_mode)
            out["is_socket"] = stat.S_ISSOCK(info.st_mode)
        try:
            endpoint = types.SimpleNamespace(api_origin=handle.api_origin)
            def ticket():
                return tickets.mint(profile_id=str(home), subject="human", purpose="interactive")
            url, protocols = gateway_ws_target(endpoint, ticket())
            out["url"] = url
            out["transport"] = gateway_ws_transport(endpoint)
            async with connect(url, subprotocols=protocols, proxy=None, **gateway_ws_transport(endpoint)) as ws:
                out["native"] = json.loads(await ws.recv())
            def sync_dial():
                url, protocols = gateway_ws_target(endpoint, ticket())
                with sync_connect(url, subprotocols=protocols, proxy=None, **gateway_ws_transport(endpoint)) as ws:
                    return json.loads(ws.recv())
            out["native_sync"] = await asyncio.to_thread(sync_dial)
            if mode == "unix":
                transport = httpx.AsyncHTTPTransport(uds=str(path))
                async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1",
                                             trust_env=False, timeout=60) as client:
                    out["status"] = (await client.get("/api/status")).status_code
                    out["rebind_http"] = (await client.get(
                        "/api/status", headers={"Host": "attacker.invalid"})).status_code
                # A second gateway's listener at the same path refuses while this one is live.
                try:
                    run_api._unix_listener_socket(path)
                    out["live_refused"] = False
                except RuntimeError as exc:
                    out["live_refused"] = "in use" in str(exc)
        finally:
            await stop_gateway_api(handle)
        out["socket_after_stop"] = path.exists()
        out["listening_total"] = {k: list(v) for k, v in seen.items()}
        print("RESULT " + json.dumps(out), file=sys.stderr, flush=True)

    asyncio.run(main())
""")


def _probe(home, mode):
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "HERMES_RUNTIME_DIR", "TMPDIR"}}
    env.update(HOME=str(home.parent), HERMES_HOME=str(home), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", _PROBE, str(home), mode], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=180, check=False)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_the_default_transport_stays_a_private_loopback_tcp_port(short_home):
    out = _probe(short_home, "tcp")
    assert out["api_origin"].startswith("http://127.0.0.1:")
    assert int(socket.AF_INET) in out["listening"]["listen"]
    assert out["socket_exists"] is False
    assert out["transport"] == {} and out["url"] == out["api_origin"].replace("http:", "ws:") + "/api/ws"
    assert out["native"] == {"handler": "handle_ws", "profile_id": str(short_home), "native": True}
    assert out["native_sync"] == out["native"]


def test_unix_transport_serves_the_same_api_on_a_private_socket_and_never_listens_on_tcp(short_home):
    out = _probe(short_home, "unix")
    path = short_home / "runtime-api.sock"
    # No TCP socket listened, nor was any server created from a host/port, at any point. The
    # listener's start binds nothing but its socket; later, loading the dashboard imports urllib3,
    # whose HAS_IPV6 probe binds (never listens on) a throwaway ::1 socket.
    assert set(out["listening"]["bind"]) == {int(socket.AF_UNIX)}, out["listening"]
    for kind in ("listening", "listening_total"):
        assert set(out[kind]["listen"]) == {int(socket.AF_UNIX)}, out[kind]
        assert out[kind]["create_server"] == [], out[kind]
    assert out["socket_exists"] and out["is_socket"] and out["socket_mode"] == 0o600
    assert out["api_origin"].startswith("http+unix://")
    from hermes_cli.gateway_runtime import api_socket_path
    assert api_socket_path(out["api_origin"]) == str(path)
    assert out["transport"] == {"unix": True, "path": str(path)}
    assert out["url"] == "ws://127.0.0.1/api/ws"
    # The ticketed native upgrade reaches the same handler, from the async and the sync client.
    assert out["native"] == {"handler": "handle_ws", "profile_id": str(short_home), "native": True}
    assert out["native_sync"] == out["native"]
    # Dashboard routes answer over the socket, behind the same Host guard.
    assert out["status"] == 200 and out["rebind_http"] == 400
    assert out["live_refused"] is True
    # Shutdown removes the socket.
    assert out["socket_after_stop"] is False


def test_a_stale_socket_is_replaced_and_a_live_one_refuses(short_home):
    from gateway import run_api
    path = run_api.runtime_api_socket_path(short_home)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()  # a dead gateway's leftover: the node stays, nothing answers
    assert stat.S_ISSOCK(path.lstat().st_mode)
    listener = run_api._unix_listener_socket(path)
    try:
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600
        listener.listen(1)
        identity = (path.lstat().st_dev, path.lstat().st_ino)
        with pytest.raises(RuntimeError, match="in use"):
            run_api._unix_listener_socket(path)
        assert (path.lstat().st_dev, path.lstat().st_ino) == identity  # the live socket is untouched
    finally:
        listener.close()
        path.unlink()


@pytest.mark.parametrize("make, message", [
    (lambda p: p.write_text("x"), "not a socket"),
    (lambda p: p.symlink_to(p.parent / "elsewhere.sock"), "symlink"),
])
def test_a_non_socket_at_the_path_refuses_the_start(short_home, make, message):
    from gateway import run_api
    path = run_api.runtime_api_socket_path(short_home)
    make(path)
    with pytest.raises(RuntimeError, match=message):
        run_api._unix_listener_socket(path)
    assert path.is_symlink() or path.read_text() == "x"


def test_a_socket_path_too_long_for_sun_path_refuses_the_start(tmp_path):
    from gateway import run_api
    with pytest.raises(RuntimeError, match="too long"):
        run_api._unix_listener_socket(tmp_path / ("d" * 120) / "runtime-api.sock")


@pytest.mark.parametrize("config, expected", [
    (None, "tcp"),
    ("gateway:\n  allow_all_users: false\n", "tcp"),
    ("gateway:\n  session_api:\n    transport: tcp\n", "tcp"),
    ("gateway:\n  session_api:\n    transport: unix\n", "unix"),
])
def test_the_transport_setting(tmp_path, config, expected):
    from gateway.webapp_listener import session_api_transport
    if config is not None:
        (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    assert session_api_transport(tmp_path) == expected


@pytest.mark.parametrize("config", [
    "gateway:\n  session_api:\n    transport: Unix\n",
    "gateway:\n  session_api: unix\n",
])
def test_an_unknown_transport_refuses_rather_than_falling_back_to_tcp(tmp_path, config):
    from gateway.webapp_listener import session_api_transport
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    with pytest.raises(RuntimeError, match="gateway.session_api.transport"):
        session_api_transport(tmp_path)


def _start_runtime_api(monkeypatch, home, config):
    from gateway import run_api, run_runtime
    (home / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    async def start(runner, *, host="127.0.0.1", port=0, webapp=None, unix_path=None):
        calls.append({"host": host, "port": port, "webapp": webapp, "unix_path": unix_path})
        origin = run_api.unix_api_origin(unix_path) if unix_path else f"http://127.0.0.1:{port or 50001}"
        return SimpleNamespace(api_origin=origin)

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    from gateway.control_socket import GatewayControlServer
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[],
                             session_control_server=GatewayControlServer(home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    return runner, calls


def test_the_gateway_starts_its_runtime_api_on_the_configured_transport(monkeypatch, short_home, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="gateway.run_runtime")
    runner, calls = _start_runtime_api(monkeypatch, short_home, "gateway:\n  session_api:\n    transport: unix\n")
    assert calls == [{"host": "127.0.0.1", "port": 0, "webapp": None,
                      "unix_path": short_home / "runtime-api.sock"}]
    assert runner.session_runtime_descriptor["api_origin"].startswith("http+unix://")
    assert runner.webapp_status == {"state": "off", "url": None, "reason": None}
    assert "transport: unix socket" in caplog.text


def test_unix_transport_keeps_the_webapp_off_and_says_why(monkeypatch, short_home):
    runner, calls = _start_runtime_api(
        monkeypatch, short_home,
        "webapp:\n  enabled: true\n  port: 9555\ngateway:\n  session_api:\n    transport: unix\n")
    assert [c["unix_path"] for c in calls] == [short_home / "runtime-api.sock"]
    assert all(c["port"] == 0 and c["webapp"] is None for c in calls)
    assert runner.webapp_status["state"] == "unavailable"
    assert "transport is unix" in runner.webapp_status["reason"]


def test_the_default_gateway_start_is_unchanged(monkeypatch, short_home, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="gateway.run_runtime")
    runner, calls = _start_runtime_api(monkeypatch, short_home, "gateway:\n  allow_all_users: false\n")
    assert calls == [{"host": "127.0.0.1", "port": 0, "webapp": None, "unix_path": None}]
    assert runner.session_runtime_descriptor["api_origin"] == "http://127.0.0.1:50001"
    assert "transport: tcp http://127.0.0.1:50001" in caplog.text


def _identify(origin, home):
    return {"runtime_protocol": 1, "served_profiles": [{"profile_id": str(home), "home": str(home)}],
            "state": "ready", "capabilities": ["session-authority-v1"], "authority_epoch": 1,
            "instance_id": "owner", "api_origin": origin, "supervisor": "none"}


def test_discovery_accepts_the_owners_own_socket_and_nothing_else(tmp_path):
    from gateway.run_api import unix_api_origin
    from hermes_cli.gateway_runtime import _endpoint
    home = (tmp_path / "home")
    home.mkdir()
    home = home.resolve()
    own = unix_api_origin(home / "runtime-api.sock")
    found = _endpoint(_identify(own, home), home)
    assert found.state == "ready" and found.endpoint.api_origin == own
    # Served by a multiplexer: the socket is in the home whose control socket answered.
    root = tmp_path / "root"
    root.mkdir()
    root = root.resolve()
    served = _endpoint(_identify(unix_api_origin(root / "runtime-api.sock"), home), home, control_home=root)
    assert served.state == "ready"
    for origin in (unix_api_origin(tmp_path / "elsewhere" / "runtime-api.sock"),
                   unix_api_origin(home / "other.sock"),
                   "http+unix://relative%2Fruntime-api.sock",
                   "http+unix://"):
        refused = _endpoint(_identify(origin, home), home)
        assert (refused.state, refused.reason_code) == ("inaccessible", "invalid_api_origin"), origin
    assert _endpoint(_identify("http://127.0.0.1:41817", home), home).state == "ready"


def test_ws_target_and_transport_in_both_modes():
    from hermes_cli.gateway_client import gateway_ws_target, gateway_ws_transport
    tcp = SimpleNamespace(api_origin="http://127.0.0.1:9")
    assert gateway_ws_target(tcp, "t")[0] == "ws://127.0.0.1:9/api/ws" and gateway_ws_transport(tcp) == {}
    unix = SimpleNamespace(api_origin="http+unix://%2Fh%2Fruntime-api.sock")
    assert gateway_ws_target(unix, "t")[0] == "ws://127.0.0.1/api/ws"
    assert gateway_ws_transport(unix) == {"unix": True, "path": "/h/runtime-api.sock"}


def test_the_ink_tui_bootstrap_refuses_a_unix_owner_by_name(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ink_gateway_bootstrap",
                                                  REPO / "ui-tui" / "scripts" / "gateway_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from hermes_cli import gateway_runtime
    endpoint = SimpleNamespace(api_origin="http+unix://%2Fh%2Fruntime-api.sock", profile_id="p",
                               instance_id="i", control_home=None)
    ready = gateway_runtime.GatewayDiscovery("ready", endpoint)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", lambda *a, **k: ready)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *a, **k: ready)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="unix socket"):
        module.bootstrap(False)

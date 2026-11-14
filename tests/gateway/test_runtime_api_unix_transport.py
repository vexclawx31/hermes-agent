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
    root = Path(tempfile.mkdtemp(prefix="u"))
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
            extra = gateway_ws_transport(endpoint)
            out["transport"] = sorted(extra)
            if "sock" in extra:
                extra["sock"].close()
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
    assert out["transport"] == [] and out["url"] == out["api_origin"].replace("http:", "ws:") + "/api/ws"
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
    assert out["transport"] == ["sock", "unix"]  # connected and authenticated before the ticket goes out
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
    with pytest.raises(FileNotFoundError):  # nothing there: never dialed
        gateway_ws_transport(unix)


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


# --- review round 1 ---------------------------------------------------------------------------------


def _listening(home):
    from gateway import run_api
    path = run_api.runtime_api_socket_path(home)
    listener = run_api._unix_listener_socket(path)
    listener.listen(1)
    return path, listener


def test_the_socket_is_made_private_without_touching_the_process_umask(monkeypatch, short_home):
    from gateway import run_api

    def no_umask(*_):
        raise AssertionError("the process umask is shared with other threads")

    monkeypatch.setattr(os, "umask", no_umask)
    path, listener = _listening(short_home)
    try:
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600 and stat.S_ISSOCK(path.lstat().st_mode)
        assert sorted(p.name for p in short_home.iterdir()) == ["runtime-api.sock"]  # staging dir removed
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:  # renamed node still answers
            peer.connect(str(path))
    finally:
        listener.close()
        path.unlink()
    assert run_api.runtime_api_socket_path(short_home) == path


def test_the_client_sends_nothing_to_a_socket_another_user_could_reach(short_home):
    from hermes_cli.gateway_client import gateway_ws_transport
    from hermes_cli.gateway_runtime_discovery import DiscoveryError
    from gateway.run_api import unix_api_origin
    path, listener = _listening(short_home)
    endpoint = SimpleNamespace(api_origin=unix_api_origin(path))
    try:
        extra = gateway_ws_transport(endpoint)
        assert extra["unix"] is True and extra["sock"].fileno() != -1  # connected, peer checked
        extra["sock"].close()
        os.chmod(path, 0o666)
        with pytest.raises(DiscoveryError) as opened:
            gateway_ws_transport(endpoint)
        assert opened.value.reason == "unsafe_control_permissions"
    finally:
        listener.close()
        path.unlink()
    target = short_home / "real.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as real:
        real.bind(str(target))
        real.listen(1)
        path.symlink_to(target)
        with pytest.raises(DiscoveryError) as linked:
            gateway_ws_transport(endpoint)
        assert linked.value.reason == "unsafe_control_path"


def test_the_client_refuses_a_listener_run_by_another_user(monkeypatch, short_home):
    from hermes_cli import gateway_runtime_discovery
    from hermes_cli.gateway_client import gateway_ws_transport
    from gateway.run_api import unix_api_origin
    path, listener = _listening(short_home)
    seen = []
    monkeypatch.setattr(gateway_runtime_discovery, "socket_peer_uid",
                        lambda sock: seen.append(sock) or os.getuid() + 1)
    try:
        with pytest.raises(gateway_runtime_discovery.DiscoveryError) as foreign:
            gateway_ws_transport(SimpleNamespace(api_origin=unix_api_origin(path)))
        assert foreign.value.reason == "unsafe_control_peer"
        assert [s.fileno() for s in seen] == [-1]  # the probed socket was closed, nothing sent on it
    finally:
        listener.close()
        path.unlink()


def _padded_home(base, length):
    """A (not created) home under *base* whose path is exactly *length* characters."""
    pad = length - len(str(base)) - 1
    assert pad > 0
    return base / ("p" * pad)


def test_a_home_that_fits_the_api_socket_but_not_the_loop_tick_socket_refuses(short_home):
    from gateway import run_api
    from gateway.control_socket import _MAX_UNIX_PATH
    from gateway.shutdown_watchdog import get_loop_tick_socket_path
    home = _padded_home(short_home, _MAX_UNIX_PATH - len("/runtime-api.sock"))
    assert len(str(run_api.runtime_api_socket_path(home))) <= _MAX_UNIX_PATH
    assert len(str(get_loop_tick_socket_path(home))) > _MAX_UNIX_PATH
    with pytest.raises(RuntimeError, match="gateway.loop-tick"):
        run_api.check_unix_socket_paths(home)
    run_api.check_unix_socket_paths(short_home)  # a short home passes


def _heartbeat_once(monkeypatch, home, config):
    """Run the loop heartbeat for one beat in *home*; returns the TCP servers it asked for."""
    import asyncio as aio
    from gateway import shutdown_watchdog
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    tcp = []
    real_start_server = aio.start_server

    async def start_server(*args, **kwargs):
        tcp.append(kwargs.get("host"))
        return await real_start_server(*args, **kwargs)

    monkeypatch.setattr(aio, "start_server", start_server)
    asyncio.run(shutdown_watchdog.loop_heartbeat_forever(home=home, should_continue=lambda: False))
    return tcp


def test_the_loop_tick_never_falls_back_to_tcp_in_unix_mode(monkeypatch, short_home, caplog):
    from gateway.control_socket import _MAX_UNIX_PATH
    home = _padded_home(short_home, _MAX_UNIX_PATH - 20)  # the tick socket cannot fit
    assert _heartbeat_once(monkeypatch, home, "gateway:\n  session_api:\n    transport: unix\n") == []
    assert "does not fall back to TCP" in caplog.text
    # Today's behaviour in tcp mode: the witness moves to a loopback port.
    assert _heartbeat_once(monkeypatch, home, "gateway:\n  session_api:\n    transport: tcp\n") == ["127.0.0.1"]


def test_an_unreadable_config_falls_back_to_tcp_with_a_warning(tmp_path, caplog):
    from gateway.webapp_listener import session_api_transport
    (tmp_path / "config.yaml").write_text("gateway: [unclosed\n", encoding="utf-8")
    assert session_api_transport(tmp_path) == "tcp"
    assert "falls back to tcp" in caplog.text and "could not be read" in caplog.text


def test_unix_mode_names_enabled_platforms_that_listen_on_tcp(monkeypatch, short_home, caplog):
    from gateway import run_api, run_runtime
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    (short_home / "config.yaml").write_text("gateway:\n  session_api:\n    transport: unix\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(short_home))

    async def start(runner, **kwargs):
        return SimpleNamespace(api_origin=run_api.unix_api_origin(kwargs["unix_path"]))

    monkeypatch.setattr(run_api, "start_gateway_api", start)
    from gateway.control_socket import GatewayControlServer
    config = GatewayConfig(platforms={Platform.API_SERVER: PlatformConfig(enabled=True),
                                      Platform.WEBHOOK: PlatformConfig(enabled=False)})
    runner = SimpleNamespace(session_runtime_descriptor={}, session_authorities=[], config=config,
                             session_control_server=GatewayControlServer(short_home))
    asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    warned = [r.getMessage() for r in caplog.records if "listen on TCP ports of their own" in r.getMessage()]
    assert len(warned) == 1 and "api_server" in warned[0] and "webhook" not in warned[0], warned


_BOOT = textwrap.dedent("""
    # Record every bind/listen and every host/port server, then run the real gateway entry point.
    import json, os, socket, sys
    record = open(sys.argv[1], "a", buffering=1)
    def note(kind, sock, address=None):
        record.write(json.dumps({"kind": kind, "family": int(sock.family),
                                 "address": address if isinstance(address, (str, tuple, list)) else None}) + "\\n")
    real_bind, real_listen = socket.socket.bind, socket.socket.listen
    def bind(self, address):
        note("bind", self, address)
        return real_bind(self, address)
    def listen(self, *args):
        note("listen", self, self.getsockname())
        return real_listen(self, *args)
    socket.socket.bind, socket.socket.listen = bind, listen
    import asyncio.base_events as base_events
    real_create_server = base_events.BaseEventLoop.create_server
    async def create_server(self, factory, host=None, port=None, *args, sock=None, **kwargs):
        if sock is None:
            record.write(json.dumps({"kind": "create_server", "host": host, "port": port}) + "\\n")
        return await real_create_server(self, factory, host, port, *args, sock=sock, **kwargs)
    base_events.BaseEventLoop.create_server = create_server
    sys.argv = ["gateway.run"]
    import runpy
    runpy.run_module("gateway.run", run_name="__main__", alter_sys=True)
""")


def test_the_real_gateway_in_unix_mode_never_listens_on_tcp(short_home):
    """The real entry point (``python -m gateway.run``), up to READY with its runtime API and the
    shutdown watchdog's loop-tick witness: every listening socket is AF_UNIX, a ticketed client
    reaches the session handler over the API socket, and a stop removes it."""
    import signal
    import time
    from gateway.run_api import runtime_api_socket_path
    from gateway.shutdown_watchdog import get_loop_tick_socket_path
    from hermes_cli.gateway_client import gateway_ws_target, gateway_ws_transport
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_discovery import query_identify
    home = short_home / "h"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False, "session_api": {"transport": "unix"}},
        "model": {"provider": "custom", "default": "unused-stub", "base_url": "http://127.0.0.1:9/v1"},
        "auxiliary": {"title_generation": {"enabled": False}},
    }), encoding="utf-8")
    events = short_home / "events.jsonl"
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "TMPDIR"}}
    env.update(HOME=str(short_home), USERPROFILE=str(short_home), HERMES_HOME=str(home), PYTHONPATH=str(REPO),
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="loopback-only")
    with (short_home / "gateway.log").open("w+") as log:
        process = subprocess.Popen([sys.executable, "-c", _BOOT, str(events)], cwd=str(REPO), env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            descriptor, tick = {}, get_loop_tick_socket_path(home, process.pid)
            deadline = time.monotonic() + 60
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = query_identify(home, timeout=2)
                except (OSError, ValueError):
                    descriptor = {}
                if descriptor.get("state") == "ready" and tick.exists():
                    break
                time.sleep(.2)
            log.flush()
            log.seek(0)
            gateway_log = log.read()
            assert descriptor.get("state") == "ready" and tick.exists(), (descriptor, gateway_log[-4000:])
            logs = "".join(p.read_text(errors="replace") for p in (home / "logs").glob("*.log"))
            assert "Gateway runtime API transport: unix socket" in gateway_log + logs
            api = runtime_api_socket_path(home)
            assert descriptor["api_origin"].startswith("http+unix://")
            assert stat.S_IMODE(api.lstat().st_mode) == 0o600
            endpoint = discover_gateway_endpoint(home, timeout=5).endpoint
            from hermes_cli.gateway_client import _session_ticket
            ticket = _session_ticket(home, endpoint)

            async def resume():
                from websockets.asyncio.client import connect
                url, protocols = gateway_ws_target(endpoint, ticket)
                async with connect(url, subprotocols=protocols, proxy=None, **gateway_ws_transport(endpoint)) as ws:
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "session.resume",
                                              "params": {"session_id": "not-registered"}}))
                    while True:
                        reply = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        if reply.get("id") == 7:
                            return reply
            assert asyncio.run(resume())["error"]["message"] == "not_found"
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=30) == 0
            assert not api.exists()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    records = [json.loads(line) for line in events.read_text().splitlines()]
    unix = int(socket.AF_UNIX)
    assert [r for r in records if r["kind"] == "listen" and r["family"] != unix] == [], records
    assert [r for r in records if r["kind"] == "create_server"] == [], records
    # A non-unix bind is allowed only as a throwaway loopback probe (urllib3's HAS_IPV6): never a
    # wildcard or a fixed port, and (above) never listened on.
    for r in records:
        if r["kind"] == "bind" and r["family"] != unix:
            host, port = r["address"][0], r["address"][1]
            assert host in {"127.0.0.1", "::1"} and port == 0, r
    # The API socket listens under the home (bound in a private staging directory there, so the
    # kernel keeps reporting that name after the rename into place).
    assert any(r["kind"] == "listen" and r["family"] == unix and str(r["address"]).startswith(str(home.resolve()))
               for r in records), records

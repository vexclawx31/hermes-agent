"""Shared-UID standalone peers launched bare, with HERMES_HOME only in the launch environment.

Real launch shape: ``python -B -I -m hermes_cli.main gateway run --external-supervisor`` with
``HERMES_HOME=<root>/profiles/<name>`` in the environment, so argv names no profile. The peer must be
recognised as its own home's gateway and never as the default home's. A REAL child process stands in for
each peer, so the environment read is the real one; only its command line is fixed to the bare gateway
argv (the child is a sleeper, not a gateway).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import host_attach, status

BARE_ARGV = "/opt/release/.venv/bin/python -B -I -m hermes_cli.main gateway run --external-supervisor"
SECRET = "synthetic-launch-secret-value"


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".hermes"
    homes = {"default": root, "lynx": root / "profiles" / "lynx", "opix": root / "profiles" / "opix"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
    children = {}

    def spawn(name, env_home):
        env = {"PATH": "/usr/bin:/bin", "FIXTURE_TOKEN": SECRET}
        if env_home is not None:
            env["HERMES_HOME"] = str(env_home)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)
        children[name] = child
        return child.pid

    real_cmdline = status._read_process_cmdline
    commands = {}
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: commands.get(pid) or real_cmdline(pid))
    monkeypatch.setattr(status, "_host_gateway_serves_home", lambda pid, home: False)
    status._clear_running_pid_cache()
    host_attach.invalidate_host_gateway_cache()

    def record(home, pid, **extra):
        payload = {"kind": "hermes-gateway", "gateway_state": "running", "pid": pid,
                   "start_time": status.get_process_start_time(pid), "hermes_home": str(home), **extra}
        (home / "gateway_state.json").write_text(json.dumps(payload))

    yield SimpleNamespace(root=root, homes=homes, spawn=spawn, commands=commands, record=record)
    for child in children.values():
        child.terminate()
        child.wait(timeout=10)
    status._clear_running_pid_cache()
    host_attach.invalidate_host_gateway_cache()


def live(fleet, name):
    status._clear_running_pid_cache()
    return status.live_gateway_pid_for_home(fleet.homes[name])


def test_bare_peer_is_its_launch_home_gateway_and_never_the_default(fleet):
    lynx = fleet.spawn("lynx", fleet.homes["lynx"])
    opix = fleet.spawn("opix", fleet.homes["opix"])
    fleet.commands.update({lynx: BARE_ARGV, opix: BARE_ARGV})
    fleet.record(fleet.homes["lynx"], lynx)
    fleet.record(fleet.homes["opix"], opix)
    fleet.record(fleet.homes["default"], lynx)          # stale/spoofed default record naming a peer PID
    assert live(fleet, "lynx") == lynx
    assert live(fleet, "opix") == opix
    assert live(fleet, "default") is None                # the default home cannot claim a named peer


def test_wrong_home_records_and_stale_start_are_refused(fleet):
    lynx = fleet.spawn("lynx", fleet.homes["lynx"])
    fleet.commands[lynx] = BARE_ARGV
    fleet.record(fleet.homes["opix"], lynx)              # opix's record names lynx's process
    assert live(fleet, "opix") is None
    fleet.record(fleet.homes["lynx"], lynx, start_time=(status.get_process_start_time(lynx) or 0) + 10_000)
    assert live(fleet, "lynx") is None                   # PID-reuse guard still wins


def test_explicit_argv_selector_stays_authoritative_over_environment(fleet):
    pid = fleet.spawn("mixed", fleet.homes["lynx"])
    fleet.commands[pid] = BARE_ARGV + " --profile opix"
    fleet.record(fleet.homes["lynx"], pid)
    fleet.record(fleet.homes["opix"], pid)
    assert live(fleet, "lynx") is None
    assert live(fleet, "opix") == pid
    fleet.commands[pid] = "HERMES_HOME=" + str(fleet.homes["opix"]) + " " + BARE_ARGV
    assert live(fleet, "lynx") is None and live(fleet, "opix") == pid


def test_absent_launch_home_is_the_default_gateway_only(fleet):
    pid = fleet.spawn("default", None)
    fleet.commands[pid] = BARE_ARGV
    fleet.record(fleet.homes["default"], pid)
    fleet.record(fleet.homes["lynx"], pid)
    assert live(fleet, "default") == pid
    assert live(fleet, "lynx") is None


@pytest.mark.parametrize("environ", ["denied", "empty"])
def test_unreadable_or_empty_environment_never_authorizes_a_named_peer(fleet, monkeypatch, environ):
    import psutil
    pid = fleet.spawn("lynx", fleet.homes["lynx"])
    fleet.commands[pid] = BARE_ARGV
    fleet.record(fleet.homes["lynx"], pid)
    real = psutil.Process

    class Denied(real):
        def environ(self):
            if environ == "denied":
                raise psutil.AccessDenied(self.pid)
            return {"HERMES_HOME": " "}
    monkeypatch.setattr(psutil, "Process", Denied)
    assert status._read_process_launch_home(pid) == ("unreadable", None)
    assert live(fleet, "lynx") is None


def test_only_the_launch_home_value_is_read_from_the_environment(fleet, caplog):
    pid = fleet.spawn("lynx", fleet.homes["lynx"])
    with caplog.at_level("DEBUG"):
        state, value = status._read_process_launch_home(pid)
    assert (state, Path(value)) == ("home", fleet.homes["lynx"])
    assert SECRET not in caplog.text


def _older_identify(fleet, answers):
    """Older-release control contract: pid, start_time, hermes_home, profile; no served set or multiplex."""
    return lambda home: answers.get(Path(home))


def test_coexisting_scan_sees_bare_standalone_peers_with_their_own_profile(fleet, monkeypatch):
    from hermes_cli import profiles
    lynx = fleet.spawn("lynx", fleet.homes["lynx"])
    opix = fleet.spawn("opix", fleet.homes["opix"])
    fleet.commands.update({lynx: BARE_ARGV, opix: BARE_ARGV})
    for name, pid in (("lynx", lynx), ("opix", opix)):
        fleet.record(fleet.homes[name], pid)
    fleet.record(fleet.homes["default"], lynx)
    answers = {fleet.homes[n]: {"pid": p, "start_time": status.get_process_start_time(p),
                                "hermes_home": str(fleet.homes[n]), "profile": n}
               for n, p in (("lynx", lynx), ("opix", opix))}
    monkeypatch.setattr(host_attach, "_identify", _older_identify(fleet, answers))
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *a, **kw: list(fleet.homes.items()))
    peers = list(host_attach._coexisting_gateways(None))
    assert sorted((p.pid, p.home, p.profiles, p.served_known) for p in peers) == sorted([
        (lynx, fleet.homes["lynx"], ("lynx",), True), (opix, fleet.homes["opix"], ("opix",), True)])
    assert all(not p.standalone for p in peers)   # absent `multiplex` is not a claimed standalone flag


@pytest.mark.parametrize("change", ["start", "profile", "home", "pid"])
def test_identify_answer_must_be_the_same_process_and_home(fleet, change):
    lynx = fleet.spawn("lynx", fleet.homes["lynx"])
    answer = {"pid": lynx, "start_time": status.get_process_start_time(lynx),
              "hermes_home": str(fleet.homes["lynx"]), "profile": "lynx"}
    peer = host_attach.HostGateway(lynx, fleet.homes["lynx"], (), served_known=False)
    assert host_attach._identity_matches(answer, peer, fleet.homes["lynx"])
    answer = {**answer, **{"start": {"start_time": answer["start_time"] + 10_000},
                           "profile": {"profile": "opix"},
                           "home": {"hermes_home": str(fleet.homes["opix"])},
                           "pid": {"pid": lynx + 1}}[change]}
    assert not host_attach._identity_matches(answer, peer, fleet.homes["lynx"])


def test_unknown_peer_answer_keeps_served_set_unknown(fleet, monkeypatch):
    from hermes_cli import profiles
    lynx = fleet.spawn("lynx", fleet.homes["lynx"])
    fleet.commands[lynx] = BARE_ARGV
    fleet.record(fleet.homes["lynx"], lynx)
    monkeypatch.setattr(host_attach, "_identify", lambda home: None)   # socket unreadable / silent
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *a, **kw: [("lynx", fleet.homes["lynx"])])
    (peer,) = list(host_attach._coexisting_gateways(None))
    assert (peer.pid, peer.served_known) == (lynx, False)

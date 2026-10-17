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
    homes = {"default": root, "alpha": root / "profiles" / "alpha", "beta": root / "profiles" / "beta"}
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
    alpha = fleet.spawn("alpha", fleet.homes["alpha"])
    beta = fleet.spawn("beta", fleet.homes["beta"])
    fleet.commands.update({alpha: BARE_ARGV, beta: BARE_ARGV})
    fleet.record(fleet.homes["alpha"], alpha)
    fleet.record(fleet.homes["beta"], beta)
    fleet.record(fleet.homes["default"], alpha)          # stale/spoofed default record naming a peer PID
    assert live(fleet, "alpha") == alpha
    assert live(fleet, "beta") == beta
    assert live(fleet, "default") is None                # the default home cannot claim a named peer


def test_wrong_home_records_and_stale_start_are_refused(fleet):
    alpha = fleet.spawn("alpha", fleet.homes["alpha"])
    fleet.commands[alpha] = BARE_ARGV
    fleet.record(fleet.homes["beta"], alpha)              # beta's record names alpha's process
    assert live(fleet, "beta") is None
    fleet.record(fleet.homes["alpha"], alpha, start_time=(status.get_process_start_time(alpha) or 0) + 10_000)
    assert live(fleet, "alpha") is None                   # PID-reuse guard still wins


def test_explicit_argv_selector_stays_authoritative_over_environment(fleet):
    pid = fleet.spawn("mixed", fleet.homes["alpha"])
    fleet.commands[pid] = BARE_ARGV + " --profile beta"
    fleet.record(fleet.homes["alpha"], pid)
    fleet.record(fleet.homes["beta"], pid)
    assert live(fleet, "alpha") is None
    assert live(fleet, "beta") == pid
    fleet.commands[pid] = "HERMES_HOME=" + str(fleet.homes["beta"]) + " " + BARE_ARGV
    assert live(fleet, "alpha") is None and live(fleet, "beta") == pid


def test_absent_launch_home_is_the_default_gateway_only(fleet):
    pid = fleet.spawn("default", None)
    fleet.commands[pid] = BARE_ARGV
    fleet.record(fleet.homes["default"], pid)
    fleet.record(fleet.homes["alpha"], pid)
    assert live(fleet, "default") == pid
    assert live(fleet, "alpha") is None


@pytest.mark.parametrize("environ", ["denied", "empty"])
def test_unreadable_or_empty_environment_never_authorizes_a_named_peer(fleet, monkeypatch, environ):
    import psutil
    pid = fleet.spawn("alpha", fleet.homes["alpha"])
    fleet.commands[pid] = BARE_ARGV
    fleet.record(fleet.homes["alpha"], pid)
    real = psutil.Process

    class Denied(real):
        def environ(self):
            if environ == "denied":
                raise psutil.AccessDenied(self.pid)
            return {"HERMES_HOME": " "}
    monkeypatch.setattr(psutil, "Process", Denied)
    assert status._read_process_launch_home(pid) == ("unreadable", None)
    assert live(fleet, "alpha") is None


def test_only_the_launch_home_value_is_read_from_the_environment(fleet, caplog):
    pid = fleet.spawn("alpha", fleet.homes["alpha"])
    with caplog.at_level("DEBUG"):
        state, value = status._read_process_launch_home(pid)
    assert (state, Path(value)) == ("home", fleet.homes["alpha"])
    assert SECRET not in caplog.text


def _older_identify(fleet, answers):
    """Older-release control contract: pid, start_time, hermes_home, profile; no served set or multiplex."""
    return lambda home: answers.get(Path(home))


def test_coexisting_scan_sees_bare_standalone_peers_with_their_own_profile(fleet, monkeypatch):
    from hermes_cli import profiles
    alpha = fleet.spawn("alpha", fleet.homes["alpha"])
    beta = fleet.spawn("beta", fleet.homes["beta"])
    fleet.commands.update({alpha: BARE_ARGV, beta: BARE_ARGV})
    for name, pid in (("alpha", alpha), ("beta", beta)):
        fleet.record(fleet.homes[name], pid)
    fleet.record(fleet.homes["default"], alpha)
    answers = {fleet.homes[n]: {"pid": p, "start_time": status.get_process_start_time(p),
                                "hermes_home": str(fleet.homes[n]), "profile": n}
               for n, p in (("alpha", alpha), ("beta", beta))}
    monkeypatch.setattr(host_attach, "_identify", _older_identify(fleet, answers))
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *a, **kw: list(fleet.homes.items()))
    peers = list(host_attach._coexisting_gateways(None))
    assert sorted((p.pid, p.home, p.profiles, p.served_known) for p in peers) == sorted([
        (alpha, fleet.homes["alpha"], ("alpha",), True), (beta, fleet.homes["beta"], ("beta",), True)])
    assert all(not p.standalone for p in peers)   # absent `multiplex` is not a claimed standalone flag


@pytest.mark.parametrize("change", ["start", "profile", "home", "pid"])
def test_identify_answer_must_be_the_same_process_and_home(fleet, change):
    alpha = fleet.spawn("alpha", fleet.homes["alpha"])
    answer = {"pid": alpha, "start_time": status.get_process_start_time(alpha),
              "hermes_home": str(fleet.homes["alpha"]), "profile": "alpha"}
    peer = host_attach.HostGateway(alpha, fleet.homes["alpha"], (), served_known=False)
    assert host_attach._identity_matches(answer, peer, fleet.homes["alpha"])
    answer = {**answer, **{"start": {"start_time": answer["start_time"] + 10_000},
                           "profile": {"profile": "beta"},
                           "home": {"hermes_home": str(fleet.homes["beta"])},
                           "pid": {"pid": alpha + 1}}[change]}
    assert not host_attach._identity_matches(answer, peer, fleet.homes["alpha"])


def test_unknown_peer_answer_keeps_served_set_unknown(fleet, monkeypatch):
    from hermes_cli import profiles
    alpha = fleet.spawn("alpha", fleet.homes["alpha"])
    fleet.commands[alpha] = BARE_ARGV
    fleet.record(fleet.homes["alpha"], alpha)
    monkeypatch.setattr(host_attach, "_identify", lambda home: None)   # socket unreadable / silent
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *a, **kw: [("alpha", fleet.homes["alpha"])])
    (peer,) = list(host_attach._coexisting_gateways(None))
    assert (peer.pid, peer.served_known) == (alpha, False)

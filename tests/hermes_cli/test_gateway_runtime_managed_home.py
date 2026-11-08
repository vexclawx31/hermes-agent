"""A home with ``gateway.managed_runtime: true`` is never stopped or spawned by a client (fork).

Hazard (v0.23 upgrade plan §5): a v0.23 client targeting a profile whose gateway is an older Hermes
stops it (``replace_incompatible_owner``) and launchd does not revive a planned stop; and a tenant's
LaunchDaemon is invisible to the LaunchAgent probe, so ``spawn_unmanaged_gateway`` starts a second,
unmanaged gateway. For a managed home the client must fail loudly instead, naming the profile.

Every process boundary here is a fake or a sleeping stand-in: nothing reaches launchd.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_runtime as runtime
from hermes_cli import gateway_runtime_managed as managed
from hermes_cli import gateway_runtime_replace as replace
from hermes_cli import gateway_runtime_service as service
from hermes_cli import gateway_runtime_start as start
from hermes_cli.gateway_runtime import GatewayDiscovery

MANAGED = "gateway:\n  standalone: true\n  managed_runtime: true\n"
_REAL_POPEN = subprocess.Popen  # the stand-in owner starts for real even under no_process_starts


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = (tmp_path / ".hermes").resolve()
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(runtime, "_update_fenced", lambda homes: False)
    monkeypatch.setattr(replace, "_install_root", lambda: root)
    return SimpleNamespace(root=root, alpha=alpha)


@pytest.fixture
def no_process_starts(monkeypatch):
    """Any start or signal is a failure: the guard must stop before the process boundary."""
    def refuse(*_a, **_k):
        pytest.fail("a managed home's gateway was started or signalled")

    monkeypatch.setattr(start.subprocess, "Popen", refuse)
    monkeypatch.setattr(start, "start_gateway_job", refuse)
    from gateway import status
    monkeypatch.setattr(status, "terminate_pid", refuse)


@pytest.fixture
def old_owner():
    proc = _REAL_POPEN([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def _old_identify(home: Path, pid: int) -> dict:
    from gateway.status import get_process_start_time
    # An older Hermes's identify: no runtime_protocol.
    return {"protocol": 1, "kind": "hermes-gateway", "pid": pid, "start_time": get_process_start_time(pid),
            "hermes_home": str(home), "supervisor": "launchd"}


# --- the signal ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (None, False),                                           # no config.yaml at all
    ("{}\n", False),
    ("gateway:\n  standalone: true\n", False),               # standalone alone is not "managed"
    ("gateway:\n  managed_runtime: false\n", False),
    ("gateway:\n  managed_runtime: true\n", True),
    ("gateway:\n  managed_runtime: 'true'\n", True),
    ("managed_runtime: true\n", False),                      # only the gateway: section counts
    ("gateway: [unbalanced\n", True),                        # unparseable: fail closed
])
def test_only_an_explicit_gateway_key_marks_a_home_managed(fleet, text, expected):
    if text is not None:
        (fleet.alpha / "config.yaml").write_text(text, encoding="utf-8")
    assert managed.home_runtime_is_managed(fleet.alpha) is expected


# --- replacement refused -------------------------------------------------------------------------

def test_replacement_of_a_managed_home_is_refused_before_anything_is_signalled(fleet, monkeypatch,
                                                                               no_process_starts, old_owner):
    (fleet.alpha / "config.yaml").write_text(MANAGED, encoding="utf-8")
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (h, _old_identify(h, old_owner.pid)))

    assert replace.replace_incompatible_owner(fleet.alpha, deadline=time.monotonic() + 5) == "owner_managed"
    assert old_owner.poll() is None
    assert not (fleet.alpha / ".gateway-planned-stop.json").exists()


def test_replacement_is_refused_when_the_answering_host_root_is_managed(fleet, monkeypatch,
                                                                        no_process_starts, old_owner):
    """A named profile served by the default multiplexer: the owner that would be stopped is the root's."""
    (fleet.root / "config.yaml").write_text("gateway:\n  managed_runtime: true\n", encoding="utf-8")
    monkeypatch.setattr(replace, "_identify",
                        lambda h, timeout: (fleet.root, _old_identify(fleet.root, old_owner.pid)))

    assert replace.replace_incompatible_owner(fleet.alpha, deadline=time.monotonic() + 5) == "owner_managed"
    assert old_owner.poll() is None


def test_ensure_fails_loudly_on_a_managed_incompatible_owner(fleet, monkeypatch, no_process_starts, old_owner):
    (fleet.alpha / "config.yaml").write_text(MANAGED, encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_gateway_endpoint",
                        lambda h, timeout=2.0: GatewayDiscovery("incompatible", reason_code="runtime_protocol"))
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (h, _old_identify(h, old_owner.pid)))

    result = runtime.ensure_gateway_runtime(fleet.alpha, timeout=5)

    assert (result.state, result.reason_code) == ("incompatible", "runtime_protocol")
    assert "alpha" in result.detail and "externally managed" in result.detail
    assert old_owner.poll() is None


# --- unmanaged spawn refused ---------------------------------------------------------------------

def test_spawn_refuses_a_managed_home(fleet, no_process_starts):
    (fleet.alpha / "config.yaml").write_text(MANAGED, encoding="utf-8")
    with pytest.raises(service.RuntimeStartError) as raised:
        start.spawn_unmanaged_gateway(fleet.alpha, deadline=time.monotonic() + 5)
    assert raised.value.reason == managed.MANAGED_RUNTIME_REASON
    assert "alpha" in raised.value.detail and "externally managed" in raised.value.detail


def test_ensure_never_spawns_for_a_managed_home_without_a_launch_agent(fleet, monkeypatch, no_process_starts):
    """A tenant LaunchDaemon: no LaunchAgent is found, and the gateway is down (mid-restart)."""
    (fleet.alpha / "config.yaml").write_text(MANAGED, encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)

    result = runtime.ensure_gateway_runtime(fleet.alpha, timeout=5)

    assert (result.state, result.reason_code) == ("inaccessible", managed.MANAGED_RUNTIME_REASON)
    assert "alpha" in result.detail


def test_ensure_never_spawns_the_host_for_a_managed_named_profile(fleet, monkeypatch, no_process_starts):
    """A managed, non-standalone named profile resolves to the host root's gateway: not spawned either."""
    (fleet.alpha / "config.yaml").write_text("gateway:\n  managed_runtime: true\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)

    result = runtime.ensure_gateway_runtime(fleet.alpha, timeout=5)

    assert (result.state, result.reason_code) == ("inaccessible", managed.MANAGED_RUNTIME_REASON)


def test_detached_restart_reports_the_managed_refusal(fleet, no_process_starts, capsys):
    (fleet.alpha / "config.yaml").write_text(MANAGED, encoding="utf-8")
    assert start.start_detached_replacement(fleet.alpha) == 1
    assert "externally managed" in capsys.readouterr().err


# --- unmanaged homes unchanged -------------------------------------------------------------------

@pytest.mark.parametrize("text", [None, "gateway:\n  standalone: true\n",
                                  "gateway:\n  standalone: true\n  managed_runtime: false\n"])
def test_unmanaged_homes_still_spawn(fleet, monkeypatch, text):
    if text is not None:
        (fleet.alpha / "config.yaml").write_text(text, encoding="utf-8")
    child = SimpleNamespace(pid=123)
    monkeypatch.setattr(start, "start_gateway_job", lambda *a, **k: None)  # never a real launchd job
    monkeypatch.setattr(start.subprocess, "Popen", lambda command, **k: child)
    assert start.spawn_unmanaged_gateway(fleet.alpha, deadline=time.monotonic() + 5) is child


def test_unmanaged_home_replacement_still_reaches_the_owner_check(fleet, monkeypatch):
    """Upstream path intact: the unmarked home goes on to identify and verify its owner."""
    asked = []
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: asked.append(h))  # None: already gone
    assert replace.replace_incompatible_owner(fleet.alpha, deadline=time.monotonic() + 5) is None
    assert asked == [fleet.alpha]

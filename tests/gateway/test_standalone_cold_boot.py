"""Standalone gateways under one OS user must all start at a cold boot (r21).

Field shape (2026-11-05, two power cuts): every unit starts at once; each claims its per-home PID,
then finds the other profiles' PIDs before their control sockets answer. Refusing on that UNKNOWN
served set made every standalone gateway exit 75 in lockstep, forever.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import host_attach, host_rendezvous as hr

REPO_ROOT = Path(__file__).resolve().parents[2]

# Named ``hermes`` and run as ``hermes --profile <name> gateway run`` so the per-home PID record and
# the live command line pass the same gateway-identity checks a real launchd unit does.
_CHILD = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path

    go, done, out = (Path(p) for p in os.environ["R21_COLD_BOOT"].split(os.pathsep))
    peers = [Path(p) for p in os.environ["R21_PEERS"].split(os.pathsep) if p]
    home = Path(os.environ["HERMES_HOME"])
    result = {"pid": os.getpid(), "home": str(home)}

    from gateway import host_attach, run

    def wait_for(predicate, what):
        deadline = time.monotonic() + 30
        while not predicate():
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out waiting for {what}")
            time.sleep(0.01)

    try:
        # start_gateway's order: the attach question, then the per-home PID claim, which takes
        # the host lock (and on losing it asks again) before any control socket is bound.
        result["first_decide"] = host_attach.decide(home).outcome
        if not run._start_gateway_claim_pid_file():
            result["claim"] = "pid-lost"
        else:
            # Worst case, every time: each peer now holds its per-home PID and none answers
            # identify (no socket is ever bound here). Ask again, as a supervisor retry would.
            wait_for(lambda: all((p / "gateway.pid").exists() for p in peers), "peer PID files")
            wait_for(go.exists, "the start signal")
            started = time.monotonic()
            decision = host_attach.decide(home)
            result["decide"] = decision.outcome
            if decision.outcome == host_attach.START:
                run._claim_host_gateway_role()
                result["claim"] = "started"
            else:
                result["claim"] = "refused"
            result["seconds"] = round(time.monotonic() - started, 2)
            from gateway import host_rendezvous as hr
            result["owns_host_lock"] = hr.owns_host_lock(hr.ROLE_GATEWAY)
            record = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
            result["record"] = record.to_json() if record is not None else None
    except SystemExit as exc:
        result["claim"] = f"exit {exc.code}"
    out.write_text(json.dumps(result))
    # Hold the per-home and host locks until the test is done, like a running gateway.
    wait_for(done.exists, "the done signal")
''')


def _isolated_env(tmp_path: Path, home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "PYTEST_"))}
    env.update({
        "HOME": str(tmp_path / "user"),
        "HERMES_HOME": str(home),
        "HERMES_GATEWAY_LOCK_DIR": str(tmp_path / "locks"),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


@pytest.mark.platforms("posix")
@pytest.mark.spawns_gateway_lookalike  # a stub child (never the gateway runtime), reaped below
@pytest.mark.parametrize("names",[("alex", "calix"), ("windx", "lynx", "opix")])
def test_standalone_profiles_cold_starting_together_all_start(tmp_path, names):
    root = tmp_path / "user" / ".hermes"
    homes = []
    for name in names:
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("gateway:\n  standalone: true\n")
        homes.append(home)
    script = tmp_path / "bin" / "hermes"
    script.parent.mkdir()
    script.write_text(_CHILD)
    go, done = tmp_path / "go", tmp_path / "done"
    children = []
    try:
        for name, home in zip(names, homes):
            env = _isolated_env(tmp_path, home)
            env["R21_COLD_BOOT"] = os.pathsep.join(str(p) for p in (go, done, tmp_path / f"{name}.json"))
            env["R21_PEERS"] = os.pathsep.join(str(h) for h in homes if h != home)
            children.append(subprocess.Popen(
                [sys.executable, str(script), "--profile", name, "gateway", "run"],
                cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 60
        while not all((h / "gateway.pid").exists() for h in homes):
            assert time.monotonic() < deadline, "children never claimed their PID files"
            assert all(c.poll() is None for c in children), _outputs(children)
            time.sleep(0.02)
        go.touch()
        while not all((tmp_path / f"{n}.json").exists() for n in names):
            assert time.monotonic() < deadline, _outputs(children)
            assert all(c.poll() is None for c in children), _outputs(children)
            time.sleep(0.02)
        results = {n: json.loads((tmp_path / f"{n}.json").read_text()) for n in names}
    finally:
        done.touch()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    for name, result in results.items():
        assert result.get("decide") == host_attach.START, (name, result)
        assert result.get("claim") == "started", (name, result)
        assert result["seconds"] < host_attach.ATTACH_CHANNEL_WAIT_S, (name, result, "nobody waits")
    # Exactly one won the host lock, and its record says it cannot multiplex.
    owners = [r for r in results.values() if r.get("owns_host_lock")]
    assert len(owners) == 1, results
    for result in results.values():  # everyone reads the same record: the owner's
        assert result["record"] is not None, result
        assert result["record"]["pid"] == owners[0]["pid"], results
        assert result["record"]["multiplex"] is False, result


def _outputs(children) -> str:
    chunks = []
    for child in children:
        if child.poll() is not None and child.stdout is not None:
            chunks.append(child.stdout.read().decode(errors="replace"))
    return "\n---\n".join(chunks) or "(children still running)"


# --- the record field ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (False, False), (True, None), (None, None), ("false", None), (0, None), ("missing", None)])
def test_record_multiplex_is_unknown_unless_literally_false(raw, expected):
    payload = {"role": "gateway", "pid": 4242, "profiles": [], "home": "/x"}
    if raw != "missing":
        payload["multiplex"] = raw
    record = hr.HostRecord.from_json(payload)
    assert record is not None and record.multiplex is expected
    assert hr.HostRecord.from_json(record.to_json()).multiplex is expected


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """One OS user's root with a standalone profile, a member profile and a custom-home standalone."""
    root = tmp_path / "hermes"
    solo = root / "profiles" / "solo"
    member = root / "profiles" / "member"
    custom = tmp_path / "elsewhere" / "custom"
    for home, cfg in ((root, "{}\n"), (solo, "gateway:\n  standalone: true\n"), (member, "{}\n"),
                      (custom, "gateway:\n  standalone: true\n")):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(cfg)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    return SimpleNamespace(root=root, solo=solo, member=member, custom=custom)


@pytest.mark.parametrize("which,expected", [("solo", False), ("custom", False), ("member", None),
                                            ("root", None)])
def test_every_record_write_carries_the_standalone_flag(fleet, monkeypatch, which, expected):
    home = getattr(fleet, which)
    monkeypatch.setenv("HERMES_HOME", str(home))
    claim = hr.publish_record(hr.ROLE_GATEWAY, profiles=(), home=str(home))
    assert claim is not None and claim.multiplex is expected
    # The settled refresh and the rescan path republish without naming the flag; it must survive.
    refreshed = hr.publish_record(hr.ROLE_GATEWAY, profiles=("solo",), home=str(home))
    assert refreshed is not None and refreshed.multiplex is expected
    stored = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert stored is not None and stored.multiplex is expected
    # The serve role never carries it.
    assert hr.publish_record(hr.ROLE_SERVE, profiles=(), home=str(home)).multiplex is None


def test_a_multiplexing_process_never_publishes_false(fleet, monkeypatch):
    """Config flipped to standalone while a multiplexer keeps running: its republish stays unknown."""
    from agent import secret_scope

    monkeypatch.setenv("HERMES_HOME", str(fleet.solo))
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    assert hr.publish_record(hr.ROLE_GATEWAY, profiles=("solo", "member"), home=str(fleet.solo)).multiplex is None


def test_identify_reports_the_live_mode(monkeypatch):
    from agent import secret_scope
    from gateway import control_socket

    for active in (True, False):
        monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda active=active: active)
        assert control_socket.build_identify_payload()["multiplex"] is active


def test_refresh_host_gateway_record_keeps_multiplex_false(fleet, monkeypatch):
    from gateway import run

    monkeypatch.setenv("HERMES_HOME", str(fleet.solo))
    monkeypatch.setattr(run, "get_hermes_home", lambda: fleet.solo)
    monkeypatch.setattr(hr, "owns_host_lock", lambda role: True)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    run._refresh_host_gateway_record(runner)
    stored = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert stored is not None and stored.multiplex is False and stored.profiles == ("solo",)


# --- readers of the record ----------------------------------------------------------------------

def _live_standalone_record(monkeypatch, home: Path):
    record = SimpleNamespace(pid=os.getpid() + 1, home=str(home), multiplex=False)
    monkeypatch.setattr(hr, "read_record", lambda role, **kw: record)
    monkeypatch.setattr(hr, "liveness_is_proven", lambda rec: True)
    monkeypatch.setattr(host_attach, "_identify", lambda h: None)  # its socket is not bound yet
    host_attach.invalidate_host_gateway_cache()
    return record


def test_probe_trusts_a_live_standalone_record_without_waiting(fleet, monkeypatch):
    _live_standalone_record(monkeypatch, fleet.solo)
    started = time.monotonic()
    owner = host_attach.host_gateway(wait_for_channel=host_attach.ATTACH_CHANNEL_WAIT_S)
    assert time.monotonic() - started < 1.0
    assert owner is not None and owner.standalone
    assert not owner.served_known, "multiplex:false says who cannot be served, not who is"


def test_probe_keeps_an_unflagged_record_unknown(fleet, monkeypatch):
    record = _live_standalone_record(monkeypatch, fleet.solo)
    record.multiplex = None
    owner = host_attach.host_gateway()
    assert owner is not None and not owner.standalone and not owner.served_known


@pytest.mark.parametrize("replace", [False, True])
def test_member_profile_starts_beside_a_standalone_owner_without_rescan(fleet, monkeypatch, replace):
    _live_standalone_record(monkeypatch, fleet.solo)
    monkeypatch.setattr(host_attach, "request_serve_profile",
                        lambda *a, **kw: pytest.fail("no rescan of a standalone owner"))
    decision = host_attach.decide(fleet.member, replace=replace)
    assert decision.outcome == host_attach.START, "never REPLACE_HOST another profile's standalone owner"


def test_a_live_multiplexer_answer_beats_a_stale_false_record(fleet, monkeypatch):
    """Record says multiplex:false, but the live owner answers that it multiplexes and serves member."""
    record = _live_standalone_record(monkeypatch, fleet.solo)
    identity = {"pid": record.pid, "hermes_home": str(fleet.solo), "profile": "solo",
                "multiplex": True, "served_profiles": ["solo", "member"]}
    monkeypatch.setattr(host_attach, "_identify", lambda h: identity)
    monkeypatch.setattr(host_attach, "_identity_matches", lambda ident, rec, home: ident is identity)
    owner = host_attach.host_gateway()
    assert owner is not None and not owner.standalone and owner.serves("member")
    assert host_attach.decide(fleet.member).outcome == host_attach.ATTACH


def test_a_standalone_restart_is_not_refused_by_its_own_old_instance(fleet, monkeypatch):
    """`gateway run --replace` (or launchd with an orphan): the live peer IS this home's old process."""
    from gateway import status

    old_pid = os.getpid() + 7
    monkeypatch.setattr(status, "live_gateway_pid_for_home",
                        lambda h: old_pid if Path(h) == fleet.solo else None)
    old = {"pid": old_pid, "hermes_home": str(fleet.solo), "profile": "solo", "multiplex": False}
    monkeypatch.setattr(host_attach, "_identify", lambda h: old if Path(h) == fleet.solo else None)
    monkeypatch.setattr(host_attach, "host_gateway", lambda **kw: None)
    assert host_attach.decide(fleet.solo, replace=True).outcome == host_attach.START


def test_owner_is_standalone_answers_from_the_record_without_rescan(fleet, monkeypatch):
    from gateway import run

    _live_standalone_record(monkeypatch, fleet.solo)
    monkeypatch.setattr(host_attach, "request_serve_profile",
                        lambda *a, **kw: pytest.fail("no rescan of a standalone owner"))
    assert run._owner_is_standalone() is True


def test_standalone_lock_loser_does_not_wait_for_the_owner(fleet, monkeypatch):
    from gateway import run

    monkeypatch.setattr(run, "get_hermes_home", lambda: fleet.solo)
    monkeypatch.setattr(hr, "claim_host_lock", lambda role: (hr.HostLockOutcome.HELD_BY_OTHER, None))
    monkeypatch.setattr(hr, "read_record", lambda role, **kw: None)
    waits = []
    unknown = host_attach.HostGateway(os.getpid() + 1, fleet.member, (), served_known=False)
    monkeypatch.setattr(host_attach, "host_gateway",
                        lambda **kw: waits.append(kw.get("wait_for_channel", 0.0)) or unknown)
    run._claim_host_gateway_role()  # returns: starts beside an owner it knows nothing about
    assert all(not wait for wait in waits)


# --- the cron ticker ----------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["solo", "custom"])
def test_standalone_cron_ticker_visits_only_its_own_home(fleet, monkeypatch, which):
    from gateway import run
    from hermes_cli import profiles

    home = getattr(fleet, which)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: which)
    homes = run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=False))
    assert homes == [(which, home)]


def test_default_home_ticker_is_unchanged(fleet, monkeypatch):
    """The default home is never standalone (it is the host), even with the key set."""
    from gateway import run
    from hermes_cli import profiles

    (fleet.root / "config.yaml").write_text("gateway:\n  standalone: true\n")
    monkeypatch.setenv("HERMES_HOME", str(fleet.root))
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    names = [name for name, _home in run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True))]
    assert names[0] == "default" and "member" in names and "solo" not in names

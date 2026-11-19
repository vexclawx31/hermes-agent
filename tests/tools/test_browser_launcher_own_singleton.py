"""A browser Hermes launched on a persistent profile leaves no stale singleton behind, and a leftover one is
recovered only on proof that it is this machine's own.

Chromium keeps ``SingletonLock`` (``<hostname>-<pid>``), ``SingletonCookie`` and ``SingletonSocket`` when it exits
on SIGTERM. Under an unchanged host name a later launch takes the stale lock over; when the machine's host name
changed in between (macOS without a fixed HostName follows the network), the lock looked like another
environment's browser and every launch was refused. Runs the real launcher against a fake browser that, like the
real one, keeps its singleton when terminated."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import FAKE_CHROMIUM
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = [pytest.mark.platforms("posix"), pytest.mark.live_system_guard_bypass]  # the fakes are ours to stop

# The shared fake releases its singleton on SIGTERM; the real browser does not. Same script, minus that unlink.
_RELEASE = '    with __import__("contextlib").suppress(OSError):\n        os.unlink(lock)\n'
assert FAKE_CHROMIUM.count(_RELEASE) == 1
_KEEPS_SINGLETON = FAKE_CHROMIUM.replace(_RELEASE, "")


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """A persistent profile dir, the fake browser, and a private launcher state root."""
    import sys
    fake = tmp_path / "chromium-keeps-singleton"
    fake.write_text(_KEEPS_SINGLETON.format(python=sys.executable))
    fake.chmod(0o755)
    shared = tmp_path / "shared-state"
    shared.mkdir(mode=0o700)
    monkeypatch.setattr(launcher, "_shared_profiles_root", lambda: shared)
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(launcher, "_SINGLETON_EXIT_WAIT_S", 10.0)
    home = tmp_path / "home"
    home.mkdir()
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    udd = tmp_path / "profile"
    started: list[int] = []

    def launch(key: str = "profile:p"):
        browser, err = launcher.launch_chromium(key, executable=str(fake), user_data_dir=str(udd), ephemeral=False,
                                                headed=False, env=dict(os.environ), timeout=30)
        if browser is not None:
            started.append(browser.pid)
        return browser, err

    yield {"udd": udd, "fake": fake, "launch": launch, "state": lambda: launcher._shared_state_dir(str(udd))}
    try:
        launcher.stop_all_browsers()
    finally:
        reset_hermes_home_override(token)
        for pid in started:
            if launcher._pid_alive(pid):
                os.kill(pid, signal.SIGKILL)


def _links(udd: Path) -> list[str]:
    return sorted(name for name in launcher._SINGLETON_LINKS if os.path.lexists(udd / name))


def _marker(state: Path) -> Path:
    return state / launcher._OWN_SINGLETON


def _forget_without_cleanup(key: str = "profile:p") -> None:
    """What a crashed Hermes leaves: nothing in memory, the launch record, the marker and the lock on disk."""
    with launcher._lock:
        launcher._browsers.pop(key, None)
        launcher._procs.pop(key, None)


def test_a_stop_leaves_no_singleton_links_and_no_marker(profile):
    browser, err = profile["launch"]()
    assert browser is not None, err
    assert _links(profile["udd"]) == ["SingletonLock"] and _marker(profile["state"]()).is_file()
    launcher.stop_browser("profile:p")
    assert not launcher._pid_alive(browser.pid)
    assert _links(profile["udd"]) == [] and not _marker(profile["state"]()).exists()


def test_after_a_host_name_change_our_own_dead_browsers_lock_is_recovered_and_a_new_one_starts(profile, monkeypatch):
    first, err = profile["launch"]()
    assert first is not None, err
    os.kill(first.pid, signal.SIGKILL)                                # the browser dies; Hermes "crashed" too
    wait_until(lambda: not launcher._pid_alive(first.pid), timeout=10)
    _forget_without_cleanup()
    for path in profile["state"]().glob("*.json"):                    # no launch record survives either
        path.unlink()
    assert _links(profile["udd"]) == ["SingletonLock"]
    monkeypatch.setattr(socket, "gethostname", lambda: "another-name-after-a-network-change")
    second, err = profile["launch"]()
    assert second is not None, err                                    # not "in use by a Chromium in another environment"
    assert second.pid != first.pid and second.owned is True
    assert os.readlink(profile["udd"] / "SingletonLock").endswith(f"-{second.pid}")


def test_a_lock_of_another_host_name_without_our_marker_is_still_refused_and_untouched(profile):
    udd = profile["udd"]
    udd.mkdir()
    os.symlink("some-other-host-4242", udd / "SingletonLock")
    os.symlink("12345", udd / "SingletonCookie")
    browser, err = profile["launch"]()
    assert browser is None and "another environment" in err
    assert os.readlink(udd / "SingletonLock") == "some-other-host-4242" and _links(udd) == ["SingletonCookie",
                                                                                           "SingletonLock"]


def test_a_live_browsers_singleton_is_never_cleared(profile):
    browser, err = profile["launch"]()
    assert browser is not None, err
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), profile["state"](), wait=0.3) is False
    assert _links(profile["udd"]) == ["SingletonLock"] and launcher._pid_alive(browser.pid)
    assert _marker(profile["state"]()).is_file()


def _records_lock(profile):
    from tools import browser_launcher_holders as holders
    return holders.records_lock(profile["state"]())


def _dead_browser_state(profile, *, pid: int, start_time, lock: str) -> Path:
    """A profile with a leftover lock and our marker for it, no browser running."""
    from tools import browser_launcher_holders as holders
    from utils import atomic_json_write
    udd = profile["udd"]
    udd.mkdir(exist_ok=True)
    os.symlink(lock, udd / "SingletonLock")
    os.symlink("777", udd / "SingletonCookie")
    target = udd / "socket-dir"
    target.mkdir()
    (target / "SingletonSocket").write_text("kept")
    os.symlink(str(target / "SingletonSocket"), udd / "SingletonSocket")
    state = profile["state"]()
    atomic_json_write(_marker(state), {"pid": pid, "start_time": start_time, **launcher._lock_identity(str(udd)),
                                       "ns": holders.ns_identity(), "machine": launcher._machine_id()}, indent=None)
    return state


def _dead_pid() -> int:
    dead = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()
    return dead.pid


def test_only_the_links_are_removed_never_what_they_point_at(profile):
    dead = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()
    state = _dead_browser_state(profile, pid=dead.pid, start_time=1, lock=f"old-host-{dead.pid}")
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is True
    assert _links(profile["udd"]) == [] and not _marker(state).exists()
    assert (profile["udd"] / "socket-dir" / "SingletonSocket").read_text() == "kept"


def test_a_recycled_pid_with_another_start_time_does_not_keep_the_dead_browsers_lock(profile):
    me = os.getpid()                                                  # alive, but not the browser the marker names
    state = _dead_browser_state(profile, pid=me, start_time=(launcher._start_time(me) or 0) + 12345,
                                lock=f"old-host-{me}")
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is True
    assert _links(profile["udd"]) == []


def test_a_live_pid_whose_start_time_is_unknown_is_treated_as_the_browser_and_left_alone(profile):
    me = os.getpid()
    state = _dead_browser_state(profile, pid=me, start_time=None, lock=f"old-host-{me}")
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state, wait=0.3) is False
    assert "SingletonLock" in _links(profile["udd"]) and _marker(state).is_file()


def test_a_lock_that_was_replaced_since_the_marker_is_not_ours_to_remove(profile):
    dead = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    dead.wait()
    state = _dead_browser_state(profile, pid=dead.pid, start_time=1, lock=f"old-host-{dead.pid}")
    os.unlink(profile["udd"] / "SingletonLock")
    os.symlink("somebody-elses-host-999", profile["udd"] / "SingletonLock")   # another browser took the profile
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is False
    assert os.readlink(profile["udd"] / "SingletonLock") == "somebody-elses-host-999"
    assert "SingletonCookie" in _links(profile["udd"])


def _rewrite_marker(state: Path, **change) -> None:
    import json
    from utils import atomic_json_write
    atomic_json_write(_marker(state), {**json.loads(_marker(state).read_text()), **change}, indent=None)


@pytest.mark.parametrize("change", [{"ns": "another-namespace:another-boot"}, {"machine": 1}])
def test_a_marker_from_another_pid_namespace_or_another_machine_proves_nothing(profile, change):
    pid = _dead_pid()
    state = _dead_browser_state(profile, pid=pid, start_time=1, lock=f"old-host-{pid}")
    _rewrite_marker(state, **change)
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is False
    assert "SingletonLock" in _links(profile["udd"])


def test_a_new_link_with_the_very_same_text_is_another_browsers_lock(profile):
    """The dead browser's pid was recycled by a browser the process scan cannot see (no ``--user-data-dir=`` in its
    argv, another OS user's, a container's): its lock reads the same, but it is another link. Never unlinked."""
    pid = _dead_pid()
    state = _dead_browser_state(profile, pid=pid, start_time=1, lock=f"same-host-{pid}")
    lock = profile["udd"] / "SingletonLock"
    os.unlink(lock)
    os.symlink(f"same-host-{pid}", lock)                              # the replacement: same text, a new link
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is False
    assert os.readlink(lock) == f"same-host-{pid}" and "SingletonCookie" in _links(profile["udd"])
    assert not _marker(state).exists()                                # that marker can never match again


def test_a_browser_process_on_the_profile_blocks_the_clear_even_when_the_marked_pid_is_dead(profile, monkeypatch):
    pid = _dead_pid()
    state = _dead_browser_state(profile, pid=pid, start_time=1, lock=f"old-host-{pid}")
    monkeypatch.setattr(launcher, "_browser_running_on", lambda udd: True)
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is False
    assert "SingletonLock" in _links(profile["udd"]) and _marker(state).is_file()


def test_a_launch_record_of_another_pid_namespace_naming_that_pid_blocks_the_clear(profile, monkeypatch):
    pid = _dead_pid()
    state = _dead_browser_state(profile, pid=pid, start_time=1, lock=f"old-host-{pid}")
    asked = []
    monkeypatch.setattr(launcher, "_foreign_record_names", lambda udd, p: asked.append(p) or True)
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state) is False
    assert asked == [pid] and "SingletonLock" in _links(profile["udd"])


def test_a_lock_replaced_while_waiting_for_the_browser_to_die_is_left_alone(profile, monkeypatch):
    from tools import browser_launcher_holders as holders
    pid = _dead_pid()
    state = _dead_browser_state(profile, pid=pid, start_time=1, lock=f"old-host-{pid}")
    lock = profile["udd"] / "SingletonLock"
    calls = []

    def dying(p, start):                                              # alive on the first look, gone on the second;
        calls.append(p)                                               # meanwhile another browser took the profile
        if len(calls) == 1:
            os.unlink(lock)
            os.symlink("new-browser-host-31337", lock)
            return True
        return False

    monkeypatch.setattr(holders, "process_alive", dying)
    with _records_lock(profile):
        assert launcher._clear_own_singleton(str(profile["udd"]), state, wait=2.0) is False
    assert os.readlink(lock) == "new-browser-host-31337"


def test_no_marker_is_written_when_the_lock_names_another_pid_than_the_launched_one(profile, monkeypatch):
    """A wrapper binary: the process Hermes started is not the one holding the singleton. No proof, no marker."""
    browser, err = profile["launch"]()
    assert browser is not None, err
    _marker(profile["state"]()).unlink()
    monkeypatch.setattr(launcher, "_singleton_owner", lambda udd: ("some-host", browser.pid + 1))
    launcher._note_own_singleton(browser)
    assert not _marker(profile["state"]()).exists()


def test_the_marker_is_not_a_launch_record_to_the_reaper(profile):
    browser, err = profile["launch"]()
    assert browser is not None, err
    marker = _marker(profile["state"]())
    assert marker.suffix != ".json"                                   # the reaper reads every *.json as a record
    launcher.reap_orphaned_browsers()
    assert marker.is_file() and launcher._pid_alive(browser.pid)

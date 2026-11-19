"""``browser.own_profile``: the operator's own browser (``browser.executable_path``) on the agent's own persistent
profile (``browser.user_data_dir``). The browser tool starts it on demand and stops it with its last session, and
launches it as a real browser (the pinned binary, no packaged-only keychain flags, no sandbox bypass, on macOS the
OS account's own HOME), with owned tabs per session. Without the opt-in a pinned binary on a persistent dir keeps
the packaged launch. Runs the real session layer + launcher against fake browser binaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

_FAKE_BROWSER = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    os.makedirs(udd, exist_ok=True)
    with open(os.path.join(udd, "launch.json"), "w") as f:
        json.dump({{"argv": sys.argv, "home": os.environ.get("HOME"), "user": os.environ.get("USER"),
                   "logname": os.environ.get("LOGNAME"), "marker": os.environ.get("OWN_ENV_MARKER")}}, f)
    with open(os.path.join(udd, "DevToolsActivePort"), "w") as f:
        f.write("{port}\\n/devtools/browser/{{}}\\n".format(os.getpid()))
    time.sleep(600)
''')


def _fake_browser(tmp_path: Path, name: str, port: int) -> Path:
    script = tmp_path / name
    script.write_text(_FAKE_BROWSER.format(python=sys.executable, port=port))
    script.chmod(0o755)
    return script


@pytest.fixture
def agent_home(tmp_path, monkeypatch):
    """One agent home; ``configure`` writes its browser settings. ``packaged`` is what the launcher resolves
    itself, ``own`` the operator's pinned binary."""
    from tools import browser_launcher as launcher
    packaged = _fake_browser(tmp_path, "packaged-chromium", 45681)
    own = _fake_browser(tmp_path, "own-browser", 45682)
    monkeypatch.setattr(launcher, "_resolve_executable", lambda: str(packaged))
    monkeypatch.setattr(launcher, "chromium_ready", lambda: (True, ""))
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: True)  # a host that WOULD add bypass flags
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launcher, "_records_dir", lambda _home: records)
    shared = tmp_path / "shared-state"
    shared.mkdir()
    monkeypatch.setattr(launcher, "_shared_profiles_root", lambda: shared)
    home = tmp_path / "agent-home"
    home.mkdir()
    profile = home / "browser-profiles" / "own"

    def configure(*, own_profile: bool, pinned: bool = True, user_data_dir: bool = True,
                  executable: str | None = None) -> None:
        browser: dict = {}
        if user_data_dir:
            browser["user_data_dir"] = "browser-profiles/own"
        if pinned:
            browser["executable_path"] = executable or str(own)
        if own_profile:
            browser["own_profile"] = True
        (home / "config.yaml").write_text(json.dumps({"browser": browser}))

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    yield {"home": home, "profile": profile, "own": own, "packaged": packaged, "configure": configure}
    try:
        from tools.browser_session_lifecycle import cleanup_all_browsers
        cleanup_all_browsers()
    finally:
        reset_hermes_home_override(token)


def _session(task: str, *, sidecar: bool = False) -> dict:
    from tools.browser_session import get_session_info
    from tools.browser_use_cli import _backend_cache_key
    return get_session_info(_backend_cache_key(task, "", sidecar))


def _launch_record(profile: Path) -> dict:
    return json.loads((profile / "launch.json").read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_gone(pid: int, seconds: float = 10.0) -> bool:
    from tests.waiting import wait_until
    try:
        wait_until(lambda: not _alive(pid), timeout=seconds)
    except (AssertionError, TimeoutError):
        return False
    return True


# ── the launch ───────────────────────────────────────────────────────────────────────────────────────

def test_the_own_profile_is_launched_as_a_real_browser_with_the_own_env(agent_home, monkeypatch):
    from tools import browser_session
    real_env = browser_session.own_profile_browser_env
    monkeypatch.setattr(browser_session, "own_profile_browser_env",
                        lambda: {**real_env(), "OWN_ENV_MARKER": "from-own-profile-env"})
    agent_home["configure"](own_profile=True)
    info = _session("task-1")
    assert info["shared_profile"] is True and info["launcher_key"] == f"profile:{agent_home['profile']}"
    launched = _launch_record(agent_home["profile"])
    argv = launched["argv"]
    assert argv[0] == str(agent_home["own"])                       # the pinned binary, not the packaged one
    assert f"--user-data-dir={agent_home['profile']}" in argv
    assert "--password-store=basic" not in argv and "--use-mock-keychain" not in argv   # the real OS store
    from tools.browser_launcher import CHROMIUM_SANDBOX_BYPASS_ARGS
    assert not set(CHROMIUM_SANDBOX_BYPASS_ARGS) & set(argv)       # never started without its own sandbox
    assert "--remote-debugging-port=0" in argv                     # no fixed port to collide on
    assert launched["marker"] == "from-own-profile-env"            # the child really got own_profile_browser_env()


def test_a_pinned_binary_on_a_persistent_dir_without_the_opt_in_keeps_the_packaged_launch(agent_home):
    """What that pair always did (the Docker / AppArmor ``executable_path`` setups): packaged flags and the
    host's sandbox bypass, the served env, the own-tab preamble only."""
    agent_home["configure"](own_profile=False)
    info = _session("task-1")
    launched = _launch_record(agent_home["profile"])
    assert "--password-store=basic" in launched["argv"]
    from tools.browser_launcher import CHROMIUM_SANDBOX_BYPASS_ARGS
    assert set(CHROMIUM_SANDBOX_BYPASS_ARGS) <= set(launched["argv"])
    assert launched["marker"] is None and "own_profile" not in info["features"]


def test_without_a_pin_the_persistent_profile_keeps_the_packaged_launch(agent_home):
    agent_home["configure"](own_profile=False, pinned=False)
    _session("task-1")
    launched = _launch_record(agent_home["profile"])
    assert launched["argv"][0] == str(agent_home["packaged"])
    assert "--password-store=basic" in launched["argv"]


@pytest.mark.parametrize("settings", [{"pinned": False}, {"user_data_dir": False}])
def test_the_opt_in_without_its_binary_or_its_dir_is_refused_and_nothing_is_launched(agent_home, settings):
    agent_home["configure"](own_profile=True, **settings)
    with pytest.raises(RuntimeError, match="needs both browser.user_data_dir .* and browser.executable_path"):
        _session("task-1")
    assert not (agent_home["profile"] / "launch.json").exists()    # never the packaged Chromium on that profile


def test_a_pinned_path_that_is_no_runnable_browser_is_refused_and_nothing_is_launched(agent_home, tmp_path):
    agent_home["configure"](own_profile=True, executable=str(tmp_path / "no-such-browser"))
    with pytest.raises(RuntimeError, match="is not a browser this process can run"):
        _session("task-1")
    assert not (agent_home["profile"] / "launch.json").exists()


def test_a_hybrid_local_sidecar_never_uses_the_signed_in_profile(agent_home):
    agent_home["configure"](own_profile=True)
    info = _session("task-1", sidecar=True)
    assert "own_profile" not in info["features"] and not info.get("shared_profile")
    assert not (agent_home["profile"] / "launch.json").exists()    # a throwaway profile, the packaged binary
    from tools.browser_owned_tabs import is_shared_cdp_session
    assert not is_shared_cdp_session(info)


# ── owned tabs ───────────────────────────────────────────────────────────────────────────────────────

def test_sessions_on_the_own_browser_are_owned_tab_sessions_like_a_cdp_override(agent_home):
    """Every session of the agent shares the one signed-in browser, so each works only in tabs it opened: the
    same guard as on a CDP override, not the best-effort own tab of a packaged shared profile."""
    agent_home["configure"](own_profile=True)
    info = _session("task-1")
    assert info["features"].get("own_profile") is True and info["features"].get("local") is True
    from tools.browser_owned_tabs import is_shared_cdp_session
    assert is_shared_cdp_session(info)
    from tools import browser_use_cli
    env: dict = {}
    browser_use_cli._mark_local_browser(env, info, "any-key")
    assert env.get(browser_use_cli._SHARED_CDP_SENTINEL) == "1"
    assert browser_use_cli._PRIVATE_BROWSER_SENTINEL not in env


def test_a_packaged_persistent_profile_keeps_its_own_tab_preamble_only(agent_home):
    agent_home["configure"](own_profile=False, pinned=False)
    info = _session("task-1")
    assert "own_profile" not in info["features"]
    from tools.browser_owned_tabs import is_shared_cdp_session
    assert not is_shared_cdp_session(info)
    from tools import browser_use_cli
    env: dict = {}
    browser_use_cli._mark_local_browser(env, info, "any-key")
    assert browser_use_cli._SHARED_CDP_SENTINEL not in env


def test_a_sessions_end_closes_its_own_tabs_first_and_only_the_last_one_stops_the_browser(agent_home, monkeypatch):
    agent_home["configure"](own_profile=True)
    first, second = _session("task-1"), _session("task-2")
    from tools.browser_launcher import get_browser
    pid = get_browser(first["launcher_key"]).pid
    released = []
    from tools import browser_owned_tabs
    # Recorded WITH the browser's state at that moment: the tabs are closed while it is still running.
    monkeypatch.setattr(browser_owned_tabs, "release_owned_tabs",
                        lambda record: released.append((record["session_name"], _alive(pid))))
    from tools.browser_session_lifecycle import cleanup_single_session
    from tools.browser_use_cli import _backend_cache_key
    cleanup_single_session(_backend_cache_key("task-1"))
    assert released == [(first["session_name"], True)] and _alive(pid)   # its tabs only; the browser stays
    cleanup_single_session(_backend_cache_key("task-2"))
    assert released[-1] == (second["session_name"], True)          # own tabs first, with the browser still up
    assert _wait_gone(pid)                                         # then the last session's end stops it


# ── start and stop ───────────────────────────────────────────────────────────────────────────────────

def test_two_sessions_share_the_one_browser_and_only_the_last_to_end_stops_it(agent_home):
    agent_home["configure"](own_profile=True)
    first, second = _session("task-1"), _session("task-2")
    assert first["cdp_url"] == second["cdp_url"] and first["launcher_key"] == second["launcher_key"]
    from tools.browser_launcher import get_browser
    pid = get_browser(first["launcher_key"]).pid
    from tools.browser_session_lifecycle import cleanup_single_session
    from tools.browser_use_cli import _backend_cache_key
    cleanup_single_session(_backend_cache_key("task-1"))
    assert _alive(pid)                                             # task-2 still uses it
    cleanup_single_session(_backend_cache_key("task-2"))
    assert _wait_gone(pid)                                         # the last session's end stops it


def test_the_next_call_after_the_stop_starts_it_again_with_a_fresh_record(agent_home):
    agent_home["configure"](own_profile=True)
    first = _session("task-1")
    first["owned_target_ids"] = ["tab-of-the-old-browser"]         # what a call on the old browser left behind
    from tools.browser_launcher import get_browser
    pid = get_browser(first["launcher_key"]).pid
    from tools.browser_session_lifecycle import cleanup_single_session
    from tools.browser_use_cli import _backend_cache_key
    cleanup_single_session(_backend_cache_key("task-1"))
    assert _wait_gone(pid)
    again = _session("task-1")
    new_pid = get_browser(again["launcher_key"]).pid
    assert new_pid != pid and _alive(new_pid)
    assert not again.get("owned_target_ids")                       # no tab id of the stopped browser survives
    assert again["features"].get("own_profile") is True


def test_a_browser_somebody_else_started_on_the_profile_is_used_and_never_stopped(agent_home, tmp_path):
    agent_home["configure"](own_profile=True)
    profile = agent_home["profile"]
    profile.mkdir(parents=True)
    # The shared fake behaves like a real browser where the launcher looks: it takes the profile's singleton
    # lock and answers /json/version on the port it writes to DevToolsActivePort.
    from tests.tools.browser_launcher_fakes import write_fake_chromium
    from tests.waiting import wait_until
    real_like = write_fake_chromium(tmp_path / "started-by-hand")
    by_hand = subprocess.Popen([str(real_like), f"--user-data-dir={profile}"], start_new_session=True)
    try:
        wait_until((profile / "DevToolsActivePort").exists, timeout=10)
        info = _session("task-1")
        from tools.browser_launcher import get_browser
        attached = get_browser(info["launcher_key"])
        assert attached.pid == by_hand.pid and attached.owned is False
        from tools.browser_session_lifecycle import cleanup_all_browsers
        cleanup_all_browsers()
        with pytest.raises(subprocess.TimeoutExpired):             # still running: not ours to stop
            by_hand.wait(timeout=1)
    finally:
        by_hand.terminate()
        by_hand.wait(timeout=10)

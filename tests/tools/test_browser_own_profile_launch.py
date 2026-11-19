"""The operator's own browser on the agent's own persistent profile (``browser.executable_path`` +
``browser.user_data_dir``): the browser tool starts it on demand and stops it with its last session, and it is
launched as a real browser (the pinned binary, no packaged-only keychain flags, on macOS the OS account's own
HOME), never as the packaged Chromium. Runs the real session layer + launcher against a fake browser binary that
records its argv and environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
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
                   "logname": os.environ.get("LOGNAME")}}, f)
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
    """One agent home whose config pins its own browser binary and profile dir; ``packaged`` is what the
    launcher would resolve without the pin."""
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

    def configure(pinned: bool) -> None:
        browser = {"user_data_dir": "browser-profiles/own"}
        if pinned:
            browser["executable_path"] = str(own)
        (home / "config.yaml").write_text(json.dumps({"browser": browser}))

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    yield {"home": home, "profile": profile, "own": own, "packaged": packaged, "configure": configure}
    try:
        from tools.browser_session_lifecycle import cleanup_all_browsers
        cleanup_all_browsers()
    finally:
        reset_hermes_home_override(token)


def _session(task: str) -> dict:
    from tools.browser_session import get_session_info
    from tools.browser_use_cli import _backend_cache_key
    return get_session_info(_backend_cache_key(task))


def _launch_record(profile: Path) -> dict:
    return json.loads((profile / "launch.json").read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_a_pinned_binary_on_the_agents_profile_is_launched_as_a_real_browser(agent_home):
    agent_home["configure"](pinned=True)
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


def test_without_a_pin_the_persistent_profile_keeps_the_packaged_launch(agent_home):
    agent_home["configure"](pinned=False)
    _session("task-1")
    launched = _launch_record(agent_home["profile"])
    assert launched["argv"][0] == str(agent_home["packaged"])
    assert "--password-store=basic" in launched["argv"]
    assert launched["home"] == os.environ.get("HOME")              # the env is untouched without a pin


def test_the_own_browser_env_uses_this_accounts_home_on_macos_only(monkeypatch, tmp_path):
    from tools import browser_session
    monkeypatch.setattr(browser_session, "build_browser_env", lambda: {"HOME": "/agent/home", "USER": "agent",
                                                                       "KEEP": "1"})
    import pwd
    account = pwd.getpwuid(os.getuid())
    monkeypatch.setattr(sys, "platform", "darwin")
    env = browser_session.own_profile_browser_env()
    assert env["HOME"] == account.pw_dir and env["USER"] == account.pw_name and env["LOGNAME"] == account.pw_name
    assert env["KEEP"] == "1"                                      # nothing else of the env changes
    monkeypatch.setattr(sys, "platform", "linux")
    assert browser_session.own_profile_browser_env() == {"HOME": "/agent/home", "USER": "agent", "KEEP": "1"}


def test_the_account_is_this_process_own_uid_never_one_from_the_environment(monkeypatch):
    from tools import browser_session
    monkeypatch.setattr(browser_session, "build_browser_env", lambda: {"HOME": "/agent/home"})
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", "/somebody/else")
    monkeypatch.setenv("USER", "somebody")
    monkeypatch.setenv("SUDO_USER", "somebody")
    import pwd
    asked = []
    real = pwd.getpwuid

    def getpwuid(uid):
        asked.append(uid)
        return real(uid)

    monkeypatch.setattr(pwd, "getpwuid", getpwuid)
    env = browser_session.own_profile_browser_env()
    assert asked == [os.getuid()] and env["HOME"] == real(os.getuid()).pw_dir


def test_an_account_without_a_usable_home_leaves_the_env_alone(monkeypatch):
    from tools import browser_session
    monkeypatch.setattr(browser_session, "build_browser_env", lambda: {"HOME": "/agent/home"})
    monkeypatch.setattr(sys, "platform", "darwin")
    import pwd

    class Account:
        pw_dir = "/nonexistent/home/of/nobody"
        pw_name = "nobody"

    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: Account())
    assert browser_session.own_profile_browser_env() == {"HOME": "/agent/home"}


def test_sessions_on_the_own_browser_are_owned_tab_sessions_like_a_cdp_override(agent_home):
    """Every session of the agent shares the one signed-in browser, so each works only in tabs it opened: the
    same guard as on a CDP override, not the best-effort own tab of a packaged shared profile."""
    agent_home["configure"](pinned=True)
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
    agent_home["configure"](pinned=False)
    info = _session("task-1")
    assert "own_profile" not in info["features"]
    from tools.browser_owned_tabs import is_shared_cdp_session
    assert not is_shared_cdp_session(info)
    from tools import browser_use_cli
    env: dict = {}
    browser_use_cli._mark_local_browser(env, info, "any-key")
    assert browser_use_cli._SHARED_CDP_SENTINEL not in env


def test_a_sessions_end_closes_its_own_tabs_before_the_browser_is_left_or_stopped(agent_home, monkeypatch):
    agent_home["configure"](pinned=True)
    first, second = _session("task-1"), _session("task-2")
    released = []
    from tools import browser_owned_tabs
    monkeypatch.setattr(browser_owned_tabs, "release_owned_tabs", lambda record: released.append(record["session_name"]))
    from tools.browser_launcher import get_browser
    pid = get_browser(first["launcher_key"]).pid
    from tools.browser_session_lifecycle import cleanup_single_session
    from tools.browser_use_cli import _backend_cache_key
    cleanup_single_session(_backend_cache_key("task-1"))
    assert released == [first["session_name"]] and _alive(pid)     # its tabs only; the other session's browser stays
    cleanup_single_session(_backend_cache_key("task-2"))
    assert released == [first["session_name"], second["session_name"]] and _wait_gone(pid)


def test_two_sessions_share_the_one_browser_and_only_the_last_to_end_stops_it(agent_home):
    agent_home["configure"](pinned=True)
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


def test_the_next_call_after_the_stop_starts_it_again(agent_home):
    agent_home["configure"](pinned=True)
    first = _session("task-1")
    from tools.browser_launcher import get_browser
    pid = get_browser(first["launcher_key"]).pid
    from tools.browser_session_lifecycle import cleanup_single_session
    from tools.browser_use_cli import _backend_cache_key
    cleanup_single_session(_backend_cache_key("task-1"))
    assert _wait_gone(pid)
    again = _session("task-1")
    new_pid = get_browser(again["launcher_key"]).pid
    assert new_pid != pid and _alive(new_pid)


def test_a_browser_somebody_else_started_on_the_profile_is_used_and_never_stopped(agent_home, tmp_path):
    agent_home["configure"](pinned=True)
    profile = agent_home["profile"]
    profile.mkdir(parents=True)
    # The shared fake behaves like a real browser where the launcher looks: it takes the profile's singleton
    # lock and answers /json/version on the port it writes to DevToolsActivePort.
    from tests.tools.browser_launcher_fakes import write_fake_chromium
    real_like = write_fake_chromium(tmp_path / "started-by-hand")
    by_hand = subprocess.Popen([str(real_like), f"--user-data-dir={profile}"], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not (profile / "DevToolsActivePort").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
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

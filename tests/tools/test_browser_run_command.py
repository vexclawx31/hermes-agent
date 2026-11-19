"""``hermes browser run -- <command>``: a helper command holds the agent's own browser (``browser.own_profile``)
exactly as a browser session does, gets its DevTools address, and lets go on exit. Every case runs the wrapper as
its own process against the shared fake browser, through the real launcher and its cross-process leases."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import pids, write_fake_chromium
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")

REPO = Path(__file__).resolve().parents[2]
_WRAPPER = ("import sys; from hermes_cli.browser_run import run; "
            "sys.exit(run(sys.argv[2:], float(sys.argv[1])))")

# The helper command: records what it was given, proves the endpoint answers, optionally starts a child of its
# own, then waits for <out>.go (or forever) and exits 7.
_HELPER = textwrap.dedent('''\
    import json, os, subprocess, sys, time, urllib.request
    out, mode = sys.argv[1], sys.argv[2]
    url, port = os.environ.get("HERMES_BROWSER_CDP_URL"), os.environ.get("HERMES_BROWSER_CDP_PORT")
    version = json.loads(urllib.request.urlopen(url + "/json/version", timeout=5).read())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"]).pid if mode == "tree" else None
    with open(out + ".tmp", "w") as f:
        json.dump({"url": url, "port": port, "ws": version["webSocketDebuggerUrl"], "pid": os.getpid(),
                   "child": child}, f)
    os.replace(out + ".tmp", out)
    while mode != "exit" and not os.path.exists(out + ".go"):
        time.sleep(0.05)
    sys.exit(7)
''')


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def own(tmp_path):
    """An agent home with an own-profile browser (the shared fake), and ``start`` to run the wrapper in it."""
    home = tmp_path / "agent-home"
    home.mkdir()
    profile = home / "browser-profiles" / "own"
    fake = write_fake_chromium(tmp_path / "own-browser")
    helper = tmp_path / "helper.py"
    helper.write_text(_HELPER)
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_GATEWAY_LOCK_DIR": str(tmp_path / "locks"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    started: list[subprocess.Popen] = []

    def configure(own_profile: bool = True) -> None:
        browser = {"user_data_dir": "browser-profiles/own", "executable_path": str(fake)}
        if own_profile:
            browser["own_profile"] = True
        (home / "config.yaml").write_text(json.dumps({"browser": browser}))

    def start(name: str, mode: str = "wait", deadline: float = 60.0) -> tuple[subprocess.Popen, Path]:
        out = tmp_path / f"{name}.json"
        proc = subprocess.Popen([sys.executable, "-c", _WRAPPER, str(deadline), sys.executable, str(helper),
                                 str(out), mode], cwd=REPO, env=env, stderr=subprocess.PIPE, text=True)
        started.append(proc)
        return proc, out

    def report(out: Path) -> dict:
        wait_until(out.exists, timeout=30)
        return json.loads(out.read_text())

    configure()
    yield {"profile": profile, "configure": configure, "start": start, "report": report,
           "browsers": lambda: pids(profile, "launches")}
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
    for pid in pids(profile, "launches"):
        if _alive(pid):
            os.kill(pid, signal.SIGTERM)


def test_the_command_gets_the_live_loopback_endpoint_and_the_browser_stops_after_it(own):
    proc, out = own["start"]("a", mode="exit")
    seen = own["report"](out)
    assert proc.wait(timeout=30) == 7                               # the command's own exit status
    assert seen["url"] == f"http://127.0.0.1:{seen['port']}"
    (browser,) = own["browsers"]()                                  # one browser was started for it
    assert seen["ws"].endswith(f"/devtools/browser/{browser}")      # and the address is that browser's
    wait_until(lambda: not _alive(browser), timeout=15)             # the last holder's exit stops it


def test_another_holder_keeps_the_browser_open(own):
    first, out_a = own["start"]("a")
    a = own["report"](out_a)
    second, out_b = own["start"]("b")
    b = own["report"](out_b)
    (browser,) = own["browsers"]()                                  # the second attached: still one browser
    assert a["port"] == b["port"]
    Path(str(out_a) + ".go").touch()
    assert first.wait(timeout=30) == 7
    assert _alive(browser) and not _alive(a["pid"])                 # the other command still holds it
    Path(str(out_b) + ".go").touch()
    assert second.wait(timeout=30) == 7
    wait_until(lambda: not _alive(browser), timeout=15)


def test_the_deadline_stops_the_commands_whole_process_tree_before_the_browser(own):
    proc, out = own["start"]("a", mode="tree", deadline=3.0)
    seen = own["report"](out)
    (browser,) = own["browsers"]()
    assert proc.wait(timeout=40) == 124
    assert "ran longer than 3 s" in proc.stderr.read()
    assert not _alive(seen["pid"]) and not _alive(seen["child"])    # nothing of the command is left
    wait_until(lambda: not _alive(browser), timeout=15)


def test_a_terminated_wrapper_stops_its_command_and_lets_go_of_the_browser(own):
    proc, out = own["start"]("a", mode="tree")
    seen = own["report"](out)
    (browser,) = own["browsers"]()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=40) == 128 + signal.SIGTERM
    assert not _alive(seen["pid"]) and not _alive(seen["child"])
    wait_until(lambda: not _alive(browser), timeout=15)


def test_without_the_own_profile_opt_in_nothing_is_started_and_the_command_never_runs(own):
    own["configure"](own_profile=False)
    proc, out = own["start"]("a", mode="exit")
    assert proc.wait(timeout=30) == 2
    assert "browser.own_profile is off" in proc.stderr.read()
    assert not out.exists() and own["browsers"]() == []


def test_an_endpoint_that_is_not_loopback_http_is_never_handed_on():
    from hermes_cli.browser_run import _endpoint
    assert _endpoint("http://127.0.0.1:9333") == ("http://127.0.0.1:9333", 9333)
    for bad in ("http://192.0.2.7:9333", "https://127.0.0.1:9333", "http://127.0.0.1", "ws://127.0.0.1:9333", ""):
        assert _endpoint(bad) is None


def test_the_cli_passes_the_command_as_an_argv_without_the_separator(monkeypatch):
    from hermes_cli import browser_run
    from hermes_cli.subcommands.browser import build_browser_parser
    calls = []
    monkeypatch.setattr(browser_run, "run", lambda command, deadline: calls.append((command, deadline)) or 0)
    parser = argparse.ArgumentParser()
    build_browser_parser(parser.add_subparsers())
    args = parser.parse_args(["browser", "run", "--timeout", "90", "--", "node", "x.js", "--flag", "a b; c"])
    assert args.func(args) == 0
    args = parser.parse_args(["browser", "run", "--", "node", "x.js"])
    assert args.func(args) == 0
    assert calls == [(["node", "x.js", "--flag", "a b; c"], 90.0),
                     (["node", "x.js"], browser_run.DEFAULT_DEADLINE)]

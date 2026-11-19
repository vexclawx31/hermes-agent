"""``hermes browser run -- <command>``: run a helper command against the agent's own browser profile
(``browser.own_profile``) while holding that browser open.

A script that talks to the browser directly (raw CDP) is not a browser-tool session: nothing would keep the
browser running under it, and it could not know the port the browser picked. This command is that missing
holder. It takes the browser exactly as a browser session does (the same launcher, the same cross-process
leases: started if it is off, attached if it runs), hands the command the DevTools address in
``HERMES_BROWSER_CDP_PORT`` / ``HERMES_BROWSER_CDP_URL`` and lets go when the command has exited: the last holder
stops the browser, any other holder keeps it. No second launcher, no keepalive.

The address is no tab boundary: a helper must open and close only its own tabs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Optional, Sequence
from urllib.parse import urlsplit

DEFAULT_DEADLINE = 600.0   # seconds; a stuck helper must not pin the browser open
TIMEOUT_EXIT = 124         # as timeout(1)
_GRACE = 5.0               # seconds between SIGTERM and SIGKILL for the command's process group
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


class _Interrupted(Exception):
    """A termination signal reached the wrapper: stop the command, then let go of the browser."""


def _fail(message: str, code: int = 2) -> int:
    print(f"hermes browser run: {message}", file=sys.stderr)
    return code


def _endpoint(http_url: str) -> Optional[tuple[str, int]]:
    """``(base url, port)`` of a loopback DevTools endpoint, else ``None``."""
    parts = urlsplit(http_url or "")
    try:
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK or not port:
        return None
    return f"http://{parts.hostname if ':' not in parts.hostname else '[' + parts.hostname + ']'}:{port}", port


def _stop_command(proc: subprocess.Popen) -> None:
    """End the command's whole process group and reap it: nothing of it outlives the browser hold."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=_GRACE)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            try:   # the leader is gone; anything it left in its group goes too
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        return


def run(command: Sequence[str], deadline: float = DEFAULT_DEADLINE) -> int:
    """Run ``command`` (an argv, never a shell line) holding the own-profile browser; its exit status, 124 when the
    deadline ended it, 2 when the browser or the command could not be started."""
    if os.name != "posix":
        return _fail("only available on POSIX hosts.")
    command = list(command)
    if not command:
        return _fail("no command given. Usage: hermes browser run [--timeout SECONDS] -- <command> [args...]")
    if not deadline or deadline <= 0:
        return _fail("--timeout must be a positive number of seconds.")
    from tools.browser_session_cloud import own_profile
    if not own_profile():
        return _fail("this profile has no own browser (browser.own_profile is off); there is nothing to hold. "
                     "No other browser is used instead.")
    from tools.browser_launcher import launch_chromium, stop_browser
    from tools.browser_session import _launch_target
    try:
        launcher_key, kwargs, _shared, own = _launch_target(f"browser-run-{os.getpid()}")
    except RuntimeError as exc:
        return _fail(str(exc))
    if not own:   # e.g. a Bot Desktop screen owns the browser on this host
        return _fail("the own-profile browser is not what this host would launch here; refusing to use another one.")

    def interrupted(signum, _frame):
        raise _Interrupted(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    proc: Optional[subprocess.Popen] = None
    held = False
    try:
        browser, err = launch_chromium(launcher_key, **kwargs)
        if browser is None:
            return _fail(err or "the browser could not be started.")
        held = True
        endpoint = _endpoint(browser.http_url)
        if endpoint is None:
            return _fail("the browser's DevTools endpoint is not a loopback http address; not handing it on.")
        env = {**os.environ, "HERMES_BROWSER_CDP_URL": endpoint[0], "HERMES_BROWSER_CDP_PORT": str(endpoint[1])}
        try:
            proc = subprocess.Popen(command, env=env, start_new_session=True)
        except OSError as exc:
            return _fail(f"could not start {command[0]}: {exc}")
        try:
            return proc.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            _stop_command(proc)
            return _fail(f"the command ran longer than {deadline:g} s and was stopped.", TIMEOUT_EXIT)
    except _Interrupted as exc:
        return 128 + int(exc.args[0])
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    finally:
        for sig in previous:   # no second interruption while cleaning up
            signal.signal(sig, signal.SIG_IGN)
        if proc is not None and proc.poll() is None:
            _stop_command(proc)
        if held:
            stop_browser(launcher_key)   # the last holder stops the browser; any other holder keeps it
        for sig, handler in previous.items():
            signal.signal(sig, handler)

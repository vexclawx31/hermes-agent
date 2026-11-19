"""``hermes browser`` subcommand parser."""

from __future__ import annotations

import argparse
import sys


def build_browser_parser(subparsers) -> None:
    """Attach the ``browser`` subcommand to ``subparsers``."""
    browser_parser = subparsers.add_parser(
        "browser", help="Browser helpers (hold the agent's own browser for a command; close a browser "
                        "locking the real profile)",
        description="run holds the agent's own browser profile (browser.own_profile) open for one "
            "command and gives it the DevTools address. "
            "Helpers for real-profile browsing (browser.use_real_profile): "
            "close-profile terminates the browser process tree holding your "
            "default profile so Hermes can copy it — DESTRUCTIVE (unsaved tabs "
            "in that browser are lost). The agent runs this only after you "
            "approve closing the browser.")
    browser_subparsers = browser_parser.add_subparsers(dest="browser_action")
    browser_close = browser_subparsers.add_parser(
        "close-profile",
        help="Close the browser locking your real profile (asks nothing — "
             "run only with the user's explicit OK; loses unsaved tabs)")
    browser_close.add_argument(
        "--browser",
        help="Override detected default browser (chrome/edge/brave/brave-origin/chromium)")

    browser_run = browser_subparsers.add_parser(
        "run",
        help="Run a command against the agent's own browser profile, holding the browser open meanwhile",
        description="Starts (or attaches to) the browser of browser.own_profile as a browser session would, "
            "runs the command with HERMES_BROWSER_CDP_URL and HERMES_BROWSER_CDP_PORT set, and lets go of "
            "the browser when the command exits (its last holder stops it). The command is run as given, "
            "without a shell; put it after --. Exit status: the command's, 124 when --timeout ended it.")
    browser_run.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                             help="Stop the command after this many seconds (default 600)")
    browser_run.add_argument("command", nargs=argparse.REMAINDER, help="-- <command> [args...]")

    def _dispatch_browser(_args):
        if getattr(_args, "browser_action", None) == "run":
            from hermes_cli.browser_run import DEFAULT_DEADLINE, run
            command = list(_args.command or [])
            if command[:1] == ["--"]:
                command = command[1:]
            return run(command, DEFAULT_DEADLINE if _args.timeout is None else _args.timeout)
        from hermes_cli.browser_connect import (
            UNSUPPORTED_CHANNEL, close_browser_holding_profile, detect_default_chromium,
            real_profile_data_dir)

        action = getattr(_args, "browser_action", None)
        if action != "close-profile":
            browser_parser.print_help()
            return 2
        browser = getattr(_args, "browser", None) or detect_default_chromium()
        if not browser or browser == UNSUPPORTED_CHANNEL:
            print("✗ No supported Chromium default browser detected.", file=sys.stderr)
            return 1
        src = real_profile_data_dir(browser)
        if not src:
            print(f"✗ Could not resolve the {browser} profile directory.", file=sys.stderr)
            return 1
        closed, msg = close_browser_holding_profile(src)
        if closed:
            print(f"✓ {msg}")
            return 0
        print(f"✗ {msg}", file=sys.stderr)
        return 1

    browser_parser.set_defaults(func=_dispatch_browser)

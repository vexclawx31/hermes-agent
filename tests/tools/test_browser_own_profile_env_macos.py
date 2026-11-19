"""``browser.own_profile`` on macOS: the own browser gets the home directory of the OS account Hermes runs as, so it
finds that account's login keychain even when Hermes itself runs with another ``HOME``. Split from
``test_browser_own_profile_launch.py``: these cases set ``sys.platform``, so they run on the macOS lane only."""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.platforms("macos")


def test_the_own_browser_env_uses_this_accounts_home_on_macos_only(monkeypatch):
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

"""External-Chrome (``browser.cdp_url``) sessions open their own tab and close only that tab.

A fake agent-browser layer models one shared Chrome: pre-existing tabs, a person's tab and two
overlapping agent sessions. No real agent-browser, Chromium or network is used.
"""

import asyncio
import itertools

import pytest

import tools.browser_tool as bt
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session
from tools.browser_supervisor import CDPSupervisor

WS_URL = "ws://127.0.0.1:9225/devtools/browser/abc123"


class FakeChrome:
    """Shared tab state plus a log of every agent-browser call, per session name."""

    def __init__(self, tabs=()):
        self.tabs = list(tabs)
        self.calls = []
        self.fail_tab_new = False
        self.legacy_tab_new = False  # agent-browser without targetId/--pin-tab
        self._ids = (f"T{n}" for n in itertools.count(1))

    def dispatch(self, task_id, session_info, browser_cmd, command, args, timeout, engine):
        self.calls.append((session_info["session_name"], command, list(args)))
        if command == "tab" and args[:1] == ["new"]:
            if self.fail_tab_new:
                return "auto", {"success": False, "error": "unknown command"}
            tid = next(self._ids)
            self.tabs.append(tid)
            if self.legacy_tab_new:
                return "auto", {"success": True, "data": {"tabId": tid}}
            return "auto", {"success": True, "data": {"targetId": tid, "tabId": "t9"}}
        if command == "tab" and args[:1] == ["close"]:
            self.tabs.remove(args[1])
            return "auto", {"success": True, "data": {"closed": True}}
        return "auto", {"success": True, "data": {}}

    def tab_calls(self, session_name=None):
        return [c for c in self.calls if c[1] == "tab" and session_name in (None, c[0])]


@pytest.fixture(autouse=True)
def _reset_browser_state():
    def _clear():
        bt._active_sessions.clear()
        bt._session_last_activity.clear()
        bt._last_active_session_key.clear()
        bt._suspect_browser_sessions.clear()

    _clear()
    yield
    _clear()


@pytest.fixture
def chrome(monkeypatch, tmp_path):
    fake = FakeChrome(tabs=["HUMAN-1"])
    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    monkeypatch.setattr(bt_session, "_dispatch_browser_command", fake.dispatch)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: WS_URL)
    monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr("tools.browser_tool_cdp._stop_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    return fake


def _session_name(task_id):
    return bt._active_sessions[task_id]["session_name"]


def test_first_command_opens_one_owned_tab(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    bt_session._run_browser_command("a", "snapshot", [])

    name = _session_name("a")
    assert chrome.tab_calls(name) == [(name, "tab", ["new"])]
    assert [c[1] for c in chrome.calls] == ["tab", "open", "snapshot"]
    assert bt._active_sessions["a"]["owned_target_ids"] == ["T1"]


def test_cleanup_closes_only_owned_tab_and_keeps_preexisting(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    name = _session_name("a")

    bt_lifecycle._cleanup_single_browser_session("a")

    assert chrome.tabs == ["HUMAN-1"]
    assert chrome.calls[-2:] == [(name, "tab", ["close", "T1"]), (name, "close", [])]
    assert "a" not in bt._active_sessions


def test_overlapping_sessions_and_human_tab_survive_other_cleanup(chrome):
    bt_session._run_browser_command("alex", "open", ["https://example.com"])
    bt_session._run_browser_command("calix", "open", ["https://example.org"])
    chrome.tabs.append("HUMAN-2")  # a person opens a tab while both sessions are live

    bt_lifecycle._cleanup_single_browser_session("alex")

    assert chrome.tabs == ["HUMAN-1", "T2", "HUMAN-2"]
    assert bt._active_sessions["calix"]["owned_target_ids"] == ["T2"]

    bt_lifecycle._cleanup_single_browser_session("calix")
    assert chrome.tabs == ["HUMAN-1", "HUMAN-2"]


def test_failed_tab_new_fails_closed_without_retrying(chrome):
    chrome.fail_tab_new = True
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    bt_session._run_browser_command("a", "snapshot", [])
    name = _session_name("a")

    bt_lifecycle._cleanup_single_browser_session("a")

    assert chrome.tab_calls(name) == [(name, "tab", ["new"])]  # tried once, never closed anything
    assert chrome.tabs == ["HUMAN-1"]


def test_non_cdp_session_never_issues_tab_commands(chrome, monkeypatch):
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp", lambda: ("", None))

    bt_session._run_browser_command("a", "open", ["https://example.com"])

    assert chrome.tab_calls() == []
    assert "owned_target_ids" not in bt._active_sessions["a"]


def test_replacement_daemon_rebinds_to_owned_tab_instead_of_opening_another(chrome, tmp_path):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    old = bt._active_sessions["a"]

    bt_session._discard_timed_out_browser_session("a", old, str(tmp_path))
    bt_session._run_browser_command("a", "snapshot", [])
    bt_session._run_browser_command("a", "snapshot", [])

    new_name = _session_name("a")
    assert new_name != old["session_name"]
    assert chrome.tab_calls(new_name) == [(new_name, "tab", ["T1"])]
    assert chrome.tabs == ["HUMAN-1", "T1"]

    bt_lifecycle._cleanup_single_browser_session("a")
    assert chrome.tabs == ["HUMAN-1"]


def test_legacy_agent_browser_undoes_its_tab_and_records_nothing(chrome):
    chrome.legacy_tab_new = True
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    name = _session_name("a")

    assert chrome.tab_calls(name) == [(name, "tab", ["new"]), (name, "tab", ["close", "T1"])]
    assert chrome.tabs == ["HUMAN-1"]
    assert bt._active_sessions["a"]["owned_target_ids"] == []


def test_owned_session_commands_pin_the_tab_but_internal_tab_commands_do_not(monkeypatch):
    argvs = []
    monkeypatch.setattr(bt_session, "_spawn_and_collect",
                        lambda task_id, info, cmd_parts, *a, **k: argvs.append(cmd_parts) or {"success": True})
    monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda _tid: None)
    info = {"session_name": "cdp_x", "cdp_url": WS_URL, "features": {"cdp_override": True}}

    bt_session._dispatch_browser_command("a", info, "agent-browser", "snapshot", [], 10, None)
    info["owned_target_ids"] = ["T1"]
    bt_session._dispatch_browser_command("a", info, "agent-browser", "snapshot", [], 10, None)
    bt_session._dispatch_browser_command("a", info, "agent-browser", "tab", ["T1"], 10, None)

    assert ["--pin-tab" in argv for argv in argvs] == [False, True, False]


class _FakeSupervisorCDP(CDPSupervisor):
    def __init__(self, targets, **kw):
        super().__init__("task", WS_URL, **kw)
        self._targets = targets
        self.attached = None

    async def _cdp(self, method, params=None, *, session_id=None, timeout=10.0):
        if method == "Target.getTargets":
            return {"result": {"targetInfos": self._targets}}
        if method == "Target.attachToTarget":
            self.attached = params["targetId"]
            return {"result": {"sessionId": "S"}}
        raise AssertionError(method)

    async def _enable_page_domains(self, sid, timeout=10.0):
        pass

    async def _install_dialog_bridge(self, sid):
        pass


@pytest.mark.parametrize("page_target_id, expected", [("T1", "T1"), (None, "HUMAN-1"), ("GONE", "HUMAN-1")])
def test_supervisor_attaches_to_owned_tab_when_present(page_target_id, expected):
    targets = [{"type": "page", "targetId": "HUMAN-1"}, {"type": "page", "targetId": "T1"}]
    sup = _FakeSupervisorCDP(targets, page_target_id=page_target_id)

    asyncio.run(sup._attach_initial_page())

    assert sup.attached == expected

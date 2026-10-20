"""External-Chrome (``browser.cdp_url``) sessions open their own tab and close only that tab.

A fake agent-browser layer models one shared Chrome: pre-existing tabs, a person's tab and two
overlapping agent sessions. No real agent-browser, Chromium or network is used.
"""

import asyncio
import itertools
import threading

import pytest

import tools.browser_tool as bt
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session
from tools.browser_supervisor import CDPSupervisor

WS_URL = "ws://127.0.0.1:9225/devtools/browser/abc123"


class FakeChrome:
    """Shared tab state (targetId → url), each daemon's bound tab, and a log of agent-browser calls."""

    def __init__(self, tabs=()):
        self.tabs = {t: "https://person.example/" for t in tabs}
        self.bound = {}  # session_name → targetId the daemon is pinned/bound to
        self.calls = []
        self.tab_new = "ok"  # ok | fail | legacy | timeout
        self._ids = (f"T{n}" for n in itertools.count(1))

    def dispatch(self, task_id, session_info, browser_cmd, command, args, timeout, engine):
        name = session_info["session_name"]
        self.calls.append((name, command, list(args)))
        if command == "tab" and args[:1] == ["new"]:
            if self.tab_new == "fail":
                return "auto", {"success": False, "error": "boom"}
            tid = next(self._ids)
            self.tabs[tid] = args[1] if len(args) > 1 else "about:blank"
            if self.tab_new == "legacy":
                return "auto", {"success": True, "data": {"tabId": tid}}
            if self.tab_new == "timeout":  # Chrome made the tab, the CLI reply never arrived
                bt_session._discard_timed_out_browser_session(task_id, session_info, "")
                return "auto", {"success": False, "error": "timed out"}
            self.bound[name] = tid
            return "auto", {"success": True, "data": {"targetId": tid, "tabId": "t9"}}
        if command == "tab" and args[:1] == ["list"]:
            tabs = [{"targetId": t, "url": u} for t, u in self.tabs.items()]
            return "auto", {"success": True, "data": {"tabs": tabs}}
        if command == "tab" and args[:1] == ["close"]:
            self.tabs.pop(args[1], None)
            return "auto", {"success": True, "data": {"closed": True}}
        if command == "tab":
            if args[0] not in self.tabs:
                return "auto", {"success": False, "error": "no such tab"}
            self.bound[name] = args[0]
            return "auto", {"success": True, "data": {}}
        if command == "close":
            return "auto", {"success": True, "data": {}}
        tid = self.bound.get(name)
        if tid is not None and tid not in self.tabs:
            return "auto", {"success": False, "code": "tab_gone", "error": "tab_gone"}
        return "auto", {"success": True, "data": {"tab": tid}}

    def tab_calls(self, session_name=None):
        return [c for c in self.calls if c[1] == "tab" and session_name in (None, c[0])]

    def page_calls(self):
        return [c for c in self.calls if c[1] not in ("tab", "close")]


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
def supervisor_calls(monkeypatch):
    calls = []
    monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", calls.append)
    return calls


@pytest.fixture
def chrome(monkeypatch, tmp_path, supervisor_calls):
    fake = FakeChrome(tabs=["HUMAN-1"])
    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    monkeypatch.setattr(bt_session, "_dispatch_browser_command", fake.dispatch)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: WS_URL)
    monkeypatch.setattr("tools.browser_tool_cdp._stop_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    return fake


def _session_name(task_id):
    return bt._active_sessions[task_id]["session_name"]


def _owned(task_id):
    return bt._active_sessions[task_id].get("owned_target_ids")


def test_first_command_opens_one_owned_tab_and_points_the_supervisor_at_it(chrome, supervisor_calls):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    bt_session._run_browser_command("a", "snapshot", [])

    name = _session_name("a")
    assert [c[2][0] for c in chrome.tab_calls(name)] == ["new"]
    assert chrome.tab_calls(name)[0][2][1].startswith("about:blank#hermes-tab-")
    assert [c[1] for c in chrome.calls] == ["tab", "open", "snapshot"]
    assert _owned("a") == ["T1"]
    assert chrome.bound[name] == "T1"
    assert supervisor_calls == ["a"]  # only after the tab exists, never at session creation


def test_cleanup_closes_only_owned_tab_and_keeps_preexisting(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    name = _session_name("a")

    bt_lifecycle._cleanup_single_browser_session("a")

    assert list(chrome.tabs) == ["HUMAN-1"]
    assert chrome.calls[-2:] == [(name, "tab", ["close", "T1"]), (name, "close", [])]
    assert "a" not in bt._active_sessions


def test_overlapping_sessions_and_human_tab_survive_other_cleanup(chrome):
    bt_session._run_browser_command("alex", "open", ["https://example.com"])
    bt_session._run_browser_command("calix", "open", ["https://example.org"])
    chrome.tabs["HUMAN-2"] = "https://person.example/2"  # a person opens a tab while both sessions are live

    bt_lifecycle._cleanup_single_browser_session("alex")

    assert list(chrome.tabs) == ["HUMAN-1", "T2", "HUMAN-2"]
    assert _owned("calix") == ["T2"]

    bt_lifecycle._cleanup_single_browser_session("calix")
    assert list(chrome.tabs) == ["HUMAN-1", "HUMAN-2"]


def test_failed_tab_new_refuses_the_command_and_retries_next_time(chrome):
    chrome.tab_new = "fail"
    first = bt_session._run_browser_command("a", "open", ["https://example.com"])

    assert first["success"] is False and "own browser tab" in first["error"]
    assert chrome.page_calls() == []  # never ran on someone else's tab

    chrome.tab_new = "ok"
    assert bt_session._run_browser_command("a", "open", ["https://example.com"])["success"]
    assert _owned("a") == ["T1"]


def test_legacy_agent_browser_undoes_its_tab_and_keeps_old_behavior(chrome):
    chrome.tab_new = "legacy"
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    bt_session._run_browser_command("a", "snapshot", [])
    name = _session_name("a")

    assert [c[2][0] for c in chrome.tab_calls(name)] == ["new", "close"]
    assert list(chrome.tabs) == ["HUMAN-1"]
    assert not _owned("a")
    bt_lifecycle._cleanup_single_browser_session("a")
    assert list(chrome.tabs) == ["HUMAN-1"]


def test_timed_out_tab_new_is_found_by_marker_and_owned_by_the_replacement(chrome):
    chrome.tab_new = "timeout"
    bt_session._run_browser_command("a", "open", ["https://example.com"])

    new_name = _session_name("a")
    assert _owned("a") == ["T1"]
    assert chrome.bound[new_name] == "T1"  # replacement daemon switched to the found tab before the command
    assert chrome.page_calls()[-1][0] == new_name  # the command ran on the replacement, not the dead daemon
    bt_lifecycle._cleanup_single_browser_session("a")
    assert list(chrome.tabs) == ["HUMAN-1"]


def test_owned_tab_closed_by_a_person_is_replaced_and_command_retried(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    del chrome.tabs["T1"]  # a person closes the agent's tab

    result = bt_session._run_browser_command("a", "snapshot", [])

    assert result["success"] and result["data"]["tab"] == "T2"
    assert _owned("a") == ["T1", "T2"]
    bt_lifecycle._cleanup_single_browser_session("a")
    assert list(chrome.tabs) == ["HUMAN-1"]


def test_replacement_daemon_rebinds_to_owned_tab_instead_of_opening_another(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    old = bt._active_sessions["a"]

    bt_session._discard_timed_out_browser_session("a", old, "")
    bt_session._run_browser_command("a", "snapshot", [])
    bt_session._run_browser_command("a", "snapshot", [])

    new_name = _session_name("a")
    assert new_name != old["session_name"]
    assert chrome.tab_calls(new_name) == [(new_name, "tab", ["T1"])]
    bt_lifecycle._cleanup_single_browser_session("a")
    assert list(chrome.tabs) == ["HUMAN-1"]


def test_failed_rebind_opens_a_new_owned_tab_never_a_foreign_one(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    old = bt._active_sessions["a"]
    del chrome.tabs["T1"]
    bt_session._discard_timed_out_browser_session("a", old, "")

    result = bt_session._run_browser_command("a", "snapshot", [])

    assert result["data"]["tab"] == "T2"
    assert _owned("a") == ["T1", "T2"]


def test_non_cdp_session_never_issues_tab_commands(chrome, monkeypatch):
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp", lambda: ("", None))

    bt_session._run_browser_command("a", "open", ["https://example.com"])

    assert chrome.tab_calls() == []
    assert "owned_target_ids" not in bt._active_sessions["a"]


def test_force_reap_closes_owned_tabs_too(chrome):
    bt_session._run_browser_command("a", "open", ["https://example.com"])

    bt_lifecycle._force_reap_browser_session("a")

    assert list(chrome.tabs) == ["HUMAN-1"]


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


@pytest.mark.parametrize("page_target_id, expected", [("T1", "T1"), (None, "HUMAN-1")])
def test_supervisor_attaches_to_owned_tab(page_target_id, expected):
    targets = [{"type": "page", "targetId": "HUMAN-1"}, {"type": "page", "targetId": "T1"}]
    sup = _FakeSupervisorCDP(targets, page_target_id=page_target_id)

    asyncio.run(sup._attach_initial_page())

    assert sup.attached == expected


@pytest.mark.parametrize("targets", [[{"type": "page", "targetId": "HUMAN-1"}], []])
def test_supervisor_refuses_foreign_or_new_tab_when_owned_tab_is_gone(targets):
    sup = _FakeSupervisorCDP(targets, page_target_id="T1")

    with pytest.raises(RuntimeError, match="gone"):
        asyncio.run(sup._attach_initial_page())

    assert sup.attached is None


def test_timed_out_tab_new_without_marker_hit_recovers_on_the_next_command(chrome, monkeypatch):
    orig = chrome.dispatch

    def dispatch(task_id, info, cmd, command, args, timeout, engine):
        if command == "tab" and args[:1] == ["list"] and chrome.tab_new == "timeout":
            return "auto", {"success": False, "error": "timed out"}
        return orig(task_id, info, cmd, command, args, timeout, engine)

    monkeypatch.setattr(bt_session, "_dispatch_browser_command", dispatch)
    chrome.tab_new = "timeout"
    assert bt_session._run_browser_command("a", "open", ["https://example.com"])["success"] is False
    assert chrome.page_calls() == []

    chrome.tab_new = "ok"
    assert bt_session._run_browser_command("a", "open", ["https://example.com"])["success"]


def test_rebind_timeout_never_opens_tabs_or_runs_commands_on_the_discarded_daemon(chrome, monkeypatch):
    bt_session._run_browser_command("a", "open", ["https://example.com"])
    bt_session._discard_timed_out_browser_session("a", bt._active_sessions["a"], "")
    replacement = bt._active_sessions["a"]["session_name"]
    orig = chrome.dispatch

    def dispatch(task_id, info, cmd, command, args, timeout, engine):
        if command == "tab" and args == ["T1"]:  # every rebind times out → record replaced again
            bt_session._discard_timed_out_browser_session(task_id, info, "")
            return "auto", {"success": False, "error": "timed out"}
        return orig(task_id, info, cmd, command, args, timeout, engine)

    monkeypatch.setattr(bt_session, "_dispatch_browser_command", dispatch)
    result = bt_session._run_browser_command("a", "snapshot", [])

    assert result["success"] is False and "retry" in result["error"]
    assert [c for c in chrome.calls if c[0] == replacement and c[2][:1] == ["new"]] == []
    assert [c for c in chrome.page_calls() if c[1] == "snapshot"] == []


def test_supervisor_lookup_without_target_keeps_the_owned_tab_supervisor(monkeypatch):
    from tools.browser_supervisor import _SupervisorRegistry

    class FakeSup:
        def __init__(self, task_id, cdp_url, dialog_policy=None, dialog_timeout_s=None, page_target_id=None):
            self.cdp_url, self.page_target_id = cdp_url, page_target_id
            self._thread = type("T", (), {"is_alive": lambda s: True})()
            self._loop = type("L", (), {"is_running": lambda s: True})()

        def start(self, timeout):
            pass

        def stop(self):
            pass

    monkeypatch.setattr("tools.browser_supervisor.CDPSupervisor", FakeSup)
    reg = _SupervisorRegistry()
    reg.get_or_start("a", "ws://x", page_target_id="T1")

    assert reg.get_or_start("a", "ws://x").page_target_id == "T1"  # vault fill / browser_exec shape
    assert reg.get_or_start("a", "ws://x", page_target_id="T2").page_target_id == "T2"


def test_focus_page_only_considers_the_owned_tab():
    targets = [{"type": "page", "targetId": "HUMAN-1", "url": "https://www.instagram.com/accounts/login/"},
               {"type": "page", "targetId": "T1", "url": "https://www.instagram.com/accounts/login/"}]
    sup = _FakeSupervisorCDP(targets, page_target_id="T1")
    loop = asyncio.new_event_loop()
    sup._loop = loop
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        result = sup.focus_page("https://www.instagram.com")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    assert result["ok"] and sup.attached == "T1"

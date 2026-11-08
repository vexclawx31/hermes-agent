"""Shared external browser (``browser.cdp_url`` / ``/browser connect``: a person's own Chrome) on v0.23's
``browser_exec``: a session works only in tabs it opened and closes only those.

A fake CDP endpoint models one shared Chrome with a person's tabs; fake harness helpers model the
daemon a call drives; a fake supervisor registry records attaches. No real browser, harness, CDP
socket or network is used.
"""

import asyncio
import atexit
import itertools
import json
import stat

import pytest

import tools.browser_use_cli as bu_cli
from tools import browser_owned_tabs as owned_tabs
from tools import browser_session as bs
from tools import browser_session_cdp as bs_cdp
from tools import browser_session_lifecycle as lifecycle
from tools import browser_supervisor
from tools.browser_supervisor import CDPSupervisor, _SupervisorRegistry

WS_URL = "ws://127.0.0.1:9225/devtools/browser/abc123"
PERSON_TABS = ("P1", "P2")
DAEMON = ("rt", "tmp", "_cdp_test")


class FakeChrome:
    """Page targets (targetId -> url) of one shared Chrome and a log of browser-level CDP calls."""

    def __init__(self, tabs=PERSON_TABS):
        self.tabs = {t: "https://person.example/" for t in tabs}
        self.calls = []
        self.closed = []
        self.fail_list = self.fail_create = False
        self.on_create = None
        self._ids = (f"T{n}" for n in itertools.count(1))

    def cdp(self, ws_url, method, params=None, timeout=10.0):
        assert ws_url == WS_URL
        self.calls.append((method, dict(params or {})))
        if method == "Target.getTargets":
            if self.fail_list:
                raise RuntimeError("socket closed")
            return {"targetInfos": [{"targetId": t, "type": "page", "url": u} for t, u in self.tabs.items()]}
        if method == "Target.createTarget":
            if self.fail_create:
                raise RuntimeError("createTarget refused")
            tid = next(self._ids)
            self.tabs[tid] = params.get("url", "about:blank")
            if self.on_create:
                self.on_create(tid)
            return {"targetId": tid}
        if method == "Target.closeTarget":
            self.closed.append(params["targetId"])
            self.tabs.pop(params["targetId"], None)
            return {"success": True}
        raise AssertionError(f"unexpected CDP call {method}")


class FakeSupervisor:
    def __init__(self, task_id, cdp_url, page_target_id=None):
        self.task_id, self.cdp_url, self.page_target_id = task_id, cdp_url, page_target_id


class FakeRegistry:
    def __init__(self):
        self.by_task, self.starts, self.stopped = {}, [], []

    def get_or_start(self, task_id, cdp_url, **kw):
        self.starts.append((task_id, kw.get("page_target_id")))
        sup = self.by_task[task_id] = FakeSupervisor(task_id, cdp_url, kw.get("page_target_id"))
        return sup

    def get(self, task_id):
        return self.by_task.get(task_id)

    def stop(self, task_id):
        self.stopped.append(task_id)
        self.by_task.pop(task_id, None)


@pytest.fixture
def chrome(monkeypatch):
    fake = FakeChrome()
    monkeypatch.setattr(owned_tabs, "_cdp", fake.cdp)
    monkeypatch.setattr(bs, "_active_sessions", {})
    monkeypatch.setattr(bs, "_session_last_activity", {})
    monkeypatch.setattr(bs, "_session_owner_homes", {})
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
    monkeypatch.setattr(bs_cdp, "get_cdp_override", lambda: WS_URL)
    monkeypatch.setattr(bs_cdp, "get_cdp_override_raw", lambda: WS_URL)
    return fake


@pytest.fixture
def registry(monkeypatch):
    reg = FakeRegistry()
    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", reg)
    return reg


@pytest.fixture
def stopped_daemons(monkeypatch):
    stopped = []
    monkeypatch.setattr(bu_cli, "stop_harness_daemon", stopped.append)
    return stopped


def _bind(key="task-a"):
    binding, err = owned_tabs.bind(key, DAEMON)
    assert err is None, err
    return binding


# ── Drives only the owned tab ────────────────────────────────────────────────────────────────────────

def test_first_call_opens_and_pins_its_own_tab(chrome, registry):
    binding = _bind()
    assert binding.pin == "T1" and binding.owned == ("T1",)
    assert binding.pin not in PERSON_TABS
    assert bs._active_sessions["task-a"]["owned_target_ids"] == ["T1"]
    assert ("Target.createTarget", {"url": "about:blank", "background": True}) in chrome.calls
    # The session record was created before its tab existed: no supervisor was started then.
    assert registry.starts == []


def test_later_calls_reuse_the_owned_tab(chrome, registry):
    assert _bind().pin == _bind().pin == "T1"
    assert [m for m, _ in chrome.calls].count("Target.createTarget") == 1


def test_the_tab_a_call_ended_on_is_the_next_pin(chrome, registry, tmp_path):
    binding = _bind()
    chrome.tabs["T7"] = "https://agent.example/"  # opened by this call's new_tab()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"opened": ["T7"], "current": "T7"}), encoding="utf-8")
    owned_tabs.record_call("task-a", binding, str(state))
    assert _bind().pin == "T7"
    assert bs._active_sessions["task-a"]["owned_target_ids"] == ["T1", "T7"]


def test_a_reported_tab_that_existed_before_the_call_is_never_adopted(chrome, registry, tmp_path):
    binding = _bind()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"opened": ["P1"], "current": "P1"}), encoding="utf-8")
    owned_tabs.record_call("task-a", binding, str(state))
    record = bs._active_sessions["task-a"]
    assert record["owned_target_ids"] == ["T1"] and record["current_target_id"] == "T1"


# ── A gone owned tab → a new owned tab or a refusal, never a foreign tab ─────────────────────────────

def test_gone_owned_tab_is_replaced_by_a_new_owned_tab(chrome, registry):
    _bind()
    del chrome.tabs["T1"]  # the person closed the agent's tab
    binding = _bind()
    assert binding.pin == "T2" and binding.pin not in PERSON_TABS
    assert bs._active_sessions["task-a"]["owned_target_ids"] == ["T2"]


def test_refuses_when_no_own_tab_can_be_opened(chrome, registry):
    chrome.fail_create = True
    binding, err = owned_tabs.bind("task-a", DAEMON)
    assert binding is None and "not using another tab" in err


def test_refuses_when_the_tabs_cannot_be_read(chrome, registry):
    _bind()
    chrome.fail_list = True
    binding, err = owned_tabs.bind("task-a", DAEMON)
    assert binding is None and "not using another tab" in err


def test_a_tab_opened_while_the_session_was_torn_down_is_closed_again(chrome, registry):
    chrome.on_create = lambda tid: bs._active_sessions.pop("task-a", None)
    binding, err = owned_tabs.bind("task-a", DAEMON)
    assert binding is None and err
    assert chrome.closed == ["T1"]


# ── Cleanup closes only owned tabs (also on force-reap) ──────────────────────────────────────────────

def _session_with_two_owned_tabs(chrome, tmp_path):
    binding = _bind()
    chrome.tabs["T7"] = "https://agent.example/"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"opened": ["T7", "P2"], "current": "T7"}), encoding="utf-8")
    owned_tabs.record_call("task-a", binding, str(state))


def test_cleanup_closes_only_the_owned_tabs(chrome, registry, stopped_daemons, tmp_path):
    _session_with_two_owned_tabs(chrome, tmp_path)
    lifecycle.cleanup_browser("task-a")
    assert sorted(chrome.closed) == ["T1", "T7"]
    assert set(PERSON_TABS) <= set(chrome.tabs)
    assert stopped_daemons == [DAEMON]  # its daemon closes the dedicated tab it opened itself
    assert "task-a" in registry.stopped and "task-a" not in bs._active_sessions


def test_force_reap_closes_only_the_owned_tabs(chrome, registry, stopped_daemons, tmp_path):
    _session_with_two_owned_tabs(chrome, tmp_path)
    lifecycle._force_reap("task-a")
    assert sorted(chrome.closed) == ["T1", "T7"]
    assert set(PERSON_TABS) <= set(chrome.tabs)
    assert stopped_daemons == [DAEMON]


def test_session_without_owned_tabs_closes_nothing(chrome, registry, stopped_daemons):
    bs.get_session_info("task-a")
    lifecycle.cleanup_browser("task-a")
    assert chrome.closed == [] and stopped_daemons == []


# ── The supervisor never attaches to a foreign tab ───────────────────────────────────────────────────

def test_ensure_cdp_supervisor_waits_for_the_own_tab_then_scopes_to_it(chrome, registry):
    bs_cdp.ensure_cdp_supervisor("task-a")  # e.g. the /browser connect probe: no session tab yet
    assert registry.starts == []
    _bind()
    bs_cdp.ensure_cdp_supervisor("task-a")
    assert registry.starts == [("task-a", "T1")]


def _fake_cdp_supervisor(page_target_id, tabs):
    sup = CDPSupervisor("t", WS_URL, page_target_id=page_target_id)
    calls = []

    async def fake_cdp(method, params=None, **_kw):
        calls.append((method, dict(params or {})))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"targetId": t, "type": "page"} for t in tabs]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "S-" + params["targetId"]}}
        raise AssertionError(method)

    async def noop(*_a, **_kw):
        return None

    sup._cdp = fake_cdp
    sup._enable_page_domains = noop
    sup._install_dialog_bridge = noop
    return sup, calls


def test_supervisor_attaches_only_its_own_tab():
    sup, calls = _fake_cdp_supervisor("T1", ["P1", "T1", "P2"])
    asyncio.run(sup._attach_initial_page())
    assert ("Target.attachToTarget", {"targetId": "T1", "flatten": True}) in calls


def test_supervisor_refuses_when_its_own_tab_is_gone():
    sup, calls = _fake_cdp_supervisor("T1", ["P1", "P2"])
    with pytest.raises(RuntimeError, match="gone"):
        asyncio.run(sup._attach_initial_page())
    assert [m for m, _ in calls] == ["Target.getTargets"]  # no attach to P1, no tab created


def test_registry_keeps_the_owned_tab_supervisor_for_unscoped_callers():
    class _Alive:
        def is_alive(self):
            return True

        def is_running(self):
            return True

    reg = _SupervisorRegistry()
    existing = CDPSupervisor("t", WS_URL, page_target_id="T1")
    existing._thread = existing._loop = _Alive()
    reg._by_task["t"] = existing
    assert reg.get_or_start("t", WS_URL) is existing


def test_vault_fill_supervisor_is_scoped_to_the_own_tab_or_refuses(chrome, registry):
    from tools.browser_vault_tool import _ensure_supervisor

    assert _ensure_supervisor("task-a") is None  # no session tab yet: the fill refuses
    registry.by_task["task-a"] = FakeSupervisor("task-a", WS_URL)  # an unscoped (first-page) one
    assert _ensure_supervisor("task-a") is None
    _bind()
    sup = _ensure_supervisor("task-a")
    assert sup.page_target_id == "T1" and registry.starts == [("task-a", "T1")]
    assert _ensure_supervisor("task-a") is sup


# ── browser_exec end to end (fake harness CLI) ───────────────────────────────────────────────────────

def _fake_cli(tmp_path, monkeypatch, body):
    script = tmp_path / "browser-harness"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(script)])
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.delenv("BU_NAME", raising=False)


_REPORT = ('input=$(cat)\ncase "$input" in *_hermes_owned_tabs*) pre=yes;; *) pre=no;; esac\n'
           'echo "name=$BU_NAME pin=$HERMES_BU_PIN_TAB owned=$HERMES_BU_OWNED_TABS pre=$pre '
           'sentinel=${_HERMES_BU_SHARED_CDP:-unset}"\n')


def test_browser_exec_pins_an_unnamed_session_to_its_own_tab(chrome, registry, tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch, _REPORT)
    out = json.loads(bu_cli.browser_exec("print(1)", task_id="task-a"))["output"]
    assert "pin=T1 owned=T1 pre=yes sentinel=unset" in out
    # never the harness "default" daemon, which attaches to the first existing tab
    assert "name=_cdp_" in out
    assert registry.starts == [("task-a", "T1")]  # the vault/dialog supervisor: own tab only

    del chrome.tabs["T1"]  # the person closed it between calls
    out = json.loads(bu_cli.browser_exec("print(1)", task_id="task-a"))["output"]
    assert "pin=T2 owned=T2" in out and registry.starts[-1] == ("task-a", "T2")


def test_browser_exec_refuses_without_running_the_code(chrome, registry, tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    _fake_cli(tmp_path, monkeypatch, f"cat > /dev/null\ntouch {marker}\n")
    chrome.fail_create = True
    result = json.loads(bu_cli.browser_exec("print(1)", session="r7k2", task_id="task-a"))
    assert "not using another tab" in result["error"]
    assert not marker.exists() and registry.starts == []


def test_browser_exec_adopts_tabs_its_code_opened(chrome, registry, stopped_daemons, tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch,
              'cat > /dev/null\necho \'{"opened": ["N1"], "current": "N1"}\' > "$HERMES_BU_TAB_STATE"\n')
    assert json.loads(bu_cli.browser_exec("new_tab()", task_id="task-a"))["success"] is True
    assert bs._active_sessions["task-a"]["owned_target_ids"] == ["T1", "N1"]
    lifecycle.cleanup_browser("task-a")
    assert sorted(chrome.closed) == ["N1", "T1"] and set(PERSON_TABS) <= set(chrome.tabs)


# ── The pin preamble inside the harness process (fake helpers) ───────────────────────────────────────

class FakeDaemon:
    """The harness helpers over one daemon attached to ``current`` (a person's tab at start)."""

    def __init__(self, current="P1", gone=()):
        self.current, self.gone, self.closed, self._ids = current, set(gone), [], (f"N{n}" for n in itertools.count(1))

    def namespace(self):
        def current_tab():
            return {"targetId": self.current, "target_id": self.current, "url": "", "title": ""}

        def switch_tab(target, activate=False):
            tid = target.get("targetId") if isinstance(target, dict) else target
            if tid in self.gone:
                raise RuntimeError("No target with given id found")
            self.current = tid
            return "S-" + tid

        def new_tab(url="about:blank"):
            self.current = next(self._ids)
            return self.current

        def close_tab(target=None):
            self.closed.append(target)

        return {"current_tab": current_tab, "switch_tab": switch_tab, "new_tab": new_tab,
                "close_tab": close_tab, "ensure_real_tab": lambda: None, "ran": []}


@pytest.fixture
def preamble_env(monkeypatch, tmp_path):
    state = tmp_path / "tab-state.json"
    monkeypatch.setenv("HERMES_BU_PIN_TAB", "T1")
    monkeypatch.setenv("HERMES_BU_OWNED_TABS", "T1")
    monkeypatch.setenv("HERMES_BU_TAB_STATE", str(state))
    exits = []
    monkeypatch.setattr(atexit, "register", exits.append)
    return state, exits


def _exec(daemon, code):
    ns = daemon.namespace()
    exec(bu_cli._OWNED_TAB_PREAMBLE + code, ns)
    return ns


def test_preamble_pins_the_call_to_the_owned_tab_before_the_model_code(preamble_env):
    daemon = FakeDaemon(current="P1")
    ns = _exec(daemon, "ran.append(current_tab()['targetId'])\n")
    assert ns["ran"] == ["T1"]


def test_preamble_refuses_foreign_tabs_and_records_opened_ones(preamble_env):
    state, exits = preamble_env
    daemon = FakeDaemon(current="T1")
    ns = _exec(daemon, "tid = new_tab('https://a.example/')\n")
    assert json.loads(state.read_text(encoding="utf-8")) == {"opened": ["N1"], "current": "N1"}
    with pytest.raises(RuntimeError, match="only switch to tabs it opened"):
        ns["switch_tab"]("P1")
    with pytest.raises(RuntimeError, match="only close tabs it opened"):
        ns["close_tab"]({"targetId": "P2"})
    assert daemon.current == "N1" and daemon.closed == []
    ns["switch_tab"]("T1")
    ns["close_tab"]("N1")
    assert daemon.closed == ["N1"]
    assert ns["ensure_real_tab"]()["targetId"] == "T1"  # never falls back to the person's first tab
    assert exits  # the final tab is reported when the call ends


def test_preamble_exits_before_the_model_code_when_it_cannot_pin(preamble_env):
    daemon = FakeDaemon(current="P1", gone={"T1"})
    ns = daemon.namespace()
    with pytest.raises(SystemExit) as exc:
        exec(bu_cli._OWNED_TAB_PREAMBLE + "ran.append('model code')\n", ns)
    assert exc.value.code == 3 and ns["ran"] == [] and daemon.current == "P1"

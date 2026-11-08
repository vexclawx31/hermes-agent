"""Fork (launch adapter), live: a real ``gateway.run`` and real ``hermes chat`` clients, loopback model.

The spike's probe (v0.23 go/no-go, "Acceptance criteria") as durable tests. One gateway per module on a
scratch HOME/HERMES_HOME, upstream's recording fake provider on 127.0.0.1 and two stdio MCP servers (the
MCP suite's wire-logging fixture server): ``meta`` opts in to ``session_meta``, ``plain`` does not. Every
client runs with the exact flag set of the installed helper it stands for (tenant intake, triage, helpdesk
ask and request, vex-ask) and is a subprocess, so what is asserted is what the gateway did: the model's
request bodies, the MCP server's inbound wire, ``state.db`` and the client's stream-json.

Each client's query carries a ``[scn:<name>]`` tag. The provider answers by tag and by how many tool
results the current turn already has, so concurrent clients never consume each other's script, and
independent flows run at once (a bypass session's managed worker takes many seconds to start).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from agent.conversation_loop import RUN_BUDGET_WRAPUP_NOTICE
from tests.fakes.fake_llm_provider import MODEL_ID, FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.gateway.fixtures.local_recovery_probe import child_env, seed_offline_models_dev
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")  # AF_UNIX control channel, POSIX signals to the client

ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER = ROOT / "tests/e2e/core/mcp_plugins/mcp_fixture_server.py"
TURN_TIMEOUT = 150
SKILL_MARKER = "TENANT-DESK-LIVE-MARKER"
SOUL_MARKER = "SOUL-LIVE-MARKER"
RULES_MARKER = "AGENTS-LIVE-MARKER"
AUTHOR = "Live Asker LA7"
TAG = re.compile(r"\[scn:([A-Za-z0-9_.-]+)\]")
META_KEY = "hermes/session_source"
SRC_ECHO = "echo SRC=[$HERMES_SESSION_SOURCE]"
LEAK = "echo $((6*7))-LEAKED"  # prints 42-LEAKED only if it ran
DISCOVERY = ["tool_call", "tool_describe", "tool_search"]


class Mcp:
    """A scripted MCP call: direct when the tool is offered, else through the ``tool_call`` bridge."""

    def __init__(self, name, args=None):
        self.name, self.args = name, args or {}

    def resolve(self, body):
        offered = {t["function"]["name"] for t in body.get("tools") or []}
        if self.name in offered or "tool_call" not in offered:
            return ToolCall(self.name, self.args)
        return ToolCall("tool_call", {"calls": [{"name": self.name, "arguments": self.args}]})


def _scenario(body):
    """(tag, tool results so far this turn) for a main request; the turn starts at the last tagged user
    message, so a resumed session's earlier turns do not count."""
    messages = body.get("messages") or []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user":
            found = TAG.search(json.dumps(message.get("content")))
            if found:
                done = sum(1 for later in messages[index + 1:] if later.get("role") == "tool")
                return found.group(1), done
    return None, 0


class Live:
    """The module's gateway, provider and helpers."""

    def __init__(self, base: Path):
        self.base = base
        self.home, self.user, self.caller, self.tmp = (base / "state", base / "user", base / "caller", base / "tmp")
        self.home.mkdir(mode=0o700)
        for path in (self.user, self.caller, self.tmp):
            path.mkdir()
        self.scripts: dict[str, list] = {}
        self.lock = threading.Lock()
        self.model = FakeLLMServer(self._respond)
        self.mcp_logs = {"meta": base / "mcp-meta.jsonl", "plain": base / "mcp-plain.jsonl"}
        self.proc = None
        self.log = (base / "gateway.log").open("a", encoding="utf-8")

    # provider
    def _respond(self, record):
        body = record["body"]
        tag, done = _scenario(body)
        with self.lock:
            steps = self.scripts.get(tag) or []
        if done >= len(steps):
            return Text(f"done:{tag}")
        step = steps[done]
        return step.resolve(body) if isinstance(step, Mcp) else step

    def script(self, tag, *steps):
        with self.lock:
            assert tag not in self.scripts, tag
            self.scripts[tag] = list(steps)

    def calls(self, tag):
        return [body for body in self.model.main_requests() if _scenario(body)[0] == tag]

    def tool_outputs(self, tag):
        """Every tool result the model was sent back in this scenario's last request, in order."""
        last = self.calls(tag)[-1]
        return [json.dumps(m.get("content"), ensure_ascii=False)
                for m in last.get("messages") or [] if m.get("role") == "tool"]

    # environment
    def env(self, **extra):
        env = child_env()
        env.update(HOME=str(self.user), USERPROFILE=str(self.user), HERMES_HOME=str(self.home),
                   PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   TMPDIR=str(self.tmp), OPENAI_API_KEY="loopback-only")
        env.update(extra)
        return env

    def configure(self):
        def mcp(name, **entry):
            return {"command": sys.executable, "args": [str(MCP_SERVER)], "connect_timeout": 60, "timeout": 60,
                    "env": {"MCPE2E_LOG": str(self.mcp_logs[name]), "MCPE2E_NAME": name,
                            "MCPE2E_CANARY": f"canary-{name}", "PYTHONPATH": str(ROOT)}, **entry}
        extra = {
            "gateway": {"multiplex_profiles": False},
            "auxiliary": {"title_generation": {"enabled": False}},
            "platform_toolsets": {"cli": ["terminal", "file", "web", "memory", "todo", "session_search", "skills",
                                          "mcp-meta", "mcp-plain"]},
            "mcp_servers": {"meta": mcp("meta", session_meta=True), "plain": mcp("plain")},
            "mcp_discovery_timeout": 120,
            "mcp_single_query_discovery_timeout": 120,
            "updates": {"check": False},
        }
        write_hermes_home(self.home, self.model.base_url, api_key="loopback-only",
                          extra_config="".join(f"{key}: {json.dumps(value)}\n" for key, value in extra.items()))
        (self.home / "SOUL.md").write_text(f"You are the live adapter profile. {SOUL_MARKER}\n")
        (self.caller / "AGENTS.md").write_text(f"Project rules. {RULES_MARKER}\n")
        skill = self.home / "skills" / "agent-ops" / "tenant-desk"
        skill.mkdir(parents=True)
        skill.joinpath("SKILL.md").write_text(
            f"---\nname: tenant-desk\ndescription: live tenant desk\n---\n\n{SKILL_MARKER}\n")
        seed_offline_models_dev(self.home)

    # gateway
    def start(self):
        self.proc = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=ROOT, env=self.env(),
                                     stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)

        def ready():
            assert self.proc.poll() is None, self.gateway_log()
            try:
                return control(self.home, "identify").get("state") == "ready"
            except (OSError, ValueError, AssertionError):
                return False
        wait_until(ready, timeout=180, interval=0.1, what="gateway ready")

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)

    def gateway_log(self):
        self.log.flush()
        return (self.base / "gateway.log").read_text(errors="replace")[-4000:]

    # clients
    def chat(self, tag, flags, **env):
        query = self.caller / f"{tag}.md"
        query.write_text(f"[scn:{tag}] hello from the live adapter test\n")
        argv = [sys.executable, "-m", "hermes_cli.main", "chat", "-Q", "--query-file", str(query), *flags]
        return Client(argv, self.caller, self.env(**env))

    def state(self, session, *extra):
        """``hermes sessions state <session> [extra]`` as the JSON it prints."""
        done = subprocess.run([sys.executable, "-m", "hermes_cli.main", "sessions", "state", session, *extra],
                              cwd=self.caller, env=self.env(), stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=90, check=False)
        assert done.returncode == 0, (done.stdout, done.stderr)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def db(self, sql, args=()):
        with sqlite3.connect(f"file:{self.home / 'state.db'}?mode=ro", uri=True) as db:
            return db.execute(sql, args).fetchall()

    def admissions(self, run_id):
        """(status, outcome) of every admission a spawner's run id named (its first submit)."""
        return self.db("SELECT status, outcome FROM session_admissions WHERE request_id = ?", (f"{run_id}-submit",))

    def settled(self, run_id):
        rows = self.admissions(run_id)
        return rows if rows and all(status == "terminal" for status, _ in rows) else None

    def answers(self, tag):
        """How many stored assistant messages carry this scenario's final answer."""
        return self.db("SELECT count(*) FROM messages WHERE role = 'assistant' AND content LIKE ?",
                       (f"%done:{tag}%",))[0][0]

    def mcp_calls(self, name):
        path = self.mcp_logs[name]
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [row for row in rows if (row.get("msg") or {}).get("method") == "tools/call"]

    # flag sets of the installed helpers (go-no-go "hermes chat callers")
    def bypass(self, toolsets):
        return ["--ignore-user-config", "--ignore-rules", "-m", MODEL_ID, "--provider", "custom",
                "--base-url", self.model.base_url, "-t", toolsets]

    def tenant(self, budget="480"):
        return [*self.bypass("terminal,file,web"), "-s", "tenant-desk", "--max-turns", "40",
                "--source", "switchboard-tenant", "--run-budget", budget, "--format", "stream-json"]

    def triage(self):
        return [*self.bypass("todo"), "--max-turns", "3", "--source", "switchboard-triage", "--format", "stream-json"]

    @staticmethod
    def request(title):
        return ["-c", title, "--create-if-missing", "--source", "helpdesk-request", "--run-budget", "480",
                "--format", "stream-json"]

    @staticmethod
    def ask(title):
        return ["-c", title, "--create-if-missing", "--source", "helpdesk-ask", "-t", "session_search",
                "--run-budget", "300", "--format", "stream-json"]

    @staticmethod
    def vex_ask(title):
        return ["-c", title, "--create-if-missing", "--source", "claude-handoff", "-s", "tenant-desk",
                "--run-budget", "480"]


class Client:
    """One ``hermes chat`` subprocess; stdout/stderr are collected line by line as they arrive."""

    def __init__(self, argv, cwd, env):
        self.argv, self.cwd, self.env = argv, cwd, env
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        self.out: list[str] = []
        self.err: list[str] = []
        self._readers = [threading.Thread(target=self._pump, args=(stream, lines), daemon=True)
                         for stream, lines in ((self.proc.stdout, self.out), (self.proc.stderr, self.err))]
        for reader in self._readers:
            reader.start()

    @staticmethod
    def _pump(stream, lines):
        for line in stream:
            lines.append(line.rstrip("\n"))

    def wait(self, timeout=TURN_TIMEOUT):
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise AssertionError(("client did not finish", self.argv[5:], self.out[-10:], self.err[-10:])) from None
        for reader in self._readers:
            reader.join(timeout=10)
        return self

    def again(self):
        """The same run once more: the same argv and env (a spawner's retry), started."""
        return Client(self.argv, self.cwd, self.env)

    @property
    def rc(self):
        return self.proc.returncode

    def events(self, kind=None):
        parsed = []
        for line in self.out:
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if kind is None or event.get("type") == kind:
                    parsed.append(event)
        return parsed

    def result(self):
        results = self.events("result")
        assert results, self.detail()
        return results[-1]

    def session_id(self):
        for event in self.events():
            if event.get("session_id"):
                return event["session_id"]
        for line in self.err:
            if line.startswith("Session: "):
                return line.split("Session: ", 1)[1].strip()
        return None

    def detail(self):
        return {"rc": self.rc, "argv": self.argv[5:9], "out": self.out[-12:], "err": self.err[-12:]}


def _tool_names(body):
    return sorted(t["function"]["name"] for t in body.get("tools") or [])


def _finish(clients, timeout=TURN_TIMEOUT):
    for client in clients.values():
        client.wait(timeout)
    return clients


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    live = Live(tmp_path_factory.mktemp("launch-adapter"))
    live.model.start()
    try:
        live.configure()
        live.start()
        yield live
    finally:
        live.stop()
        live.model.stop()
        live.log.close()


@pytest.fixture(scope="module")
def flows(live):
    """The independent flows of the scope, preload, attribution and budget checks, run at once."""
    live.script("tenant", ToolCall("terminal", {"command": SRC_ECHO}))
    live.script("triage",
                ToolCall("tool_call", {"calls": [{"name": "terminal", "arguments": {"command": LEAK}}]}),
                ToolCall("tool_call", {"calls": [{"name": "todo_list", "arguments": {}}]}))
    live.script("ask", ToolCall("tool_call", {"calls": [{"name": "session_search", "arguments": {"query": "x"}}]}))
    slow = [ToolCall("terminal", {"command": "sleep 3; echo step"}) for _ in range(2)]
    for tag in ("budget-worker", "budget-classic", "budget-long"):
        live.script(tag, *slow)
    author = {"id": "U-live", "name": AUTHOR, "verified": True}
    return _finish({
        "tenant": live.chat("tenant", live.tenant(), HERMES_TURN_AUTHOR=json.dumps(author)),
        # Control: --safe-mode with -s stays refused (the bypass allows -s; safe mode does not).
        "safe": live.chat("safe", ["--safe-mode", "-s", "tenant-desk", "--format", "stream-json"]),
        "triage": live.chat("triage", live.triage()),
        "ask": live.chat("ask", live.ask("ask:live-scope")),
        "budget-worker": live.chat("budget-worker", live.tenant(budget="5")),
        "budget-classic": live.chat("budget-classic", ["--source", "claude-handoff", "--run-budget", "5"]),
        "budget-long": live.chat("budget-long", live.tenant()),
    })


def _turn_authors(node):
    if isinstance(node, dict):
        found = [node["turn_author"]] if "turn_author" in node else []
        return found + [a for value in node.values() for a in _turn_authors(value)]
    if isinstance(node, list):
        return [a for value in node for a in _turn_authors(value)]
    return []


def test_tenant_intake_gets_only_its_toolsets_its_skill_and_its_label(live, flows):
    """Tenant intake (``--ignore-user-config --ignore-rules -t terminal,file,web -s tenant-desk``): exactly
    those toolsets (plus v0.23's discovery tools), no SOUL/AGENTS/MCP, the skill preloaded and reported as a
    ``system/skills`` event, the ``origin`` label in the tools' env, and the turn author stored unverified."""
    tenant, safe = flows["tenant"], flows["safe"]
    assert tenant.rc == 0, tenant.detail()
    assert tenant.result()["text"] == "done:tenant"
    assert safe.rc != 0 and not live.calls("safe"), safe.detail()

    first = live.calls("tenant")[0]
    assert _tool_names(first) == sorted(["patch", "read_file", "search_files", "terminal", "web_extract",
                                         "web_search", "write_file", *DISCOVERY])
    prompt = json.dumps(first.get("messages"))
    assert SKILL_MARKER in prompt
    assert SOUL_MARKER not in prompt and RULES_MARKER not in prompt
    assert AUTHOR not in prompt  # turn data, not prompt text
    skills = [e for e in tenant.events("system") if e.get("subtype") == "skills"]
    assert [(e["loaded"], e["missing"]) for e in skills] == [(["tenant-desk"], [])], tenant.out
    assert skills[0]["session_id"] == tenant.result()["session_id"]
    assert "SRC=[switchboard-tenant]" in live.tool_outputs("tenant")[0]

    stored = [json.loads(p) for (p,) in live.db(
        "SELECT payload_json FROM session_admissions WHERE payload_json LIKE ?", (f"%{AUTHOR}%",))]
    authors = [a for payload in stored for a in _turn_authors(payload)]
    assert len(stored) == 1 and len(authors) == 1, stored
    assert authors[0]["id"] == "U-live" and authors[0]["name"] == AUTHOR, authors
    assert "verified" not in authors[0], authors  # the client sent verified: true; it is never stored


def test_restricted_toolsets_hold_through_tool_call(live, flows):
    """Triage (``-t todo``) and a helpdesk ask (``-t session_search``): v0.23 puts the toolset behind
    ``tool_search``/``tool_call``; the bridge refuses an out-of-scope tool without running it, and an
    in-scope call through the bridge works."""
    triage, ask = flows["triage"], flows["ask"]
    assert triage.rc == 0 and ask.rc == 0, (triage.detail(), ask.detail())
    assert _tool_names(live.calls("triage")[0]) == DISCOVERY
    assert _tool_names(live.calls("ask")[0]) == DISCOVERY
    refused, in_scope = live.tool_outputs("triage")
    assert "'terminal' is not available" in refused and "42-LEAKED" not in refused, refused
    assert "not available" not in in_scope and "not a known tool" not in in_scope, in_scope
    (searched,) = live.tool_outputs("ask")
    assert '\\"success\\": true' in searched, searched


def test_run_budget_is_per_session_on_both_paths(live, flows):
    """``--run-budget 5`` is a creation field the gateway freezes and enforces with no profile change: the
    80% wrap-up notice reaches the model on the worker path (tenant flags) and the in-process path (classic
    ``-Q``), and not in a concurrent session with a 480 s budget."""
    for tag in ("budget-worker", "budget-classic", "budget-long"):
        assert flows[tag].rc == 0, flows[tag].detail()

    def noticed(tag):
        return any(RUN_BUDGET_WRAPUP_NOTICE in json.dumps(body.get("messages"), ensure_ascii=False)
                   for body in live.calls(tag))
    assert noticed("budget-worker") and noticed("budget-classic")
    assert not noticed("budget-long")


def test_origin_is_frozen_through_resume_and_restart_and_reaches_tools_and_opted_in_mcp(live):
    """A helpdesk request's ``--source helpdesk-request`` is the session's typed ``origin``: the tools' env
    and an opted-in MCP server's ``_meta`` carry it, per call (one shared server process tells a concurrent
    vex-ask's ``claude-handoff`` apart), on resume and after a gateway restart. A server that did not opt in
    gets no session meta."""
    steps = (ToolCall("terminal", {"command": SRC_ECHO}), Mcp("mcp__meta__noargs_probe"),
             Mcp("mcp__plain__noargs_probe"))
    for tag in ("prov-1", "prov-resume", "prov-restart"):
        live.script(tag, *steps)
    live.script("prov-handoff", Mcp("mcp__meta__noargs_probe"))

    def meta_sources():
        return [((row["msg"].get("params") or {}).get("_meta") or {}).get(META_KEY) for row in live.mcp_calls("meta")]

    def check(tag, client):
        assert client.rc == 0, client.detail()
        outputs = live.tool_outputs(tag)
        assert "SRC=[helpdesk-request]" in outputs[0], outputs
        assert "canary-meta" in outputs[1] and "canary-plain" in outputs[2], outputs
        return client.result()["session_id"]

    first = _finish({"request": live.chat("prov-1", live.request("request:live-prov")),
                     "handoff": live.chat("prov-handoff", live.vex_ask("claude:live-prov"))})
    sid = check("prov-1", first["request"])
    assert first["handoff"].rc == 0, first["handoff"].detail()
    assert sorted(meta_sources()) == ["claude-handoff", "helpdesk-request"], live.mcp_calls("meta")
    assert len({row["pid"] for row in live.mcp_calls("meta")}) == 1  # one shared server: per call, not ancestry
    plain = live.mcp_calls("plain")
    assert len(plain) == 1 and META_KEY not in json.dumps(plain[0]["msg"]), plain

    assert check("prov-resume", live.chat("prov-resume", live.request("request:live-prov")).wait()) == sid
    live.stop()
    live.start()
    assert check("prov-restart", live.chat("prov-restart", live.request("request:live-prov")).wait()) == sid
    assert meta_sources()[2:] == ["helpdesk-request", "helpdesk-request"], live.mcp_calls("meta")
    assert all(META_KEY not in json.dumps(row["msg"]) for row in live.mcp_calls("plain")), live.mcp_calls("plain")
    assert live.db("SELECT id FROM sessions WHERE title = 'request:live-prov'") == [(sid,)]


def test_lifetime_detach_retry_interrupt_and_sigterm(live):
    """A spawner's run (``HERMES_CHAT_RUN_ID``) on both paths (classic titled request = in-process, tenant
    intake = managed worker):

    - client SIGKILL mid-turn only detaches: the turn runs on to its end exactly once, and a retry with the
      same run id gets the same session and admission back with the result, without a model call;
    - a spawner that times out asks the owner: the killed client's session still reports running, and
      ``hermes sessions state --interrupt --wait`` stops it before its next model call (a retry then
      returns interrupted, 130, without running); an attached client ends when the owner interrupts;
    - SIGTERM to the client (a spawner's hard limit) stops the turn on the gateway, also for an untitled
      session, and the client exits 130 with an interrupted result.
    """
    flags = {"life-classic": live.request("request:live-life"), "life-worker": live.tenant(),
             "stop-classic": live.request("request:live-stop"), "stop-worker": live.tenant(), "term": live.tenant()}
    for tag in flags:
        # The detached turns must outlast the kill; the stopped ones would run far past every deadline below.
        live.script(tag, ToolCall("terminal", {"command": "sleep 12; echo slow" if tag.startswith("life")
                                               else "sleep 90; echo late"}))
    clients = {tag: live.chat(tag, argv, HERMES_CHAT_RUN_ID=f"run-{tag}") for tag, argv in flags.items()}
    actions = {"life-classic": signal.SIGKILL, "life-worker": signal.SIGKILL, "stop-worker": signal.SIGKILL,
               "term": signal.SIGTERM, "stop-classic": None}
    pending, signalled = dict(actions), {}

    def act_on_tool_start():
        """Signal each client the moment its stream reports the tool running on the gateway."""
        for tag in list(pending):
            client = clients[tag]
            if client.events("tool_use"):
                if pending[tag] is not None:
                    client.proc.send_signal(pending[tag])
                signalled[tag] = time.monotonic()
                del pending[tag]
            elif client.proc.poll() is not None:
                raise AssertionError((tag, client.detail()))
        return not pending
    wait_until(act_on_tool_start, timeout=TURN_TIMEOUT, interval=0.05, what="every tool running")
    sid = {tag: clients[tag].session_id() for tag in ("life-worker", "stop-worker", "term")}
    sid.update({"life-classic": "request:live-life", "stop-classic": "request:live-stop"})
    for tag in ("life-classic", "life-worker", "stop-worker"):
        clients[tag].wait(30)

    # Timeout on the worker path: still running after the kill; the owner stops it on request.
    running = live.state(sid["stop-worker"])
    assert running["running"] and [p["status"] for p in running["pending"]] == ["started"], running
    stopped = live.state(sid["stop-worker"], "--interrupt", "--wait", "30")
    assert stopped["interrupted"] and not stopped["running"], stopped
    # In process, with the client still attached: the owner's interrupt ends that client too.
    stopped = live.state(sid["stop-classic"], "--interrupt", "--wait", "30")
    assert stopped["interrupted"] and not stopped["running"], stopped
    clients["stop-classic"].wait(60)
    # SIGTERM: the client itself stopped the turn on the gateway and reports it.
    term = clients["term"].wait(45)
    assert term.rc == 130 and term.result()["exit_code"] == 130, term.detail()
    assert not live.state(sid["term"])["running"]
    for tag in ("stop-worker", "stop-classic", "term"):
        assert live.settled(f"run-{tag}") == [("terminal", "interrupted")], (tag, live.admissions(f"run-{tag}"))
        assert time.monotonic() - signalled[tag] < 80, tag  # well before its 90 s tool would have ended

    # Detached turns run on to their end, once.
    for tag in ("life-classic", "life-worker"):
        assert wait_until(lambda tag=tag: live.settled(f"run-{tag}"), timeout=TURN_TIMEOUT,
                          what=f"{tag} settled") == [("terminal", "completed")], tag
    retries = _finish({tag: clients[tag].again() for tag in ("life-classic", "life-worker", "stop-worker")})
    for tag in ("life-classic", "life-worker"):
        retry = retries[tag]
        assert retry.rc == 0 and retry.result()["text"] == f"done:{tag}", retry.detail()
        assert retry.result()["session_id"] == live.state(sid[tag])["session_id"]
    retry = retries["stop-worker"]
    assert retry.rc == 130 and retry.result()["exit_code"] == 130, retry.detail()

    for tag in flags:
        done = tag.startswith("life")
        assert len(live.calls(tag)) == (2 if done else 1), (tag, len(live.calls(tag)))
        assert live.answers(tag) == (1 if done else 0), tag
        assert len(live.admissions(f"run-{tag}")) == 1, (tag, live.admissions(f"run-{tag}"))
        state = live.state(sid[tag])
        assert not state["running"] and not state["pending"], (tag, state)
    assert live.proc.poll() is None  # the gateway itself is untouched by its clients' signals

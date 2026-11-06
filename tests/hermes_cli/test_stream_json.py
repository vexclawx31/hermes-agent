"""``hermes chat -q … --format stream-json`` emits a parseable JSONL event stream and nothing else on stdout."""

import json
import signal

import pytest

from hermes_cli.stream_json import StreamJsonEmitter


def _events(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line]


def test_emitter_event_stream_is_valid_jsonl(capsys):
    emitter = StreamJsonEmitter(model="test-model", session_id="s-1")
    emitter.on_text_delta("hel")
    emitter.on_text_delta("\n  ")  # whitespace deltas are part of the answer and must be forwarded verbatim
    emitter.on_text_delta("lo")
    emitter.on_text_delta(None)  # the turn-end sentinel the agent sends
    emitter.on_text_delta("")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "x"}, tool_call_id="call-a")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "y"}, tool_call_id="call-b")
    emitter.on_tool_progress("reasoning.available", "_thinking", "hmm", None)  # not part of the protocol
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-b", result="y")
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-a", duration=0.5,
                             is_error=False, result="x" * 6000)
    code = emitter.emit_result({"final_response": "", "failed": True, "error": "boom", "input_tokens": 3}, exit_code=0)

    events = _events(capsys)
    assert [e["type"] for e in events] == ["system", "text", "text", "text", "tool_use", "tool_use", "tool_result",
                                           "tool_result", "result"]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "hel\n  lo"
    assert events[0]["subtype"] == "init" and events[0]["model"] == "test-model"
    assert events[4]["input"] == {"path": "x"} and events[4]["tool_call_id"] == "call-a"
    # concurrent same-name calls: each result pairs with its own start, not the last-started one
    assert [e["tool_call_id"] for e in events if e["type"] == "tool_result"] == ["call-b", "call-a"]
    assert events[6]["duration_ms"] < 500
    assert events[7]["duration_ms"] == 500 and events[7]["output"].endswith("...") and len(events[7]["output"]) == 5003
    assert code == 1 and events[-1] == {**events[-1], "exit_code": 1, "error": "boom", "session_id": "s-1"}
    assert events[-1]["tokens"]["input"] == 3
    assert all("timestamp" in e for e in events)


def _run_stream_json_chat(monkeypatch, capsys, run_conversation, credentials_ok=True, launch_skills=None,
                          preload_result=None):
    """parser → cmd_chat → cli.main → quiet single-query path with a deterministic fake agent."""
    import cli
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    real_finalize = cli.HermesCLI.__dict__["finalize_preloaded_skills"]

    class FakeAgent:
        model = "test-model"
        session_id = "session-123"

        def run_conversation(self, **_kwargs):
            return run_conversation(self)

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.session_id = "session-123"
            self.conversation_history = []
            self.agent = None
            self._active_agent_route_signature = None
            self.tool_progress_mode = None
            if launch_skills is not None:
                self.launch_skills = launch_skills

        def _claim_active_session(self, *_a, **_k):
            return True

        def _ensure_runtime_credentials(self):
            return credentials_ok

        def _resolve_turn_agent_config(self, _query):
            return {"signature": "r", "model": None, "runtime": None, "request_overrides": None}

        def _init_agent(self, **_kwargs):
            if preload_result is not None:
                # The real finalize, as HermesCLI._init_agent runs it before building the agent.
                self.system_prompt = ""
                self._preload_skills_thread = _FinishedThread()
                self._preload_skills_result = preload_result
                real_finalize(self)
            self.agent = FakeAgent()
            return True

        def chat(self, _query, images=None):
            print("human output")  # must never reach stdout under stream-json

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_entry, "_resolve_use_tui", lambda _args: pytest.fail("TUI resolution consulted"))
    monkeypatch.setattr(cli_entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(cli_entry, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(cli_entry, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(cli_entry, "_confirm_startup_expensive_model_override", lambda _a: None)
    monkeypatch.setattr(cli_entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr("hermes_cli.quiet_single_query.continue_quiet_notify_completions", lambda *_a, **_k: None)

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "-q", "hello", "--format", "stream-json"])
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(args)
    return exc_info.value.code, _events(capsys)


class _FinishedThread:
    """A preload thread that has already finished."""

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


def _ok_turn(agent):
    agent.stream_delta_callback("hello")
    agent.tool_progress_callback("tool.started", "read_file", "p", {"path": "f"})
    agent.tool_progress_callback("tool.completed", "read_file", None, None, duration=0.01, result="contents")
    return {"final_response": "hello", "failed": False}


def _interrupted_turn(_agent):
    raise KeyboardInterrupt


@pytest.mark.parametrize("turn, credentials_ok, exit_code, types", [
    (_ok_turn, True, 0, ["system", "system", "text", "tool_use", "tool_result", "result"]),
    (_interrupted_turn, True, 130, ["system", "system", "result"]),
    (_ok_turn, False, 1, ["system", "result"]),  # credentials fail before the agent exists: no skills event
])
def test_chat_stream_json_implies_quiet_and_closes_with_result(monkeypatch, capsys, turn, credentials_ok, exit_code,
                                                                types):
    """No ``-Q`` needed; stdout is only JSONL; the stream always ends in a ``result`` carrying the exit code."""
    code, events = _run_stream_json_chat(monkeypatch, capsys, turn, credentials_ok=credentials_ok)
    assert code == exit_code
    assert [e["type"] for e in events] == types
    assert events[-1]["exit_code"] == exit_code and events[-1]["session_id"] == "session-123"


def test_chat_stream_json_reports_loaded_and_missing_skills(monkeypatch, capsys):
    """``system/skills`` follows ``init`` once the agent exists and names what `-s` really loaded."""
    _code, events = _run_stream_json_chat(
        monkeypatch, capsys, _ok_turn,
        launch_skills={"loaded": ["switchboard-intake"], "missing": ["typo-skill"]})
    assert [(e["type"], e.get("subtype")) for e in events[:2]] == [("system", "init"), ("system", "skills")]
    skills = events[1]
    assert skills["loaded"] == ["switchboard-intake"] and skills["missing"] == ["typo-skill"]
    assert skills["session_id"] == "session-123"
    assert [e for e in events if e.get("subtype") == "skills"] == [skills]


def test_chat_stream_json_real_finalize_partial_miss(monkeypatch, capsys):
    """Through the real finalize_preloaded_skills: the event reflects the preload result."""
    code, events = _run_stream_json_chat(
        monkeypatch, capsys, _ok_turn, preload_result=("skill prompt", ["tenant-desk"], ["typo-skill"]))
    assert code == 0
    assert events[1]["subtype"] == "skills"
    assert events[1]["loaded"] == ["tenant-desk"] and events[1]["missing"] == ["typo-skill"]


def test_chat_stream_json_all_skills_missing_closes_protocol(monkeypatch, capsys):
    """Every -s skill unknown: stdout stays JSONL-only, init → skills (nothing loaded) → failing result."""
    code, events = _run_stream_json_chat(
        monkeypatch, capsys, _ok_turn, preload_result=("", [], ["switchboard-intake"]))
    assert code == 1
    assert [(e["type"], e.get("subtype")) for e in events] == [("system", "init"), ("system", "skills"),
                                                               ("result", None)]
    assert events[1]["loaded"] == [] and events[1]["missing"] == ["switchboard-intake"]
    assert events[2]["exit_code"] == 1 and "switchboard-intake" in events[2]["error"]


def test_chat_stream_json_skills_event_is_empty_without_skills(monkeypatch, capsys):
    """No `-s`: the event still comes, with empty lists, so a consumer never has to time out on it."""
    _code, events = _run_stream_json_chat(monkeypatch, capsys, _ok_turn)
    assert events[1]["subtype"] == "skills" and events[1]["loaded"] == [] and events[1]["missing"] == []


@pytest.mark.parametrize("argv, message", [
    (["chat", "--format", "stream-json"], "requires -q/--query"),
    (["chat", "-q", "hi", "--format", "stream-json", "--tui"], "cannot be used with --tui"),
    (["--tui", "chat", "-q", "hi", "--format", "stream-json"], "cannot be used with --tui"),
])
def test_chat_stream_json_rejects_interactive_combinations(monkeypatch, capsys, argv, message):
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(cli_entry, "_launch_tui", lambda *_a, **_k: pytest.fail("TUI launched"))
    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(parser.parse_args(argv))
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err

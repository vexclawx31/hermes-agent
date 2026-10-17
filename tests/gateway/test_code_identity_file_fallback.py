"""gateway_state code_sha falls back to the file-only HEAD reader when git identity is unreadable."""

from gateway import status


def test_falls_back_to_file_head_when_git_identity_missing(monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info.get_code_identity", lambda: {"sha": None, "version": "0.22.2"})
    monkeypatch.setattr("gateway.code_skew.current_code_sha", lambda: "a" * 40)
    assert status._get_code_identity_fields() == {"code_sha": "a" * 40, "code_version": "0.22.2"}


def test_git_identity_wins_when_present(monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info.get_code_identity", lambda: {"sha": "b" * 40, "version": "0.22.2"})

    def boom():
        raise AssertionError("file fallback must not run when git identity resolved")

    monkeypatch.setattr("gateway.code_skew.current_code_sha", boom)
    assert status._get_code_identity_fields() == {"code_sha": "b" * 40, "code_version": "0.22.2"}


def test_fallback_never_spawns_git(monkeypatch, tmp_path):
    import subprocess

    def no_spawn(*a, **k):
        raise AssertionError("fallback must not spawn a subprocess")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("c" * 40 + "\n")
    monkeypatch.setattr("gateway.code_skew._PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("hermes_cli.version_info.get_code_identity", lambda: {"sha": None, "version": "x"})
    monkeypatch.setattr(subprocess, "run", no_spawn)
    monkeypatch.setattr(subprocess, "Popen", no_spawn)
    assert status._get_code_identity_fields()["code_sha"] == "c" * 40


def test_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("x")

    monkeypatch.setattr("hermes_cli.version_info.get_code_identity", broken)
    assert status._get_code_identity_fields() == {}

"""Opt-in own-worktree exemption for the protected instruction-file gate.

``security.protected_instruction_files_allow_own_worktrees`` + ``security.own_worktree_roots`` let
the agent edit AGENTS.md / CLAUDE.md / SOUL.md under explicitly configured roots only. There is no
git auto-detection: the agent can write ``.git`` / ``worktrees/<n>/gitdir`` files, so those writes
are gated instead. A symlink in either direction across a root boundary keeps protection, and
project-local ``.hermes`` config stays gated.
"""

import os

import yaml

import tools.file_tools_write_guards as ft


def _write_security(**security):
    with open(os.path.join(os.environ["HERMES_HOME"], "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"security": security}, fh)


def _opt_in(root):
    _write_security(protected_instruction_files_allow_own_worktrees=True, own_worktree_roots=[str(root)])


def test_default_keeps_instruction_files_protected_under_a_root(tmp_path):
    _write_security(own_worktree_roots=[str(tmp_path)])  # roots alone do nothing
    assert ft._protected_instruction_reason(str(tmp_path / "wt" / "AGENTS.md")) == "AGENTS.md"


def test_opt_in_exempts_only_paths_under_configured_roots(tmp_path):
    root = tmp_path / "worktrees"
    (root / "job").mkdir(parents=True)
    _opt_in(root)
    assert ft._protected_instruction_reason(str(root / "job" / "AGENTS.md")) is None
    assert ft._protected_instruction_reason(str(tmp_path / "worktrees-sibling" / "AGENTS.md")) == "AGENTS.md"
    assert ft._protected_instruction_reason(str(root / "job" / ".hermes" / "config.yaml"))


def test_spoofed_git_metadata_is_gated_and_detects_nothing(tmp_path):
    """Writing a fake ``.git`` file or ``worktrees/<n>/gitdir`` back-link is itself gated, and a
    linked-worktree layout outside the configured roots earns no exemption."""
    root = tmp_path / "worktrees"
    root.mkdir()
    _opt_in(root)
    fake = tmp_path / "fake"
    assert ft._protected_instruction_reason(str(fake / ".git"))
    assert ft._protected_instruction_reason(str(tmp_path / "repo.git" / "worktrees" / "x" / "gitdir"))
    assert ft._protected_instruction_reason(str(root / "job" / ".git" / "hooks" / "pre-commit"))
    assert ft._protected_instruction_reason(str(fake / "SOUL.md")) == "SOUL.md"


def test_symlink_across_root_boundary_keeps_protection(tmp_path):
    root = tmp_path / "worktrees"
    (root / "job").mkdir(parents=True)
    outside = tmp_path / "main"
    outside.mkdir()
    (outside / "AGENTS.md").write_text("main rules")
    (root / "job" / "AGENTS.md").symlink_to(outside / "AGENTS.md")   # in-root link -> outside target
    (outside / "CLAUDE.md").symlink_to(root / "job" / "CLAUDE.md")   # outside link -> in-root target
    _opt_in(root)
    assert ft._protected_instruction_reason(str(root / "job" / "AGENTS.md")) == "AGENTS.md"
    assert ft._protected_instruction_reason(str(outside / "CLAUDE.md")) == "CLAUDE.md"

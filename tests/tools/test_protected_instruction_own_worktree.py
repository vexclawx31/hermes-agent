"""Opt-in own-worktree exemption for the protected instruction-file gate.

``security.protected_instruction_files_allow_own_worktrees`` lets the agent edit AGENTS.md /
CLAUDE.md / SOUL.md inside its own work area: a LINKED git worktree (never the main checkout) or a
path under ``security.own_worktree_roots``. Default stays protected, and project-local ``.hermes``
config stays gated even inside an own worktree.
"""

import os
import subprocess

import pytest
import yaml

import tools.file_tools_write_guards as ft


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.fixture
def repo_with_worktree(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", "-b", "main", cwd=main)
    (main / "README").write_text("x")
    _git("add", "README", cwd=main)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init", cwd=main)
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "feature", str(worktree), cwd=main)
    return main, worktree


def _write_security(**security):
    home = os.environ["HERMES_HOME"]
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"security": security}, fh)


def test_default_keeps_worktree_instruction_files_protected(repo_with_worktree):
    _main, worktree = repo_with_worktree
    _write_security()
    assert ft._protected_instruction_reason(str(worktree / "AGENTS.md")) == "AGENTS.md"


def test_opt_in_exempts_linked_worktree_but_not_main_checkout(repo_with_worktree):
    main, worktree = repo_with_worktree
    _write_security(protected_instruction_files_allow_own_worktrees=True)
    assert ft._protected_instruction_reason(str(worktree / "AGENTS.md")) is None
    assert ft._protected_instruction_reason(str(worktree / "sub" / "CLAUDE.md")) is None
    assert ft._protected_instruction_reason(str(main / "AGENTS.md")) == "AGENTS.md"
    # Project-local .hermes config steers behaviour too and is never exempt.
    assert ft._protected_instruction_reason(str(worktree / ".hermes" / "config.yaml"))


def test_forged_git_file_is_not_a_worktree(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / ".git").write_text(f"gitdir: {tmp_path / 'nowhere' / 'worktrees' / 'x'}\n")
    _write_security(protected_instruction_files_allow_own_worktrees=True)
    assert ft._protected_instruction_reason(str(fake / "SOUL.md")) == "SOUL.md"


def test_configured_root_is_exempt_only_under_that_root(tmp_path):
    scratch = tmp_path / "scratch"
    (scratch / "job").mkdir(parents=True)
    _write_security(protected_instruction_files_allow_own_worktrees=True, own_worktree_roots=[str(scratch)])
    assert ft._protected_instruction_reason(str(scratch / "job" / "AGENTS.md")) is None
    assert ft._protected_instruction_reason(str(tmp_path / "scratch-sibling" / "AGENTS.md")) == "AGENTS.md"

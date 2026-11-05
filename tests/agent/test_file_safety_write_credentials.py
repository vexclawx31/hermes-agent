"""Secret stores under HERMES_HOME are write-denied; control files stay writable (#110464).

``get_read_block_error`` refuses every credential store. The write side is deliberately
narrower — #45947 freed ``auth.json`` / ``config.yaml`` / ``webhook_subscriptions.json`` so
the user can ask to edit them — but the secret *material* it meant to keep blocked had
drifted: ``auth/google_oauth.json``, the plaintext Bitwarden cache, ``vault/`` and
``browser-profile/`` were writable through ``write_file`` / ``patch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import agent.file_safety as fs

SECRET_STORES = (
    "auth/google_oauth.json", "cache/bws_cache.json", "vault/vault.key", "browser-profile/Default/Cookies",
)
WRITABLE_CONTROL_FILES = ("auth.json", "config.yaml", "webhook_subscriptions.json")


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Profile HERMES_HOME plus a distinct global root, both patched."""
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, profile


def _touch(base: Path, rel: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_read_denied_secret_stores_are_write_denied_on_profile_and_root(hermes_layout):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in SECRET_STORES:
            path = _touch(base, rel)
            assert fs.get_read_block_error(str(path)), f"fixture drift: not read-denied: {path}"
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_control_files_and_lookalikes_outside_home_stay_writable(hermes_layout, tmp_path):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in WRITABLE_CONTROL_FILES:
            assert fs.is_write_denied(str(_touch(base, rel))) is False, f"#45947 regression: {rel}"
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "cache/bws_cache.json"))) is False
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "vault/vault.key"))) is False


class TestProfileHomeProcessHome:
    """With the process HOME pinned to ``{HERMES_HOME}/home`` (TERMINAL_HOME_MODE=profile,
    containers, spawned workers) the write guards must still cover every home a write can
    land in: the OS user's real home, the profile home and ``~name`` accounts."""

    @pytest.fixture()
    def profile_home_env(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        (profile / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HOME", str(profile / "home"))
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: profile.parent)
        return profile

    def test_every_home_is_guarded(self, profile_home_env):
        import pwd

        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        for rel in (".aws/credentials", ".ssh/id_ed25519", ".netrc", ".config/gh/hosts.yml"):
            assert fs.is_write_denied(str(real_home / rel)), rel
            assert fs.is_write_denied(str(profile_home_env / "home" / rel)), rel
        assert fs.is_write_denied("~/.aws/credentials")
        assert fs.is_write_denied("~root/.ssh/authorized_keys")
        # ``~/.ssh/config`` stays approval-gated (not hard-denied) on the real home too.
        assert fs.is_write_approval_required(str(real_home / ".ssh" / "config"))
        assert fs.is_write_denied(str(real_home / ".ssh" / "config")) is False

    def test_benign_paths_stay_writable(self, profile_home_env, tmp_path):
        import pwd

        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        for benign in (tmp_path / "scratch" / "notes.txt", real_home / "projects" / "notes.md"):
            assert fs.is_write_denied(str(benign)) is False, benign
            assert fs.is_write_approval_required(str(benign)) is False, benign
        assert fs.is_write_denied("~nosuchuser-hopefully/.ssh/authorized_keys") is False


# --- Identity matching: spellings a string comparison misses ---------------------------------------


def _case_insensitive_key(path):
    """Simulated case-insensitive file system for ``fs._file_key`` (meaningful on Linux CI too)."""
    p = Path(path)
    if not p.exists():
        return None
    return (0, hash(str(p.resolve()).casefold()))


def _ensure(base: Path, rel: str) -> Path:
    """Create base/<rel> unless it exists (on a case-insensitive FS a case variant already does)."""
    p = base / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("variant", encoding="utf-8")
    return p


def _case_insensitive_fs(tmp_path: Path) -> bool:
    (tmp_path / "CaseProbe").write_text("x", encoding="utf-8")
    return (tmp_path / "caseprobe").exists()


EXISTING_VARIANTS = [
    (".env", ".ENV"),
    ("cache/bws_cache.json", "CACHE/BWS_Cache.json"),
    ("auth/google_oauth.json", "Auth/Google_OAuth.json"),
    ("vault/vault.key", "VAULT/vault.key"),
    ("browser-profile/Default/Cookies", "Browser-Profile/Default/Cookies"),
    ("mcp-tokens/srv.json", "MCP-Tokens/srv.json"),
    ("state.db", "STATE.DB"),
]


@pytest.mark.parametrize("store, variant", EXISTING_VARIANTS)
def test_case_variant_of_existing_store_write_denied(hermes_layout, monkeypatch, store, variant):
    root, profile = hermes_layout
    monkeypatch.setattr(fs, "_file_key", _case_insensitive_key)
    for base in (profile, root):
        _touch(base, store)
        assert fs.is_write_denied(str(_ensure(base, variant))), f"{variant} under {base}"


@pytest.mark.parametrize("variant", [".ENV", "AUTH/Google_OAuth.json", "Cache/BWS_CACHE.enc.json",
                                     "VAULT/new.key", "MCP-TOKENS/new.json", "Sessions/x.jsonl"])
def test_case_variant_of_missing_store_write_denied(hermes_layout, variant):
    """Nothing exists yet: the nearest existing parent's identity plus the folded rest of the name."""
    root, profile = hermes_layout
    for base in (profile, root):
        assert not (base / variant).exists()
        assert fs.is_write_denied(str(base / variant)), f"{variant} under {base}"


def test_case_variant_inside_existing_denied_dir_write_denied(hermes_layout, monkeypatch):
    _, profile = hermes_layout
    monkeypatch.setattr(fs, "_file_key", _case_insensitive_key)
    (profile / "vault").mkdir()
    (profile / "VAULT").mkdir(exist_ok=True)
    assert fs.is_write_denied(str(profile / "VAULT" / "brand-new.key"))


def test_normalization_variant_write_denied(tmp_path, monkeypatch):
    """A Hermes root under a non-ASCII name, written through its decomposed (NFD) spelling."""
    nfc = "café_root"
    nfd = "café_root"
    root = tmp_path / nfc
    profile = root / "profiles" / "p"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    assert fs.is_write_denied(str(tmp_path / nfd / ".env"))
    assert fs.is_write_denied(str(tmp_path / nfd / "profiles" / "p" / "vault" / "k"))


def test_case_variant_of_home_credential_dir_write_denied(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: tmp_path / "hh")
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: tmp_path / "hh")
    assert fs.is_write_denied(str(home / ".SSH" / "id_rsa"))
    assert fs.is_write_denied(str(home / ".AWS" / "credentials"))
    assert fs.is_write_denied(str(home / ".NETRC"))


def test_unrelated_and_control_file_variants_stay_writable(hermes_layout, monkeypatch):
    """Only spellings of a write-denied store are refused; #45947 control files stay writable."""
    _, profile = hermes_layout
    monkeypatch.setattr(fs, "_file_key", _case_insensitive_key)
    _touch(profile, ".env")
    _touch(profile, "auth.json")
    for rel in ("AUTH.JSON", "Config.YAML", "notes/.Env.example", "cache/other.json", "Vaulted/x.key",
                "new-dir/file.txt"):
        assert fs.is_write_denied(str(_ensure(profile, rel) if rel != "new-dir/file.txt"
                                      else profile / rel)) is False, rel


def test_hardlink_to_store_write_denied(hermes_layout):
    """Real identity on any OS: a hardlink is the same store under an unlisted name."""
    _, profile = hermes_layout
    store = _touch(profile, ".env")
    alias = profile / "innocent.txt"
    try:
        os.link(store, alias)
    except (OSError, NotImplementedError):
        pytest.skip("hardlinks not supported here")
    assert fs.is_write_denied(str(alias))


def test_real_case_insensitive_file_system_write(hermes_layout, tmp_path):
    """No simulation: on a case-insensitive FS (macOS, Windows) .ENV is .env."""
    if not _case_insensitive_fs(tmp_path):
        pytest.skip("case-sensitive file system; covered by the simulated tests")
    _, profile = hermes_layout
    _touch(profile, ".env")
    _touch(profile, "vault/vault.key")
    assert fs.is_write_denied(str(profile / ".ENV"))
    assert fs.is_write_denied(str(profile / "Vault" / "VAULT.KEY"))
    assert fs.is_write_denied(str(profile / "VAULT" / "fresh.key"))
    assert fs.is_write_denied(str(profile / "AUTH.JSON")) is False

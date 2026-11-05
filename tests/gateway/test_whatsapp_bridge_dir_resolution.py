"""Tests for resolve_whatsapp_bridge_dir() — read-only install tree handling.

Regression coverage for #49561: in the Docker image the install tree
(/opt/hermes/scripts/whatsapp-bridge) is read-only, so `npm install` fails
with EACCES. The resolver must detect the read-only install dir and mirror the
bridge source into a writable HERMES_HOME location instead.
"""
import importlib
from pathlib import Path

import pytest

from gateway.platforms import whatsapp_common


def _seed_install_tree(install_bridge: Path) -> None:
    """Create a minimal fake bridge source tree."""
    install_bridge.mkdir(parents=True, exist_ok=True)
    (install_bridge / "bridge.js").write_text("// bridge\n")
    (install_bridge / "package.json").write_text('{"name": "whatsapp-bridge"}\n')


def test_readonly_install_mirrors_to_hermes_home(tmp_path, monkeypatch):
    """A read-only install tree is mirrored into a writable HERMES_HOME."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setattr(
        whatsapp_common, "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: hermes_home
    )

    # Simulate a read-only install tree. chmod(0o555) is unreliable under
    # root (CI/Docker bypass permission bits), so force the write probe to
    # fail by raising on the .write_test touch for the install dir only.
    _real_touch = Path.touch

    def _fake_touch(self, *a, **kw):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return _real_touch(self, *a, **kw)

    monkeypatch.setattr(Path, "touch", _fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    expected = hermes_home / "scripts" / "whatsapp-bridge"
    assert resolved == expected
    # Source was mirrored, not symlinked.
    assert (expected / "bridge.js").read_text() == "// bridge\n"
    assert (expected / "package.json").exists()
    assert (expected / whatsapp_common._BRIDGE_MARKER).read_text().strip() == (
        whatsapp_common._bridge_tree_hash(install_bridge))


# ---------------------------------------------------------------- copy refresh
# A release upgrade must refresh a stale HERMES_HOME copy (an agent ran an
# Oct 18 bridge for weeks because the copy was made once and reused forever).


def _release(tmp_path: Path, bridge_js: str = "// v2\n", lock: str = "{\"lock\": 1}\n") -> Path:
    release = tmp_path / "release" / "whatsapp-bridge"
    release.mkdir(parents=True)
    (release / "bridge.js").write_text(bridge_js)
    (release / "package.json").write_text('{"name": "whatsapp-bridge"}\n')
    (release / "package-lock.json").write_text(lock)
    return release


def _copy_of(release: Path, dest: Path, *, bridge_js: str | None = None, marker: str | None = None) -> Path:
    """An existing HERMES_HOME copy with an installed node_modules (and the adapter's dep stamp)."""
    dest.mkdir(parents=True)
    for name in ("bridge.js", "package.json", "package-lock.json"):
        (dest / name).write_bytes((release / name).read_bytes())
    if bridge_js is not None:
        (dest / "bridge.js").write_text(bridge_js)
    (dest / "node_modules" / "baileys").mkdir(parents=True)
    (dest / "node_modules" / "baileys" / "index.js").write_text("// dep\n")
    (dest / "node_modules" / ".hermes-pkg-hash").write_text("stamp")
    if marker is not None:
        (dest / whatsapp_common._BRIDGE_MARKER).write_text(marker)
    return dest


def _siblings(copy: Path) -> list[str]:
    return sorted(p.name for p in copy.parent.iterdir() if p.name != ".whatsapp-bridge.lock")


def test_tree_hash_is_deterministic_and_ignores_runtime_artefacts(tmp_path):
    release = _release(tmp_path)
    before = whatsapp_common._bridge_tree_hash(release)
    (release / "node_modules").mkdir()
    (release / "node_modules" / "x.js").write_text("x")
    (release / ".write_test").write_text("")
    (release / whatsapp_common._BRIDGE_MARKER).write_text("whatever")
    assert whatsapp_common._bridge_tree_hash(release) == before
    (release / "bridge.js").write_text("// v3\n")
    assert whatsapp_common._bridge_tree_hash(release) != before


def test_stale_copy_is_refreshed_and_keeps_node_modules(tmp_path, caplog):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    old_hash = whatsapp_common._bridge_tree_hash(copy)
    new_hash = whatsapp_common._bridge_tree_hash(release)

    with caplog.at_level("INFO", logger=whatsapp_common.logger.name):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text().strip() == new_hash
    # package.json + lockfile identical: the installed deps (and their stamp) carry over.
    assert (copy / "node_modules" / "baileys" / "index.js").read_text() == "// dep\n"
    assert (copy / "node_modules" / ".hermes-pkg-hash").read_text() == "stamp"
    refreshed = [r for r in caplog.records if "refreshed bridge copy" in r.getMessage()]
    assert len(refreshed) == 1 and refreshed[0].levelname == "INFO"
    assert old_hash in refreshed[0].getMessage() and new_hash in refreshed[0].getMessage()
    assert _siblings(copy) == ["whatsapp-bridge"]  # no temp/old dirs left behind


@pytest.mark.parametrize("changed", ["package.json", "package-lock.json"])
def test_dependency_change_drops_node_modules(tmp_path, changed):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge")
    (release / changed).write_text('{"changed": true}\n')

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / changed).read_text() == '{"changed": true}\n'
    # No node_modules: the adapter's _ensure_bridge_deps / the CLI installers run npm install.
    assert not (copy / "node_modules").exists()
    assert _siblings(copy) == ["whatsapp-bridge"]


def test_identical_copy_without_marker_only_gains_marker(tmp_path, caplog, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge")
    inode = copy.stat().st_ino

    def _no_refresh(*_a, **_kw):
        raise AssertionError("identical copy must not be swapped")

    monkeypatch.setattr(whatsapp_common, "_refresh_whatsapp_bridge_copy", _no_refresh)
    with caplog.at_level("INFO", logger=whatsapp_common.logger.name):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert copy.stat().st_ino == inode
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text().strip() == whatsapp_common._bridge_tree_hash(release)
    assert not [r for r in caplog.records if "refreshed bridge copy" in r.getMessage()]


def test_matching_marker_is_the_cheap_path(tmp_path, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge",
                    marker=whatsapp_common._bridge_tree_hash(release))
    hashed = []
    real_hash = whatsapp_common._bridge_tree_hash
    monkeypatch.setattr(whatsapp_common, "_bridge_tree_hash", lambda root: hashed.append(root) or real_hash(root))

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert hashed == [release]  # the copy itself is not rehashed
    assert not (copy.parent / ".whatsapp-bridge.lock").exists()


def test_wrong_marker_still_refreshes(tmp_path):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge",
                    bridge_js="// v1\n", marker="0123456789abcdef")

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "bridge.js").read_text() == "// v2\n"


def test_failed_refresh_leaves_old_copy_intact(tmp_path, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    real_replace = whatsapp_common.os.replace

    def _flaky_replace(src, dst):
        # Fail the final swap of the new tree into place; the rollback rename must still work.
        if Path(dst) == copy and ".whatsapp-bridge.new-" in Path(src).name:
            raise OSError("simulated swap failure")
        return real_replace(src, dst)

    monkeypatch.setattr(whatsapp_common.os, "replace", _flaky_replace)
    with pytest.raises(OSError):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    monkeypatch.setattr(whatsapp_common.os, "replace", real_replace)

    # Old tree is back in place, node_modules included, and no temp dirs are left.
    assert (copy / "bridge.js").read_text() == "// v1\n"
    assert (copy / "node_modules" / "baileys" / "index.js").exists()
    assert _siblings(copy) == ["whatsapp-bridge"]


def test_concurrent_syncs_produce_one_consistent_copy(tmp_path):
    import threading

    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    barrier = threading.Barrier(4)
    errors = []

    def _run():
        barrier.wait()
        try:
            whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
        except Exception as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / "node_modules" / "baileys" / "index.js").exists()
    assert whatsapp_common._bridge_tree_hash(copy) == whatsapp_common._bridge_tree_hash(release)
    assert _siblings(copy) == ["whatsapp-bridge"]


def test_resolver_keeps_existing_copy_when_refresh_fails(tmp_path, monkeypatch):
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)
    hermes_home = tmp_path / "hermes_home"
    existing = hermes_home / "scripts" / "whatsapp-bridge"
    existing.mkdir(parents=True)
    (existing / "bridge.js").write_text("// old\n")
    monkeypatch.setattr(whatsapp_common, "__file__", str(install_root / "gateway" / "platforms" / "whatsapp_common.py"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    _real_touch = Path.touch

    def _fake_touch(self, *a, **kw):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return _real_touch(self, *a, **kw)

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "touch", _fake_touch)
    monkeypatch.setattr(whatsapp_common, "_refresh_whatsapp_bridge_copy", _boom)

    assert whatsapp_common.resolve_whatsapp_bridge_dir() == existing
    assert (existing / "bridge.js").read_text() == "// old\n"



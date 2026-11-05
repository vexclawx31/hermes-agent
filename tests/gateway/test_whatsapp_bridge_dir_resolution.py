"""Tests for resolve_whatsapp_bridge_dir() — read-only install tree handling.

Regression coverage for #49561: in the Docker image the install tree
(/opt/hermes/scripts/whatsapp-bridge) is read-only, so `npm install` fails
with EACCES. The resolver must detect the read-only install dir and mirror the
bridge source into a writable HERMES_HOME location instead.
"""
import importlib
import os
import shutil
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
    assert whatsapp_common.whatsapp_bridge_marker_hash(expected) == (
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


def _marker_text(release: Path) -> str:
    """The marker a refresh from ``release`` writes."""
    return f"{whatsapp_common._bridge_tree_hash(release)} {whatsapp_common._bridge_deps_hash(release)}\n"


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
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)
    # package.json unchanged: the installed deps (and their stamp) carry over.
    assert (copy / "node_modules" / "baileys" / "index.js").read_text() == "// dep\n"
    assert (copy / "node_modules" / ".hermes-pkg-hash").read_text() == "stamp"
    refreshed = [r for r in caplog.records if "refreshed bridge copy" in r.getMessage()]
    assert len(refreshed) == 1 and refreshed[0].levelname == "INFO"
    assert old_hash in refreshed[0].getMessage() and new_hash in refreshed[0].getMessage()
    assert _siblings(copy) == ["whatsapp-bridge"]  # no temp/old dirs left behind


@pytest.mark.parametrize("changed,kept", [("package.json", False), ("package-lock.json", True)])
def test_first_refresh_keeps_node_modules_unless_package_json_changed(tmp_path, changed, kept):
    """No marker yet (every pre-r21 copy): reuse follows package.json; the deps stamp decides installs."""
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge")
    (release / changed).write_text('{"changed": true}\n')

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / changed).read_text() == '{"changed": true}\n'
    # Dropped: the adapter's _ensure_bridge_deps / the CLI installers run npm install.
    assert (copy / "node_modules" / ".hermes-pkg-hash").exists() is kept
    assert _siblings(copy) == ["whatsapp-bridge"]


def test_marked_copy_compares_release_dependencies_not_the_copy_lockfile(tmp_path):
    """npm rewrites the copy's lockfile; reuse compares the release fingerprint recorded at refresh."""
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", marker=_marker_text(release))
    (copy / "package-lock.json").write_text('{"rewritten by": "npm"}\n')
    (release / "bridge.js").write_text("// v3\n")  # next release: JS only

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "bridge.js").read_text() == "// v3\n"
    assert (copy / "node_modules" / ".hermes-pkg-hash").exists()

    (release / "package-lock.json").write_text('{"lock": 2}\n')  # next release: new lockfile
    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "package-lock.json").read_text() == '{"lock": 2}\n'
    assert not (copy / "node_modules").exists()


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
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)
    assert not [r for r in caplog.records if "refreshed bridge copy" in r.getMessage()]


def test_matching_marker_is_the_cheap_path(tmp_path, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", marker=_marker_text(release))
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


def test_corrupt_marker_counts_as_missing(tmp_path):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    (copy / whatsapp_common._BRIDGE_MARKER).write_bytes(b"\xff\xfe\x00garbage\x80")

    assert whatsapp_common._read_bridge_marker(copy) == ("", "")
    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / "node_modules" / ".hermes-pkg-hash").exists()


def _kill_at(monkeypatch, when):
    """Make the refresh die (like a SIGKILL: no cleanup runs) at ``when``: "between-renames" (old copy
    renamed aside, new tree not yet in place) or "before-rmtree" (swap done, old tree not removed)."""
    real_replace, real_rmtree = whatsapp_common.os.replace, shutil.rmtree

    class _Killed(BaseException):
        pass

    def _replace(src, dst):
        if when == "between-renames" and ".whatsapp-bridge.new-" in Path(src).name and Path(dst).name == "whatsapp-bridge":
            raise _Killed
        return real_replace(src, dst)

    def _rmtree(path, *a, **kw):
        if ".whatsapp-bridge." in Path(path).name:
            if when == "before-rmtree" and ".old-" in Path(path).name:
                raise _Killed
            if when == "between-renames":
                return None  # a killed process never reaches its cleanup
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(whatsapp_common.os, "replace", _replace)
    monkeypatch.setattr("shutil.rmtree", _rmtree)
    return _Killed, lambda: (monkeypatch.setattr(whatsapp_common.os, "replace", real_replace),
                             monkeypatch.setattr("shutil.rmtree", real_rmtree))


def test_kill_between_renames_is_recovered_on_next_start(tmp_path, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    killed, restore = _kill_at(monkeypatch, "between-renames")
    with pytest.raises(killed):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    restore()
    assert not copy.exists()  # the crash window: old copy aside, node_modules in the new tree

    # A read-only caller already puts the old copy and its node_modules back.
    whatsapp_common._sync_whatsapp_bridge_copy(release, copy, refresh=False)
    assert (copy / "bridge.js").read_text() == "// v1\n"
    assert (copy / "node_modules" / "baileys" / "index.js").exists()
    assert _siblings(copy) == ["whatsapp-bridge"]

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / "node_modules" / "baileys" / "index.js").exists()


def test_kill_after_swap_cleans_leftovers_on_cheap_path(tmp_path, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    killed, restore = _kill_at(monkeypatch, "before-rmtree")
    with pytest.raises(killed):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    restore()
    assert any(".old-" in name for name in _siblings(copy))

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)  # marker matches: cheap path + cleanup

    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / "node_modules" / "baileys" / "index.js").exists()
    assert _siblings(copy) == ["whatsapp-bridge"]


def test_read_only_resolution_never_replaces_an_existing_copy(tmp_path):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy, refresh=False)

    assert (copy / "bridge.js").read_text() == "// v1\n"
    assert not (copy / whatsapp_common._BRIDGE_MARKER).exists()
    assert not (copy.parent / ".whatsapp-bridge.lock").exists()


def test_read_only_resolution_creates_a_missing_copy(tmp_path):
    release = _release(tmp_path)
    copy = tmp_path / "home" / "scripts" / "whatsapp-bridge"

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy, refresh=False)

    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)


def test_symlink_in_release_is_refused(tmp_path):
    release = _release(tmp_path)
    (tmp_path / "outside.js").write_text("// not part of the release\n")
    (release / "linked.js").symlink_to(tmp_path / "outside.js")
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")

    with pytest.raises(OSError, match="symlink"):
        whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    assert (copy / "bridge.js").read_text() == "// v1\n"


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


# --- sealed (read-only) releases: the mirror inherits their modes (r21 follow-up) -------------------

def _seal(root: Path) -> None:
    """Make a tree read-only like a sealed release: dirs r-x, files r--, leaving node_modules alone."""
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if "node_modules" in path.relative_to(root).parts:
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _owner_writable(root: Path) -> bool:
    import stat
    return all(p.stat().st_mode & stat.S_IWUSR for p in [root, *root.rglob("*")]
               if "node_modules" not in p.relative_to(root).parts)


@pytest.fixture
def unseal(tmp_path):
    """Give every test tree back its write bits so pytest can remove tmp_path."""
    yield
    for path in [tmp_path, *tmp_path.rglob("*")]:
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


root_bypasses_modes = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                         reason="root ignores permission bits")


@root_bypasses_modes
def test_read_only_copy_of_a_sealed_release_is_refreshed(tmp_path, unseal):
    """The field case: a mirror made from a sealed release is r-x, so the marker write, the swap and
    the cleanup all failed with PermissionError and the stale copy was used forever."""
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge", bridge_js="// v1\n")
    _seal(release)
    _seal(copy)

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / "bridge.js").read_text() == "// v2\n"
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)
    assert (copy / "node_modules" / ".hermes-pkg-hash").read_text() == "stamp"  # deps carried over
    assert _owner_writable(copy), "the mirror is the home's working copy: npm and the next refresh write in it"
    assert _siblings(copy) == ["whatsapp-bridge"]  # the old copy was removed, not left as .old-*


@root_bypasses_modes
def test_first_copy_of_a_sealed_release_is_owner_writable(tmp_path, unseal):
    release = _release(tmp_path)
    _seal(release)
    copy = tmp_path / "home" / "scripts" / "whatsapp-bridge"

    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)

    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)
    assert _owner_writable(copy)


@root_bypasses_modes
def test_identical_read_only_copy_only_gains_the_marker(tmp_path, unseal, monkeypatch):
    release = _release(tmp_path)
    copy = _copy_of(release, tmp_path / "home" / "scripts" / "whatsapp-bridge")
    _seal(release)
    _seal(copy)

    def _no_refresh(*_a, **_kw):
        raise AssertionError("an identical copy must not be swapped")

    monkeypatch.setattr(whatsapp_common, "_refresh_whatsapp_bridge_copy", _no_refresh)
    whatsapp_common._sync_whatsapp_bridge_copy(release, copy)
    assert (copy / whatsapp_common._BRIDGE_MARKER).read_text() == _marker_text(release)



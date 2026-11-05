"""Google Chat dependency installation crosses PM's declared-feature interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

import pm
from plugins.platforms.google_chat import oauth


def test_stale_google_transitives_are_reported_missing(monkeypatch):
    installed = {
        "google-cloud-pubsub": "2.39.2",
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(oauth, "_distribution_version", fake_version)

    stale = {spec.split("==")[0] for spec in oauth._missing_required_packages()}
    assert {"google-auth", "httplib2", "pyasn1"} <= stale


def _all_pins_installed(monkeypatch, *, skip=()):
    from packaging.requirements import Requirement

    installed = {}
    for spec in oauth._REQUIRED_PACKAGES:
        requirement = Requirement(spec)
        if requirement.name not in skip:
            installed[requirement.name] = next(iter(requirement.specifier)).version

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(oauth, "_distribution_version", fake_version)


def _fake_environment(monkeypatch, *, sys_platform, platform_machine):
    """Evaluate markers as if running on another platform (packaging reads these at evaluate time)."""
    from packaging import markers

    real = markers.default_environment

    def fake_default_environment():
        env = dict(real())
        env.update(sys_platform=sys_platform, platform_machine=platform_machine)
        return env

    monkeypatch.setattr(markers, "default_environment", fake_default_environment)


def test_marker_excluded_package_is_not_required_on_windows_arm64(monkeypatch):
    """pyproject never installs google-cloud-pubsub on win32-arm64, so it must not be reported missing."""
    _all_pins_installed(monkeypatch, skip={"google-cloud-pubsub"})
    _fake_environment(monkeypatch, sys_platform="win32", platform_machine="ARM64")

    assert oauth._missing_required_packages() == []


def test_marker_matching_package_is_still_required_elsewhere(monkeypatch):
    _all_pins_installed(monkeypatch, skip={"google-cloud-pubsub"})
    _fake_environment(monkeypatch, sys_platform="darwin", platform_machine="arm64")

    missing = oauth._missing_required_packages()
    assert [spec.split("==")[0] for spec in missing] == ["google-cloud-pubsub"]


def test_required_packages_match_pyproject_pins_and_markers():
    """Each probe spec is the exact pyproject entry (pin AND marker) from the declared extras."""
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    declared = {}
    for extra in oauth._DEPENDENCY_EXTRAS:
        for spec in extras[extra]:
            requirement = Requirement(spec)
            declared[requirement.name] = requirement
    for spec in oauth._REQUIRED_PACKAGES:
        probe = Requirement(spec)
        assert probe.name in declared, spec
        assert str(probe) == str(declared[probe.name]), spec


def test_installer_repairs_stale_transitives_through_pm(monkeypatch, capsys):
    states = iter([["google-auth==2.55.1", "httplib2==0.32.0", "pyasn1==0.6.4"], []])
    monkeypatch.setattr(oauth, "_missing_required_packages", lambda: next(states))
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))

    assert oauth.install_deps() is True
    assert calls == [(["google", "google-chat"], {"explicit": True})]
    assert "restart" in capsys.readouterr().out.lower()


def test_ensure_deps_surfaces_install_reason(monkeypatch):
    """A refused install must reach the registry's log with its reason, not a bare False."""
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)

    def blocked(extra):
        raise RuntimeError(f"extra {extra!r}: lazy installs are disabled")

    monkeypatch.setattr(pm, "ensure_import", blocked)
    with pytest.raises(RuntimeError, match="lazy installs are disabled"):
        adapter.ensure_google_chat_deps()


def test_ensure_deps_requests_every_extra_before_reporting_a_restart(monkeypatch):
    """A successful install of the FIRST extra raises InstallError("restart Hermes…"); stopping
    there left the second extra uninstalled, so the restart landed right back here. Both are
    requested in one pass and the first failure is what the registry sees."""
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)
    requested = []

    def installs_then_needs_restart(extra):
        requested.append(extra)
        raise pm.InstallError("venv", f"{extra} installed; restart Hermes to activate the new dependency environment")

    monkeypatch.setattr(pm, "ensure_import", installs_then_needs_restart)
    with pytest.raises(pm.InstallError, match="google installed"):
        adapter.ensure_google_chat_deps()
    assert requested == ["google", "google-chat"]

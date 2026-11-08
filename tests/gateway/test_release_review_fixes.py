"""Fork (v0.23 release, second independent review): the fixes it asked for, pinned.

- S1: a spawner's HERMES_CHAT_RUN_ID never reaches a gateway its client autostarts.
- S4: a managed home's owner is never stopped or replaced by a manual command (``gateway restart``'s
  stop-and-respawn fallback, ``gateway run --replace``): refused before anything stops.
- A durable plugin row whose stored author no longer re-validates is refused, never run as the human.
"""

import pytest


def test_run_id_is_stripped_from_an_autostarted_gateway():
    from hermes_cli.gateway_runtime_start import DAEMON_STRIPPED_ENV
    assert "HERMES_CHAT_RUN_ID" in DAEMON_STRIPPED_ENV


def _managed_home(tmp_path, managed=True):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(f"gateway:\n  managed_runtime: {'true' if managed else 'false'}\n",
                                      encoding="utf-8")
    return home


def test_a_manual_owner_change_on_a_managed_home_is_refused(tmp_path, capsys):
    from hermes_cli.gateway_runtime_managed import refuse_manual_owner_change
    home = _managed_home(tmp_path)
    with pytest.raises(SystemExit) as exc:
        refuse_manual_owner_change(home, "gateway restart")
    assert exc.value.code == 1
    assert "externally managed" in capsys.readouterr().err


def test_an_unmanaged_home_is_left_to_upstream(tmp_path):
    from hermes_cli.gateway_runtime_managed import refuse_manual_owner_change
    refuse_manual_owner_change(_managed_home(tmp_path, managed=False), "gateway restart")  # returns


def test_run_replace_refuses_before_any_guard_or_stop(tmp_path, monkeypatch):
    from hermes_cli import gateway
    home = _managed_home(tmp_path)
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)

    def stopped(*_a, **_k):
        raise AssertionError("reached the owner checks: the managed home was not refused first")
    monkeypatch.setattr(gateway, "_attach_to_host_gateway_or_guard", stopped)
    with pytest.raises(SystemExit) as exc:
        gateway.run_gateway(replace=True)
    assert exc.value.code == 1


def test_a_plugin_row_whose_author_no_longer_validates_is_refused():
    from types import SimpleNamespace

    from gateway.session_automation import restore_producer_descriptor
    from hermes_state_runtime import RuntimeStoreError
    event = SimpleNamespace(metadata={}, allow_gateway_control=True)
    forged = {"origin": "plugin", "plugin": "switchboard-line", "id": "plugin:switchboard-line/x",
              "is_bot": True, "verified": True}
    with pytest.raises(RuntimeStoreError):
        restore_producer_descriptor(event, {"control": False, "turn_author": forged})
    assert "turn_author" not in event.metadata

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


@pytest.mark.parametrize("value, expected", [
    ("helpdesk-request", "helpdesk-request"), ("switchboard-tenant", "switchboard-tenant"),
    ("slack", None), ("cli", None), ("", None), ("Not A Label!", None),
])
def test_a_nested_client_inherits_only_a_valid_provenance_label(monkeypatch, value, expected):
    """#531 review item 4: a `hermes chat` run from a request or tenant turn's terminal keeps that turn's
    origin (the gateway binds HERMES_SESSION_SOURCE in its tools' env), so the door's guards see it."""
    from hermes_cli.gateway_session_store import _source_fields, inherited_origin
    monkeypatch.setenv("HERMES_SESSION_SOURCE", value)
    assert inherited_origin() == expected
    assert _source_fields("cli") == {"source": "cli", **({"origin": expected} if expected else {})}


def test_an_inherited_origin_never_refuses_a_resume(monkeypatch):
    """Only creation carries the inherited origin: the resume check compares explicit flags alone."""
    import argparse

    from hermes_cli.gateway_chat import check_resume_policy
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "helpdesk-request")
    args = argparse.Namespace(source=None, in_dir=None)
    check_resume_policy(args, {"info": {"launch_request": {}}})  # no refusal


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

"""Recall scope: labels, scopes, containment and the trusted turn scope (agent/recall_scope.py)."""
import pytest

from agent.recall_scope import (
    UNKNOWN_LABEL, enforcement_enabled, make_label, normalize_pattern, scope_for_source, scope_from_patterns,
    sender_label,
)

J, P = "UJAMAL0001", "UPAUL00002"
DM_J, JOINT = "DJAMAL0001", "CJOINT0001"


def _sender(user_id, chat_id, trusted=True, platform="slack"):
    return {"sender": {"v": 1, "platform": platform, "user_id": user_id, "chat_id": chat_id, "trusted": trusted}}


def test_dm_scope_allows_only_that_person_anywhere():
    scope = scope_for_source("slack", J, DM_J, "dm")
    assert scope.allows({f"slack:{J}@{DM_J}", f"slack:{J}@{JOINT}"})
    assert not scope.allows({f"slack:{J}@{DM_J}", f"slack:{P}@{JOINT}"})  # one foreign label taints
    assert not scope.allows({f"telegram:{J}@{DM_J}"})


def test_shared_chat_scope_allows_only_that_chat():
    scope = scope_for_source("slack", J, JOINT, "group")
    assert scope.allows({f"slack:{J}@{JOINT}", f"slack:{P}@{JOINT}"})
    assert not scope.allows({f"slack:{J}@{DM_J}"})  # private DM evidence never reaches a shared audience


@pytest.mark.parametrize("prov", [set(), {UNKNOWN_LABEL}, {f"slack:{J}@{DM_J}", UNKNOWN_LABEL}, None])
def test_empty_or_unknown_provenance_never_matches(prov):
    assert not scope_for_source("slack", J, DM_J, "dm").allows(prov)


def test_bot_and_incomplete_sources_get_no_scope():
    assert scope_for_source("slack", J, DM_J, "dm", is_bot=True) is None
    assert scope_for_source("slack", "", DM_J, "dm") is None
    assert scope_for_source("slack", "U:1", DM_J, "dm") is None  # reserved characters


def test_declared_patterns():
    scope = scope_from_patterns([f"slack:{J}@{JOINT}", f"slack:{P}@{JOINT}"], origin="cron")
    assert scope.allows({f"slack:{P}@{JOINT}"}) and not scope.allows({f"slack:{P}@{DM_J}"})
    assert scope_from_patterns([f"slack:{J}@*", "garbage"]) is None
    assert scope_from_patterns([]) is None
    assert normalize_pattern("slack:*@*") is None  # never "everyone everywhere"
    assert normalize_pattern("*:U1@C1") is None


def test_labels_and_sender_metadata():
    assert make_label("Slack", J, DM_J) == f"slack:{J}@{DM_J}"
    assert make_label("slack", J, None) == UNKNOWN_LABEL
    assert sender_label(_sender(J, DM_J)) == f"slack:{J}@{DM_J}"
    assert sender_label(_sender(J, DM_J, trusted=False)) == UNKNOWN_LABEL
    assert sender_label('{"sender": "nope"}') == UNKNOWN_LABEL
    assert sender_label(None) == UNKNOWN_LABEL


def test_scope_containment_and_partitions():
    jamal_anywhere = scope_from_patterns([f"slack:{J}@*"])
    jamal_joint = scope_from_patterns([f"slack:{J}@{JOINT}"])
    joint = scope_from_patterns([f"slack:{J}@{JOINT}", f"slack:{P}@{JOINT}"])
    assert jamal_anywhere.covers(jamal_joint) and not jamal_joint.covers(jamal_anywhere)
    assert joint.covers(jamal_joint) and not jamal_anywhere.covers(joint)
    assert jamal_anywhere.partition() != joint.partition()
    assert jamal_anywhere.partition() == scope_from_patterns([f"slack:{J}@*"], origin="cron").partition()


class _Ingress:
    def __init__(self, user_id, chat_id, chat_type="dm"):
        self.platform, self.user_id, self.chat_id, self.chat_type = "slack", user_id, chat_id, chat_type


class _Source:
    def __init__(self, user_id, chat_id, is_bot=False):
        self.platform, self.user_id, self.chat_id, self.is_bot = "slack", user_id, chat_id, is_bot


def _turn_scope(monkeypatch, ingress, source):
    import gateway.ingress_identity as ingress_mod
    from gateway.sender_attribution import trusted_turn_scope
    monkeypatch.setattr(ingress_mod, "get_trusted_ingress", lambda: ingress)
    return trusted_turn_scope(source)


def test_trusted_turn_scope_one_to_one_dm_vs_group_dm(monkeypatch):
    dm = _turn_scope(monkeypatch, _Ingress(J, DM_J), _Source(J, DM_J))
    assert dm.patterns == {f"slack:{J}@*"}
    # Slack types a group DM (mpim) as "dm" too: it must get the chat's scope, not Jamal's.
    group_dm = _turn_scope(monkeypatch, _Ingress(J, "GMPIM00001"), _Source(J, "GMPIM00001"))
    assert group_dm.patterns == {"slack:*@GMPIM00001"}


def test_trusted_turn_scope_requires_matching_native_ingress(monkeypatch):
    assert _turn_scope(monkeypatch, None, _Source(J, DM_J)) is None
    assert _turn_scope(monkeypatch, _Ingress(J, DM_J), _Source(P, DM_J)) is None  # source claims another sender
    assert _turn_scope(monkeypatch, _Ingress(J, DM_J), _Source(J, "DOTHER0001")) is None
    assert _turn_scope(monkeypatch, _Ingress(J, DM_J), _Source(J, DM_J, is_bot=True)) is None


def test_enforcement_flag():
    assert enforcement_enabled({"privacy": {"recall_scope": "enforce"}})
    assert not enforcement_enabled({"privacy": {"recall_scope": "off"}})
    assert not enforcement_enabled({})


def test_unreadable_config_keeps_the_last_known_answer(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {"privacy": {"recall_scope": "enforce"}})
    assert enforcement_enabled()

    def _boom():
        raise OSError("config unreadable")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert enforcement_enabled()  # an enforcing profile stays enforced
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})
    assert not enforcement_enabled()
    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert not enforcement_enabled()  # a profile that never opted in is unaffected


def test_defaults_fallback_on_a_broken_config_is_not_an_off_switch(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod
    from hermes_cli.config_read_errors import FailedConfigRead
    from agent.recall_scope import enforcement_epoch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broken = FailedConfigRead({"privacy": {"recall_scope": "off"}}, error=ValueError("bad yaml"))
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: broken)
    assert not enforcement_enabled()  # never enforced here, no epoch: unaffected
    enforcement_epoch(True, tmp_path, now=10.0)
    assert enforcement_enabled()  # an enforcing profile (epoch on disk) stays enforced
    assert (tmp_path / "state" / "recall_scope_epoch.json").exists()

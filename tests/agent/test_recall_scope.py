"""Recall scope: provenance ⊆ scope, unknown provenance never matches (agent/recall_scope.py)."""
import pytest

from agent.recall_scope import (
    UNKNOWN_LABEL, enforcement_enabled, make_label, message_provenance, messages_provenance,
    normalize_pattern, provenance_from_metadata, provenance_metadata, scope_for_source, scope_from_patterns,
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


def test_message_provenance_inherits_and_taints():
    rows = [
        {"role": "user", "content": "hi", "display_metadata": _sender(J, JOINT)},
        {"role": "assistant", "content": "relaying Pauline"},
        {"role": "user", "content": "x", "display_metadata": _sender(P, JOINT)},
    ]
    assert messages_provenance(rows) == {f"slack:{J}@{JOINT}", f"slack:{P}@{JOINT}"}
    assert message_provenance({"role": "user", "content": "legacy"}) == {UNKNOWN_LABEL}
    assert message_provenance({"role": "user", "_compressed_summary": True, "content": "s"}) == {UNKNOWN_LABEL}
    summary = {"role": "user", "_compressed_summary": True, "display_metadata": provenance_metadata(
        {f"slack:{J}@{DM_J}"})}
    assert message_provenance(summary) == {f"slack:{J}@{DM_J}"}
    assert provenance_metadata(()) == {"provenance": [UNKNOWN_LABEL]}
    assert provenance_from_metadata({"provenance": []}) is None
    assert provenance_from_metadata({"provenance": [3]}) == {UNKNOWN_LABEL}


def test_enforcement_flag():
    assert enforcement_enabled({"privacy": {"recall_scope": "enforce"}})
    assert not enforcement_enabled({"privacy": {"recall_scope": "off"}})
    assert not enforcement_enabled({})

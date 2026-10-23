"""Attribution aids under recall-scope enforcement: the summary prompt rule and Slack backfill ids."""
from agent.context_compressor import ContextCompressor


def test_summary_rule_only_when_enforced(monkeypatch):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: False)
    assert ContextCompressor._speaker_attribution_rule() == ""
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: True)
    rule = ContextCompressor._speaker_attribution_rule()
    assert "SPEAKER ATTRIBUTION" in rule and "(author unknown)" in rule


def test_slack_user_id_pattern():
    from plugins.platforms.slack.adapter import _SLACK_USER_ID
    assert _SLACK_USER_ID.fullmatch("U0AFPKV3C5T") and _SLACK_USER_ID.fullmatch("W012ABC")
    assert not _SLACK_USER_ID.fullmatch("B012ABC") and not _SLACK_USER_ID.fullmatch("u0afpkv3c5t")
    assert not _SLACK_USER_ID.fullmatch("U0AF>evil")

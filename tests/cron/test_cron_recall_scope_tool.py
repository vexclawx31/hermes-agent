"""The model may not repoint a cron job that declares a recall scope (tools/cronjob_tools.py)."""

import json

import pytest

from tools.cronjob_tools import _scoped_job_refusal

SCOPED = {"id": "abc123", "recall_scope": ["slack:UJAMAL0001@*"]}


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: True)


@pytest.mark.parametrize("action,args", [
    ("update", {"prompt": "send it to Pauline"}),
    ("update", {"deliver": "slack:UPAUL00002"}),
    ("remove", {}),
    ("run", {"prompt": "and include Jamal's diary"}),
    ("run_now", {"prompt": "x"}),
    ("trigger", {"prompt": "x"}),
])
def test_scoped_job_cannot_be_repointed(enforce, action, args):
    out = json.loads(_scoped_job_refusal(SCOPED, action, args))
    assert out["success"] is False and "recall scope" in out["error"]


def test_scoped_job_can_still_run_as_configured(enforce):
    assert _scoped_job_refusal(SCOPED, "run", {"prompt": ""}) is None
    assert _scoped_job_refusal(SCOPED, "run_now", {}) is None


def test_unscoped_jobs_and_unenforced_profiles_are_unaffected(monkeypatch):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: True)
    assert _scoped_job_refusal({"id": "def456"}, "update", {"prompt": "x"}) is None
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: False)
    assert _scoped_job_refusal(SCOPED, "update", {"prompt": "x"}) is None

"""Cron recall scope: run docs record the scope they ran under, and context_from injects only
outputs whose recorded scope the reading job covers (agent/recall_scope.py)."""

import pytest

JAMAL = ["slack:UJAMAL0001@*"]
JOINT = ["slack:UJAMAL0001@CJOINT0001", "slack:UPAUL00002@CJOINT0001"]


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: True)


def _archive(job_id, name, scope_patterns, answer):
    from cron.jobs import OUTPUT_DIR
    from cron.scheduler import _run_doc_header
    out = OUTPUT_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)
    header = _run_doc_header({"recall_scope": scope_patterns} if scope_patterns else {}, "t", job_id, "p")
    (out / name).write_text(f"{header}## Response\n\n{answer}\n", encoding="utf-8")


def _job(**fields):
    from cron.jobs import create_job
    job = create_job(prompt="continue", schedule="every 1h")
    job.update(fields)
    return job


def test_header_records_scope_only_when_declared():
    from cron.scheduler import _run_doc_header
    assert "**Recall scope:** slack:UJAMAL0001@*" in _run_doc_header({"recall_scope": JAMAL}, "t", "ab", "p")
    assert "Recall scope" not in _run_doc_header({}, "t", "ab", "p")
    assert "Recall scope" not in _run_doc_header({"recall_scope": ["bogus"]}, "t", "ab", "p")


def test_self_continuity_skips_legacy_and_wider_outputs(cron_env, enforce):
    from cron.scheduler import _build_job_prompt
    job = _job(recall_scope=JAMAL)
    job["context_from"] = ["self"]
    _archive(job["id"], "2026-10-01_08-00-00.md", JAMAL, "Jamal-scoped note")
    _archive(job["id"], "2026-10-02_08-00-00.md", JOINT, "joint note with Pauline")
    _archive(job["id"], "2026-10-03_08-00-00.md", None, "legacy unattributed note")
    prompt = _build_job_prompt(job)
    assert "Jamal-scoped note" in prompt
    assert "Pauline" not in prompt and "legacy" not in prompt


def test_job_without_scope_gets_nothing_under_enforcement(cron_env, enforce):
    from cron.scheduler import _build_job_prompt
    job = _job()
    job["context_from"] = ["self"]
    _archive(job["id"], "2026-10-01_08-00-00.md", JAMAL, "Jamal-scoped note")
    assert "Jamal-scoped note" not in _build_job_prompt(job)


def test_joint_job_cannot_read_jamal_private_output(cron_env, enforce):
    from cron.scheduler import _build_job_prompt
    source = _job(recall_scope=JAMAL)
    _archive(source["id"], "2026-10-01_08-00-00.md", JAMAL, "private Jamal note")
    reader = _job(recall_scope=JOINT, context_from=[source["id"]])
    assert "private Jamal note" not in _build_job_prompt(reader)


def test_unenforced_profiles_keep_previous_behaviour(cron_env):
    from cron.scheduler import _build_job_prompt
    job = _job()
    job["context_from"] = ["self"]
    _archive(job["id"], "2026-10-03_08-00-00.md", None, "legacy unattributed note")
    assert "legacy unattributed note" in _build_job_prompt(job)


def test_cron_agent_scope_comes_from_the_job():
    from agent.recall_scope import scope_from_patterns
    scope = scope_from_patterns(JAMAL, origin="cron")
    assert scope.allows({"slack:UJAMAL0001@DJAMAL0001"}) and not scope.allows({"slack:UPAUL00002@CJOINT0001"})

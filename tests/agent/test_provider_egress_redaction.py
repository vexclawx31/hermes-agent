import pytest

from agent import redact


TOKEN = "ghp_" + "A" * 40


def test_provider_payload_redacts_nested_copy_without_mutating_source():
    source = [{"role": "tool", "content": {"parts": [f"TOKEN={TOKEN}"]}}]

    result = redact.redact_provider_payload(source)

    assert TOKEN not in result[0]["content"]["parts"][0]
    assert source[0]["content"]["parts"][0].endswith(TOKEN)


def test_provider_payload_reuses_clean_objects():
    source = [{"role": "user", "content": "ordinary text"}]
    assert redact.redact_provider_payload(source) is source


def test_provider_payload_fails_closed_when_redactor_errors(monkeypatch):
    monkeypatch.setattr(redact, "redact_sensitive_text", lambda *_a, **_kw: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(redact.ProviderEgressRedactionError):
        redact.redact_provider_payload([{"content": TOKEN}])

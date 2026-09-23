"""Provider-egress redaction on the compressor's direct custom/base-URL transport.

Summarization replays tool output and injected context verbatim, so the raw-client path must
apply the same mandatory egress policy the auxiliary ``call_llm`` lanes apply at their wire
boundary — without mutating the caller's request.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics


SECRET = "ghp_" + "B" * 40
_SUMMARY = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))])


def _custom_endpoint_compressor():
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(
        summarization_model="custom-model", summary_target_tokens=100, max_retries=1)
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    return compressor


def test_direct_transport_redacts_replayed_tool_output():
    compressor = _custom_endpoint_compressor()
    compressor.client = MagicMock()
    compressor.client.chat.completions.create.return_value = _SUMMARY

    result = compressor._generate_summary(
        f"<tool_response>token={SECRET}</tool_response>", TrajectoryMetrics())

    sent = compressor.client.chat.completions.create.call_args.kwargs
    assert result.startswith("[CONTEXT SUMMARY]:")
    assert SECRET not in str(sent)
    assert "token=" in sent["messages"][0]["content"]  # only the credential is masked


@pytest.mark.asyncio
async def test_direct_async_transport_redacts_injected_context(monkeypatch):
    compressor = _custom_endpoint_compressor()
    client = MagicMock()

    async def _create(**kwargs):
        _create.kwargs = kwargs
        return _SUMMARY

    client.chat.completions.create = _create
    monkeypatch.setattr(compressor, "_get_async_client", lambda: client)

    await compressor._generate_summary_async(
        f"injected context: please use {SECRET}", TrajectoryMetrics())

    assert SECRET not in str(_create.kwargs)


def test_redaction_copies_rather_than_mutating_the_request():
    compressor = _custom_endpoint_compressor()
    kwargs = {"model": "custom-model", "messages": [{"role": "user", "content": f"key {SECRET}"}]}

    redacted = compressor._redacted_provider_request(kwargs)

    assert SECRET not in str(redacted)
    assert kwargs["messages"][0]["content"].endswith(SECRET)


def test_redaction_failure_sends_nothing(monkeypatch):
    """An unredactable request must never reach the provider (the retry loop reports failure)."""
    from agent import redact

    compressor = _custom_endpoint_compressor()
    compressor.client = MagicMock()
    monkeypatch.setattr(
        redact, "redact_sensitive_text", lambda *_a, **_kw: (_ for _ in ()).throw(ValueError()))
    metrics = TrajectoryMetrics()

    summary = compressor._generate_summary(f"tool output {SECRET}", metrics)

    compressor.client.chat.completions.create.assert_not_called()
    assert metrics.summarization_errors == 1
    assert summary.startswith("[CONTEXT SUMMARY]:")


def test_known_provider_lane_still_routes_through_call_llm(monkeypatch):
    """Preserved behaviour: detected providers keep using the auxiliary wire boundary."""
    import agent.auxiliary_client as auxiliary_client

    compressor = _custom_endpoint_compressor()
    compressor._use_call_llm = True
    compressor._llm_provider = "openrouter"
    compressor.client = None
    seen = {}

    def _call_llm(**kwargs):
        seen.update(kwargs)
        return _SUMMARY

    monkeypatch.setattr(auxiliary_client, "call_llm", _call_llm)

    compressor._generate_summary("ordinary tool output", TrajectoryMetrics())

    assert seen["provider"] == "openrouter"
    assert seen["model"] == "custom-model"

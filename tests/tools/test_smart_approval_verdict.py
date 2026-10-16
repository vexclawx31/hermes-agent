"""Smart-approval guardian: policy mode and strict verdict parsing.

Contracts:
  * ``approvals.smart_policy_mode: replace`` drops the built-in rules but never the
    untrusted-input preamble or the one-word answer contract.
  * The guardian gets enough output budget for a reasoning model to reach a verdict
    (an empty answer at 16 tokens escalated every flagged command, #117428).
  * Verdict parsing is strict: only an exact verdict word (reply or last line) counts.
"""

from unittest.mock import MagicMock, patch

import pytest

from tools import approval_smart
from tools.approval_smart import _parse_verdict, _smart_approve


def _response(text):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


def _system_prompt(mock_call_llm):
    return mock_call_llm.call_args.kwargs["messages"][0]["content"]


@pytest.mark.parametrize("text,verdict", [
    ("APPROVE", "approve"),
    ("**DENY**", "deny"),
    ("approve.", "approve"),
    ("The command only lists files in the repo.\nAPPROVE", "approve"),
    ("ESCALATE", "escalate"),
    ("do not APPROVE", "escalate"),
    ("APPROVE or DENY", "escalate"),
    ("APPROVE\nbecause it is fine", "escalate"),
    ("", "escalate"),
])
def test_parse_verdict_is_strict(text, verdict):
    assert _parse_verdict(text) == verdict


@patch("tools.approval_context._get_approval_config")
@patch("agent.auxiliary_client.call_llm")
def test_guardian_budget_fits_reasoning_preamble(mock_call_llm, mock_cfg):
    mock_cfg.return_value = {}
    mock_call_llm.return_value = _response("Lists a directory.\nAPPROVE")
    assert _smart_approve("ls -la", "flagged") == "approve"
    assert mock_call_llm.call_args.kwargs["max_tokens"] >= 128


@patch("tools.approval_context._get_approval_config")
@patch("agent.auxiliary_client.call_llm")
def test_replace_mode_swaps_rules_but_keeps_trust_boundary(mock_call_llm, mock_cfg):
    policy = "APPROVE builds inside ~/work worktrees."
    mock_call_llm.return_value = _response("ESCALATE")

    mock_cfg.return_value = {"smart_policy": policy, "smart_policy_mode": "replace"}
    _smart_approve("make test", "flagged")
    replaced = _system_prompt(mock_call_llm)

    assert policy in replaced
    assert approval_smart._DEFAULT_RULES not in replaced
    assert replaced.startswith(approval_smart._PREAMBLE)
    assert replaced.rstrip().endswith(approval_smart._ANSWER_FORMAT)

    mock_cfg.return_value = {"smart_policy": policy}
    _smart_approve("make test", "flagged")
    appended = _system_prompt(mock_call_llm)
    assert approval_smart._DEFAULT_RULES in appended and policy in appended


@patch("tools.approval_context._get_approval_config")
@patch("agent.auxiliary_client.call_llm")
def test_unknown_policy_mode_warns_and_appends(mock_call_llm, mock_cfg):
    mock_cfg.return_value = {"smart_policy": "X", "smart_policy_mode": " Overwrite "}
    mock_call_llm.return_value = _response("ESCALATE")
    with patch("tools.approval_smart.logger") as mock_logger:
        _smart_approve("make test", "flagged")
    assert mock_logger.warning.called
    assert " Overwrite " in mock_logger.warning.call_args.args  # the raw value, as configured
    assert approval_smart._DEFAULT_RULES in _system_prompt(mock_call_llm)

"""Routing metadata is not privileged ingress authority."""

from types import SimpleNamespace
import pytest
from gateway.ingress_identity import get_trusted_ingress
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    reset_session_vars,
)
from gateway.session import SessionSource
from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.cronjob_job_args import _origin_from_env


@pytest.mark.parametrize("profile", [None, "", "alpha", False, 0])
def test_routing_equivalence_without_live_authority(profile):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123456789",
        chat_type="group",
        user_id="U123456789",
        scope_id="T123456789",
        profile=profile,
        message_id="1788265045.536689",
        thread_id="1788265045.536689",
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    tokens = runner._set_session_env(
        SimpleNamespace(source=source, session_key="fixture")
    )
    try:
        assert get_session_env("HERMES_SESSION_PROFILE") == (profile or "")
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == source.message_id
        assert get_session_env("HERMES_SESSION_THREAD_ID") == source.thread_id
        assert _origin_from_env()["thread_id"] is None
        assert get_trusted_ingress() is None
    finally:
        clear_session_vars(tokens)
    assert get_trusted_ingress() is None
    reset_session_vars()
    assert get_trusted_ingress() is None

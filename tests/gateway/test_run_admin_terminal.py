"""Authenticated admin terminal pipeline, with protected helper admission fixtures."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from gateway.runtime_context import RunEnvironment, bind_environment
from tests.tools.test_sibling_service_policy import admission, pipeline, COMMAND

@pytest.mark.parametrize('final_valid', [True, False])
def test_revalidates_after_approval(pipeline, monkeypatch, final_valid):
    from tools import run_admin_policy, run_admin_sink, terminal_tool
    checks = Mock(side_effect=[True, final_valid])
    sink = Mock(return_value=SimpleNamespace(stdout='ok', stderr='', returncode=3))
    monkeypatch.setattr(run_admin_policy, 'admit_run', checks)
    monkeypatch.setattr(run_admin_sink, 'run_helper', sink)
    scope = RunEnvironment({'PAPERCLIP_API_KEY': 'synthetic'})
    with bind_environment(scope):
        result = json.loads(terminal_tool.terminal_tool(COMMAND))
    assert checks.call_count == 2
    pipeline.approval.assert_called_once()
    if final_valid:
        sink.assert_called_once()
        assert result['exit_code'] == 3 and result['service_control']
        assert sink.call_args.kwargs['env'] == {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin', 'LANG':'C'}
    else:
        sink.assert_not_called()
        assert result['status'] == 'blocked'
        pipeline.finalizer.assert_not_called()
    scope.close()


def test_denied_approval_never_rechecks_or_executes(pipeline, monkeypatch):
    from tools import run_admin_policy, run_admin_sink, terminal_tool
    checks = Mock(return_value=True)
    sink = Mock()
    monkeypatch.setattr(run_admin_policy, 'admit_run', checks)
    monkeypatch.setattr(run_admin_sink, 'run_helper', sink)
    pipeline.approval.side_effect = terminal_tool._Rejected(json.dumps({'status':'blocked'}))
    scope = RunEnvironment({'PAPERCLIP_API_KEY':'synthetic'})
    with bind_environment(scope):
        assert json.loads(terminal_tool.terminal_tool(COMMAND))['status'] == 'blocked'
    checks.assert_called_once()
    sink.assert_not_called()
    scope.close()

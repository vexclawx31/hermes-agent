import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tools import sibling_service_policy as p
from tools import terminal_tool as t

POLICY = dict(schema=2, profile='fixture', own_target='gui/501/example.host',
              helper='/usr/local/sbin/example-service', targets={'sibling': 'system/example.sibling'})
COMMAND = '/usr/bin/sudo -n /usr/local/sbin/example-service restart sibling'

@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(p.sys, 'platform', 'darwin')
    loader = Mock(return_value=POLICY.copy())
    monkeypatch.setattr(p, 'load_policy', loader)
    monkeypatch.setattr(p, '_protected_helper', lambda path: None)
    monkeypatch.setattr(p, '_hosting_identity', lambda own: True)
    monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: 'fixture')
    monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: os.getpid())
    return loader

@pytest.fixture
def pipeline(monkeypatch, admission):
    plan = SimpleNamespace(config={}, env_type='local', cwd='/', effective_task_id='fixture',
                           effective_timeout=15, promoted_from_foreground_timeout=None)
    monkeypatch.setattr(t, '_plan_execution', lambda *a, **kw: plan)
    monkeypatch.setattr(t, '_acquire_env', lambda *a, **kw: object())
    def blocked(*a, **kw):
        raise t._Rejected('{"status":"blocked"}')
    generic = Mock(side_effect=blocked)
    monkeypatch.setattr(t, '_pre_exec_block', generic)
    approval = Mock(return_value=t._ApprovalVerdict())
    monkeypatch.setattr(t, '_run_approval_guards', approval)
    sink = Mock(return_value=SimpleNamespace(stdout='ok', stderr='', returncode=0))
    monkeypatch.setattr(subprocess, 'run', sink)
    finalizer = Mock(side_effect=lambda **kw: json.dumps({'output':kw['result']['output'], 'exit_code':kw['result']['returncode']}))
    monkeypatch.setattr(t, 'finalize_foreground_result', finalizer)
    return SimpleNamespace(plan=plan, generic=generic, approval=approval, sink=sink, finalizer=finalizer)

@pytest.mark.parametrize('op', ['status', 'restart'])
@pytest.mark.parametrize('force', [False, True])
def test_public_sink_and_approval_contract(pipeline, admission, op, force, caplog):
    with caplog.at_level('INFO'):
        result = json.loads(t.terminal_tool(COMMAND.replace('restart', op), force=force))
    assert result['service_control'] and not result['health_verified']
    pipeline.sink.assert_called_once_with(['/usr/bin/sudo', '-n', POLICY['helper'], op, 'sibling'],
        shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, cwd='/', env=p.ENV)
    pipeline.approval.assert_called_once()
    assert pipeline.approval.call_args.kwargs['force'] == force
    pipeline.generic.assert_not_called()
    admission.assert_called_once()
    pipeline.finalizer.assert_called_once()
    assert 'Sibling control finished' in caplog.text
    assert 'force' not in t.TERMINAL_SCHEMA['parameters']['properties']

@pytest.mark.parametrize('status', ['blocked', 'pending_approval'])
def test_approval_denial_never_executes(pipeline, status):
    pipeline.approval.side_effect = t._Rejected(json.dumps({'status': status}))
    assert json.loads(t.terminal_tool(COMMAND))['status'] == status
    pipeline.sink.assert_not_called()

@pytest.mark.parametrize('kwargs', [{'background': True}, {'pty': True}, {'_host_local': True}])
def test_nonforeground_and_host_override_refused(pipeline, kwargs):
    assert json.loads(t.terminal_tool(COMMAND, **kwargs))['status'] == 'blocked'
    pipeline.sink.assert_not_called()

def test_promoted_refused(pipeline):
    pipeline.plan.promoted_from_foreground_timeout = 1800
    assert json.loads(t.terminal_tool(COMMAND))['status'] == 'blocked'
    pipeline.sink.assert_not_called()

def test_admission_timeout(pipeline, monkeypatch):
    import agent.deadline as d
    original = d.run_bounded_sync
    def bounded(fn, timeout, **kw):
        if kw['label'] == 'terminal.sibling-admission':
            return SimpleNamespace(timed_out=True, value=None)
        return original(fn, timeout, **kw)
    monkeypatch.setattr(d, 'run_bounded_sync', bounded)
    assert json.loads(t.terminal_tool(COMMAND))['status'] == 'blocked'
    pipeline.sink.assert_not_called()

def test_sink_timeout_is_uncertain(pipeline):
    pipeline.sink.side_effect = subprocess.TimeoutExpired(COMMAND, 15)
    result = json.loads(t.terminal_tool(COMMAND))
    assert result['exit_code'] is None and result['timed_out'] and result['outcome'] == 'unknown'
    assert not result['health_verified']
    pipeline.sink.assert_called_once()
    pipeline.finalizer.assert_not_called()

@pytest.mark.parametrize('command', [COMMAND+';', COMMAND+' ', COMMAND.replace(' sibling', ' unknown'),
    COMMAND.replace(' -n', ''), COMMAND.replace('/usr/local/sbin/', ''),
    '/bin/launchctl kickstart -k gui/501/example.host', COMMAND+'\ntrue', COMMAND.replace(' sibling', ' example.host')])
def test_malformed_and_unknown_use_original_guard(pipeline, command):
    assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    pipeline.generic.assert_called_once()
    pipeline.sink.assert_not_called()

@pytest.mark.parametrize('failure', ['policy', 'profile', 'pid', 'helper', 'ancestry'])
def test_identity_and_policy_failure_preserve_guard(pipeline, monkeypatch, failure):
    if failure == 'policy':
        monkeypatch.setattr(p, 'load_policy', Mock(side_effect=FileNotFoundError))
    elif failure == 'profile':
        monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: 'other')
    elif failure == 'pid':
        monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: -1)
    elif failure == 'helper':
        monkeypatch.setattr(p, '_protected_helper', Mock(side_effect=ValueError))
    else:
        monkeypatch.setattr(p, '_hosting_identity', lambda own: False)
    assert json.loads(t.terminal_tool(COMMAND, force=True))['status'] == 'blocked'
    pipeline.sink.assert_not_called()
    pipeline.generic.assert_called_once()

@pytest.mark.parametrize('mode', ['remote', 'platform', 'ordinary'])
def test_cheap_filter_skips_admission_worker(pipeline, monkeypatch, mode):
    import agent.deadline as d
    original = d.run_bounded_sync
    labels = []
    def bounded(fn, timeout, **kw):
        labels.append(kw['label'])
        return original(fn, timeout, **kw)
    monkeypatch.setattr(d, 'run_bounded_sync', bounded)
    if mode == 'remote':
        pipeline.plan.env_type = 'ssh'
    elif mode == 'platform':
        monkeypatch.setattr(p.sys, 'platform', 'linux')
    t.terminal_tool('printf ok' if mode == 'ordinary' else COMMAND)
    assert labels == ['terminal.pre-exec-guard']
    pipeline.sink.assert_not_called()

@pytest.mark.parametrize('change', [ {'own_target':'gui/0501/example.host'},
    {'targets':{'sibling':'system/example.host'}}, {'targets':{'-bad':'system/other'}},
    {'helper':'relative/path'}, {'helper':'/usr/../bin/thing'}, {'schema':True},
    {'targets':{'a':'system/other','b':'system/other'}}])
def test_policy_rejects_aliases_and_self(change):
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY, **change})

def test_policy_generic_profile_and_request():
    assert p.validate_policy(POLICY) == POLICY
    assert p._request(POLICY, 'sibling', 'status')[-2:] == ['status', 'sibling']

@pytest.mark.parametrize('case', ['wrapper', 'direct', 'wrong', 'nested', 'duplicate', 'uid', 'failed', 'cycle', 'depth'])
def test_full_hosting_identity(monkeypatch, case):
    monkeypatch.setattr(os, 'getuid', lambda: 501)
    monkeypatch.setattr(os, 'getpid', lambda: 1048)
    parents = {1048:1037, 1037:884, 884:1}
    job = 884
    if case == 'direct':
        job=1048; parents={1048:1}
    if case == 'wrong': job=999
    if case == 'cycle': parents[1037]=1048
    if case == 'depth': parents={i:i-1 for i in range(1048,1038,-1)}
    output = f'gui/501/example.host = {{\n\tpid = {job}\n\tresources = {{\n\t\tpid = 999\n\t}}\n}}'
    if case == 'nested': output='job = {\n\tresources = {\n\t\tpid = 884\n\t}\n}'
    if case == 'duplicate': output+='\n\tpid = 884'
    run=Mock(return_value=SimpleNamespace(returncode=1 if case=='failed' else 0, stdout=output))
    monkeypatch.setattr(subprocess, 'run', run)
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(ppid=lambda:parents[pid]))
    assert p._hosting_identity('gui/502/example.host' if case=='uid' else POLICY['own_target']) == (case in {'wrapper','direct'})
    if case != 'uid':
        run.assert_called_once_with(['/bin/launchctl','print', POLICY['own_target']],shell=False,
                                    capture_output=True,text=True,timeout=3,cwd='/',env=p.ENV)

def test_protected_file_loader_rejects_user_file(tmp_path):
    path=tmp_path/'policy.json'; path.write_text(json.dumps(POLICY)); path.chmod(0o444)
    with pytest.raises(ValueError): p.load_policy(path)

def test_real_approval_denial_and_internal_force(monkeypatch):
    monkeypatch.setattr(t, '_check_all_guards', lambda *a, **kw: {'approved':False})
    with pytest.raises(t._Rejected): t._run_approval_guards(COMMAND,'local',{},force=False)
    assert t._run_approval_guards(COMMAND,'local',{},force=True).approved_run

@pytest.mark.parametrize('rc', [0, 1, 2])
def test_real_finalizer_preserves_helper_exit(pipeline, monkeypatch, rc):
    from tools.terminal_tool_result import finalize_foreground_result
    monkeypatch.setattr(t, 'finalize_foreground_result', finalize_foreground_result)
    pipeline.sink.return_value = SimpleNamespace(stdout='synthetic output', stderr='', returncode=rc)
    assert json.loads(t.terminal_tool(COMMAND))['exit_code'] == rc

@pytest.mark.parametrize('force', [False, True])
def test_credential_run_refused_before_admission(pipeline, admission, force):
    from gateway.runtime_context import bind_environment, validate_environment
    scope = validate_environment({'PAPERCLIP_API_KEY': 'synthetic-sibling-unit-12345'},
        {'gateway': {'api_server': {'run_environment_allowlist': ['PAPERCLIP_API_KEY']}}}, 'local')
    with bind_environment(scope):
        result = json.loads(t.terminal_tool(COMMAND, force=force))
        # Non-candidates keep the ordinary guarded path under the same scope.
        assert json.loads(t.terminal_tool('printf ok'))['status'] == 'blocked'
    assert result['status'] == 'blocked' and 'credential runs' in result['error']
    admission.assert_not_called()
    pipeline.approval.assert_not_called()
    pipeline.sink.assert_not_called()
    pipeline.generic.assert_called_once()

def test_workdir_refused(pipeline):
    assert json.loads(t.terminal_tool(COMMAND, workdir='/'))['status'] == 'blocked'
    pipeline.sink.assert_not_called()

def test_real_identity_api_signatures():
    import inspect
    from gateway.status import get_running_pid
    from hermes_cli.profiles import get_active_profile_name
    inspect.signature(get_running_pid).bind(cleanup_stale=False)
    inspect.signature(get_active_profile_name).bind()


# ---- schema 3 rollout registry ----

import copy

PLAN = 'a' * 64
RECEIPT = 'b' * 64
POLICY3 = {**POLICY, 'schema': 3,
           'rollout': {'targets': {'one': 'gui/501/example.one', 'two': 'gui/501/example.two'}, 'timeout_seconds': 1500}}
HELPER = '/usr/bin/sudo -n /usr/local/sbin/example-service '
APPLY = HELPER + 'apply ' + PLAN + ' one,two'
ROLLBACK = HELPER + 'rollback ' + PLAN + ' ' + RECEIPT + ' two,one'


@pytest.fixture
def rollout(pipeline, admission):
    # Direct terminal_tool calls get an exactly adequate executor budget; executor wiring is tested below.
    from agent.deadline import ToolBudget, tool_budget
    admission.return_value = copy.deepcopy(POLICY3)
    with tool_budget(ToolBudget(1500 + t.SIBLING_ROLLOUT_OUTER_MARGIN_S)):
        yield pipeline


@pytest.mark.parametrize('command,argv', [
    (APPLY, ['apply', PLAN, 'one,two']),
    (ROLLBACK, ['rollback', PLAN, RECEIPT, 'two,one']),
    (HELPER + 'apply ' + PLAN + ' two', ['apply', PLAN, 'two'])])
def test_rollout_admitted_with_protected_timeout_and_normal_approval(rollout, command, argv):
    result = json.loads(t.terminal_tool(command, timeout=15))
    rollout.sink.assert_called_once_with(['/usr/bin/sudo', '-n', POLICY['helper'], *argv],
        shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1500, cwd='/', env=p.ENV)
    rollout.approval.assert_called_once()
    assert rollout.approval.call_args.kwargs['force'] is False
    rollout.generic.assert_not_called()
    rollout.finalizer.assert_called_once()
    assert result['service_control'] and not result['health_verified']


def test_schema3_keeps_status_restart_scope_and_timeout(rollout):
    json.loads(t.terminal_tool(COMMAND))
    assert rollout.sink.call_args.args[0] == ['/usr/bin/sudo', '-n', POLICY['helper'], 'restart', 'sibling']
    assert rollout.sink.call_args.kwargs['timeout'] == 15
    rollout.sink.reset_mock()
    # A rollout-registry name is not restartable, and a restart target is not rollout-eligible.
    for command in (HELPER + 'restart one', HELPER + 'apply ' + PLAN + ' sibling'):
        assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


@pytest.mark.parametrize('command', [
    HELPER + 'apply ' + PLAN + ' one,unknown',          # mixed set
    HELPER + 'apply ' + PLAN + ' unknown',              # unknown
    HELPER + 'apply ' + PLAN + ' host',                 # self by name
    HELPER + 'apply ' + PLAN + ' one,one',              # duplicate
    HELPER + 'apply ' + PLAN.upper() + ' one',          # malformed digest
    HELPER + 'apply ' + PLAN[:-1] + ' one',
    HELPER + 'rollback ' + PLAN + ' one',               # missing receipt digest
    HELPER + 'apply ' + PLAN + ' ' + RECEIPT + ' one',  # extra digest
    HELPER + 'apply ' + PLAN + ' one,',
    HELPER + 'apply ' + PLAN + ' one, two',
    HELPER + 'apply ' + PLAN + ' one;true',
    HELPER + 'apply ' + PLAN + ' one\ntrue',
    HELPER + 'activate ' + PLAN + ' one',
    '/usr/bin/sudo -n /usr/local/sbin/other-helper apply ' + PLAN + ' one'])
def test_rollout_refusals_use_original_guard(rollout, command):
    assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


def test_self_in_rollout_registry_refused_at_validation_and_request():
    with pytest.raises(ValueError, match='hosting label'):
        p.validate_policy({**POLICY3, 'rollout': {'targets': {'host': 'gui/501/example.host'}, 'timeout_seconds': 600}})
    forged = copy.deepcopy(POLICY3)
    forged['rollout']['targets']['host'] = 'gui/502/example.host'  # bypasses validation: same label, other domain
    with pytest.raises(ValueError, match='hosting service'):
        p._rollout_request(forged, 'apply', [PLAN], ['one', 'host'])


def test_schema2_policy_never_admits_rollout(pipeline):
    for command in (APPLY, ROLLBACK):
        assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    pipeline.sink.assert_not_called()
    assert pipeline.generic.call_count == 2


@pytest.mark.parametrize('status', ['blocked', 'pending_approval'])
def test_rollout_denial_and_pending_never_execute(rollout, status):
    rollout.approval.side_effect = t._Rejected(json.dumps({'status': status}))
    assert json.loads(t.terminal_tool(APPLY))['status'] == status
    rollout.sink.assert_not_called()


@pytest.mark.parametrize('kwargs', [{'background': True}, {'pty': True}, {'workdir': '/'}, {'_host_local': True}])
def test_rollout_requires_plain_foreground(rollout, kwargs):
    assert json.loads(t.terminal_tool(APPLY, **kwargs))['status'] == 'blocked'
    rollout.sink.assert_not_called()


def test_rollout_refused_in_credential_run_before_admission(rollout, admission):
    from gateway.runtime_context import bind_environment, validate_environment
    scope = validate_environment({'PAPERCLIP_API_KEY': 'synthetic-rollout-unit-12345'},
        {'gateway': {'api_server': {'run_environment_allowlist': ['PAPERCLIP_API_KEY']}}}, 'local')
    with bind_environment(scope):
        result = json.loads(t.terminal_tool(APPLY))
    assert result['status'] == 'blocked' and 'credential runs' in result['error']
    admission.assert_not_called()
    rollout.approval.assert_not_called()
    rollout.sink.assert_not_called()


def test_rollout_timeout_is_unknown_without_retry(rollout):
    rollout.sink.side_effect = subprocess.TimeoutExpired(APPLY, 1500)
    result = json.loads(t.terminal_tool(APPLY))
    assert result['outcome'] == 'unknown' and result['exit_code'] is None and not result['health_verified']
    rollout.sink.assert_called_once()
    rollout.finalizer.assert_not_called()


@pytest.mark.parametrize('rc', [0, 1, 2])
def test_rollout_real_finalizer_preserves_operator_exit(rollout, monkeypatch, rc):
    from tools.terminal_tool_result import finalize_foreground_result
    monkeypatch.setattr(t, 'finalize_foreground_result', finalize_foreground_result)
    rollout.sink.return_value = SimpleNamespace(stdout='{"status":"stopped-for-review"}', stderr='', returncode=rc)
    assert json.loads(t.terminal_tool(APPLY))['exit_code'] == rc


@pytest.mark.parametrize('failure', ['profile', 'pid', 'helper', 'ancestry'])
def test_rollout_identity_failure_preserves_guard(rollout, monkeypatch, failure):
    if failure == 'profile':
        monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: 'other')
    elif failure == 'pid':
        monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: -1)
    elif failure == 'helper':
        monkeypatch.setattr(p, '_protected_helper', Mock(side_effect=ValueError))
    else:
        monkeypatch.setattr(p, '_hosting_identity', lambda own: False)
    assert json.loads(t.terminal_tool(APPLY, force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


@pytest.mark.parametrize('change', [
    {'rollout': None}, {'rollout': {'targets': {'one': 'gui/501/example.one'}}},
    {'rollout': {'targets': {'one': 'gui/501/example.one'}, 'timeout_seconds': True}},
    {'rollout': {'targets': {'one': 'gui/501/example.one'}, 'timeout_seconds': 59}},
    {'rollout': {'targets': {'one': 'gui/501/example.one'}, 'timeout_seconds': 3601}},
    {'rollout': {'targets': {}, 'timeout_seconds': 600}},
    {'rollout': {'targets': {'sibling': 'gui/501/example.one'}, 'timeout_seconds': 600}},   # name also restartable
    {'rollout': {'targets': {'one': 'system/example.sibling'}, 'timeout_seconds': 600}},   # service also restartable
    {'rollout': {'targets': {'a': 'gui/501/x', 'b': 'gui/501/x'}, 'timeout_seconds': 600}},
    {'rollout': {'targets': {'one': 'gui/501/example.one'}, 'timeout_seconds': 600, 'extra': 1}}])
def test_schema3_policy_rejections(change):
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY3, **change})


def test_schema_and_rollout_key_must_agree():
    assert p.validate_policy(copy.deepcopy(POLICY3))['schema'] == 3
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY, 'rollout': POLICY3['rollout']})
    with pytest.raises(ValueError):
        p.validate_policy({k: v for k, v in POLICY3.items() if k != 'rollout'})


# ---- outer executor deadline must contain the protected rollout bound ----

import concurrent.futures
import threading
import time

from agent.deadline import ToolBudget, current_tool_budget, tool_budget

NEEDED = 1500 + t.SIBLING_ROLLOUT_OUTER_MARGIN_S


@pytest.mark.parametrize('budget', [None, ToolBudget(420), ToolBudget(NEEDED - 1)])
def test_rollout_refused_without_an_outer_budget_that_contains_it(pipeline, admission, budget):
    admission.return_value = copy.deepcopy(POLICY3)
    if budget is None:
        result = json.loads(t.terminal_tool(APPLY))   # unknown bound: refuse
    else:
        with tool_budget(budget):
            result = json.loads(t.terminal_tool(APPLY))
    assert result['status'] == 'blocked' and 'outer tool deadline' in result['error']
    assert str(NEEDED) in result['error']
    pipeline.approval.assert_not_called()   # no prompt for a call that could not finish
    pipeline.sink.assert_not_called()
    pipeline.generic.assert_not_called()


@pytest.mark.parametrize('budget', [ToolBudget(NEEDED), ToolBudget(None), ToolBudget(0)])
def test_rollout_admitted_with_adequate_or_disabled_outer_deadline(pipeline, admission, budget):
    admission.return_value = copy.deepcopy(POLICY3)
    with tool_budget(budget):
        json.loads(t.terminal_tool(APPLY))
    pipeline.approval.assert_called_once()
    assert pipeline.sink.call_args.kwargs['timeout'] == 1500


def test_rollout_uses_remaining_not_nominal_budget(pipeline, admission):
    admission.return_value = copy.deepcopy(POLICY3)
    late = ToolBudget(4000)
    late.arm(4000, time.monotonic() + NEEDED - 30, lambda: 0.0)   # a late start in a shared batch deadline
    with tool_budget(late):
        assert json.loads(t.terminal_tool(APPLY))['status'] == 'blocked'
    pipeline.sink.assert_not_called()
    waited = ToolBudget(4000)
    waited.arm(4000, time.monotonic() + NEEDED - 30, lambda: 60.0)  # approval waits extend the executor deadline
    with tool_budget(waited):
        json.loads(t.terminal_tool(APPLY))
    pipeline.sink.assert_called_once()


@pytest.mark.parametrize('via_kernel', [False, True])
def test_execute_code_rpc_cannot_inherit_rollout_budget(pipeline, admission, monkeypatch, via_kernel):
    import model_tools
    from tools.code_execution_rpc import _default_dispatch
    from tools.code_kernel import CellAuthority

    admission.return_value = copy.deepcopy(POLICY3)
    monkeypatch.setattr(model_tools, 'handle_function_call',
                        lambda name, args, **kw: t.terminal_tool(**args))
    outer = ToolBudget(NEEDED)
    with tool_budget(outer):
        dispatch = CellAuthority('rpc-budget-test').dispatch if via_kernel else _default_dispatch('rpc-budget-test')
        result = json.loads(dispatch('terminal', {'command': APPLY}))
        assert current_tool_budget() is outer  # clearing is scoped, not global
    assert result['status'] == 'blocked' and 'outer tool deadline' in result['error']
    pipeline.approval.assert_not_called()
    pipeline.sink.assert_not_called()
    pipeline.generic.assert_not_called()


def test_status_restart_need_no_outer_budget(pipeline):
    assert current_tool_budget() is None
    json.loads(t.terminal_tool(COMMAND))
    pipeline.sink.assert_called_once()


class _Agent:
    """The attributes the real executors touch; nothing tool-specific."""

    def __init__(self):
        self._tool_worker_threads_lock = threading.Lock()
        self._tool_worker_threads = set()
        self._interrupt_requested = False
        self.log_prefix = ''

    def _touch_activity(self, *args):
        pass

    def interrupt(self, *args):
        pass

    def _vprint(self, *args, **kwargs):
        pass


@pytest.fixture
def executors(rollout, monkeypatch):
    """Real sequential / concurrent executors with real config resolution; the middleware hop that would
    reach the registry calls terminal_tool directly. The rollout fixture's budget is removed."""
    from agent import tool_executor as te
    from agent import deadline as d
    token = d._TOOL_BUDGET.set(None)
    config = {}
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: config)
    monkeypatch.delenv('HERMES_CONCURRENT_TOOL_TIMEOUT_S', raising=False)

    def middleware(agent, *, function_args, **kwargs):
        return te._ManagedToolResult(result=t.terminal_tool(function_args['command']), args=function_args,
                                     middleware_trace=[], blocked=False, dispatched=True)
    monkeypatch.setattr(te, '_run_agent_tool_execution_middleware', middleware)
    yield SimpleNamespace(te=te, config=config, pipeline=rollout)
    d._TOOL_BUDGET.reset(token)


def _sequential(te, command):
    return json.loads(te._run_sequential_tool_execution_middleware(
        _Agent(), function_name='terminal', function_args={'command': command}, effective_task_id='task',
        tool_call_id='call-1', execute=lambda args: None).result)


def _concurrent(te, commands):
    calls = [te._ParsedCall(SimpleNamespace(id=f'call-{i}', function=SimpleNamespace(name='terminal', arguments='{}')),
                            'terminal', {'command': c}, [], None, None) for i, c in enumerate(commands)]
    batch = te._ConcurrentBatch(_Agent(), [], 'task', calls, te._resolve_concurrent_tool_timeout())
    batch.run()
    return [json.loads(r.result) for r in batch.results]


def test_real_sequential_executor_default_deadline_refuses_rollout(executors):
    assert executors.te._resolve_sequential_tool_timeout() == 420.0
    result = _sequential(executors.te, APPLY)
    assert result['status'] == 'blocked' and 'available: 4' in result['error']
    executors.pipeline.approval.assert_not_called()
    executors.pipeline.sink.assert_not_called()


def test_real_sequential_executor_configured_deadline_admits_rollout(executors):
    executors.config['timeouts'] = {'tools': {'sequential_call': NEEDED + 30, 'concurrent_batch': NEEDED + 30}}
    _sequential(executors.te, APPLY)
    executors.pipeline.approval.assert_called_once()
    executors.pipeline.sink.assert_called_once()


def test_real_concurrent_executor_publishes_its_shared_deadline(executors):
    results = _concurrent(executors.te, [APPLY, COMMAND])
    assert results[0]['status'] == 'blocked' and 'available: 4' in results[0]['error']
    assert [c.args[0][3] for c in executors.pipeline.sink.call_args_list] == ['restart']   # status/restart unaffected
    executors.pipeline.sink.reset_mock()
    executors.config['timeouts'] = {'tools': {'concurrent_batch': NEEDED + 30}}
    _concurrent(executors.te, [APPLY])
    executors.pipeline.sink.assert_called_once()


def test_real_prepared_terminal_slot_publishes_and_arms_budget(executors, monkeypatch):
    from agent import terminal_approval_batch as tab
    te = executors.te
    seen = []
    monkeypatch.setattr(te, '_resolve_sequential_dispatch', lambda agent, ref, messages: SimpleNamespace(execute=None))
    monkeypatch.setattr(te, '_run_agent_tool_execution_middleware',
                        lambda agent, **kw: seen.append(current_tool_budget()) or te._ManagedToolResult('ok', {}, [], False, True))
    agent = _Agent()
    parsed = te._ParsedCall(SimpleNamespace(id='call-1', function=SimpleNamespace(name='terminal', arguments='{}')),
                            'terminal', {'command': APPLY}, [], None, None)
    batch = SimpleNamespace(agent=agent, task_id='task', messages=[], cancelled=threading.Event(),
                            authorization_gate=te._ConcurrentToolAuthorizationGate())
    slot = tab._TerminalSlot(batch, parsed, 0)
    slot.budget = ToolBudget(420)
    slot.run()
    assert seen == [slot.budget]
    # The sequential runner arms the prepared worker's published budget with the deadline it enforces.
    future = concurrent.futures.Future()
    future.set_result(te._ManagedToolResult('ok', {}, [], False, True))
    prepared = SimpleNamespace(batch=SimpleNamespace(authorization_gate=batch.authorization_gate, executor=None),
                               tids=[], future=future, budget=ToolBudget(420))
    monkeypatch.setattr(tab, 'take_prepared_call', lambda call_id: prepared)
    executors.config['timeouts'] = {'tools': {'sequential_call': 700}}
    te._run_sequential_tool_execution_middleware(agent, function_name='terminal', function_args={'command': APPLY},
                                                 effective_task_id='task', tool_call_id='call-1', execute=None)
    assert prepared.budget.timeout_s == 700 and 690 < prepared.budget.remaining() <= 700


SHARED = {**POLICY3, 'rollout': {'targets': {'one': 'gui/501/example.one', 'sibling': 'system/example.sibling'},
                                  'timeout_seconds': 1500}}


def test_identical_overlap_is_valid_and_admits_both_operations(rollout, admission):
    assert p.validate_policy(copy.deepcopy(SHARED))['rollout']['targets']['sibling'] == POLICY['targets']['sibling']
    admission.return_value = copy.deepcopy(SHARED)
    json.loads(t.terminal_tool(COMMAND))                                    # restart sibling: unchanged scope
    json.loads(t.terminal_tool(HELPER + 'apply ' + PLAN + ' one,sibling'))   # the same service, now rollout-eligible
    assert [c.args[0][3:] for c in rollout.sink.call_args_list] == [['restart', 'sibling'], ['apply', PLAN, 'one,sibling']]
    assert rollout.approval.call_count == 2


@pytest.mark.parametrize('rollout_targets', [
    {'sibling': 'gui/501/example.other'},                                     # same name, different service
    {'alias': 'system/example.sibling'},                                      # same service, different name
    {'sibling': 'system/example.sibling', 'alias': 'system/example.sibling'},  # one service, two names
    {'sibling': 'system/example.sibling', 'host': 'gui/501/example.host'}])    # hosting label still refused
def test_conflicting_overlap_refused(rollout_targets):
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY3, 'rollout': {'targets': rollout_targets, 'timeout_seconds': 600}})


def test_shared_name_does_not_widen_mixed_or_unknown_sets(rollout, admission):
    admission.return_value = copy.deepcopy(SHARED)
    for command in (HELPER + 'apply ' + PLAN + ' sibling,unknown', HELPER + 'apply ' + PLAN + ' sibling,host',
                    HELPER + 'apply ' + PLAN + ' sibling,sibling', HELPER + 'restart one'):
        assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


# ---- schema 3 optional consumers list ----

CONSUMED = {**POLICY3, 'consumers': {'names': ['viewer', 'tool-cli'], 'timeout_seconds': 1500}}


@pytest.mark.parametrize('command,argv', [
    (HELPER + 'consumers-apply ' + PLAN + ' viewer,tool-cli', ['consumers-apply', PLAN, 'viewer,tool-cli']),
    (HELPER + 'consumers-rollback ' + PLAN + ' ' + RECEIPT + ' tool-cli', ['consumers-rollback', PLAN, RECEIPT, 'tool-cli'])])
def test_consumer_requests_admitted_with_protected_timeout(rollout, admission, command, argv):
    admission.return_value = copy.deepcopy(CONSUMED)
    result = json.loads(t.terminal_tool(command))
    rollout.sink.assert_called_once_with(['/usr/bin/sudo', '-n', POLICY['helper'], *argv],
        shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1500, cwd='/', env=p.ENV)
    rollout.approval.assert_called_once()
    assert result['service_control'] and not result['health_verified']


@pytest.mark.parametrize('command', [
    HELPER + 'consumers-apply ' + PLAN + ' viewer,unknown',      # mixed
    HELPER + 'consumers-apply ' + PLAN + ' one',                 # a rollout name is not a consumer
    HELPER + 'apply ' + PLAN + ' viewer',                        # a consumer name is not a rollout target
    HELPER + 'restart viewer',                                   # nor restartable
    HELPER + 'consumers-apply ' + PLAN + ' viewer,viewer',
    HELPER + 'consumers-apply ' + PLAN + ' fixture',             # the hosting profile
    HELPER + 'consumers-rollback ' + PLAN + ' viewer',           # missing receipt digest
    HELPER + 'consumers-apply ' + PLAN.upper() + ' viewer'])
def test_consumer_refusals_use_original_guard(rollout, admission, command):
    admission.return_value = copy.deepcopy(CONSUMED)
    assert json.loads(t.terminal_tool(command, force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


def test_policies_without_consumers_never_admit_consumer_requests(rollout, admission):
    for policy in (POLICY3, POLICY):
        admission.return_value = copy.deepcopy(policy)
        assert json.loads(t.terminal_tool(HELPER + 'consumers-apply ' + PLAN + ' viewer', force=True))['status'] == 'blocked'
    rollout.sink.assert_not_called()


@pytest.mark.parametrize('status', ['blocked', 'pending_approval'])
def test_consumer_denial_and_pending_never_execute(rollout, admission, status):
    admission.return_value = copy.deepcopy(CONSUMED)
    rollout.approval.side_effect = t._Rejected(json.dumps({'status': status}))
    assert json.loads(t.terminal_tool(HELPER + 'consumers-apply ' + PLAN + ' viewer'))['status'] == status
    rollout.sink.assert_not_called()


def test_consumer_request_needs_an_outer_budget_and_refuses_credential_runs(pipeline, admission):
    admission.return_value = copy.deepcopy(CONSUMED)
    command = HELPER + 'consumers-apply ' + PLAN + ' viewer'
    assert 'outer tool deadline' in json.loads(t.terminal_tool(command))['error']          # unknown budget
    from gateway.runtime_context import bind_environment, validate_environment
    scope = validate_environment({'PAPERCLIP_API_KEY': 'synthetic-consumer-unit-12345'},
        {'gateway': {'api_server': {'run_environment_allowlist': ['PAPERCLIP_API_KEY']}}}, 'local')
    with bind_environment(scope), tool_budget(ToolBudget(None)):
        assert 'credential runs' in json.loads(t.terminal_tool(command))['error']
    pipeline.approval.assert_not_called()
    pipeline.sink.assert_not_called()


@pytest.mark.parametrize('consumers', [
    {'names': ['viewer'], 'timeout_seconds': 59},
    {'names': ['viewer']},
    {'names': [], 'timeout_seconds': 600},
    {'names': ['viewer', 'viewer'], 'timeout_seconds': 600},
    {'names': ['Viewer'], 'timeout_seconds': 600},
    {'names': ['sibling'], 'timeout_seconds': 600},     # restart name
    {'names': ['one'], 'timeout_seconds': 600},         # rollout name
    {'names': ['fixture'], 'timeout_seconds': 600},     # hosting profile
    {'names': ['viewer'], 'timeout_seconds': 600, 'services': {}}])
def test_consumers_list_rejections(consumers):
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY3, 'consumers': consumers})


def test_consumers_key_is_schema3_only_and_optional():
    assert p.validate_policy(copy.deepcopy(CONSUMED))['consumers']['names'] == ['viewer', 'tool-cli']
    assert p.validate_policy(copy.deepcopy(POLICY3)) and p.validate_policy(copy.deepcopy(POLICY))
    with pytest.raises(ValueError):
        p.validate_policy({**POLICY, 'consumers': CONSUMED['consumers']})


def test_canonical_consumer_request_binds_runtime_identity(monkeypatch):
    monkeypatch.setattr(p, 'load_policy', lambda path: copy.deepcopy(CONSUMED))
    request = p.canonical_consumer_request('/fixture', runtime_profile='fixture', runtime_service=POLICY['own_target'],
                                           operation='consumers-apply', digests=[PLAN], names=['viewer'])
    assert ' '.join(request) == HELPER + 'consumers-apply ' + PLAN + ' viewer' and request.timeout == 1500
    with pytest.raises(ValueError, match='identity'):
        p.canonical_consumer_request('/fixture', runtime_profile='other', runtime_service=POLICY['own_target'],
                                     operation='consumers-apply', digests=[PLAN], names=['viewer'])


def test_canonical_rollout_request_binds_runtime_identity(monkeypatch):
    monkeypatch.setattr(p, 'load_policy', lambda path: copy.deepcopy(POLICY3))
    request = p.canonical_rollout_request('/fixture', runtime_profile='fixture', runtime_service=POLICY['own_target'],
                                          operation='rollback', digests=[PLAN, RECEIPT], names=['two', 'one'])
    assert ' '.join(request) == ROLLBACK and request.timeout == 1500
    with pytest.raises(ValueError, match='identity'):
        p.canonical_rollout_request('/fixture', runtime_profile='other', runtime_service=POLICY['own_target'],
                                    operation='apply', digests=[PLAN], names=['one'])

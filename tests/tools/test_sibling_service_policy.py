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

def test_workdir_refused(pipeline):
    assert json.loads(t.terminal_tool(COMMAND, workdir='/'))['status'] == 'blocked'
    pipeline.sink.assert_not_called()

def test_real_identity_api_signatures():
    import inspect
    from gateway.status import get_running_pid
    from hermes_cli.profiles import get_active_profile_name
    inspect.signature(get_running_pid).bind(cleanup_stale=False)
    inspect.signature(get_active_profile_name).bind()

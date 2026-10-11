import base64
import json
import time
import subprocess
from types import SimpleNamespace
import pytest
from gateway.runtime_context import RunEnvironment
from tools import run_admin_policy as policy

RUN='11111111-1111-4111-8111-111111111111'
TASK='22222222-2222-4222-8222-222222222222'

@pytest.fixture
def admitted(monkeypatch):
    pin=dict(schema=1,api_base='http://127.0.0.1:1234',agent_id='admin',company_id='company',profile='default')
    claims=dict(sub='admin',company_id='company',run_id=RUN,adapter_type='hermes_gateway',exp=time.time()+60)
    encode=lambda c: 'header.'+base64.urlsafe_b64encode(json.dumps(c).encode()).decode().rstrip('=')+'.signature'
    scope=RunEnvironment(dict(PAPERCLIP_API_KEY=encode(claims),PAPERCLIP_API_URL=pin['api_base'],PAPERCLIP_RUN_ID=RUN,PAPERCLIP_TASK_ID=TASK,PAPERCLIP_AGENT_ID='admin',PAPERCLIP_COMPANY_ID='company'))
    replies={
      '/api/agents/me':dict(id='admin',companyId='company',adapterType='hermes_gateway'),
      '/api/heartbeat-runs/'+RUN:dict(id=RUN,agentId='admin',companyId='company',status='running',contextSnapshot={'issueId':TASK}),
      '/api/issues/'+TASK:dict(id=TASK,companyId='company',checkoutRunId=RUN,executionRunId=RUN,assigneeAgentId='admin',status='in_progress')}
    monkeypatch.setattr(policy,'load_policy',lambda:pin)
    monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name',lambda:'default')
    monkeypatch.setattr(policy,'_get',lambda base,route,key:replies[route])
    return scope,replies,claims,encode

def test_authorized(admitted):
    assert policy.admit_run(admitted[0])

@pytest.mark.parametrize('field,value',[('sub','tenant'),('company_id','wrong'),('run_id',TASK),('exp',0),('adapter_type','claude_local')])
def test_bad_claims(admitted,field,value):
    scope,replies,claims,encode=admitted
    claims[field]=value;scope.values['PAPERCLIP_API_KEY']=encode(claims)
    assert not policy.admit_run(scope)

@pytest.mark.parametrize('kind,field,value',[
 ('agent','id','tenant'),('agent','trustPreset','low_trust_review'),
 ('run','status','succeeded'),('run','companyId','wrong'),
 ('issue','checkoutRunId',TASK),('issue','status','blocked'),('issue','assigneeAgentId','tenant')])
def test_server_mismatch(admitted,kind,field,value):
    scope,replies,*_=admitted
    route={'agent':'/api/agents/me','run':'/api/heartbeat-runs/'+RUN,'issue':'/api/issues/'+TASK}[kind]
    replies[route][field]=value
    assert not policy.admit_run(scope)

def test_closed_and_spoofed_origin(admitted):
    scope,*_=admitted
    scope.values['PAPERCLIP_API_URL']='http://attacker.invalid'
    assert not policy.admit_run(scope)
    scope.close();assert not policy.admit_run(scope)

def test_helper_tracks_lifetime_without_credentials():
    import sys
    from tools.run_admin_sink import run_helper
    scope=RunEnvironment({'PAPERCLIP_API_KEY':'synthetic-secret'})
    result=run_helper(scope,[sys.executable,'-c','import os,sys; print("PAPERCLIP_API_KEY" in os.environ); sys.exit(3)'],timeout=5,capture_output=True,text=True,env={'PATH':'/usr/bin:/bin'},cwd='/')
    assert result.returncode==3 and result.stdout.strip()=='False'
    assert not scope._processes
    scope.close()

@pytest.mark.parametrize('final_now', [160.0, 161.0])
def test_expiry_during_server_validation_denied(admitted, monkeypatch, final_now):
    scope, replies, claims, encode = admitted
    claims['exp'] = 160.0
    scope.values['PAPERCLIP_API_KEY'] = encode(claims)
    clock = [100.0]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    def get(base, route, key):
        if route == '/api/issues/' + TASK:
            clock[0] = final_now
        return replies[route]
    monkeypatch.setattr(policy, '_get', get)
    assert not policy.admit_run(scope)

@pytest.mark.parametrize('expiry', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_expiry_denied(admitted, expiry):
    scope, replies, claims, encode = admitted
    claims['exp'] = expiry
    scope.values['PAPERCLIP_API_KEY'] = encode(claims)
    assert not policy.admit_run(scope)

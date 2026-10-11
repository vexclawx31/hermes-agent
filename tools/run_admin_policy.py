"""Opt-in authenticated run admission for the existing protected admin sink.

No environment-variable or model-provided identity authorizes administration.
The API origin and principal are pinned in root-owned read-only policy.
"""
import json
import base64
import math
import os
import stat
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit

POLICY_PATH = '/Library/Hermes/managed/run-admin.json'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def load_policy():
    from tools.sibling_service_policy import load_policy as unused
    path = Path(POLICY_PATH)
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
            st = os.fstat(directory)
            if st.st_uid != 0 or st.st_mode & 0o022:
                raise ValueError('unsafe admin policy ancestor')
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1 or st.st_mode & 0o222 or st.st_size > 8192:
            raise ValueError('unsafe admin policy')
        with os.fdopen(fd, 'r', closefd=False) as stream:
            policy = json.load(stream)
    finally:
        os.close(fd)
    if (not isinstance(policy, dict) or set(policy) != {'schema','api_base','agent_id','company_id','profile'}
            or type(policy['schema']) is not int or policy['schema'] != 1
            or any(not isinstance(policy[k], str) or not policy[k] for k in ('api_base','agent_id','company_id','profile'))):
        raise ValueError('invalid admin policy')
    u = urlsplit(policy['api_base'])
    if (u.scheme not in {'http','https'} or not u.hostname or u.username or u.password
            or u.query or u.fragment or u.path not in ('','/api')
            or u.scheme == 'http' and u.hostname not in {'localhost','127.0.0.1','::1'}):
        raise ValueError('unsafe admin API origin')
    return policy

def _get(base, route, key):
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + route, headers={'Authorization': 'Bearer' + chr(32) + key})
    with opener.open(req, timeout=2) as response:
        raw = response.read(262145)
        if len(raw) > 262144:
            raise ValueError('oversized identity response')
        return json.loads(raw)

def admit_run(scope):
    """Authenticate fresh at every helper operation; any uncertainty refuses."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        with scope._lock:
            if scope._closed:
                return False
            values = dict(scope.values)
        policy = load_policy()
        if get_active_profile_name() != policy['profile']:
            return False
        base = policy['api_base'].rstrip('/').removesuffix('/api')
        if values.get('PAPERCLIP_API_URL','').rstrip('/').removesuffix('/api') != base:
            return False
        key = values['PAPERCLIP_API_KEY']
        # Signature/authentication is verified by the pinned server below, not by this decode.
        if key.count('.') != 2:
            return False
        claims = json.loads(base64.urlsafe_b64decode(key.split('.')[1] + '==='))
        import time
        from uuid import UUID
        run_id, task_id = values['PAPERCLIP_RUN_ID'], values['PAPERCLIP_TASK_ID']
        if str(UUID(run_id)) != run_id or str(UUID(task_id)) != task_id:
            return False
        if (claims.get('sub') != policy['agent_id'] or claims.get('company_id') != policy['company_id']
                or claims.get('run_id') != run_id or claims.get('adapter_type') != 'hermes_gateway'
                or type(claims.get('exp')) not in (int,float)
                or not math.isfinite(claims['exp']) or claims['exp'] <= time.time()
                or values.get('PAPERCLIP_AGENT_ID') != policy['agent_id']
                or values.get('PAPERCLIP_COMPANY_ID') != policy['company_id']):
            return False
        identity = _get(base, '/api/agents/me', key)
        run = _get(base, '/api/heartbeat-runs/' + run_id, key)
        issue = _get(base, '/api/issues/' + task_id, key)
        if (identity.get('id') != policy['agent_id'] or identity.get('companyId') != policy['company_id']
                or identity.get('trustPreset', 'standard') != 'standard'
                or identity.get('adapterType') != 'hermes_gateway'
                or run.get('id') != run_id or run.get('agentId') != policy['agent_id']
                or run.get('companyId') != policy['company_id'] or run.get('status') != 'running'
                or issue.get('id') != task_id or issue.get('companyId') != policy['company_id']
                or issue.get('checkoutRunId') != run_id or issue.get('executionRunId') != run_id
                or issue.get('assigneeAgentId') != policy['agent_id'] or issue.get('status') != 'in_progress'
                or (run.get('contextSnapshot') or {}).get('issueId') != task_id):
            return False
        with scope._lock:
            return (not scope._closed and claims['exp'] > time.time()
                    and dict(scope.values) == values)
    except Exception:
        return False

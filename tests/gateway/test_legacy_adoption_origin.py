"""Fork (launch adapter): resuming a pre-v0.23 local session under a provenance label.

``hermes chat -Q -c <title> --create-if-missing --source <label>`` (``session.create`` with ``title`` and
``origin``) and ``hermes chat --resume <id> --source <label>`` (``session.resume`` with ``origin``) adopt
a legacy row the first time they open it. The adoption freezes the client's validated label as the new
policy's ``origin``, so the conversation's turns carry it (``turn_origin_label`` / ``launch_origin_source``)
and help-desk doors that require a turn source accept its MCP calls. Only the label carries over: every
other creation field of the request is ignored, as an unlabelled adoption ignores it. A session whose
policy is already frozen keeps its origin (present or absent); ``session.resume`` refuses a different
label, the titled create-if-missing leaves it as it was. ``switchboard-tenant`` is never minted by
adoption, and a refused label refuses the adoption: nothing is written.

Real: SessionStore/SessionAuthority/AuthorityConnection dispatch, ``resolve_local_target`` ->
``adopt_legacy_session``, the committed policy receipt and ``gateway.run_turn_origin``.
Seam: a SimpleNamespace runner. No endpoint is contacted.
"""
import json

import pytest
import pytest_asyncio

from tests.gateway.test_follow_profile_followups import _profile
from tests.gateway.test_session_follow_profile_config import _owner

TITLE = 'claude:2026-11-legacy-project'


@pytest.fixture
def home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    profile = get_hermes_home()
    profile.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', profile)
    _profile(profile, 'profile-model', 1, 'profile-key')
    return profile


@pytest_asyncio.fixture
async def owner(home):
    store, runner, authority, connection = _owner(home)
    runner._peek_session_state = lambda key: None
    # The native owner's transport: the only actor that may adopt pre-authority history.
    authority._native_legacy_transports = {connection.actor.transport_id: connection.actor}
    try:
        yield runner, authority, connection
    finally:
        await connection.close()
        store.close_all_db_handles()


def _legacy(db, home, sid, title=None, **row):
    """A row as the pre-v0.23 CLI wrote it: ``source='cli'``, unrouted, ended."""
    db.create_session(sid, source=row.pop('source', 'cli'), cwd=str(home), model='legacy-model',
                      model_config={'max_iterations': 90, 'reasoning_config': {'enabled': True, 'effort': 'high'}},
                      **row)
    if title:
        db.set_session_title(sid, title)
    db.end_session(sid, 'cli_close')


async def _call(connection, method, **params):
    return await connection.dispatch({'id': method, 'method': method, 'params': params})


async def _create_titled(connection, home, title, request_id, **extra):
    return await _call(connection, 'session.create', request_id=request_id, source='cli', cwd=str(home),
                       title=title, **extra)


def _stored(authority, sid):
    """The committed local policy receipt, or None when nothing was written for *sid*."""
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
        legacy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,)).fetchone()
    if row is None:
        assert legacy is None
        return None
    return json.loads(row[0])['policy']


def _origin(authority, sid):
    policy = _stored(authority, sid)
    return json.loads(policy['request_json']).get('origin') if policy else None


def _turn_labels(runner, authority, sid):
    from gateway.run_turn_origin import launch_origin_source, turn_origin_label
    source = authority.sessions[sid].source
    return turn_origin_label(runner, source), launch_origin_source(runner, source)


def _refused(response, reason='invalid_params'):
    assert 'error' in response, response
    data = response['error'].get('data') or {}
    assert data.get('reason') == reason, response
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['create-if-missing', 'resume-id', 'resume-title'])
async def test_an_adopted_legacy_session_takes_the_resuming_clients_label(home, owner, door):
    runner, authority, connection = owner
    _legacy(authority.db, home, 'legacy-1', title=TITLE)
    if door == 'create-if-missing':
        response = await _create_titled(connection, home, TITLE, 'c1', origin='helpdesk-ask')
    elif door == 'resume-id':
        response = await _call(connection, 'session.resume', session_id='legacy-1', origin='helpdesk-ask')
    else:
        response = await _call(connection, 'session.resume', session_id='', title=TITLE, origin='helpdesk-ask')
    assert 'result' in response, response
    assert response['result']['session_id'] == 'legacy-1'
    assert _origin(authority, 'legacy-1') == 'helpdesk-ask'
    assert response['result']['info']['launch_request'].get('origin') == 'helpdesk-ask'
    # The next turn is labelled: MCP ``_meta`` and HERMES_SESSION_SOURCE both read the frozen origin.
    assert _turn_labels(runner, authority, 'legacy-1') == ('helpdesk-ask', {'source': 'helpdesk-ask'})


@pytest.mark.asyncio
async def test_an_unlabelled_resume_adopts_exactly_as_before(home, owner):
    runner, authority, connection = owner
    _legacy(authority.db, home, 'legacy-1', title=TITLE)
    response = await _create_titled(connection, home, TITLE, 'c1')
    assert 'result' in response, response
    policy = _stored(authority, 'legacy-1')
    assert 'origin' not in json.loads(policy['request_json'])
    assert _turn_labels(runner, authority, 'legacy-1') == ('', {})


@pytest.mark.asyncio
async def test_the_label_is_all_a_labelled_adoption_takes_from_the_request(home, owner):
    """Two identical legacy rows: one adopted unlabelled, one by a labelled create-if-missing that also
    asks for other toolsets, another model, YOLO, ignored rules and another cwd. The policies differ by
    the origin only: adoption never relaxes the row's restrictions or gains anything else."""
    _runner, authority, connection = owner
    other = home / 'elsewhere'
    other.mkdir()
    _legacy(authority.db, home, 'plain', title='plain chat')
    _legacy(authority.db, home, 'labelled', title='labelled chat')
    assert 'result' in await _create_titled(connection, home, 'plain chat', 'p1')
    response = await _call(connection, 'session.create', request_id='l1', source='cli', cwd=str(other),
                           title='labelled chat', origin='helpdesk-ask', toolsets=['terminal', 'web'],
                           model='other-model', yolo=True, ignore_rules=True)
    assert 'result' in response, response
    plain, labelled = _stored(authority, 'plain'), _stored(authority, 'labelled')
    plain_request = json.loads(plain['request_json'])
    labelled_request = json.loads(labelled.pop('request_json'))
    assert labelled_request.pop('origin') == 'helpdesk-ask'
    assert labelled_request == plain_request
    plain.pop('request_json')
    # The secret references hash the whole policy (its request included) and name the session id.
    for ref in ('credential_ref', 'config_secret_ref'):
        assert (plain.pop(ref) is None) == (labelled.pop(ref) is None)
    assert json.dumps(labelled, sort_keys=True).replace('labelled', 'X') == \
        json.dumps(plain, sort_keys=True).replace('plain', 'X')
    assert labelled['cwd'] == str(home) and 'other-model' not in json.dumps(labelled)


@pytest.mark.asyncio
async def test_a_frozen_origin_resists_relabelling(home, owner):
    runner, authority, connection = owner
    response = await _call(connection, 'session.create', request_id='fresh', source='cli', cwd=str(home),
                           origin='helpdesk-ask')
    sid = response['result']['session_id']
    # session.resume refuses a different label; repeating the same one is fine.
    data = _refused(await _call(connection, 'session.resume', session_id=sid, origin='other-label'))
    assert 'origin' in json.dumps(data)
    assert 'result' in await _call(connection, 'session.resume', session_id=sid, origin='helpdesk-ask')
    # A titled create-if-missing leaves an existing session's creation fields as they are.
    authority.db.set_session_title(sid, 'fresh chat')
    response = await _create_titled(connection, home, 'fresh chat', 'again', origin='other-label')
    assert response['result']['session_id'] == sid
    assert _origin(authority, sid) == 'helpdesk-ask'
    assert _turn_labels(runner, authority, sid)[0] == 'helpdesk-ask'


@pytest.mark.asyncio
async def test_an_already_adopted_unlabelled_session_stays_unlabelled(home, owner):
    """Adoption happens once. A session adopted without a label (before this fix, or by an unlabelled
    resume) has a frozen policy whose origin is absent, and a later labelled resume never adds one."""
    runner, authority, connection = owner
    _legacy(authority.db, home, 'legacy-1', title=TITLE)
    assert 'result' in await _call(connection, 'session.resume', session_id='legacy-1')
    _refused(await _call(connection, 'session.resume', session_id='legacy-1', origin='helpdesk-ask'))
    assert 'result' in await _create_titled(connection, home, TITLE, 'c2', origin='helpdesk-ask')
    assert _origin(authority, 'legacy-1') is None
    assert _turn_labels(runner, authority, 'legacy-1') == ('', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['create-if-missing', 'resume-id'])
@pytest.mark.parametrize('label', ['switchboard-tenant', 'cli', 'tui', 'slack', 'local', 'cron', 'gateway',
                                   'Bad_Label', 'x', 'helpdesk ask', '', 7])
async def test_tenant_reserved_and_invalid_labels_refuse_the_adoption_and_write_nothing(home, owner, door, label):
    _runner, authority, connection = owner
    _legacy(authority.db, home, 'legacy-1', title=TITLE)
    if door == 'create-if-missing':
        response = await _create_titled(connection, home, TITLE, 'c1', origin=label)
    else:
        response = await _call(connection, 'session.resume', session_id='legacy-1', origin=label)
    assert 'error' in response, response
    assert _stored(authority, 'legacy-1') is None
    # Still the owner's legacy row, adoptable by an unlabelled (or valid) resume afterwards.
    assert 'result' in await _call(connection, 'session.resume', session_id='legacy-1', origin='helpdesk-ask')
    assert _origin(authority, 'legacy-1') == 'helpdesk-ask'


def test_switchboard_tenant_is_refused_by_the_adoption_check_itself():
    from gateway.session_local_migration import adoption_origin
    from hermes_state_runtime import RuntimeStoreError
    assert adoption_origin(None) is None
    assert adoption_origin('helpdesk-ask') == 'helpdesk-ask'
    for label in ('switchboard-tenant', 'cli', 'whatsapp', 'BAD'):
        with pytest.raises(RuntimeStoreError) as caught:
            adoption_origin(label)
        assert (caught.value.reason, caught.value.field) == ('invalid_params', 'origin')


@pytest.mark.asyncio
async def test_a_failed_adoption_never_mints_a_labelled_policy(home, owner):
    """Fail closed: a row adoption refuses (another login's stamp) stays unadopted under a valid label."""
    _runner, authority, connection = owner
    _legacy(authority.db, home, 'foreign', title='foreign chat', user_id='someone-else')
    response = await _call(connection, 'session.resume', session_id='foreign', origin='helpdesk-ask')
    assert 'error' in response, response
    response = await _create_titled(connection, home, 'foreign chat', 'c1', origin='helpdesk-ask')
    assert _stored(authority, 'foreign') is None
    if 'result' in response:  # not found for this owner: the titled create made a fresh labelled session
        assert response['result']['session_id'] != 'foreign'

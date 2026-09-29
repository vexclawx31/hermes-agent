"""Scoped durable projection survives scope teardown and resumed-session reads."""
import json
from types import SimpleNamespace
from agent.session_persistence import SessionPersistenceMixin
from gateway.runtime_context import bind_environment, validate_environment
from hermes_state import SessionDB


def test_real_persistence_redacts_nested_rows_without_changing_execution(tmp_path):
    secret = 'synthetic-persistence-authority-12345'
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('scoped', source='api')
    agent = SimpleNamespace(_session_db=db, _session_db_created=True,
        _persist_disabled=False, session_id='scoped', _session_persist_lock=None,
        _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0)
    agent._flush_messages_to_session_db_unlocked = SessionPersistenceMixin._flush_messages_to_session_db_unlocked.__get__(agent)
    messages = [{'role': 'assistant', 'content': secret, 'reasoning': secret,
        'api_content': 'wire ' + secret,
        'display_metadata': {'nested': [secret]},
        'tool_calls': [{'id': 't1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({'command': secret})}}]}]
    scope = validate_environment({'PAPERCLIP_API_KEY': secret},
        {'gateway': {'api_server': {'run_environment_allowlist': ['PAPERCLIP_API_KEY']}}}, 'local')
    try:
        with bind_environment(scope):
            assert SessionPersistenceMixin._flush_messages_to_session_db(agent, messages) is True
        scope.close()
        rows = db.get_messages('scoped')
        assert secret not in json.dumps(rows)
        resumed = db.get_messages_as_conversation('scoped')
        assert secret not in json.dumps(resumed)
        assert '[REDACTED]' in json.dumps(resumed)
        assert messages[0]['content'] == secret
        assert secret in messages[0]['tool_calls'][0]['function']['arguments']
    finally:
        db.close()

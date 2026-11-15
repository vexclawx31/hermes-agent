"""Tests for gateway/mirror.py — session mirroring."""

import importlib
import json
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


def _setup_sessions(tmp_path, sessions_data):
    """Helper to write a fake sessions.json and patch module-level paths."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_file = sessions_dir / "sessions.json"
    index_file.write_text(json.dumps(sessions_data), encoding="utf-8")
    return sessions_dir, index_file


class TestFindSessionId:
    def test_finds_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "new": {
                "session_id": "sess_new",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"


class TestMirrorToSession:


    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_alice"

    def test_no_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False


    def test_failed_sqlite_write_reports_false(self, tmp_path):
        """A mirror whose transcript write raises must not report success (#10130)."""
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "dm": {
                "session_id": "sess_dm",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            },
        })
        broken_db = MagicMock()
        broken_db.find_session_by_origin.return_value = None  # resolve via sessions.json
        broken_db.append_message.side_effect = OSError("disk full")

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("hermes_state_registry.acquire", return_value=broken_db), \
             patch("hermes_state_registry.release_or_close"):
            result = mirror_to_session("telegram", "123", "Hello!")

        assert result is False
        broken_db.append_message.assert_called_once()


class TestAppendToSqlite:
    def test_connection_is_released_after_use(self, tmp_path):
        """Verify _append_to_sqlite returns the shared SessionDB reference."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()
        released = []

        with patch("hermes_state_registry.acquire", return_value=mock_db), \
             patch(
                 "hermes_state_registry.release_or_close",
                 side_effect=lambda db: released.append(db),
             ):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        assert released == [mock_db], (
            "the shared handle must be released exactly once after use"
        )


class TestSessionsIndexProfileScoping:
    """#112844: the fallback index must follow the active profile, not the launch one."""

    @staticmethod
    def _write_index(home, session_id):
        d = home / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        (d / "sessions.json").write_text(json.dumps({
            "agent:main:telegram:dm": {
                "session_id": session_id,
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        }), encoding="utf-8")

    def test_fallback_follows_active_profile_home(self, tmp_path, monkeypatch):
        """A profile switched in after import must be read, not the launch profile's index.

        The module captures ``sessions.json`` under the home that was live at import. Under the
        multiplexed gateway one process serves every profile, so a lookup for another profile
        must not resolve against the launch profile's index and return its session id.
        """
        launch_home, active_home = tmp_path / "launch", tmp_path / "active"
        self._write_index(launch_home, "sess_launch")
        self._write_index(active_home, "sess_active")

        # Re-import the module with the launch home live: this is the import-time capture.
        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        importlib.reload(mirror_mod)
        try:
            # A request for a different profile is now served by the same process.
            monkeypatch.setenv("HERMES_HOME", str(active_home))
            assert mirror_mod._find_session_id("telegram", "12345") == "sess_active"
        finally:
            monkeypatch.undo()
            importlib.reload(mirror_mod)

    def test_patched_constant_still_wins(self, tmp_path, monkeypatch):
        """Tests that patch ``_SESSIONS_INDEX`` keep overriding the live home."""
        patched = tmp_path / "patched"
        self._write_index(patched, "sess_patched")

        monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", patched / "sessions" / "sessions.json")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "elsewhere"))

        assert mirror_mod._find_session_id("telegram", "12345") == "sess_patched"


PHONE, LID = "33700000001", "131900000000001"


def _lid_mapping(tmp_path):
    """The bridge's lid-mapping files for one person (phone <-> LID), as the WhatsApp adapter writes them."""
    session_dir = tmp_path / "home" / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / f"lid-mapping-{PHONE}.json").write_text(json.dumps(LID), encoding="utf-8")
    (session_dir / f"lid-mapping-{LID}_reverse.json").write_text(json.dumps(PHONE), encoding="utf-8")
    return tmp_path / "home"


def _dm_session(chat_id):
    return {"agent:main:whatsapp:dm:" + PHONE: {
        "session_id": "sess_dm", "updated_at": "2026-11-15T19:00:00",
        "origin": {"platform": "whatsapp", "chat_id": chat_id, "user_id": chat_id, "chat_type": "dm"}}}


class TestWhatsAppDmAliases:
    """2026-11-15: an owner notice sent to the phone JID was never mirrored into the owner's DM session, which the
    bridge had keyed by his LID, so the DM turn that followed had no idea what had just been sent."""

    def _mirror(self, tmp_path, monkeypatch, session_chat, target, user_id):
        monkeypatch.setenv("HERMES_HOME", str(_lid_mapping(tmp_path)))
        sessions_dir, index_file = _setup_sessions(tmp_path, _dm_session(session_chat))
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            ok = mirror_to_session("whatsapp", target, "the ski research", source_label="owner-dm", user_id=user_id)
        return ok, mock_sqlite

    def test_a_phone_jid_send_reaches_a_lid_keyed_dm_session(self, tmp_path, monkeypatch):
        ok, sqlite = self._mirror(tmp_path, monkeypatch, f"{LID}@lid", f"{PHONE}@s.whatsapp.net",
                                  f"{PHONE}@s.whatsapp.net")
        assert ok is True and sqlite.call_args[0][0] == "sess_dm"

    def test_a_lid_send_reaches_a_phone_keyed_dm_session(self, tmp_path, monkeypatch):
        ok, sqlite = self._mirror(tmp_path, monkeypatch, f"{PHONE}@s.whatsapp.net", f"{LID}@lid", None)
        assert ok is True and sqlite.call_args[0][0] == "sess_dm"

    def test_a_bare_phone_target_resolves_too(self, tmp_path, monkeypatch):
        ok, sqlite = self._mirror(tmp_path, monkeypatch, f"{LID}@lid", PHONE, PHONE)
        assert ok is True and sqlite.call_args[0][0] == "sess_dm"

    def test_without_a_mapping_nothing_is_guessed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
        sessions_dir, index_file = _setup_sessions(tmp_path, _dm_session(f"{LID}@lid"))
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("whatsapp", f"{PHONE}@s.whatsapp.net", "x") is False
        mock_sqlite.assert_not_called()

    def test_another_persons_dm_is_never_matched(self, tmp_path, monkeypatch):
        """Aliases come only from the mapping files: a different number keeps no session."""
        ok, sqlite = self._mirror(tmp_path, monkeypatch, f"{LID}@lid", "33799999999@s.whatsapp.net",
                                  "33799999999@s.whatsapp.net")
        assert ok is False
        sqlite.assert_not_called()

    def test_groups_and_other_platforms_get_no_alternates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(_lid_mapping(tmp_path)))
        assert mirror_mod._whatsapp_dm_alternates("120363000000000001@g.us", None) == []
        assert mirror_mod._whatsapp_dm_alternates("status@broadcast", None) == []
        sessions_dir, index_file = _setup_sessions(tmp_path, _dm_session(f"{LID}@lid"))
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("signal", PHONE, "x") is False
        mock_sqlite.assert_not_called()

    def test_a_different_user_id_is_kept_not_rewritten(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(_lid_mapping(tmp_path)))
        alts = mirror_mod._whatsapp_dm_alternates(f"{PHONE}@s.whatsapp.net", "someone-else")
        assert alts == [(f"{LID}@lid", "someone-else")]
        alts = mirror_mod._whatsapp_dm_alternates(f"{PHONE}@s.whatsapp.net", f"{PHONE}@s.whatsapp.net")
        assert alts == [(f"{LID}@lid", f"{LID}@lid")]
        # the peer's other id as user_id (phone target, LID user) also follows the candidate
        alts = mirror_mod._whatsapp_dm_alternates(f"{PHONE}@s.whatsapp.net", f"{LID}@lid")
        assert alts == [(f"{LID}@lid", f"{LID}@lid")]

    def test_only_the_mapped_namespace_is_tried(self, tmp_path, monkeypatch):
        """Vex (PR #69 review): the same digits in the other namespace are not the same person."""
        monkeypatch.setenv("HERMES_HOME", str(_lid_mapping(tmp_path)))
        assert mirror_mod._whatsapp_dm_alternates(f"{PHONE}@s.whatsapp.net", None) == [(f"{LID}@lid", None)]
        assert mirror_mod._whatsapp_dm_alternates(f"{LID}@lid", None) == [(f"{PHONE}@s.whatsapp.net", None)]
        assert mirror_mod._whatsapp_dm_alternates(PHONE, None) == [(f"{PHONE}@s.whatsapp.net", None),
                                                                   (f"{LID}@lid", None)]

    def test_same_digits_in_the_other_namespace_are_never_matched_without_a_mapping(self, tmp_path, monkeypatch):
        """No mapping: a phone-JID send must not land in an unrelated session whose LID has the same digits."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
        assert mirror_mod._whatsapp_dm_alternates(f"{PHONE}@s.whatsapp.net", None) == []
        assert mirror_mod._whatsapp_dm_alternates(f"{PHONE}@lid", None) == []
        sessions_dir, index_file = _setup_sessions(tmp_path, _dm_session(f"{PHONE}@lid"))
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("whatsapp", f"{PHONE}@s.whatsapp.net", "x",
                                     user_id=f"{PHONE}@s.whatsapp.net") is False
        mock_sqlite.assert_not_called()


class TestWhatsAppDmAliasesInStateDb:
    """The same lookups through a real SessionDB (state.db), not the sessions.json fallback."""

    def _db(self, tmp_path, monkeypatch):
        import hermes_state
        from hermes_state import SessionDB

        home = _lid_mapping(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)
        monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", home / "sessions" / "absent.json")
        return SessionDB(db_path=db_path)

    @staticmethod
    def _dm(db, sid, peer, started, active):
        db.create_session(sid, "whatsapp", user_id=peer, session_key=f"agent:main:whatsapp:dm:{PHONE}:{sid}",
                          chat_id=peer, chat_type="dm")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (started, active, sid)))

    def test_mixed_ids_mirror_into_the_lid_session(self, tmp_path, monkeypatch):
        db = self._db(tmp_path, monkeypatch)
        try:
            self._dm(db, "sess_lid", f"{LID}@lid", 100.0, 200.0)
        finally:
            db.close()
        with patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("whatsapp", f"{PHONE}@s.whatsapp.net", "the ski research",
                                     user_id=f"{PHONE}@s.whatsapp.net") is True
        assert mock_sqlite.call_args[0][0] == "sess_lid"

    def test_with_phone_and_lid_sessions_the_most_recently_active_wins(self, tmp_path, monkeypatch):
        """A bare-number target can match both a phone-keyed and a LID-keyed DM session of the same person:
        the conversation active most recently gets the mirror, whatever the alias order."""
        db = self._db(tmp_path, monkeypatch)
        try:
            self._dm(db, "sess_phone_old", f"{PHONE}@s.whatsapp.net", 100.0, 150.0)
            self._dm(db, "sess_lid_new", f"{LID}@lid", 120.0, 900.0)
        finally:
            db.close()
        with patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("whatsapp", PHONE, "x", user_id=PHONE) is True
        assert mock_sqlite.call_args[0][0] == "sess_lid_new"

    def test_an_exact_match_is_unchanged(self, tmp_path, monkeypatch):
        """A send that already names the session's own id never looks at aliases."""
        db = self._db(tmp_path, monkeypatch)
        try:
            self._dm(db, "sess_phone", f"{PHONE}@s.whatsapp.net", 100.0, 150.0)
            self._dm(db, "sess_lid", f"{LID}@lid", 120.0, 900.0)
        finally:
            db.close()
        with patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session("whatsapp", f"{PHONE}@s.whatsapp.net", "x",
                                     user_id=f"{PHONE}@s.whatsapp.net") is True
        assert mock_sqlite.call_args[0][0] == "sess_phone"

"""Recall scope on memory: built-in store partitions, provider gating, session epoch (agent/recall_scope.py)."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.recall_scope import apply_turn_recall_scope, enforcement_epoch, scope_for_source
from tools.memory_tool_store import MemoryStore

J_DM = scope_for_source("slack", "UJAMAL0001", "DJAMAL0001", "dm")
JOINT = scope_for_source("slack", "UJAMAL0001", "CJOINT0001", "group")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _enforce(monkeypatch, on=True):
    monkeypatch.setattr("agent.recall_scope.enforcement_enabled", lambda config=None: on)


def test_store_partitions_by_scope_and_never_reads_legacy(home):
    legacy = home / "memories"
    legacy.mkdir()
    (legacy / "USER.md").write_text("Pauline said the legacy secret", encoding="utf-8")
    store = MemoryStore()
    store.load_from_disk()
    assert "legacy secret" in (store.format_for_system_prompt("user") or "")

    assert store.bind_recall_scope(True, J_DM)
    assert store.format_for_system_prompt("user") is None
    assert store.add("user", "Jamal likes tea")["success"]
    assert store.bind_recall_scope(True, JOINT)
    assert store.user_entries == []  # the joint chat never sees Jamal's DM partition
    assert store.bind_recall_scope(True, J_DM)
    assert store.user_entries == ["Jamal likes tea"]
    assert (legacy / "USER.md").read_text(encoding="utf-8") == "Pauline said the legacy secret"
    assert not store.bind_recall_scope(True, J_DM)  # unchanged partition: no reload


def test_store_without_scope_is_off_and_unenforced_restores_legacy(home):
    store = MemoryStore()
    store.bind_recall_scope(True, None)
    assert not store.target_enabled("memory") and not store.target_enabled("user")
    assert store.format_for_system_prompt("memory") is None
    store.bind_recall_scope(False)
    assert store.target_enabled("memory") and store.target_enabled("user")


class _Provider:
    def __init__(self, name, scoped=False):
        self.name, self.supports_recall_scope, self.bound, self.prefetched = name, scoped, [], []

    def get_tool_schemas(self):
        return [{"name": f"{self.name}_recall", "description": "", "parameters": {}}]

    def prefetch(self, query, *, session_id=""):
        self.prefetched.append(query)
        return f"{self.name} recall"

    def queue_prefetch(self, query, *, session_id=""):
        pass

    def bind_recall_scope(self, scope, *, enforced):
        self.bound.append((scope, enforced))

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"ok": True})


def _manager(*providers):
    manager = MemoryManager()
    for p in providers:
        manager._providers.append(p)
        for schema in p.get_tool_schemas():
            manager._tool_to_provider[schema["name"]] = p
    return manager


def test_unscoped_provider_is_withheld_under_enforcement():
    legacy, scoped = _Provider("legacy"), _Provider("scoped", scoped=True)
    manager = _manager(legacy, scoped)
    manager.bind_recall_scope(True, J_DM)
    assert manager._recall_providers() == [scoped]
    assert scoped.bound[-1] == (J_DM, True)
    assert "unavailable" in json.loads(manager.handle_tool_call("legacy_recall", {}))["error"]
    assert json.loads(manager.handle_tool_call("scoped_recall", {}))["ok"]


def test_no_scope_withholds_every_provider_and_off_restores():
    scoped = _Provider("scoped", scoped=True)
    manager = _manager(scoped)
    manager.bind_recall_scope(True, None)
    assert manager._recall_providers() == []
    assert "unavailable" in json.loads(manager.handle_tool_call("scoped_recall", {}))["error"]
    manager.bind_recall_scope(False, J_DM)
    assert manager._recall_providers() == [scoped] and scoped.bound[-1] == (None, False)


def test_apply_turn_recall_scope_binds_both_and_invalidates_prompt(home, monkeypatch):
    _enforce(monkeypatch)
    scoped = _Provider("scoped", scoped=True)
    agent = SimpleNamespace(_recall_scope=J_DM, _memory_store=MemoryStore(), _memory_manager=_manager(scoped),
                            _cached_system_prompt="old prompt with legacy memory")
    apply_turn_recall_scope(agent)
    assert agent._cached_system_prompt is None
    assert agent._memory_store._scope_partition == J_DM.partition()
    assert scoped.bound[-1] == (J_DM, True)
    agent._recall_scope = "not a scope"  # only a real RecallScope counts
    apply_turn_recall_scope(agent)
    assert agent._memory_store._scope_partition == "" and scoped.bound[-1] == (None, True)


def test_epoch_created_once_and_forgotten_when_off(home):
    first = enforcement_epoch(True, home, now=100.0)
    assert first == 100.0 and enforcement_epoch(True, home, now=200.0) == 100.0
    assert enforcement_epoch(False, home) is None
    assert enforcement_epoch(True, home, now=300.0) == 300.0


def test_sessions_predating_the_epoch_are_replaced(home, monkeypatch):
    from gateway.session_lifecycle import _predates_recall_scope_epoch
    _enforce(monkeypatch)
    since = enforcement_epoch(True)
    old = SimpleNamespace(created_at=datetime.fromtimestamp(since) - timedelta(hours=1))
    new = SimpleNamespace(created_at=datetime.fromtimestamp(since) + timedelta(seconds=1))
    assert _predates_recall_scope_epoch(old) and not _predates_recall_scope_epoch(new)
    _enforce(monkeypatch, on=False)
    assert not _predates_recall_scope_epoch(old)


class _SessionDB:
    def __init__(self, rows):
        self.rows = rows

    def get_session(self, session_id):
        return self.rows.get(session_id)


def test_a_fork_sharing_the_store_with_the_same_scope_leaves_it_bound(home, monkeypatch):
    _enforce(monkeypatch)
    store = MemoryStore()
    parent = SimpleNamespace(_recall_scope=J_DM, _memory_store=store, _memory_manager=None, _cached_system_prompt="p")
    apply_turn_recall_scope(parent)
    store.add("user", "Jamal likes tea")
    fork = SimpleNamespace(_recall_scope=parent._recall_scope, _memory_store=store, _memory_manager=None,
                           _cached_system_prompt="f")
    apply_turn_recall_scope(fork)
    assert store._scope_partition == J_DM.partition() and store.user_entries == ["Jamal likes tea"]
    assert fork._cached_system_prompt == "f"  # no rebind, no invalidation


def test_prompt_trailer_names_the_partition_and_a_mismatch_is_not_restored(home, monkeypatch):
    from agent.system_prompt import recall_scope_trailer_value
    store = MemoryStore()
    agent = SimpleNamespace(_memory_store=store)
    assert recall_scope_trailer_value(agent) is None  # unenforced: no trailer line at all
    store.bind_recall_scope(True, J_DM)
    assert recall_scope_trailer_value(agent) == J_DM.partition()
    store.bind_recall_scope(True, None)
    assert recall_scope_trailer_value(agent) == "none"


def test_scoped_store_refuses_to_stage_writes(home):
    from tools.memory_tool import _scoped_staging_refusal
    store = MemoryStore()
    assert _scoped_staging_refusal(store) is None
    store.bind_recall_scope(True, J_DM)
    assert "can't be staged" in json.loads(_scoped_staging_refusal(store))["error"]


def test_scope_change_drains_queued_syncs_first():
    order = []
    manager = _manager(_Provider("scoped", scoped=True))
    manager.flush_pending = lambda timeout=None: order.append("drain") or True
    manager.bind_recall_scope(True, J_DM)
    manager.bind_recall_scope(True, J_DM)  # same scope: no drain
    manager.bind_recall_scope(True, JOINT)
    assert order == ["drain", "drain"]


def test_undrainable_backlog_withholds_recall_for_the_turn():
    scoped = _Provider("scoped", scoped=True)
    manager = _manager(scoped)
    manager.flush_pending = lambda timeout=None: False
    manager.bind_recall_scope(True, J_DM)
    assert manager._recall_providers() == [] and scoped.bound[-1] == (None, True)


DM_KEY, JOINT_KEY = "agent:calix:slack:dm:DJAMAL0001", "agent:calix:slack:group:CJOINT0001"


def test_route_check_covers_the_compression_lineage_only(home, monkeypatch):
    from agent.recall_scope import session_route_mismatch
    _enforce(monkeypatch)
    db = _SessionDB({
        "jamal-dm": {"session_key": DM_KEY, "parent_session_id": None, "end_reason": "compression"},
        "jamal-child": {"session_key": DM_KEY, "parent_session_id": "jamal-dm"},  # compression child
        "rekeyed-child": {"session_key": JOINT_KEY, "parent_session_id": "jamal-dm"},  # compressed, then moved
        "orphan": {"session_key": DM_KEY, "parent_session_id": "missing"},
    })
    assert not session_route_mismatch(db, "jamal-dm", DM_KEY)
    assert not session_route_mismatch(db, "jamal-child", DM_KEY)
    assert session_route_mismatch(db, "jamal-dm", JOINT_KEY)  # a DM conversation served on a joint route
    assert session_route_mismatch(db, "jamal-child", JOINT_KEY)
    assert session_route_mismatch(db, "rekeyed-child", JOINT_KEY)  # its compressed history is the DM's
    assert session_route_mismatch(db, "orphan", DM_KEY)  # unprovable lineage
    assert not session_route_mismatch(db, "not-created-yet", DM_KEY)
    _enforce(monkeypatch, on=False)
    assert not session_route_mismatch(db, "jamal-dm", JOINT_KEY)


def test_reset_and_branch_successors_start_their_own_conversation(home, monkeypatch):
    # R1: a reset/branch successor's parent didn't end by compression, so its rows aren't replayed
    # and must not condemn the successor forever.
    from agent.recall_scope import session_lineage_predates_epoch, session_route_mismatch
    _enforce(monkeypatch)
    since = enforcement_epoch(True)
    thread_key = DM_KEY + ":thread:1700000000.000100"
    db = _SessionDB({
        "legacy": {"session_key": DM_KEY, "started_at": since - 100, "parent_session_id": None,
                   "end_reason": "recall_scope_enforced"},
        "successor": {"session_key": DM_KEY, "started_at": since + 1, "parent_session_id": "legacy"},
        "channel": {"session_key": JOINT_KEY, "started_at": since + 2, "parent_session_id": None},
        "thread-branch": {"session_key": thread_key, "started_at": since + 3, "parent_session_id": "channel"},
    })
    assert not session_lineage_predates_epoch(db, "successor")
    assert not session_route_mismatch(db, "successor", DM_KEY)
    assert not session_route_mismatch(db, "thread-branch", thread_key)


def test_switch_session_never_adopts_another_routes_conversation(home, monkeypatch):
    from gateway.session import SessionStore
    _enforce(monkeypatch)
    db = _SessionDB({"jamal-dm": {"session_key": DM_KEY, "parent_session_id": None},
                     "joint-old": {"session_key": JOINT_KEY, "parent_session_id": None},
                     "unkeyed": {"session_key": None, "parent_session_id": None}})
    store = SessionStore.__new__(SessionStore)
    store._db_for_key = lambda key: db
    assert store._foreign_route_under_recall_scope(JOINT_KEY, "jamal-dm")
    assert not store._foreign_route_under_recall_scope(JOINT_KEY, "joint-old")
    assert not store._foreign_route_under_recall_scope(JOINT_KEY, "unkeyed")  # e.g. a model-switch child
    assert store.switch_session(JOINT_KEY, "jamal-dm") is None
    _enforce(monkeypatch, on=False)
    assert not store._foreign_route_under_recall_scope(JOINT_KEY, "jamal-dm")


def test_resume_and_branch_are_refused_under_enforcement(monkeypatch):
    from gateway.slash_commands_session import _recall_scope_refusal
    _enforce(monkeypatch)
    assert "turned off" in _recall_scope_refusal("/resume") and "turned off" in _recall_scope_refusal("/branch")
    _enforce(monkeypatch, on=False)
    assert _recall_scope_refusal("/resume") is None


def test_an_unscoped_turn_does_not_change_what_the_next_turn_may_replay(home, monkeypatch):
    # The route key is fixed at session creation: no turn (scoped or not) can rewrite it.
    from agent.recall_scope import session_route_mismatch
    _enforce(monkeypatch)
    dm = "agent:calix:slack:dm:DJAMAL0001"
    db = _SessionDB({"jamal-dm": {"session_key": dm, "parent_session_id": None,
                                  "system_prompt": "...\n\nRecall scope: none"}})
    assert not session_route_mismatch(db, "jamal-dm", dm)


def test_lineage_check_follows_parents_to_the_root(home, monkeypatch):
    from agent.recall_scope import session_lineage_predates_epoch
    _enforce(monkeypatch)
    since = enforcement_epoch(True)
    db = _SessionDB({
        "legacy": {"started_at": since - 100, "parent_session_id": None, "end_reason": "compression"},
        "child-of-legacy": {"started_at": since + 5, "parent_session_id": "legacy"},
        "fresh": {"started_at": since + 1, "parent_session_id": None, "end_reason": "compression"},
        "child-of-fresh": {"started_at": since + 9, "parent_session_id": "fresh"},
        "orphan": {"started_at": since + 9, "parent_session_id": "missing"},
    })
    assert session_lineage_predates_epoch(db, "legacy")
    assert session_lineage_predates_epoch(db, "child-of-legacy")  # an old conversation compressed after the epoch
    assert not session_lineage_predates_epoch(db, "fresh")
    assert not session_lineage_predates_epoch(db, "child-of-fresh")
    assert session_lineage_predates_epoch(db, "orphan")  # unprovable lineage fails closed
    assert not session_lineage_predates_epoch(db, "not-created-yet")  # nothing to replay
    _enforce(monkeypatch, on=False)
    assert not session_lineage_predates_epoch(db, "legacy")

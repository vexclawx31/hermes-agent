"""Deferred-action receipts: resume ONE exact, otherwise-authorized tool call that a capability switch blocked.

Flow (design-stage2.md Part B, project 2026-11-switchboard-vex-messages):

1. mint, in a native turn only (TrustedIngress bound), by a tool that registered the blocking switch as resumable.
   The identity (profile, workspace, adapter, requester, original message, origin chat/thread) comes from
   TrustedIngress, never from arguments; the tool supplies its exact, complete action. State ``pending`` (inert).
2. notice: the core posts a fixed-template notice in the origin thread through the adapter and reads that exact
   message back (bot ownership, chat/thread, content hash). Only then ``noticed``. An uncertain notice is never
   resent; the receipt stays pending and expires. The notice informs; it is not consent.
3. link: in the same native turn the plugin that sent the blocker request (switchboard_request ``defers``) records
   its ref B on the receipt.
4. activate, by the host at dispatch: an injected event whose TurnSpec carries a resume marker signed by the
   configured Vex key (ed25519), naming this event's ref M, the blocker B, the receipt and a single-use marker id,
   with the event's origin equal to the receipt's. Then ``active`` for that turn only; the marker is spent.
5. redeem, by the same plugin and tool, in that turn: expiry re-checked and ``active -> claimed`` in one
   transaction; returns the stored action. The tool re-checks current policy and executes exactly that action.
6. settle: ``done`` (operation-specific evidence), ``failed`` (definite refusal) or ``unknown`` (ambiguous).

Lifecycle: an active turn that ends without redeeming goes back to ``noticed`` (a new marker is needed); at start,
``active`` rows go back to ``noticed`` and ``claimed`` rows without a settle become ``unknown``, never runnable.
Cancel (native turn, the requester) races claim under the same write lock. Expiry at most 24 h.

Storage: SQLite under the profile's ``state/deferred/`` (0700 dir, 0600 files); every row is HMAC'd with a durable
per-profile key in the same dir. This is a trusted-host-code boundary only: it does not isolate the gateway from a
shell running as its own uid, and must not be described as doing so. Off unless the profile config sets
``gateway.deferred.enabled: true`` and pins ``gateway.deferred.resume_signer`` (base64 raw ed25519 public key).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

DEFERRED_API_VERSION = 1
MAX_TTL = 24 * 3600
STATES = ("pending", "noticed", "active", "claimed", "done", "failed", "unknown", "cancelled", "expired")
FINAL = frozenset({"done", "failed", "unknown", "cancelled", "expired"})
_ID = re.compile(r"df-[0-9a-f]{16}\Z")
_REF = re.compile(r"hd-[0-9a-f]{16}\Z")
_MARKER = re.compile(r"[0-9a-f]{16}\Z")
_TOOL = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SWITCH = re.compile(r"[a-z][a-z0-9_.]{0,63}\Z")
NOTICE_TEMPLATE = ("Deferred {id}: once the setting `{blocked_by}` is on, I can do this once, exactly as written "
                   "(until {expires} UTC):\n> {summary}\nSay \"cancel {id}\" to stop it.")


class DeferredError(Exception):
    """A refused deferred-action operation (nothing changed)."""


# ---- config ----

def settings(home: Path) -> Optional[dict]:
    """``gateway.deferred`` of the profile at *home*, when enabled with a pinned signer; else None."""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.plugins import _plugin_home_scope
        with _plugin_home_scope(home):
            cfg = load_config_readonly() or {}
    except Exception:
        return None
    gw = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
    block = gw.get("deferred") if isinstance(gw.get("deferred"), dict) else None
    if not block or block.get("enabled") is not True or not isinstance(block.get("resume_signer"), str):
        return None
    try:
        raw = base64.b64decode(block["resume_signer"], validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) != 32:
        return None
    return {"signer": raw}


# ---- the store ----

_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
  id TEXT PRIMARY KEY, state TEXT NOT NULL, doc TEXT NOT NULL, mac TEXT NOT NULL,
  created_at REAL NOT NULL, expires_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS markers (marker TEXT PRIMARY KEY, receipt TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS log (seq INTEGER PRIMARY KEY AUTOINCREMENT, receipt TEXT NOT NULL, at REAL NOT NULL,
  from_state TEXT, to_state TEXT NOT NULL, note TEXT);
"""


class Store:
    """One profile's receipts. Every transition is one BEGIN IMMEDIATE transaction."""

    def __init__(self, home: Path, clock=time.time):
        self.dir = Path(home) / "state" / "deferred"
        self.clock = clock
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.path = self.dir / "deferred.db"
        self._key = self._load_key()
        with _LOCK:  # executescript commits on its own, so the schema runs outside tx()
            db = sqlite3.connect(self.path, timeout=30)
            try:
                db.executescript(_SCHEMA)
                os.chmod(self.path, 0o600)
            finally:
                db.close()

    def _load_key(self) -> bytes:
        path = self.dir / "receipt.key"
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            tmp = self.dir / f".receipt.key.{secrets.token_hex(4)}"
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(32))
            try:
                os.link(tmp, path)  # first writer wins; never overwrite an existing key
            except FileExistsError:
                pass
            finally:
                tmp.unlink(missing_ok=True)
            raw = path.read_bytes()
        if len(raw) != 32:
            raise DeferredError("receipt key is malformed")
        return raw

    @contextmanager
    def tx(self):
        with _LOCK:
            db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            try:
                os.chmod(self.path, 0o600)
                db.row_factory = sqlite3.Row
                db.execute("BEGIN IMMEDIATE")
                try:
                    yield db
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
            finally:
                db.close()

    def _mac(self, rid: str, state: str, doc: str) -> str:
        return hmac.new(self._key, f"{rid}\x1f{state}\x1f{doc}".encode(), hashlib.sha256).hexdigest()

    def _row(self, db, rid: str) -> Optional[dict]:
        row = db.execute("SELECT * FROM receipts WHERE id = ?", (rid,)).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row["mac"], self._mac(rid, row["state"], row["doc"])):
            raise DeferredError("receipt failed its integrity check")
        doc = json.loads(row["doc"])
        return {"id": rid, "state": row["state"], "expires_at": row["expires_at"], **doc}

    def _set(self, db, rec: dict, state: str, note: str = "", **changes) -> dict:
        doc = {k: v for k, v in rec.items() if k not in ("id", "state", "expires_at")}
        doc.update(changes)
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        db.execute("UPDATE receipts SET state = ?, doc = ?, mac = ?, updated_at = ? WHERE id = ?",
                   (state, raw, self._mac(rec["id"], state, raw), self.clock(), rec["id"]))
        db.execute("INSERT INTO log (receipt, at, from_state, to_state, note) VALUES (?, ?, ?, ?, ?)",
                   (rec["id"], self.clock(), rec["state"], state, note))
        return {**rec, **changes, "state": state}

    def create(self, doc: dict, ttl: float) -> dict:
        rid = f"df-{secrets.token_hex(8)}"
        now = self.clock()
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        with self.tx() as db:
            db.execute("INSERT INTO receipts VALUES (?, 'pending', ?, ?, ?, ?, ?)",
                       (rid, raw, self._mac(rid, "pending", raw), now, now + ttl, now))
            db.execute("INSERT INTO log (receipt, at, from_state, to_state, note) VALUES (?, ?, NULL, 'pending', '')",
                       (rid, now))
        return {"id": rid, "state": "pending", "expires_at": now + ttl, **doc}

    def get(self, rid: str) -> Optional[dict]:
        if not isinstance(rid, str) or not _ID.fullmatch(rid):
            return None
        with self.tx() as db:
            return self._expire(db, self._row(db, rid))

    def _expire(self, db, rec):
        if rec is not None and rec["state"] not in FINAL and rec["state"] != "claimed" and self.clock() >= rec["expires_at"]:
            return self._set(db, rec, "expired", "past expires_at")
        return rec

    def transition(self, rid: str, allowed_from: tuple, to: str, note: str = "", check=None, **changes) -> dict:
        """Atomic: the receipt must be in *allowed_from* (after expiry), and ``check(rec)`` must pass."""
        if not isinstance(rid, str) or not _ID.fullmatch(rid):
            raise DeferredError("not a deferred-action id")
        with self.tx() as db:
            rec = self._expire(db, self._row(db, rid))
            if rec is None:
                raise DeferredError("no such deferred action")
            if rec["state"] not in allowed_from:
                raise DeferredError(f"the deferred action is {rec['state']}")
            if check is not None:
                check(rec)
            return self._set(db, rec, rec["state"] if to is None else to, note, **changes)

    def spend_marker(self, marker: str, rid: str) -> None:
        """Record a resume marker; refused if it was used before (single use)."""
        with self.tx() as db:
            try:
                db.execute("INSERT INTO markers VALUES (?, ?, ?)", (marker, rid, self.clock()))
            except sqlite3.IntegrityError:
                raise DeferredError("resume marker already used") from None

    def recover(self) -> None:
        """At start: active -> noticed; claimed without a settle -> unknown (never runnable again)."""
        with self.tx() as db:
            for row in db.execute("SELECT id FROM receipts WHERE state IN ('active', 'claimed')").fetchall():
                rec = self._row(db, row["id"])
                self._set(db, rec, "noticed" if rec["state"] == "active" else "unknown", "recovered at start")


_STORES: dict[str, Store] = {}
_STORES_LOCK = threading.Lock()


def store_for(home: Path) -> Store:
    key = str(Path(home).resolve())
    with _STORES_LOCK:
        st = _STORES.get(key)
        if st is None:
            st = _STORES[key] = Store(Path(home))
            st.recover()
        return st


# ---- resume markers (signed by the Vex door) ----

def marker_payload(m: Mapping[str, Any]) -> bytes:
    fields = {k: m[k] for k in ("v", "event_ref", "blocker", "receipt", "target", "marker", "expires_at")}
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()


def verify_marker(signer: bytes, m: Any) -> dict:
    """The validated marker, or DeferredError. Shape first, then the ed25519 signature."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    want = {"v", "event_ref", "blocker", "receipt", "target", "marker", "expires_at", "sig"}
    if not isinstance(m, Mapping) or set(m) != want or m["v"] != 1:
        raise DeferredError("malformed resume marker")
    if not (isinstance(m["event_ref"], str) and _REF.fullmatch(m["event_ref"]) and isinstance(m["blocker"], str)
            and _REF.fullmatch(m["blocker"]) and isinstance(m["receipt"], str) and _ID.fullmatch(m["receipt"])
            and isinstance(m["marker"], str) and _MARKER.fullmatch(m["marker"]) and isinstance(m["target"], str)
            and isinstance(m["expires_at"], (int, float)) and isinstance(m["sig"], str)):
        raise DeferredError("malformed resume marker")
    try:
        Ed25519PublicKey.from_public_bytes(signer).verify(base64.b64decode(m["sig"], validate=True), marker_payload(m))
    except (InvalidSignature, ValueError, TypeError):
        raise DeferredError("resume marker signature is invalid") from None
    return dict(m)


# ---- what the host calls ----

def activate(home: Path, *, plugin_id: str, profile: str, event_ref: str, origin_chat: str, origin_thread: str,
             marker: Any) -> str:
    """At dispatch, for one injected event: verify everything, spend the marker, make the receipt active.
    Returns the receipt id (bound to the turn) or raises DeferredError (nothing changed except a spent marker
    after a valid signature)."""
    cfg = settings(home)
    if cfg is None:
        raise DeferredError("deferred actions are off for this profile")
    m = verify_marker(cfg["signer"], marker)
    now = time.time()
    if now >= m["expires_at"]:
        raise DeferredError("resume marker expired")
    if m["event_ref"] != event_ref:
        raise DeferredError("resume marker names another Switchboard event")
    st = store_for(home)

    def check(rec):
        if rec.get("plugin_link") is None or rec.get("blocker") != m["blocker"]:
            raise DeferredError("resume marker names another blocker request")
        if rec["profile"] != profile or m["target"].rsplit(":", 1)[-1] != profile:
            raise DeferredError("resume marker names another profile")
        if (rec["origin_chat"], rec["origin_thread"]) != (origin_chat, origin_thread):
            raise DeferredError("the event's origin is not the receipt's origin")
        if rec.get("plugin_link") != plugin_id:
            raise DeferredError("the receipt was linked by another plugin")

    st.get(m["receipt"])  # expire first, so an expired receipt never spends a marker
    st.spend_marker(m["marker"], m["receipt"])
    st.transition(m["receipt"], ("noticed",), "active", f"marker {m['marker']} event {event_ref}", check)
    return m["receipt"]


def release_unredeemed(home: Path, rid: str) -> None:
    """Turn ended: an active receipt that wasn't claimed goes back to noticed."""
    try:
        store_for(home).transition(rid, ("active",), "noticed", "turn ended without redemption")
    except DeferredError:
        pass


class DeferredAPI:
    """``PluginContext.deferred``: the operations an adopting plugin may call, bound to its plugin id and home."""

    def __init__(self, plugin_id: str, home: Path):
        self._plugin = plugin_id
        self._home = Path(home)
        self._resumable: dict[str, frozenset] = {}

    @property
    def enabled(self) -> bool:
        return settings(self._home) is not None

    def register(self, tool: str, switches) -> None:
        """Declare which capability switches block *tool* resumably (nothing else can be minted)."""
        if not _TOOL.fullmatch(tool or "") or not switches or not all(_SWITCH.fullmatch(s or "") for s in switches):
            raise DeferredError("invalid deferred registration")
        self._resumable[tool] = frozenset(switches)

    def _store(self) -> Store:
        if not self.enabled:
            raise DeferredError("deferred actions are off for this profile")
        return store_for(self._home)

    async def mint(self, tool: str, action: Mapping[str, Any], blocked_by: str, summary: str,
                   ttl: float = MAX_TTL) -> dict:
        """In a native turn: record the exact action, post the notice, read it back. Returns {id, state}."""
        from gateway.ingress_identity import get_trusted_ingress
        if blocked_by not in self._resumable.get(tool, ()):
            raise DeferredError(f"{blocked_by} is not a resumable switch for {tool}")
        ti = get_trusted_ingress()
        if ti is None:
            raise DeferredError("a deferred action can only be recorded in a native turn")
        if not isinstance(action, Mapping) or not isinstance(summary, str) or not summary.strip():
            raise DeferredError("a deferred action needs its exact action and a summary")
        st = self._store()
        doc = {"plugin": self._plugin, "tool": tool, "blocked_by": blocked_by,
               "action": json.loads(json.dumps(dict(action), sort_keys=True)),
               "summary": summary[:500], "profile": ti.profile, "platform": ti.platform, "workspace": ti.scope_id,
               "requester": ti.user_id, "message_id": ti.message_id, "origin_chat": ti.chat_id,
               "origin_thread": _thread_of(ti), "blocker": None, "plugin_link": None}
        rec = st.create(doc, min(float(ttl), MAX_TTL))
        notice_hook = getattr(ti.adapter, "deferred_notice", None)
        if notice_hook is None:
            return {"id": rec["id"], "state": "pending", "note": "this platform can't post a verified notice"}
        text = NOTICE_TEMPLATE.format(id=rec["id"], blocked_by=blocked_by, summary=doc["summary"],
                                      expires=time.strftime("%Y-%m-%d %H:%M", time.gmtime(rec["expires_at"])))
        try:
            evidence = await notice_hook(ti, text)  # posts, then reads that exact message back
        except Exception:
            evidence = None
        if not isinstance(evidence, Mapping) or evidence.get("verified") is not True:
            return {"id": rec["id"], "state": "pending", "note": "the notice couldn't be verified; it isn't resent"}
        st.transition(rec["id"], ("pending",), "noticed", "notice verified",
                      notice={"ts": evidence.get("ts"), "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        return {"id": rec["id"], "state": "noticed"}

    def link(self, rid: str, blocker: str) -> None:
        """In the same native turn: the Switchboard request this action waits on."""
        from gateway.ingress_identity import get_trusted_ingress
        ti = get_trusted_ingress()
        if ti is None or not _REF.fullmatch(blocker or ""):
            raise DeferredError("link needs a native turn and a Switchboard ref")

        def check(rec):
            if rec["message_id"] != ti.message_id or rec["profile"] != ti.profile or rec["blocker"] is not None:
                raise DeferredError("that deferred action wasn't recorded in this turn, or is linked already")
        self._store().transition(rid, ("pending", "noticed"), None, f"linked to {blocker}", check,
                                 blocker=blocker, plugin_link=self._plugin)

    def redeem(self, rid: str, tool: str) -> dict:
        """In the resumed turn: claim and return the stored action (once)."""
        from gateway.injected_turn import get_injected_turn
        turn = get_injected_turn()
        if turn is None or turn.resume != rid:
            raise DeferredError("this turn doesn't resume that deferred action")

        def check(rec):
            if rec["plugin"] != self._plugin or rec["tool"] != tool:
                raise DeferredError("that deferred action belongs to another tool")
        rec = self._store().transition(rid, ("active",), "claimed", "claimed", check, claim=secrets.token_hex(8))
        return {"id": rid, "claim": rec["claim"], "action": rec["action"], "requester": rec["requester"],
                "workspace": rec["workspace"], "origin_chat": rec["origin_chat"], "origin_thread": rec["origin_thread"]}

    def settle(self, rid: str, claim: str, outcome: str, evidence: Optional[Mapping[str, Any]] = None) -> None:
        if outcome not in ("done", "failed", "unknown"):
            raise DeferredError("outcome is done, failed or unknown")
        if outcome == "done" and not evidence:
            raise DeferredError("done needs operation-specific evidence")

        def check(rec):
            if rec.get("claim") != claim:
                raise DeferredError("another claim")
        self._store().transition(rid, ("claimed",), outcome, outcome, check, evidence=dict(evidence or {}))

    def cancel(self, rid: str) -> None:
        """In a native turn, by the original requester (adopters may add their owners)."""
        from gateway.ingress_identity import get_trusted_ingress
        ti = get_trusted_ingress()
        if ti is None:
            raise DeferredError("cancel needs a native turn")

        def check(rec):
            if rec["requester"] != ti.user_id or rec["profile"] != ti.profile:
                raise DeferredError("only the requester can cancel it")
        self._store().transition(rid, ("pending", "noticed", "active"), "cancelled", f"by {ti.user_id}", check)


def _thread_of(ti) -> str:
    """The origin thread of a native turn as the session binds it ("" for a top-level message): the same value a
    Switchboard binding captures, so the resume event's origin compares equal."""
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_THREAD_ID", "") or ""

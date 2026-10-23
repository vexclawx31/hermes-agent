#!/usr/bin/env python3
"""Session Search Tool - long-term conversation recall over the SQLite session DB.

Single-shape tool; the mode is inferred from the args: DISCOVERY (``query``;
FTS5 deduped by lineage, adaptive detail hydrates only the top result),
SCROLL (``session_id`` + ``around_message_id``; ±window around the anchor),
READ (``session_id`` alone; whole session or head/tail), BROWSE (no args).
No LLM calls — every shape returns actual DB messages.
"""

import contextvars
import json
import logging
import re
import time
from datetime import datetime, timezone, UTC
from typing import Any, Dict, List, Optional, Union

from hermes_state_common import _BOUNDARY_END_REASONS
from hermes_time import safe_strftime

# Hidden from browsing/searching — integrations (HERMES_SESSION_SOURCE=tool), delegate
# subagent runs, kanban workers are not the user's history.
_HIDDEN_SESSION_SOURCES = ("kanban", "subagent", "tool")
# Searchable but DEMOTED below interactive sessions: cron vocabulary dominates bare
# BM25 and starves out the user's own sessions ("recall blindness").
# Automation sources that are kept searchable but DEMOTED below interactive sessions in discover ranking.
# Cron jobs run on a schedule and accumulate large volumes of repetitive vocabulary (recurring project
# names, dates, "session", summaries); under bare BM25 they dominate the top-N FTS rows and starve out the
# user's own interactive sessions, producing "recall blindness" where only cron sessions surface (#19434).
# Demoting — not excluding — keeps cron content reachable when it's the only match, while interactive
# sessions always win when both match.
_DEMOTED_SESSION_SOURCES = ("cron",)

# Read-shape per-message content cap. #69334 capped discovery bookends (1200) and
# scroll windows (4000) but left ``_read_session`` returning whole messages, so a
# single archived tool result stored as a message could come back verbatim - one
# read returned 74K chars and took a request from ~50K to ~89K tokens in a step.
# Bounding message COUNT (head/tail) is not enough when content per message is
# unbounded; the agent can scroll around a message for detail (#114344).
_READ_MAX_CONTENT = 2000
# FTS rows scanned before dedup-by-lineage — well above the distinct sessions a query
# returns, so interactive matches buried under cron hits survive the demotion pass.
_DISCOVER_SCAN_LIMIT = 300
# exclude_session_ids: ids already inspected this task; capped so a runaway list can't fan out lineage walks.
_EXCLUDE_SESSION_IDS_CAP = 20
# Relative time bounds: "7d" / "24h" / "2w" = now minus N hours/days/weeks.
_RELATIVE_BOUND_RE = re.compile(r"^(\d+)\s*(h|d|w)$", re.IGNORECASE)
_RELATIVE_UNIT_SECONDS = {"h": 3600, "d": 86400, "w": 604800}
# Raw FTS rows are only a plan input; the response hydrates its own window/bookends.
_DISCOVER_SEARCH_FIELDS = ("id", "session_id", "role", "snippet", "source", "model", "session_started")
# Compaction handoff summaries (agent/context_compressor.py); excluded from bookends.
_COMPACTION_PREFIXES = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:")
# /new, /reset, idle/daily expiry and CLI /new ("new_session") end the predecessor WITHOUT
# carrying its transcript forward — unlike compression continuations and live delegation
# children. The store's boundary set, so the two cannot drift.
_FRESH_RESET_END_REASONS = _BOUNDARY_END_REASONS


def _quiet(fn, default, msg, *log_args, with_exc: bool = False):
    """``fn()``, or *default* after debug-logging *msg* (+ the exception when *with_exc*)."""
    try:
        return fn()
    except Exception as e:
        logging.debug(msg, *(log_args + (e,) if with_exc else log_args), exc_info=True)
        return default


def _loud(fn, log_msg, error_prefix, *log_args):
    """``(fn(), None)``, or ``(None, tool_error_json)`` after an error-level log — for DB
    calls whose failure the model must see."""
    try:
        return fn(), None
    except Exception as e:
        logging.error(log_msg, *log_args, e, exc_info=True)
        return None, tool_error(f"{error_prefix}: {e}", success=False)


def _format_timestamp(ts: float | str | None) -> str:
    """Unix timestamp -> readable date; ISO strings pass through; "unknown" for None."""
    if ts is None:
        return "unknown"
    if isinstance(ts, str) and not ts.replace(".", "").replace("-", "").isdigit():
        return ts
    return _quiet(lambda: safe_strftime(datetime.fromtimestamp(float(ts)), "%B %d, %Y at %I:%M %p"), str(ts),
                  "Failed to format timestamp %s: %s", ts, with_exc=True)


def _get_session_meta(db, session_id: str) -> dict:
    """``db.get_session`` that degrades to ``{}`` on error."""
    return _quiet(lambda: db.get_session(session_id), None,
                  "get_session failed for %s: %s", session_id, with_exc=True) or {}


def _session_meta_block(meta: dict[str, Any]) -> dict[str, Any]:
    return {"when": _format_timestamp(meta.get("started_at")), "source": meta.get("source"),
            "model": meta.get("model"), "title": meta.get("title")}


def _ok(**payload) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False)


def _is_compaction_summary(content: str) -> bool:
    return bool(content) and content.lstrip().startswith(_COMPACTION_PREFIXES)


def _resolve_to_parent(db, session_id: str) -> tuple[str, bool]:
    """Walk parent_session_id to the root -> ``(root_id, has_compression_hop)``; the flag
    separates a compression-split lineage (parent summarised away) from a delegation
    lineage (child still visible to the parent)."""
    visited: set[str] = set()
    cur, has_compression = session_id, False
    while cur and cur not in visited:
        visited.add(cur)
        s = _get_session_meta(db, cur)
        has_compression = has_compression or s.get("end_reason") == "compression"
        if not s.get("parent_session_id"):
            break
        cur = s["parent_session_id"]
    return cur, has_compression


def _resolve_lineage(db, session_id: str) -> str:
    return _resolve_to_parent(db, session_id)[0]


def _parse_iso_bound(value: Optional[str]) -> Optional[int]:
    """Parse an ISO date/datetime OR a relative duration into a UTC unix timestamp.

    ISO: a date-only value (``YYYY-MM-DD``) is midnight UTC on that day (the SQL
    ``before`` predicate is exclusive, so ``before=2026-07-01`` keeps June and drops
    July 1 00:00). Relative: ``"7d"``, ``"24h"``, ``"2w"`` = now minus N
    hours/days/weeks, so ``after="7d"`` is the last week and ``before="7d"`` is
    everything older than a week.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if rel := _RELATIVE_BOUND_RE.match(text):
        return int(time.time()) - int(rel.group(1)) * _RELATIVE_UNIT_SECONDS[rel.group(2).lower()]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"invalid time bound: {value!r} (expected ISO date/datetime like "
                         "2026-07-01, or a relative duration like 7d, 24h, 2w)") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _coerce_started_ts(value: Any) -> Optional[int]:
    """``sessions.started_at`` as an int unix timestamp (numeric or ISO text); None if unusable."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    try:
        return int(float(text))
    except (TypeError, ValueError):
        pass
    try:
        return _parse_iso_bound(text)
    except ValueError:
        return None


def _in_time_window(started_ts: Optional[int], after_ts: Optional[int], before_ts: Optional[int]) -> bool:
    """``after_ts <= started_ts < before_ts``; an unknown start fails any bound."""
    if after_ts is None and before_ts is None:
        return True
    if started_ts is None:
        return False
    return (after_ts is None or started_ts >= after_ts) and (before_ts is None or started_ts < before_ts)


def _normalize_exclude_session_ids(raw: Any) -> list[str]:
    """Deduped, stripped session ids from a str or list, capped at ``_EXCLUDE_SESSION_IDS_CAP``."""
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else []
    out: list[str] = []
    for item in items:
        sid = item.strip() if isinstance(item, str) else ""
        if sid and sid not in out:
            out.append(sid)
        if len(out) >= _EXCLUDE_SESSION_IDS_CAP:
            break
    return out


def _excluded_lineage_roots(db, exclude_session_ids: list[str]) -> set[str]:
    """The excluded ids plus their lineage roots, so a child id also hides its parent chain."""
    roots: set[str] = set()
    for sid in exclude_session_ids:
        roots.add(sid)
        roots.add(_resolve_lineage(db, sid) or sid)
    return roots


def _same_lineage(db, a: str, b: str) -> bool:
    a_root = _resolve_lineage(db, a)
    return bool(a_root and a_root == _resolve_lineage(db, b))


def _session_left_live_context(db, session_id: str) -> bool:
    """True when the transcript left everyone's live context: ``compression``
    (summarised into the child) or a fresh reset (child starts empty). Live delegation
    children (``end_reason is None``) and ``branched`` parents (copied verbatim into
    the branch) ARE the current context, so they stay excluded from recall."""
    end_reason = (session_id and _get_session_meta(db, session_id).get("end_reason")) or None
    return end_reason == "compression" or end_reason in _FRESH_RESET_END_REASONS


def _get_message_storage_state(db, message_id) -> Optional[dict[str, Any]]:
    """Owning session and visibility flags for *message_id* (None if missing/error)."""
    def _lookup():
        with db._lock:
            return db._conn.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?", (message_id,)).fetchone()
    row = message_id and _quiet(_lookup, None, "message storage-state lookup failed for %s", message_id)
    return dict(row) if row else None


def _is_compacted_state(state: Optional[dict[str, Any]]) -> bool:
    """Compaction archives are ``active=0, compacted=1``; rewind/undo rows are
    ``active=0, compacted=0`` and must stay hidden."""
    return state is not None and state["active"] == 0 and state["compacted"] == 1


def _is_compacted_message(db, message_id) -> bool:
    """True for a compaction-archived row: no longer in live context, so discoverable
    even on the current session. False on any error."""
    return _is_compacted_state(_get_message_storage_state(db, message_id))


# Set per session_search call under ``privacy.recall_scope: attribute`` (agent/recall_scope.py) to the
# call's author-id redactor: every shaped user row then names its author from the persisted trusted sender.
_ATTRIBUTE_AUTHORS: contextvars.ContextVar[Any] = contextvars.ContextVar("session_search_attribute", default=None)


def _author_tag(m: dict[str, Any], redact: Any = None) -> dict[str, Any]:
    """Who wrote a stored user row: its trusted persisted sender, else ``{"verified": False}``.
    Display names and text prefixes in the content never count."""
    meta = m.get("display_metadata")
    if isinstance(meta, str):
        meta = _quiet(lambda: json.loads(meta), {}, "sender metadata decode failed")
    sender = meta.get("sender") if isinstance(meta, dict) else None
    if not isinstance(sender, dict) or sender.get("trusted") is not True or not sender.get("user_id"):
        return {"verified": False}
    user_id = redact(sender.get("platform"), sender.get("user_id")) if redact else sender.get("user_id")
    tag = {"platform": sender.get("platform"), "user_id": user_id, "name": sender.get("name"),
           "chat_type": sender.get("chat_type"), "verified": True}
    return {k: v for k, v in tag.items() if v is not None}


def _shape_message(m: dict[str, Any], anchor_id: Optional[int] = None,
                   max_content_len: Optional[int] = None) -> dict[str, Any]:
    """Slim a message row; keeps ``content`` even when empty (tool-call-only turns)."""
    content = m.get("content")
    if isinstance(content, str) and "\x1b" in content:  # archived terminal output carries ANSI
        from tools.ansi_strip import strip_ansi
        content = strip_ansi(content)
    entry = {"id": m.get("id"), "role": m.get("role"), "content": content, "timestamp": m.get("timestamp")}
    entry.update({k: m.get(k) for k in ("tool_name", "tool_calls", "tool_call_id") if m.get(k)})
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    if max_content_len and content and len(content) > max_content_len:
        entry.update(content=content[:max_content_len] + "…", content_truncated=True,
                     original_content_chars=len(content))
    if m.get("role") == "user" and (redact := _ATTRIBUTE_AUTHORS.get()) is not None:
        entry["author"] = _author_tag(m, redact)
    return {k: v for k, v in entry.items() if v is not None or k == "content"}


def _session_link(session_id: str, profile: str | None = None) -> str:
    """The reference the agent writes for a session — same value the desktop composer
    emits, so it renders as a titled link. The profile segment is omitted when it
    can't be named confidently (a bare id still resolves, just not across profiles)."""
    def _active():
        from hermes_cli.profiles import get_active_profile_name
        resolved = get_active_profile_name()
        return "" if resolved == "custom" else resolved
    name = (profile or "").strip() or _quiet(_active, "", "get_active_profile_name failed for session link")
    return f"@session:{name}/{session_id}" if name else f"@session:{session_id}"


def _discovery_entry(lineage_root: Optional[str], **fields) -> dict[str, Any]:
    """Canonical key order; ``parent_session_id`` set when the hit lives in a child."""
    entry = {k: fields[k] for k in (
        "session_id", "when", "source", "model", "title", "matched_role", "match_message_id", "snippet",
        "bookend_start", "messages", "bookend_end", "messages_before", "messages_after", "detail")}
    if lineage_root and lineage_root != entry["session_id"]:
        entry["parent_session_id"] = lineage_root
    return entry


def _title_match_result(db, query: str, current_lineage_root: Optional[str]) -> Optional[dict[str, Any]]:
    """Discovery-shaped result when the query matches a session title, else None."""
    title_query = query.strip().strip("`'\"")  # models often quote a remembered title
    session_id = title_query and _quiet(lambda: db.resolve_session_by_title(title_query), None,
                                        "resolve_session_by_title failed for %r", title_query)
    if not session_id:
        return None
    lineage_root = _resolve_lineage(db, session_id)
    # Same-lineage title hits are in-context only while the session is live;
    # /new-reset and compression-ended parents are not.
    if current_lineage_root and lineage_root == current_lineage_root and not _session_left_live_context(db, session_id):
        return None
    session_meta = _quiet(lambda: db.get_session(lineage_root) or db.get_session(session_id), None,
                          "get_session failed for title match %s", session_id) or {}
    if session_meta.get("source") in _HIDDEN_SESSION_SOURCES:
        return None
    messages = _quiet(lambda: db.get_messages(session_id), [], "get_messages failed for title match %s", session_id)
    anchor_id = messages[0].get("id") if messages else None
    view = {} if anchor_id is None else _quiet(
        lambda: db.get_anchored_view(session_id, anchor_id, window=5, bookend=3), {},
        "get_anchored_view failed for title match %s/%s", session_id, anchor_id)
    title = session_meta.get("title") or title_query
    # Same caps as FTS hits (_bookend / _hydrate_hit): a title match is a discovery entry too.
    def shape(key, fallback, anchor=None, max_content_len=1200):
        return [_shape_message(m, anchor_id=anchor, max_content_len=max_content_len)
                for m in (view.get(key) or fallback)]
    return {**_discovery_entry(
        lineage_root, session_id=session_id, when=_format_timestamp(session_meta.get("started_at")),
        source=session_meta.get("source", "unknown"), model=session_meta.get("model") or "unknown",
        title=title, matched_role="session_title", match_message_id=anchor_id,
        snippet=f"Session title matched: {title}",
        bookend_start=shape("bookend_start", messages[:3]),
        messages=shape("window", messages[:5], anchor_id, max_content_len=4000),
        bookend_end=shape("bookend_end", messages[-3:]), messages_before=view.get("messages_before", 0),
        messages_after=view.get("messages_after", max(len(messages) - 5, 0)), detail="full"),
        "_lineage_root": lineage_root}


def _discover_payload(db, query: str, detail: str, results: list, **extra) -> str:
    """Discovery response; notes FTS backfill progress so the agent can explain thin
    results instead of treating them as ground truth."""
    status = _quiet(db.fts_rebuild_status, None, "fts_rebuild_status failed")
    rebuild = {} if status is None else {"index_rebuild": {"percent": status["percent"], "note": (
        f"The search index is rebuilding in the background ({status['percent']}% done, "
        f"{status['indexed']:,} of {status['total']:,} messages). Results from older messages "
        f"may be incomplete until it finishes.")}}
    return _ok(mode="discover", query=query, detail=detail, results=results, count=len(results), **extra, **rebuild)


def _bookend(view: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [_shape_message(m, max_content_len=1200) for m in (view.get(key) or [])
            if not _is_compaction_summary(m.get("content", ""))]


def _hydrate_hit(db, lineage_root: str, match_info: dict[str, Any], result_detail: str) -> Optional[dict[str, Any]]:
    """Discovery result from a surviving FTS row; None (dropped) if the view can't load."""
    hit_sid, msg_id = match_info.get("session_id") or lineage_root, match_info.get("id")
    try:
        view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
    except Exception as e:
        logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
        return None
    session_meta, full = _get_session_meta(db, lineage_root), result_detail == "full"
    return _discovery_entry(
        lineage_root, session_id=hit_sid,
        when=_format_timestamp(session_meta.get("started_at") or match_info.get("session_started")),
        source=session_meta.get("source") or match_info.get("source", "unknown"),
        model=session_meta.get("model") or match_info.get("model") or "unknown",
        title=session_meta.get("title") or None, matched_role=match_info.get("role"),
        match_message_id=msg_id, snippet=match_info.get("snippet") or "",
        bookend_start=_bookend(view, "bookend_start") if full else [],
        messages=[_shape_message(m, anchor_id=msg_id, max_content_len=4000)
                  for m in (view.get("window") or []) if full or m.get("id") == msg_id],
        bookend_end=_bookend(view, "bookend_end") if full else [],
        messages_before=view.get("messages_before", 0), messages_after=view.get("messages_after", 0),
        detail=result_detail)


def _discover(db, query: str, role_filter: Optional[list[str]], limit: int, sort: Optional[str],
              detail: str, current_session_id: str | None = None, link_profile: str | None = None,
              after_ts: Optional[int] = None, before_ts: Optional[int] = None,
              exclude_session_ids: Optional[list[str]] = None) -> str:
    """Discovery shape: FTS5 plus adaptive or full result hydration."""
    current_lineage_root = _resolve_lineage(db, current_session_id) if current_session_id else None
    excluded_roots = _excluded_lineage_roots(db, exclude_session_ids or [])
    title_result = _title_match_result(db, query, current_lineage_root)
    # FTS rows are time-bounded in SQL (_search_filter_clauses); the title match bypasses that
    # query, so it is the one place the window is re-checked in Python.
    if title_result:
        title_sid, title_root = title_result["session_id"], title_result.get("_lineage_root") or title_result["session_id"]
        title_started = _coerce_started_ts((_get_session_meta(db, title_root) or _get_session_meta(db, title_sid)).get("started_at"))
        if {title_sid, title_root} & excluded_roots or not _in_time_window(title_started, after_ts, before_ts):
            title_result = None
    raw_results, err = _loud(lambda: db.search_messages(
        query=query, role_filter=role_filter or ["user", "assistant"],
        exclude_sources=list(_HIDDEN_SESSION_SOURCES), limit=_DISCOVER_SCAN_LIMIT, offset=0, sort=sort,
        fields=_DISCOVER_SEARCH_FIELDS, after_ts=after_ts, before_ts=before_ts), "FTS5 search failed: %s", "Search failed")
    if err:
        return err
    # Demote cron rows below interactive ones BEFORE dedup so a high-volume cron corpus
    # can't starve the user's own sessions out of the top `limit`; stable sort keeps BM25
    # order within each class.
    raw_results = sorted(raw_results, key=lambda r: (r.get("source") or "") in _DEMOTED_SESSION_SOURCES)
    # See #19434.
    if not raw_results and not title_result:
        return _discover_payload(db, query, detail, [], message=(
            "No matching sessions found. FTS5 ANDs all terms by default — "
            "broaden with OR (`alpha OR beta`), exact-match with quoted "
            "phrases, exclude with NOT, or prefix-match with `deploy*`."))
    seen_sessions: dict[str, dict[str, Any]] = {}
    results = [title_result] if title_result else []
    if title_result and (title_lineage := title_result.pop("_lineage_root", None)):
        seen_sessions[title_lineage] = {"_title_only": True}
    # Dedupe by lineage (lineage_root -> first surviving FTS row) up to `limit`. The raw
    # owning session_id stays on the row — only it pairs validly with the FTS match id.
    # Current-lineage hits are skipped UNLESS the transcript left live context
    # (compression-ended, /new-reset predecessor, or an in-place compacted row on the
    # SAME session); a live delegation child (end_reason=None) stays excluded.
    for r in raw_results:
        if len(seen_sessions) >= limit:
            break
        raw_sid, resolved_sid = r["session_id"], _resolve_lineage(db, r["session_id"])
        if raw_sid in excluded_roots or resolved_sid in excluded_roots:
            continue
        # Skip the current session lineage — UNLESS the hit's transcript has left live context. Three
        # sub-cases: Legacy compression rotation: the FTS hit lives in a session that itself ended with
        # end_reason='compression'. That session's content has been replaced by a summary in the
        # continuation child, so it must stay discoverable. /new-reset (and idle/daily/CLI new_session): the
        # predecessor was ended without carrying any transcript into the child. Same lineage root, but the
        # prior conversation is NOT in the active context — hiding it made gateway recall go blind after
        # every /new (#85756). A live delegation child has end_reason=None, so it stays excluded. In-place
        # compaction: the FTS hit lives on the SAME session_id as the current session, but the matched
        # message row is an archived (active=0, compacted=1) row. The live-context load filters active=1, so
        # that content is no longer in context — let it through.
        is_compacted_hit = _is_compacted_message(db, r.get("id"))
        if current_lineage_root and resolved_sid == current_lineage_root and not (
                _session_left_live_context(db, raw_sid) or is_compacted_hit):
            continue
        if current_session_id and raw_sid == current_session_id and not is_compacted_hit:
            continue
        seen_sessions.setdefault(resolved_sid, {**r, "_lineage_root": resolved_sid})
    for lineage_root, match_info in seen_sessions.items():
        if match_info.get("_title_only"):
            continue
        # Adaptive: only the top-ranked result is fully hydrated.
        entry = _hydrate_hit(db, lineage_root, match_info, "full" if detail == "full" or not results else "compact")
        if entry is not None:
            results.append(entry)
    for entry in results:
        entry["link"] = _session_link(entry["session_id"], link_profile)
    return _discover_payload(db, query, detail, results, sessions_searched=len(seen_sessions), link_hint=(
        "When referring the user to a session, write its `link` value "
        "verbatim inline mid-sentence (it renders as a titled link) — never "
        "as markdown, in backticks, on its own line, or next to the "
        "title/id/date. To read more around a compact result, scroll: "
        "session_search(session_id=..., around_message_id=match_message_id)."))


def _resolve_profile_db(profile: str):
    """Another profile's ``state.db`` opened read-only (safe on a live DB); None = current."""
    if profile is None or not str(profile).strip():
        return None
    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB
    canon = profiles_mod.normalize_profile_name(profile)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise ValueError(f"profile '{canon}' does not exist")
    return SessionDB(db_path=profiles_mod.get_profile_dir(canon) / "state.db", read_only=True)


def _read_session(db, session_id: str, head: int = 20, tail: int = 10, link_profile: str | None = None) -> str:
    """Read shape: whole session, or ``head`` + ``tail`` messages with a scroll pointer."""
    meta = _get_session_meta(db, session_id)
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    rows, err = _loud(lambda: db.get_messages(session_id), "get_messages failed for %s: %s", "failed to load session",
                      session_id)
    if err:
        return err
    shaped = [_shape_message(m, max_content_len=_READ_MAX_CONTENT) for m in rows]
    total, truncated = len(shaped), len(shaped) > head + tail
    return _ok(mode="read", session_id=session_id, link=_session_link(session_id, link_profile),
               session_meta=_session_meta_block(meta), message_count=total, truncated=truncated,
               messages=shaped[:head] + shaped[-tail:] if truncated else shaped,
               **({"message": (f"Session has {total} messages; showing first {head} + last {tail}. "
                               "Pass around_message_id (any id above) to scroll the middle.")} if truncated else {}))


def _read_scoped(db, sid: str, profile: Optional[str]) -> str:
    """Read shape scoped to ONE store: the caller's profile, or the profile it named.

    A miss is a miss. Profiles are isolated islands, so a bare id never falls through to
    a scan of every other profile's ``state.db`` — that returned another profile's full
    transcript to any caller holding the id (#106761). The hint tells the model how to
    ask properly: ``@session:<profile>/<id>`` or ``profile=``.
    """
    result = _read_session(db, sid, link_profile=profile)
    if json.loads(result).get("success") is not False or profile:
        return result
    return tool_error(f"session_id not found in this profile: {sid}. If it belongs to another "
                      "profile, pass profile=<name> (or the @session:<profile>/<id> link).", success=False)


def _list_recent_sessions(db, limit: int, current_session_id: str | None = None, link_profile: str | None = None) -> str:
    """Browse shape: metadata for the most recent sessions (no LLM, no FTS5)."""
    def _browse():
        # Never use list_sessions_rich(order_by_last_active=True) here: it walks every
        # compression chain and derives activity/previews before LIMIT, which can
        # monopolise a gateway callback for minutes on a multi-GB state.db. The
        # bounded browse query preselects an indexed candidate set and carries a
        # cooperative SQLite VM cancellation deadline. Fail closed rather than
        # silently falling back to the whole-database query shape.
        bounded_list = getattr(db, "list_recent_sessions_bounded", None)
        if bounded_list is None:
            raise RuntimeError("session database does not support bounded recent-session browse")
        sessions = bounded_list(
            limit=limit + 15,  # extra so we can skip current / compression roots
            exclude_sources=list(_HIDDEN_SESSION_SOURCES), timeout_seconds=3.0)
        current_root, has_compression_hop = (
            _resolve_to_parent(db, current_session_id) if current_session_id else (None, False))
        # Compression continuation: the root was summarised into the live child, so hide
        # it. /new-reset children carry no transcript — keep that root browsable.
        hidden = {current_session_id, current_root if has_compression_hop and current_root else None}
        results = [{
            "session_id": s.get("id", ""), "link": _session_link(s.get("id", ""), link_profile),
            "title": s.get("title") or None, **{k: s.get(k, "") for k in ("source", "started_at", "last_active")},
            "message_count": s.get("message_count", 0), "preview": s.get("preview", "")}
            for s in [x for x in sessions if x.get("id", "") not in hidden][:limit]]
        return _ok(mode="browse", results=results, count=len(results), message=(
            f"Showing {len(results)} most recent sessions. Pass a query= to search, "
            "or session_id+around_message_id to scroll."))

    out, err = _loud(_browse, "Error listing recent sessions: %s", "Failed to list recent sessions")
    return err or out


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def _anchor_in_live_context(db, anchor_state, anchor_sid: str, current_session_id: str) -> bool:
    """True when the scroll anchor is still in the caller's active context (reject).
    Same-lineage history that LEFT live context (compacted rows, compression-ended
    parents, /new-reset predecessors) passes, so scroll never rejects a discovery result.
    Rewind/undo rows (active=0, compacted!=1) never count as out-of-context history."""
    if not _same_lineage(db, anchor_sid, current_session_id) or _is_compacted_state(anchor_state):
        return False
    return (anchor_state is not None and anchor_state["active"] == 0) or not _session_left_live_context(db, anchor_sid)


def _scroll(db, session_id: str, around_message_id: int, window: int = 5,
            current_session_id: str | None = None) -> str:
    """Scroll shape: a window centered on an anchor (no FTS5, no bookends)."""
    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)
    window = _clamp_int(window, 5, 1, 20)
    # Locate the anchor BEFORE the current-lineage guard (see _anchor_in_live_context).
    anchor_state = _get_message_storage_state(db, around_message_id)
    owning = (anchor_state or {}).get("session_id")
    if current_session_id and _anchor_in_live_context(db, anchor_state, owning or session_id, current_session_id):
        return tool_error("scroll rejected: anchor lives in the current session lineage (already in your active context)", success=False)
    session_meta = _get_session_meta(db, session_id)
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    view, err = _loud(lambda: db.get_messages_around(session_id, around_message_id, window=window),
                      "get_messages_around failed: %s", "failed to load messages")
    if err:
        return err
    messages = view.get("window") or []
    extra = {}
    if not messages and owning and owning != session_id:
        # Lineage rebind: the caller paired a parent session_id with a message id
        # living in a descendant — serve the owner's window transparently.
        rebind_view = _same_lineage(db, session_id, owning) and _quiet(
            lambda: db.get_messages_around(owning, around_message_id, window=window),
            None, "rebind get_messages_around failed: %s", with_exc=True)
        if rebind_view and rebind_view.get("window"):
            extra["warning"] = (f"around_message_id {around_message_id} lives in {owning} "
                                f"(child of {session_id}); rebound transparently")
            view, messages, session_id = rebind_view, rebind_view["window"], owning
            session_meta = _get_session_meta(db, owning) or session_meta
    if not messages:
        return tool_error(f"around_message_id {around_message_id} not in session_id {session_id}", success=False)
    return _ok(
        mode="scroll", session_id=session_id, around_message_id=around_message_id,
        session_meta=_session_meta_block(session_meta), window=window,
        messages=[_shape_message(m, anchor_id=around_message_id) for m in messages],
        messages_before=view.get("messages_before", 0), messages_after=view.get("messages_after", 0),
        hint=("Scroll forward: re-call with around_message_id = the LAST message's "
              "id; backward: the FIRST message's id (the boundary message repeats "
              "as an orientation marker). messages_before/messages_after < window "
              "means you've hit that end of the session."), **extra)


# ---------------------------------------------------------------------------
# Sender-scoped mode: MESSAGE-level attribution from the persisted
# ``display_metadata.sender`` object (gateway/sender_attribution.py). Session
# participation never qualifies a message; unknown attribution fails closed.
# ---------------------------------------------------------------------------
_SENDER_NOTE = (
    "Sender-scoped: only user messages whose persisted platform sender matches the filter are "
    "returned. Assistant replies, other participants' messages, session titles, previews and "
    "messages with missing, untrusted or ambiguous sender attribution are omitted.")
_SENDER_BROWSE_SCAN = 60


class _SenderFilter:
    """Model-chosen sender filters, intersected with the turn's enforced recall scope when there is one.
    ``platform``/``user_id`` are None only under a scope (agent/recall_scope.py), which then decides alone."""
    __slots__ = ("chat_id", "min_timestamp", "platform", "require_unthreaded", "scope", "user_id")

    def __init__(self, platform: Optional[str], user_id: Optional[str], chat_id: Optional[str],
                 require_unthreaded: bool, scope: Any = None):
        self.platform, self.user_id, self.chat_id = platform, user_id, chat_id
        self.require_unthreaded = require_unthreaded
        self.scope = scope
        # Under a scope: rows written before enforcement began are legacy and never returned.
        self.min_timestamp: Optional[float] = None

    def describe(self) -> dict[str, Any]:
        out = {"sender_platform": self.platform, "sender_user_id": self.user_id,
               "sender_chat_id": self.chat_id, "require_unthreaded": self.require_unthreaded}
        if self.scope is not None:
            out["recall_scope"] = self.scope.describe()
        return out

    def matches(self, display_metadata: Any) -> bool:
        """True only for a trusted, structurally complete sender object matching every filter."""
        if isinstance(display_metadata, str):
            try:
                display_metadata = json.loads(display_metadata)
            except (TypeError, ValueError):
                return False
        sender = display_metadata.get("sender") if isinstance(display_metadata, dict) else None
        if not isinstance(sender, dict) or sender.get("trusted") is not True:
            return False
        if self.scope is not None:
            from agent.recall_scope import sender_label
            if not self.scope.allows({sender_label(display_metadata)}):
                return False
        elif not self.platform or not self.user_id:
            return False
        if self.platform and str(sender.get("platform") or "").strip().lower() != self.platform:
            return False
        if self.user_id and str(sender.get("user_id") or "") != self.user_id:
            return False
        if self.chat_id is not None and str(sender.get("chat_id") or "") != self.chat_id:
            return False
        if self.require_unthreaded and sender.get("thread_reply") is not False:
            return False
        return True


def _parse_sender_filter(sender_platform, sender_user_id, sender_chat_id, require_unthreaded,
                         recall_scope=None, enforce_scope=False):
    """``(filter_or_None, error_json_or_None)``. Any sender arg turns the mode on; an incomplete
    filter is refused rather than silently widened to an unscoped search. Under an enforced recall
    scope the mode is always on, the scope always applies and model args can only narrow it; with
    enforcement on and no scope the search is refused (unknown requester fails closed)."""
    if enforce_scope:
        if recall_scope is None:
            return None, tool_error("session_search is unavailable: this turn has no verified recall scope "
                                    "(sender attribution is enforced on this profile)", success=False)
        flt, err = _parse_sender_filter(sender_platform, sender_user_id, sender_chat_id, require_unthreaded)
        if err:
            return None, err
        if flt is None:
            flt = _SenderFilter(None, None, None, False)
        flt.scope = recall_scope
        from agent.recall_scope import enforcement_epoch
        flt.min_timestamp = enforcement_epoch(True)
        return flt, None
    def _s(v):
        return v.strip() if isinstance(v, str) else ("" if v is None else str(v).strip())
    platform, user_id, chat_id = _s(sender_platform).lower(), _s(sender_user_id), _s(sender_chat_id)
    flag = require_unthreaded
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "1", "yes")
    if not (platform or user_id or chat_id or flag):
        return None, None
    if not platform or not user_id:
        return None, tool_error("sender-scoped search requires both sender_platform and sender_user_id "
                                "(refusing to widen to an unscoped search)", success=False)
    return _SenderFilter(platform, user_id, chat_id or None, bool(flag)), None


def _sender_rows(db, session_id: str, flt: "_SenderFilter") -> list[dict[str, Any]]:
    """Visible user rows in *session_id* whose persisted sender matches *flt*, in id order.
    Visible = live (active=1) or compaction-archived (active=0, compacted=1); rewind/undo rows stay hidden.
    SQL only narrows to rows that carry a sender; ``flt.matches`` decides."""
    def _q():
        with db._lock:
            return db._conn.execute(
                "SELECT id, session_id, role, content, timestamp, display_metadata FROM messages "
                "WHERE session_id = ? AND role = 'user' AND (active = 1 OR compacted = 1) "
                "AND CASE WHEN json_valid(display_metadata) "
                "THEN json_extract(display_metadata, '$.sender.user_id') END = coalesce(?, "
                "CASE WHEN json_valid(display_metadata) "
                "THEN json_extract(display_metadata, '$.sender.user_id') END) ORDER BY id",
                (session_id, flt.user_id)).fetchall()
    rows = [dict(r) for r in _q()]
    floor = flt.min_timestamp
    return [r for r in rows if flt.matches(r.get("display_metadata"))
            and (floor is None or (isinstance(r.get("timestamp"), (int, float)) and r["timestamp"] >= floor))]


def _count_session_messages(db, session_id: str) -> int:
    def _q():
        with db._lock:
            return db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)",
                (session_id,)).fetchone()[0]
    return int(_quiet(_q, 0, "count messages failed for %s", session_id) or 0)


def _shape_sender_message(row: dict[str, Any], anchor_id: Optional[int] = None,
                          max_content_len: Optional[int] = 4000) -> dict[str, Any]:
    entry = _shape_message(row, anchor_id=anchor_id, max_content_len=max_content_len)
    meta = row.get("display_metadata")
    if isinstance(meta, str):
        meta = _quiet(lambda: json.loads(meta), {}, "sender metadata decode failed")
    sender = dict((meta or {}).get("sender") or {})
    sender.pop("untrusted_reasons", None)
    sender.pop("name", None)  # display name: shown only in attribution's author tag
    entry["sender"] = sender
    entry["attribution"] = "verified_sender"
    return entry


def _sender_meta_block(meta: dict[str, Any]) -> dict[str, Any]:
    """Session metadata without the title: titles are generated from any participant's text."""
    return {"when": _format_timestamp(meta.get("started_at")), "source": meta.get("source")}


def _window_around(rows: list[dict[str, Any]], anchor_id: int, window: int) -> tuple[list, int, int]:
    ids = [r["id"] for r in rows]
    idx = ids.index(anchor_id)
    lo, hi = max(0, idx - window), min(len(rows), idx + window + 1)
    return rows[lo:hi], lo, len(rows) - hi


def _sender_discover(db, query: str, flt: "_SenderFilter", limit: int, sort: Optional[str], detail: str,
                     current_session_id: Optional[str], link_profile: Optional[str],
                     after_ts: Optional[int], before_ts: Optional[int],
                     exclude_session_ids: Optional[list[str]]) -> str:
    """Discovery restricted to verified-sender user rows (no title match, no bookends, no assistant rows)."""
    current_lineage_root = _resolve_lineage(db, current_session_id) if current_session_id else None
    excluded_roots = _excluded_lineage_roots(db, exclude_session_ids or [])
    raw_results, err = _loud(lambda: db.search_messages(
        query=query, role_filter=["user"], exclude_sources=list(_HIDDEN_SESSION_SOURCES),
        limit=_DISCOVER_SCAN_LIMIT, offset=0, sort=sort, fields=_DISCOVER_SEARCH_FIELDS,
        after_ts=after_ts, before_ts=before_ts), "FTS5 search failed: %s", "Search failed")
    if err:
        return err
    raw_results = sorted(raw_results, key=lambda r: (r.get("source") or "") in _DEMOTED_SESSION_SOURCES)
    verified_cache: dict[str, dict[int, dict[str, Any]]] = {}
    seen: dict[str, dict[str, Any]] = {}
    unverified_hits = 0
    for r in raw_results:
        if len(seen) >= limit:
            break
        raw_sid = r["session_id"]
        resolved_sid = _resolve_lineage(db, raw_sid)
        if raw_sid in excluded_roots or resolved_sid in excluded_roots:
            continue
        is_compacted_hit = _is_compacted_message(db, r.get("id"))
        if current_lineage_root and resolved_sid == current_lineage_root and not (
                _session_left_live_context(db, raw_sid) or is_compacted_hit):
            continue
        if current_session_id and raw_sid == current_session_id and not is_compacted_hit:
            continue
        if raw_sid not in verified_cache:
            verified_cache[raw_sid] = {row["id"]: row for row in _quiet(
                lambda: _sender_rows(db, raw_sid, flt), [], "sender rows failed for %s", raw_sid)}
        if r.get("id") not in verified_cache[raw_sid]:
            unverified_hits += 1
            continue
        seen.setdefault(resolved_sid, {**r, "_lineage_root": resolved_sid})
    results = []
    for lineage_root, hit in seen.items():
        sid, mid = hit["session_id"], hit["id"]
        rows = list(verified_cache[sid].values())
        full = detail == "full" or not results
        window_rows, before, after = _window_around(rows, mid, 5 if full else 0)
        meta = _get_session_meta(db, lineage_root) or _get_session_meta(db, sid)
        entry = {"session_id": sid, "when": _format_timestamp(meta.get("started_at") or hit.get("session_started")),
                 "source": meta.get("source") or hit.get("source", "unknown"), "matched_role": "user",
                 "match_message_id": mid,
                 "messages": [_shape_sender_message(m, anchor_id=mid) for m in window_rows],
                 "sender_messages_before": before, "sender_messages_after": after,
                 "detail": "full" if full else "compact", "link": _session_link(sid, link_profile)}
        if lineage_root != sid:
            entry["parent_session_id"] = lineage_root
        results.append(entry)
    return _discover_payload(db, query, detail, results, sessions_searched=len(seen),
                             sender_filter=flt.describe(), attribution_note=_SENDER_NOTE,
                             # Under a scope, whether others wrote a match is itself out of scope.
                             **({} if flt.scope is not None else {"omitted_unverified_hits": unverified_hits}))


def _sender_read(db, session_id: str, flt: "_SenderFilter", head: int = 20, tail: int = 10,
                 link_profile: Optional[str] = None) -> str:
    meta = _get_session_meta(db, session_id)
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    rows, err = _loud(lambda: _sender_rows(db, session_id, flt), "sender read failed for %s: %s",
                      "failed to load session", session_id)
    if err:
        return err
    shaped = [_shape_sender_message(m, max_content_len=_READ_MAX_CONTENT) for m in rows]
    total, truncated = len(shaped), len(shaped) > head + tail
    return _ok(mode="read", session_id=session_id, link=_session_link(session_id, link_profile),
               session_meta=_sender_meta_block(meta), message_count=total, truncated=truncated,
               # Under a recall scope, how many messages others wrote is itself out of scope.
               **({} if flt.scope is not None else
                  {"omitted_messages": max(_count_session_messages(db, session_id) - total, 0)}),
               messages=shaped[:head] + shaped[-tail:] if truncated else shaped,
               sender_filter=flt.describe(), attribution_note=_SENDER_NOTE)


def _sender_scroll(db, session_id: str, around_message_id, flt: "_SenderFilter", window: int = 5,
                   current_session_id: Optional[str] = None) -> str:
    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)
    window = _clamp_int(window, 5, 1, 20)
    anchor_state = _get_message_storage_state(db, around_message_id)
    owning = (anchor_state or {}).get("session_id")
    if current_session_id and _anchor_in_live_context(db, anchor_state, owning or session_id, current_session_id):
        return tool_error("scroll rejected: anchor lives in the current session lineage (already in your active context)", success=False)
    if owning and owning != session_id and _same_lineage(db, session_id, owning):
        session_id = owning
    meta = _get_session_meta(db, session_id)
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    rows, err = _loud(lambda: _sender_rows(db, session_id, flt), "sender scroll failed: %s", "failed to load messages")
    if err:
        return err
    if around_message_id not in {r["id"] for r in rows}:
        return tool_error(f"around_message_id {around_message_id} is not a verified message from this sender "
                          f"in session_id {session_id}", success=False)
    window_rows, before, after = _window_around(rows, around_message_id, window)
    return _ok(mode="scroll", session_id=session_id, around_message_id=around_message_id,
               session_meta=_sender_meta_block(meta), window=window,
               messages=[_shape_sender_message(m, anchor_id=around_message_id) for m in window_rows],
               sender_messages_before=before, sender_messages_after=after,
               sender_filter=flt.describe(), attribution_note=_SENDER_NOTE)


def _sender_browse(db, flt: "_SenderFilter", limit: int, current_session_id: Optional[str],
                   link_profile: Optional[str]) -> str:
    def _browse():
        bounded_list = getattr(db, "list_recent_sessions_bounded", None)
        if bounded_list is None:
            raise RuntimeError("session database does not support bounded recent-session browse")
        sessions = bounded_list(limit=_SENDER_BROWSE_SCAN, exclude_sources=list(_HIDDEN_SESSION_SOURCES),
                                timeout_seconds=3.0)
        results = []
        for s in sessions:
            sid = s.get("id", "")
            if not sid or sid == current_session_id:
                continue
            rows = _quiet(lambda: _sender_rows(db, sid, flt), [], "sender rows failed for %s", sid)
            if not rows:
                continue
            results.append({"session_id": sid, "link": _session_link(sid, link_profile),
                            "source": s.get("source", ""), "started_at": s.get("started_at", ""),
                            "sender_message_count": len(rows),
                            "last_sender_message_id": rows[-1]["id"],
                            "last_sender_message_at": rows[-1].get("timestamp")})
            if len(results) >= limit:
                break
        return _ok(mode="browse", results=results, count=len(results), sender_filter=flt.describe(),
                   attribution_note=_SENDER_NOTE, message=(
                       f"Showing {len(results)} recent sessions containing verified messages from this "
                       "sender. Read one with session_id (same sender filters)."))

    out, err = _loud(_browse, "Error listing sender sessions: %s", "Failed to list recent sessions")
    return err or out


def _dispatch(query, role_filter, limit, db, current_session_id, session_id,
              around_message_id, window, sort, profile, detail, owned_dbs,
              after=None, before=None, exclude_session_ids=None, sender=None) -> str:
    """Mode dispatch (see module docstring); scroll wins when an anchor is set.
    Profile DBs opened here are appended to *owned_dbs* for the caller to close."""
    # A raw `@session:<profile>/<id>` link as session_id: ids never contain "/", so
    # split on it and adopt the embedded profile only when none was passed.
    if isinstance(session_id, str) and "/" in session_id:
        emb_profile, _, emb_id = session_id.partition("/")
        if emb_id:
            session_id = emb_id
            if emb_profile and (profile is None or not str(profile).strip()):
                profile = emb_profile
    if sender is not None and sender.scope is not None and isinstance(profile, str) and profile.strip():
        return tool_error("session_search can't read another profile while sender attribution is enforced "
                          "on this one", success=False)
    # Cross-profile: swap in the named profile's DB (read-only) for every shape;
    # current-lineage guards key off ids that won't collide, so they stay inert.
    try:
        profile_db = _resolve_profile_db(profile)
    except Exception as e:
        return tool_error(f"profile '{profile}': {e}", success=False)
    if profile_db is not None:
        db, current_session_id = profile_db, None
        owned_dbs.append(profile_db)
    if isinstance(session_id, str) and session_id.strip():
        if sender is not None:
            if around_message_id is not None:
                return _sender_scroll(db, session_id.strip(), around_message_id, sender, window, current_session_id)
            return _sender_read(db, session_id.strip(), sender, link_profile=profile)
        if around_message_id is not None:
            return _scroll(db, session_id.strip(), around_message_id, window, current_session_id)
        return _read_scoped(db, session_id.strip(), profile)
    limit = _clamp_int(limit, 3, 1, 10)
    if not query or not isinstance(query, str) or not query.strip():
        if sender is not None:
            return _sender_browse(db, sender, limit, current_session_id, link_profile=profile)
        return _list_recent_sessions(db, limit, current_session_id, link_profile=profile)
    sort_norm = sort.strip().lower() if isinstance(sort, str) else None
    try:
        after_ts, before_ts = _parse_iso_bound(after), _parse_iso_bound(before)
    except ValueError as e:
        return tool_error(str(e), success=False)
    if sender is not None:
        return _sender_discover(
            db, query.strip(), sender, limit, sort_norm if sort_norm in ("newest", "oldest") else None,
            "full" if isinstance(detail, str) and detail.strip().lower() == "full" else "adaptive",
            current_session_id, profile, after_ts, before_ts, _normalize_exclude_session_ids(exclude_session_ids))
    return _discover(
        db=db, query=query.strip(), limit=limit, sort=sort_norm if sort_norm in ("newest", "oldest") else None,
        role_filter=([r.strip() for r in role_filter.split(",") if r.strip()] or None) if isinstance(role_filter, str) else None,
        detail="full" if isinstance(detail, str) and detail.strip().lower() == "full" else "adaptive",
        current_session_id=current_session_id, link_profile=profile, after_ts=after_ts, before_ts=before_ts,
        exclude_session_ids=_normalize_exclude_session_ids(exclude_session_ids))


def session_search(query: str = "", role_filter: str | None = None, limit: int = 3, db=None,
                   current_session_id: str | None = None, session_id: str | None = None, around_message_id: int | None = None,
                   window: int = 5, sort: str | None = None, profile: str | None = None, detail: str = "adaptive",
                   after: str | None = None, before: str | None = None, exclude_session_ids: Optional[list[str]] = None,
                   sender_platform: str | None = None, sender_user_id: str | None = None,
                   sender_chat_id: str | None = None, require_unthreaded: bool | None = None,
                   recall_scope: Any = None, enforce_scope: bool | None = None) -> str:
    """Run session search, closing DBs opened here. Positional order is frozen for old callers;
    new parameters are appended after ``detail``."""
    from hermes_state import format_session_db_unavailable
    from hermes_state_registry import acquire, release_or_close
    if enforce_scope is None:
        from agent.recall_scope import enforcement_enabled
        enforce_scope = enforcement_enabled()
    if enforce_scope and isinstance(profile, str) and profile.strip():
        return tool_error("session_search can't read another profile while sender attribution is enforced "
                          "on this one", success=False)
    sender, sender_err = _parse_sender_filter(sender_platform, sender_user_id, sender_chat_id, require_unthreaded,
                                              recall_scope=recall_scope, enforce_scope=enforce_scope)
    if sender_err:
        return sender_err
    owned_dbs: list[Any] = []
    if db is None:
        db = _quiet(acquire, None, "SessionDB unavailable for session_search")
        if db is None:
            return tool_error(format_session_db_unavailable(), success=False)
        owned_dbs.append(db)
    # Enforced searches already return each row's verified sender (_shape_sender_message).
    from agent.recall_scope import attribution_enabled, author_id_redactor
    attribute = not enforce_scope and _quiet(attribution_enabled, False, "attribution mode check failed")
    attribute_token = _ATTRIBUTE_AUTHORS.set(author_id_redactor() if attribute else None)
    try:
        return _dispatch(query, role_filter, limit, db, current_session_id, session_id,
                         around_message_id, window, sort, profile, detail, owned_dbs,
                         after=after, before=before, exclude_session_ids=exclude_session_ids, sender=sender)
    finally:
        _ATTRIBUTE_AUTHORS.reset(attribute_token)
        for owned_db in reversed(owned_dbs):
            _quiet(lambda: release_or_close(owned_db), None, "Failed to close session_search SessionDB")


def check_session_search_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import _default_db_path
        return _default_db_path().parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Recall past conversations: search or read old Hermes sessions (FTS5), or "
        "scroll inside one. Four shapes, picked by args: `query` = discovery "
        "(top-N matching sessions, top result fully hydrated); `session_id` + "
        "`around_message_id` = scroll (window of messages around an anchor); "
        "`session_id` alone = read a whole session — how you resolve an "
        "`@session:<profile>/<id>` link (split on '/' into profile + id); no "
        "args = browse recent sessions. Results are actual DB messages, no LLM. "
        "Searches conversation history ONLY — when the user gave a direct "
        "source (URL, file, contact, live system), inspect that first; never "
        "conclude 'not found' from history alone. Use for questions about past "
        "conversations: 'what did we do about X', 'where did we leave Y'. When "
        "referring the user to a session, write its `link` value verbatim "
        "inline (it renders as a titled link)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (discovery shape). Keywords, phrases, or boolean "
                    "expressions to find in past sessions. Omit to browse recent "
                    "sessions. Ignored when session_id + around_message_id are set "
                    "(scroll shape)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Discovery shape only. Max sessions to return (default 3, max 10). "
                    "Bump to 5–10 when the topic likely spans several sessions and you "
                    "want to pick the right one to scroll into."
                ),
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": (
                    "Discovery shape only. Temporal bias on top of FTS5 ranking: omit "
                    "for relevance-only (exploratory recall), 'newest' for "
                    "\"where did we leave X\", 'oldest' for \"how did X start\"."
                ),
            },
            "detail": {
                "type": "string",
                "enum": ["adaptive", "full"],
                "description": (
                    "Discovery shape only. 'adaptive' (default) fully hydrates the "
                    "top-ranked result and returns only the exact anchor message for "
                    "lower-ranked results. 'full' returns bookends and the complete "
                    "anchored window for every result."
                ),
                "default": "adaptive",
            },
            "after": {
                "type": "string",
                "description": (
                    "Discovery shape only. Inclusive lower bound on session start "
                    "time. ISO date/datetime (e.g. 2026-06-01) or relative duration "
                    "(7d, 24h, 2w = within the last N). Use only when the user names "
                    "a time frame. sort is a ranking bias, not a bound."
                ),
            },
            "before": {
                "type": "string",
                "description": (
                    "Discovery shape only. Exclusive upper bound on session start "
                    "time. ISO date/datetime (a date-only value is midnight UTC that "
                    "day) or relative duration (7d = older than a week). Use only "
                    "when the user names a time frame."
                ),
            },
            "exclude_session_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Discovery shape only. Session ids already inspected this task. "
                    "Those sessions and their lineage are omitted so a later query "
                    "explores instead of repeating the same hit. Cap 20."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Scroll shape. Session to read inside. Use the session_id returned "
                    "from a prior discovery call. Must be paired with "
                    "around_message_id."
                ),
            },
            "around_message_id": {
                "type": "integer",
                "description": (
                    "Scroll shape. Message id to center the window on — use "
                    "match_message_id from a discovery result, or any id from a "
                    "prior window."
                ),
            },
            "window": {
                "type": "integer",
                "description": (
                    "Scroll shape only. Messages to return on each side of the anchor "
                    "(anchor itself always included). Clamped to [1, 20]. Default 5."
                ),
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated roles to include. Discovery defaults to "
                    "'user,assistant' (tool output is usually noise). Pass "
                    "'user,assistant,tool' to include tool output (debugging tool "
                    "behaviour) or 'tool' to search tool output only."
                ),
            },
            "sender_platform": {
                "type": "string",
                "description": (
                    "Optional sender scope (all shapes). Platform of the person whose OWN "
                    "messages you want, e.g. 'slack'. Requires sender_user_id. When set, only "
                    "user messages with a verified persisted sender matching every sender_* "
                    "filter are returned; assistant replies, other participants, titles and "
                    "previews are omitted, and unattributed messages are excluded."
                ),
            },
            "sender_user_id": {
                "type": "string",
                "description": "Optional sender scope. Platform user id (e.g. Slack U…). Requires sender_platform.",
            },
            "sender_chat_id": {
                "type": "string",
                "description": "Optional sender scope. Only messages the sender posted in this chat/channel id.",
            },
            "require_unthreaded": {
                "type": "boolean",
                "description": "Optional sender scope. Only top-level messages (exclude thread replies).",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Optional. Read sessions from another Hermes profile's database "
                    "(read-only). Use when resolving an `@session:<profile>/<id>` link: "
                    "pass the profile segment here with session_id as the id segment. "
                    "Omit to use the current profile."
                ),
            },
        },
        "required": [],
    },
}


from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "", limit=args.get("limit", 3), window=args.get("window", 5),
        detail=args.get("detail", "adaptive"), db=kw.get("db"), current_session_id=kw.get("current_session_id"),
        **{k: args.get(k) for k in ("role_filter", "session_id", "around_message_id", "sort", "profile",
                                    "after", "before", "exclude_session_ids", "sender_platform",
                                    "sender_user_id", "sender_chat_id", "require_unthreaded")}),
    check_fn=check_session_search_requirements,
    emoji="🔍")

"""
Transport-agnostic WhatsApp behavior shared by the Baileys bridge adapter and the
Cloud API adapter: allow-list / DM / group gating, mention detection, quoted-reply-
to-bot detection, broadcast filtering, WhatsApp markdown conversion, chunk budgeting.

Mixin contract — the host adapter sets these on ``self`` before calling any mixin
method: ``config`` (PlatformConfig), ``name``, ``_dm_policy`` / ``_group_policy``
("open" | "allowlist" | "disabled" | "pairing"), ``_allow_from`` / ``_group_allow_from`` (set[str]),
``_mention_patterns`` (list[re.Pattern]), ``_reply_prefix`` (Optional[str]). Optional:
``_groups`` (``groups.<chat_id|*>.allow_from`` per-group sender lists) and ``_owners`` (set[str]).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms._shared import (
    decode_json_list_literal as _decode_json_list_literal, extra_or_secret as _extra_or_wsecret,
    get_scoped_secret as _get_wsecret
)
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin


logger = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes", "on"}


def _stash(pattern: str, text: str, tag: str) -> tuple[str, list[str]]:
    """Replace every ``pattern`` match with a ``\\x00<tag><n>\\x00`` placeholder."""
    saved: list[str] = []

    def keep(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"\x00{tag}{len(saved) - 1}\x00"

    return re.sub(pattern, keep, text), saved


def _header_to_bold(m: re.Match) -> str:
    """``# Header`` → ``*Header*``, stripping already-bolded ``*...*`` so ``# **Title**``
    doesn't render with literal asterisks."""
    inner = m.group(1).strip()
    while len(inner) > 1 and inner.startswith("*") and inner.endswith("*"):
        inner = inner[1:-1].strip()
    return f"*{inner}*"


class WhatsAppBehaviorMixin(OwnAccessPolicyMixin):
    """Shared behavior for all WhatsApp adapters (Baileys + Cloud API); owns no state
    of its own — see the module docstring for the host adapter's attribute contract."""

    # Practical UX limit, not the ~65K protocol max (long messages are unreadable on mobile).
    MAX_MESSAGE_LENGTH: int = 4096
    ALLOW_ALL_ENV_PREFIX = "WHATSAPP"
    supports_code_blocks = True  # WhatsApp renders fenced code blocks (monospace)

    DEFAULT_REPLY_PREFIX: str = "☤ *Hermes Agent*\n────────────\n"

    _OUTBOUND_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u2060\u2063\ufeff]")
    _OUTBOUND_ODD_SPACE_RE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]")

    @classmethod
    def _sanitize_outbound_text(cls, content: str) -> str:
        """Strip zero-width format chars (WORD JOINER etc.) and normalize odd unicode
        spaces — WhatsApp renders them as mojibake prefixes. Emoji joiners are kept."""
        if not content:
            return content
        return cls._OUTBOUND_ODD_SPACE_RE.sub(" ", cls._OUTBOUND_INVISIBLE_CHARS_RE.sub("", content))

    def _effective_reply_prefix(self) -> str:
        """Prefix for outgoing replies in self-chat mode (Cloud API overrides to ``""``)."""
        if (_get_wsecret("WHATSAPP_MODE", default="self-chat") or "self-chat") != "self-chat":
            return ""
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        env_prefix = _get_wsecret("WHATSAPP_REPLY_PREFIX")
        if env_prefix is not None:
            return env_prefix.replace("\\n", "\n")
        return self.DEFAULT_REPLY_PREFIX

    def _outgoing_chunk_limit(self) -> int:
        """Reserve room for the reply prefix; floor keeps space for pagination/fence repair."""
        return max(1024, self.MAX_MESSAGE_LENGTH - len(self._effective_reply_prefix()))

    def _whatsapp_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is None:
            configured = _get_wsecret("WHATSAPP_REQUIRE_MENTION", default="false") or "false"
        if isinstance(configured, str):
            return configured.lower() in _TRUTHY
        return bool(configured)

    def _whatsapp_free_response_chats(self) -> set[str]:
        """``extra.free_response_chats`` (blank = unset) else the scoped env CSV."""
        return self._coerce_allow_list(
            _extra_or_wsecret(self.config.extra, "free_response_chats", "WHATSAPP_FREE_RESPONSE_CHATS"))

    @staticmethod
    def _coerce_allow_list(raw) -> set[str]:
        """Parse allow_from / group_allow_from from config (list or JSON-list string) or env var (CSV)."""
        if raw is None:
            return set()
        raw = _decode_json_list_literal(raw)
        parts = raw if isinstance(raw, list) else str(raw).split(",")
        return {str(part).strip() for part in parts if str(part).strip()}

    @staticmethod
    def _select_allowlist(extra: dict[str, Any], config_keys, env_keys, read_env) -> tuple[Optional[str], Any]:
        """``(source, raw)`` by key *presence*: a config key wins (an explicit empty list stays authoritative),
        then the first truthy env carrier; ``(None, None)`` when neither is set."""
        for key in config_keys:
            if key in extra:
                return "config", extra.get(key)
        for env in env_keys:
            raw = read_env(env)
            if raw:
                return env, raw
        return None, None

    def _select_dm_allowlist(self, extra: dict[str, Any], env_keys, read_env) -> Any:
        """Raw DM allowlist; records the winning source in ``_dm_allowlist_source`` so live DM checks keep
        the same precedence."""
        self._dm_allowlist_source, raw = self._select_allowlist(extra, ("allow_from", "allowFrom"), env_keys, read_env)
        return raw

    def _live_dm_allow_from(self) -> set[str]:
        """Allowlist currently enforced for DM intake / strict DM auth. Env-seeded adapters re-read
        the same key so pairing approve/revoke takes effect without restart; a removed key (sole-entry
        revoke) means empty, not the construction snapshot. Config-seeded adapters keep the in-memory
        set (pairing revoke purges it in place) — a stale env value must not broaden access."""
        source = getattr(self, "_dm_allowlist_source", None)
        if isinstance(source, str) and source != "config":
            # Scoped read: under multiplex os.environ is the DEFAULT profile's allowlist. None = key
            # absent (revoked) → empty, never the construction snapshot.
            live = _get_wsecret(source)
            return self._coerce_allow_list(live) if live is not None else set()
        return set(self._allow_from or ())

    # ------------------------------------------------------------------ JID helpers
    @staticmethod
    def _normalize_whatsapp_id(value: Optional[str]) -> str:
        if not value:
            return ""
        # Device-qualified ids (`<user>:<device>@lid`) must equal their bare form.
        return re.sub(r":\d+(?=@)", "", str(value).strip())

    @staticmethod
    def _is_broadcast_chat(chat_id: str) -> bool:
        """Status updates (Stories) and Channel/Newsletter broadcasts — never reply
        (answering a Story spams the status feed; Channel posts aren't addressable)."""
        cid = (chat_id or "").strip().lower()
        return cid == "status@broadcast" or cid.endswith(("@broadcast", "@newsletter"))

    # ------------------------------------------------------------------ gating
    @staticmethod
    def _matches_whatsapp_allowlist(candidate: str, allow_from) -> bool:
        """Match a WhatsApp identifier against an allowlist across phone/LID forms. Inbound senders
        arrive as ``<id>@lid`` while allowlists hold phone numbers (or vice versa), so resolve both
        sides through the bridge's lid-mapping files via ``gateway.whatsapp_identity``."""
        if not allow_from:
            return False
        if candidate in allow_from:
            return True
        from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier
        candidate_aliases = expand_whatsapp_aliases(candidate)
        if not candidate_aliases:
            return False
        return any(
            entry == "*"
            or normalize_whatsapp_identifier(entry) in candidate_aliases
            or expand_whatsapp_aliases(entry) & candidate_aliases
            for entry in allow_from
        )

    def _entry_matches(self, entries, target: str) -> bool:
        return self._matches_whatsapp_allowlist(target, entries)

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = (_get_wsecret("WHATSAPP_MENTION_PATTERNS", default="") or "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    # Plain text: one pattern per line, else comma-separated.
                    patterns = [p.strip() for p in raw.splitlines() if p.strip()]
                    patterns = patterns or [p.strip() for p in raw.split(",") if p.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning("[%s] whatsapp mention_patterns must be a list or string; got %s", self.name, type(patterns).__name__)
            return []
        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid WhatsApp mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d WhatsApp mention pattern(s)", self.name, len(compiled))
        return compiled

    def _bot_ids_from_message(self, data: dict[str, Any]) -> set[str]:
        return {nid for c in (data.get("botIds") or []) if (nid := self._normalize_whatsapp_id(c))}

    def _message_is_reply_to_bot(self, data: dict[str, Any]) -> bool:
        quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
        return bool(quoted_participant) and quoted_participant in self._bot_ids_from_message(data)

    def _message_mentions_bot(self, data: dict[str, Any]) -> bool:
        bot_ids = self._bot_ids_from_message(data)
        if not bot_ids:
            return False
        mentioned = {nid for c in (data.get("mentionedIds") or []) if (nid := self._normalize_whatsapp_id(c))}
        if mentioned & bot_ids:
            return True
        lower_body = str(data.get("body") or "").lower()
        return any(
            bare and (f"@{bare}" in lower_body or bare in lower_body)
            for bare in (bot_id.split("@", 1)[0].lower() for bot_id in bot_ids)
        )

    def _message_matches_mention_patterns(self, data: dict[str, Any]) -> bool:
        body = str(data.get("body") or "")
        return any(pattern.search(body) for pattern in self._mention_patterns or ())

    def _clean_bot_mention_text(self, text: str, data: dict[str, Any]) -> str:
        if not text:
            return text
        cleaned = text
        for bot_id in self._bot_ids_from_message(data):
            bare_id = bot_id.split("@", 1)[0]
            if bare_id:
                cleaned = re.sub(rf"@{re.escape(bare_id)}\b[,:\-]*\s*", "", cleaned)
        return cleaned.strip() or text

    def _whatsapp_require_native_mention(self) -> bool:
        """``extra.require_native_mention``: a group turn needs the bot in the message's native
        ``mentionedJid`` metadata. Replies, ``/`` commands, regex patterns, typed numbers and
        free-response chats no longer wake the bot in groups."""
        configured = self.config.extra.get("require_native_mention", False)
        if isinstance(configured, str):
            return configured.strip().lower() in _TRUTHY
        return configured is True

    def _message_natively_mentions_bot(self, data: dict[str, Any]) -> bool:
        bot_ids = self._bot_ids_from_message(data)
        mentioned = {nid for c in (data.get("mentionedIds") or []) if (nid := self._normalize_whatsapp_id(c))}
        return bool(bot_ids and mentioned & bot_ids)

    def _resolve_group_cfg(self, chat_id: str) -> dict[str, Any]:
        """``groups`` entry for a chat: exact key, then case-insensitive key, then ``"*"``."""
        groups = getattr(self, "_groups", None)
        if not isinstance(groups, dict):
            return {}
        lowered = chat_id.lower()
        candidates = (groups.get(chat_id), next((v for k, v in groups.items() if isinstance(k, str) and k.lower() == lowered and isinstance(v, dict)), None), groups.get("*"))
        return next((c for c in candidates if isinstance(c, dict)), {})

    @property
    def group_sender_rules_active(self) -> bool:
        """True when ``groups`` restricts group senders, so the runner re-checks each sender here."""
        groups = getattr(self, "_groups", None)
        return isinstance(groups, dict) and any(isinstance(v, dict) for v in groups.values())

    def _is_group_allowed(self, chat_id: str, sender_id: str = "") -> bool:
        """Chat-level ``group_policy`` plus per-group ``groups.<id>.allow_from`` sender lists. With sender
        rules configured, a chat with no matching entry (and no ``"*"``) or an empty list is denied."""
        if not super()._is_group_allowed(chat_id):
            return False
        if not self.group_sender_rules_active:
            return True
        group_cfg = self._resolve_group_cfg(chat_id)
        sender_allow = self._coerce_allow_list(group_cfg.get("allow_from", group_cfg.get("allowFrom")))
        principal = str(sender_id or "").strip()
        return bool(principal and sender_allow and self._matches_whatsapp_allowlist(principal, sender_allow))

    def _is_whatsapp_owner(self, user_id: Optional[str]) -> bool:
        """Whether ``user_id`` is one of ``extra.owners`` (phone/LID aliases match)."""
        owners = getattr(self, "_owners", None) or set()
        principal = str(user_id or "").strip()
        return bool(principal and owners and "*" not in owners and self._matches_whatsapp_allowlist(principal, owners))

    # Guest overrides are exact: the runner keeps only the listed concrete toolsets (no composites,
    # default-on plugin toolsets or MCP servers).
    toolsets_for_source_exact = True

    def _group_is_trusted(self, chat_id: str) -> bool:
        """``groups.<chat_id>.trust: trusted`` (exact or case-insensitive key; never the ``"*"``
        default: trust is granted group by group)."""
        groups = getattr(self, "_groups", None)
        if not isinstance(groups, dict) or not chat_id:
            return False
        lowered = chat_id.lower()
        cfg = groups.get(chat_id)
        if not isinstance(cfg, dict):
            cfg = next((v for k, v in groups.items()
                        if isinstance(k, str) and k != "*" and k.lower() == lowered and isinstance(v, dict)), None)
        return isinstance(cfg, dict) and str(cfg.get("trust", "")).strip().lower() == "trusted"

    def annotate_group_trust(self, source) -> None:
        """Mark a trusted group's source: one shared session and a verified sender role. The role
        comes from the sender id (``extra.owners``), never the display name."""
        if getattr(source, "chat_type", "") != "group" or not self._group_is_trusted(str(source.chat_id or "")):
            return
        source.shared_group = True
        source.sender_role = "owner" if self._is_whatsapp_owner(getattr(source, "user_id", None)) else "member"

    def toolsets_for_source(self, source) -> Optional[list[str]]:
        """``extra.guest_toolsets`` for any caller that is not an owner; owners keep the platform toolsets.
        In a trusted group every member gets the owner's toolsets."""
        if getattr(source, "chat_type", "") == "group" and self._group_is_trusted(str(getattr(source, "chat_id", "") or "")):
            return None
        guest = self.config.extra.get("guest_toolsets")
        if guest is None and not getattr(self, "_owners", None):
            return None
        if self._is_whatsapp_owner(getattr(source, "user_id", None)):
            return None
        # With owners configured, a missing/blank guest list means no tools, never the platform set.
        return [str(t) for t in self._coerce_allow_list(guest)]

    def _text_batch_key(self, event) -> str:
        """Never merge one participant's burst into another's when owners/guests get different tools."""
        key = super()._text_batch_key(event)
        source = getattr(event, "source", None)
        if getattr(self, "_owners", None) and getattr(source, "chat_type", "") == "group":
            return f"{key}:{getattr(source, 'user_id', '') or ''}"
        return key

    def _guest_isolation_broken(self) -> bool:
        """Owner/guest separation needs per-participant group sessions; a shared one would load the
        owner's transcript (and its tool results) into a guest turn."""
        return bool(getattr(self, "_owners", None)) and self.config.extra.get("group_sessions_per_user", True) is False

    def _should_process_message(self, data: dict[str, Any]) -> bool:
        if data.get("observerOnly"):
            return False  # bridge-marked observer events (own edits/revokes): transport observers only, never a turn
        chat_id = str(data.get("chatId") or "")
        # Broadcast pseudo-chats are filtered even in self-chat mode (fromMe events).
        if self._is_broadcast_chat(chat_id):
            return False
        if data.get("upsertType") == "history":
            return False  # history sync: archive-only, never a conversation turn
        if not data.get("isGroup", False):
            # DMs that pass the policy gate are always processed
            return self._is_dm_intake_allowed(str(data.get("senderId") or data.get("from") or ""))
        sender = str(data.get("senderId") or "")
        if not self._is_group_allowed(chat_id, sender):
            return False
        if self._guest_isolation_broken() and not self._is_whatsapp_owner(sender) and not self._group_is_trusted(chat_id):
            return False
        if data.get("upsertType") == "append":
            return False  # history/offline replay: archive-only, never a late conversation turn
        if self._whatsapp_require_native_mention():
            # The bridge emits poll votes only for polls it sent itself (clarify-as-poll answers).
            return data.get("nativeType") == "pollUpdateMessage" or self._message_natively_mentions_bot(data)
        # Group messages: check mention / free-response settings
        if chat_id in self._whatsapp_free_response_chats() or not self._whatsapp_require_mention():
            return True
        return (
            str(data.get("body") or "").strip().startswith("/")
            or self._message_is_reply_to_bot(data)
            or self._message_mentions_bot(data)
            or self._message_matches_mention_patterns(data)
        )

    # ------------------------------------------------------------------ formatting
    def format_message(self, content: str) -> str:
        """Convert markdown to WhatsApp syntax (*bold*, _italic_, ~strike~); fenced and
        inline code are protected via placeholder substitution."""
        if not content:
            return content
        result, fences = _stash(r"```[\s\S]*?```", self._sanitize_outbound_text(content), "FENCE")
        result, codes = _stash(r"`[^`\n]+`", result, "CODE")
        # Italic *text* → _text_ BEFORE bold so **bold** doesn't become italic;
        # lookarounds skip list bullets and bold delimiters.
        result = re.sub(r"(?<!\*)\*(?!\s|\*)([^*\n]*?\S[^*\n]*?)\*(?!\*)", r"_\1_", result)
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
        result = re.sub(r"__(.+?)__", r"*\1*", result)
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)
        result = re.sub(r"^#{1,6}\s+(.+)$", _header_to_bold, result, flags=re.MULTILINE)
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)  # [text](url) → text (url)
        for tag, saved in (("FENCE", fences), ("CODE", codes)):
            for i, original in enumerate(saved):
                result = result.replace(f"\x00{tag}{i}\x00", original)
        return result


# Runtime artefacts that never take part in the bridge content hash or the refresh copy.
_BRIDGE_SKIP_DIRS = frozenset({"node_modules", ".git", "__pycache__"})
# "<tree hash> <deps hash>" of the release the HERMES_HOME copy was made from; bridge.js reports the
# tree hash as /health ``treeHash``.
_BRIDGE_MARKER = ".hermes-release-hash"
_BRIDGE_SKIP_FILES = frozenset({".write_test", ".DS_Store", _BRIDGE_MARKER})
_BRIDGE_DEP_FILES = ("package.json", "package-lock.json")
_HASH16 = re.compile(r"[0-9a-f]{16}")


def _bridge_tree_hash(root: Path) -> str:
    """Deterministic SHA-256 (first 16 hex) over relative paths + bytes of the bridge files,
    skipping node_modules and runtime artefacts. Symlinks are refused (copytree would follow them)."""
    import hashlib
    if not root.is_dir():
        raise FileNotFoundError(f"bridge directory missing: {root}")
    digest = hashlib.sha256()

    def _raise(err: OSError) -> None:
        raise err

    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise):
        dirnames[:] = sorted(d for d in dirnames if d not in _BRIDGE_SKIP_DIRS)
        for name in (*dirnames, *sorted(f for f in filenames if f not in _BRIDGE_SKIP_FILES)):
            path = Path(dirpath) / name
            if path.is_symlink():
                raise OSError(f"symlink in WhatsApp bridge tree: {path}")
            if name in dirnames:
                continue
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def _bridge_deps_hash(root: Path) -> str:
    """Fingerprint of the files npm installs from (package.json + lockfile; a missing file counts)."""
    import hashlib
    digest = hashlib.sha256()
    for name in _BRIDGE_DEP_FILES:
        path = root / name
        digest.update(name.encode() + b"\0" + (path.read_bytes() if path.exists() else b"\x01missing") + b"\0")
    return digest.hexdigest()[:16]


def _read_bridge_marker(bridge: Path) -> tuple[str, str]:
    """``(tree_hash, deps_hash)`` from the copy's marker; ``("", "")`` when missing, unreadable or corrupt."""
    try:
        tokens = (bridge / _BRIDGE_MARKER).read_text(encoding="utf-8-sig").split()
    except (OSError, ValueError):  # ValueError: UnicodeDecodeError on a corrupt marker
        return "", ""
    if len(tokens) == 2 and all(_HASH16.fullmatch(t) for t in tokens):
        return tokens[0], tokens[1]
    return "", ""


def whatsapp_bridge_marker_hash(bridge_dir: Path) -> str:
    """Release tree hash recorded in a HERMES_HOME bridge copy ("" for an in-place tree or no marker)."""
    return _read_bridge_marker(bridge_dir)[0]


def _write_bridge_marker(bridge: Path, tree_hash: str, deps_hash: str) -> None:
    (bridge / _BRIDGE_MARKER).write_text(f"{tree_hash} {deps_hash}\n", encoding="utf-8")


def _same_bytes(a: Path, b: Path) -> bool:
    """Both missing, or both present with identical bytes."""
    try:
        return a.read_bytes() == b.read_bytes()
    except FileNotFoundError:
        return not a.exists() and not b.exists()


def _bridge_leftovers(copy: Path) -> list[Path]:
    """Sibling temp dirs of a refresh (``.<name>.new-*`` being built, ``.<name>.old-*`` swapped out)."""
    return sorted((*copy.parent.glob(f".{copy.name}.new-*"), *copy.parent.glob(f".{copy.name}.old-*")))


def _recover_interrupted_refresh(copy: Path) -> None:
    """Undo a refresh killed mid-way (caller holds the lock): put a swapped-out copy back, return
    node_modules it had moved into the new tree, then drop the leftovers."""
    import shutil
    olds = sorted(copy.parent.glob(f".{copy.name}.old-*"))
    if not copy.exists() and len(olds) == 1:
        os.replace(olds[0], copy)
        logger.warning("[whatsapp] restored bridge copy %s after an interrupted refresh", copy)
    if copy.is_dir() and not (copy / "node_modules").exists():
        donor = next((d for d in copy.parent.glob(f".{copy.name}.new-*") if (d / "node_modules").is_dir()), None)
        if donor is not None:
            os.replace(donor / "node_modules", copy / "node_modules")
    for stale in _bridge_leftovers(copy):
        shutil.rmtree(stale, ignore_errors=True)


def _refresh_whatsapp_bridge_copy(install_bridge: Path, copy: Path, release_hash: str, deps_hash: str,
                                  keep_modules: bool) -> None:
    """Replace ``copy`` with a fresh mirror of ``install_bridge``: build a sibling temp dir, then swap
    it in by rename. ``keep_modules`` moves the copy's node_modules over; otherwise the new copy has
    none, so the existing npm install paths run."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix=f".{copy.name}.new-", dir=copy.parent))
    old = tmp.with_name(tmp.name.replace(".new-", ".old-", 1))
    moved_modules = False
    try:
        shutil.copytree(install_bridge, tmp, dirs_exist_ok=True, ignore=lambda _d, names: [
            n for n in names if n in _BRIDGE_SKIP_DIRS or n in _BRIDGE_SKIP_FILES])
        _write_bridge_marker(tmp, release_hash, deps_hash)
        if keep_modules and (copy / "node_modules").is_dir():
            os.replace(copy / "node_modules", tmp / "node_modules")
            moved_modules = True
        if copy.exists():
            os.replace(copy, old)
        try:
            os.replace(tmp, copy)
        except OSError:
            if old.exists():
                os.replace(old, copy)  # roll back to the previous copy
            raise
    except BaseException:
        if moved_modules and copy.exists() and not (copy / "node_modules").exists():
            os.replace(tmp / "node_modules", copy / "node_modules")
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _sync_whatsapp_bridge_copy(install_bridge: Path, copy: Path, *, refresh: bool = True) -> None:
    """Create ``copy`` if missing and, with ``refresh``, make it match the release bridge.

    Cheap path: no refresh leftovers and (refresh) the marker names the release hash, or (no refresh)
    the copy exists. Otherwise, under a file lock: recover an interrupted refresh, re-check, hash the
    copy itself (a missing or wrong marker on an identical copy only rewrites the marker) and refresh
    when the content differs. node_modules is kept when the release's dependency fingerprint equals the
    one recorded at the last refresh; a copy without a marker keeps it when package.json is unchanged
    and leaves the decision to the deps stamp (``node_modules/.hermes-pkg-hash``)."""
    import time
    from gateway.status import _release_file_lock, _try_acquire_file_lock
    release_hash = _bridge_tree_hash(install_bridge) if refresh else ""
    if not _bridge_leftovers(copy) and (
            _read_bridge_marker(copy)[0] == release_hash if refresh else copy.is_dir()):
        return
    copy.parent.mkdir(parents=True, exist_ok=True)
    with open(copy.parent / f".{copy.name}.lock", "a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + 60
        while not _try_acquire_file_lock(handle):
            if time.monotonic() > deadline:
                logger.warning("[whatsapp] bridge copy %s is locked by another process; not refreshing it", copy)
                return
            time.sleep(0.1)
        try:
            _recover_interrupted_refresh(copy)
            if not refresh and copy.is_dir():
                return  # read-only callers never replace an existing copy
            release_hash = release_hash or _bridge_tree_hash(install_bridge)
            deps_hash = _bridge_deps_hash(install_bridge)
            old_tree, old_deps = _read_bridge_marker(copy)
            if old_tree == release_hash:
                return  # another process refreshed it while we waited
            old_hash = _bridge_tree_hash(copy) if copy.is_dir() else ""  # "" = no copy yet
            if old_hash == release_hash:
                _write_bridge_marker(copy, release_hash, deps_hash)
                return
            keep = old_deps == deps_hash if old_deps else _same_bytes(install_bridge / "package.json", copy / "package.json")
            _refresh_whatsapp_bridge_copy(install_bridge, copy, release_hash, deps_hash, keep)
            if old_hash:
                logger.info("[whatsapp] refreshed bridge copy %s from the release (%s -> %s)", copy, old_hash, release_hash)
        finally:
            _release_file_lock(handle)


def resolve_whatsapp_bridge_dir(refresh: bool = False) -> Path:
    """Bridge directory for CLI and adapter. A read-only install tree (e.g. Docker
    /opt/hermes) is mirrored to HERMES_HOME so npm install works. Only the adapter at gateway
    start passes ``refresh=True`` to bring the mirror up to date with the release; doctor, setup
    and the pairing route only create a missing mirror, so they never swap a running gateway's
    bridge or race its npm install (see ``_sync_whatsapp_bridge_copy``)."""
    from hermes_constants import get_hermes_home
    install_bridge = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"
    hermes_home_bridge = get_hermes_home() / "scripts" / "whatsapp-bridge"
    try:
        (install_bridge / ".write_test").touch()
        (install_bridge / ".write_test").unlink()
        return install_bridge
    except OSError:
        pass
    try:
        _sync_whatsapp_bridge_copy(install_bridge, hermes_home_bridge, refresh=refresh)
        return hermes_home_bridge
    except OSError as exc:  # shutil.Error is an OSError too
        if hermes_home_bridge.is_dir():
            logger.warning("[whatsapp] could not refresh bridge copy %s (%s); using it as is", hermes_home_bridge, exc)
            return hermes_home_bridge
        return install_bridge

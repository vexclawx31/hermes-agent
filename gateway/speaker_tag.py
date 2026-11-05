"""Unforgeable speaker tags for shared multi-user sessions.

The gateway prefixes each turn in a shared session with ``[Name]`` (plus ``| <role>`` or ``| Slack user <@U…>``).
A display name or a message body must not be able to imitate that tag: a name can't contain the tag's brackets,
field bar or mention angle brackets, and a body line can't start with an opening bracket. Look-alikes count too
(fullwidth forms, other vertical bars, non-ASCII brackets such as 【 ⟦ ⁅), and so do line breaks other than LF and
invisible leading characters (zero-width and combining marks, variation selectors, Hangul and Braille fillers).
Plain ASCII names and bodies are left byte-identical, except that a name's "<" and ">" become "/".
"""

import unicodedata

_TAG_CHARS = frozenset("[]|<>")
_OPEN_BRACKET = frozenset("[")
# Vertical bars NFKC does not fold to "|": box-drawing, divides, ornament, dental click, halfwidth forms.
_BAR_LOOKALIKES = frozenset(map(chr, (0x2502, 0x2223, 0x2758, 0x01C0, 0xFFE8)))
# Default_Ignorable_Code_Point ranges (Unicode DerivedCoreProperties) plus the Braille blank: render as nothing.
_INVISIBLE_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x2800, 0x2800),
    (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)


def _folds_to(ch: str, targets: frozenset) -> bool:
    return ch in targets or unicodedata.normalize("NFKC", ch) in targets


def _is_bracket_lookalike(ch: str, categories: tuple) -> bool:
    """A non-ASCII bracket (Ps opening, Pe closing) reads like the tag's square bracket."""
    return not ch.isascii() and unicodedata.category(ch) in categories


def neutralize_speaker_name(name: str) -> str:
    """Replace the tag's brackets, field bar and mention angle brackets, and their look-alikes, with "/"."""
    return "".join(
        "/" if ch in _BAR_LOOKALIKES or _folds_to(ch, _TAG_CHARS) or _is_bracket_lookalike(ch, ("Ps", "Pe")) else ch
        for ch in name
    )


def _is_invisible(ch: str) -> bool:
    if ch.isspace() or unicodedata.category(ch) in ("Cf", "Mn", "Me"):
        return True
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _INVISIBLE_RANGES)


def _starts_like_a_tag(line: str) -> bool:
    for ch in line:
        if _is_invisible(ch):
            continue
        return _folds_to(ch, _OPEN_BRACKET) or _is_bracket_lookalike(ch, ("Ps",))
    return False


def escape_speaker_tag_lines(text: object) -> str:
    """Prefix a backslash to every line (any Unicode line break) whose first visible character is an opening
    bracket. A non-string (an off-the-wire value) is rendered with ``str()``; None is empty."""
    text = "" if text is None else str(text)
    return "".join("\\" + line if _starts_like_a_tag(line) else line for line in text.splitlines(keepends=True))


def safe_speaker_name(name: object) -> str:
    """A display name ready for any speaker label (live turn tag or history/backfill line): one
    inert line (no newlines or controls, length-capped) with no tag brackets, field bar or mention brackets."""
    # Local import: gateway.session is heavy and must not load with this leaf module.
    from gateway.session import neutralize_untrusted_inline_text

    return neutralize_speaker_name(neutralize_untrusted_inline_text(name))


def id_tagged_text(name: object, user_id: object, body: object) -> str:
    """``[name|user_id]\\n<body>``: the observed-history attribution some adapters use in place of the runner's
    tag (Telegram observe mode, Yuanbao groups), neutralized the same way."""
    uid = safe_speaker_name(user_id or "unknown")
    return f"[{safe_speaker_name(name or uid)}|{uid}]\n{escape_speaker_tag_lines(body)}"

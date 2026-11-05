"""Unforgeable speaker tags for shared multi-user sessions.

The gateway prefixes each turn in a shared session with ``[Name]`` (plus ``| <role>`` or ``| Slack user <@U…>``).
A display name or a message body must not be able to imitate that tag: a name can't contain the tag's brackets or
field bar, and a body line can't start with an opening bracket. Look-alikes count too (fullwidth brackets and bar,
other vertical bars), and so do line breaks other than LF and invisible leading characters (zero-width space, BOM).
Plain ASCII names and bodies are left byte-identical.
"""

import unicodedata

_TAG_CHARS = frozenset("[]|")
_OPEN_BRACKET = frozenset("[")
# Vertical bars NFKC does not fold to "|": box-drawing, divides, ornament, dental click, halfwidth forms.
_BAR_LOOKALIKES = frozenset(map(chr, (0x2502, 0x2223, 0x2758, 0x01C0, 0xFFE8)))


def _folds_to(ch: str, targets: frozenset) -> bool:
    return ch in targets or unicodedata.normalize("NFKC", ch) in targets


def neutralize_speaker_name(name: str) -> str:
    """Replace the tag's brackets and field bar, and their look-alikes, with "/"."""
    return "".join("/" if ch in _BAR_LOOKALIKES or _folds_to(ch, _TAG_CHARS) else ch for ch in name)


def _starts_like_a_tag(line: str) -> bool:
    for ch in line:
        if ch.isspace() or unicodedata.category(ch) == "Cf":
            continue
        return _folds_to(ch, _OPEN_BRACKET)
    return False


def escape_speaker_tag_lines(text: str) -> str:
    """Prefix a backslash to every line (any Unicode line break) whose first visible character is an opening bracket."""
    return "".join("\\" + line if _starts_like_a_tag(line) else line for line in text.splitlines(keepends=True))


def safe_speaker_name(name: object) -> str:
    """A display name ready for any speaker label (live turn tag or history/backfill line): one
    inert line (no newlines or controls, length-capped) with no tag brackets or field bar."""
    # Local import: gateway.session is heavy and must not load with this leaf module.
    from gateway.session import neutralize_untrusted_inline_text

    return neutralize_speaker_name(neutralize_untrusted_inline_text(name))

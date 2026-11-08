"""The sudo privilege-flag gate parses sudo's real option grammar.

The old regex read ANY single-letter flag after the word "sudo" as sudo's own
(`sudo -n /usr/bin/shasum -a 256 f` -> `-a`), and matched "sudo" inside path
segments (`agent-sudo-scope --apply` -> `--a...`). Option parsing in sudo stops
at `--` or the first non-option word (the command), so only flags before it count.
"""

import time

import pytest

from tools.approval import DEFAULT_HUMAN_REQUIRED
from tools.approval_detection import (
    _approval_key_aliases,
    detect_dangerous_command,
    detect_dangerous_keys,
)

PRIVILEGE = "sudo with privilege flag (stdin/askpass/shell/list)"
COMBINED = "sudo with combined-flag privilege escalation"


@pytest.mark.parametrize("cmd", [
    # The target program's own flags are data for sudo.
    "sudo -n -u upwind-agents /usr/bin/shasum -a 256 f",
    "sudo -n /usr/bin/env -i PATH=/usr/bin id",
    "sudo -n ln -s a b",
    "sudo -n ls -l /",
    "sudo -nu root ls -l",
    "sudo -u root -- ls -l",
    "sudo -n -g staff /bin/ls -s",
    "sudo -D / -u root ls -l",
    "sudo --user=root ls -la",
    "sudo --user root ls -S",
    "sudo -u -S id",  # -u owns the next word, even option-shaped
    "sudo -uS id",  # user "S"
    "sudo -- -S",
    "sudo -a pam id",  # BSD auth type, not --askpass
    "sudo -n /usr/bin/shasum -a 256 f && sudo -n ls -l",
    "/usr/local/bin/agent-sudo-scope --apply -s",
    # "sudo" inside a path or argument is not a sudo invocation.
    "/Users/vex/bin/agent-sudo-scope --agent dumx --apply",
    "~/.local/bin/agent-sudo-scope --apply --agent x",
    "pseudosudo -S id",
    "sudo whoami",
])
def test_not_sudo_own_privilege_flag(cmd):
    dangerous, key, _ = detect_dangerous_command(cmd)
    assert not (dangerous and key in (PRIVILEGE, COMBINED)), cmd
    assert PRIVILEGE not in detect_dangerous_keys(cmd)
    assert COMBINED not in detect_dangerous_keys(cmd)


@pytest.mark.parametrize("cmd,expected", [
    ("echo pw | sudo -S whoami", PRIVILEGE),
    ("sudo -A id", PRIVILEGE),
    ("sudo -s", PRIVILEGE),
    ("sudo -i", PRIVILEGE),
    ("sudo -l", PRIVILEGE),
    ("sudo -u root -i", PRIVILEGE),
    ("sudo -E -H -s", PRIVILEGE),
    ("sudo -nS id", COMBINED),
    ("sudo -nus", None),  # -u owns "s" as its value; no gated flag
    ("sudo --stdin id", PRIVILEGE),
    ("sudo --st id", PRIVILEGE),  # unambiguous long prefix
    ("sudo --askpass id", PRIVILEGE),
    ("sudo --shell", PRIVILEGE),
    ("sudo --login", PRIVILEGE),
    ("sudo --list", PRIVILEGE),
    ("/usr/bin/sudo -S id", PRIVILEGE),
    ("env FOO=1 sudo -s", PRIVILEGE),
    ("true && sudo -S id", PRIVILEGE),
    # Review round: forms the old regex caught that a naive parser would miss.
    ("sudo FOO=1 -s", PRIVILEGE),  # VAR=value among sudo's options
    ("echo pw | sudo X=1 -S id", PRIVILEGE),
    ("sudo -u root HOME=/root -i", PRIVILEGE),
    ("sudo 2>/dev/null -S id", PRIVILEGE),  # redirections are removed by the shell
    ("echo pw | sudo </dev/stdin -S id", PRIVILEGE),
    ("sudo >out.txt -s", PRIVILEGE),
    ("SUDO -S id", PRIVILEGE),  # case-insensitive filesystem
    ("/USR/BIN/SUDO -S id", PRIVILEGE),
    ("echo id | xargs sudo -S", PRIVILEGE),
    ("find . -exec sudo -S id \\;", PRIVILEGE),
    ("caffeinate sudo -S id", PRIVILEGE),
    ("$(which sudo) -S id", PRIVILEGE),
    ("`which sudo` -S id", PRIVILEGE),
    ('echo "$(sudo -S id)"', PRIVILEGE),
    ("eval 'sudo -S id'", PRIVILEGE),
    ("env -S 'sudo -S id'", PRIVILEGE),
    ("sudo -h host -S id", PRIVILEGE),
    ("sudo --log id", PRIVILEGE),  # ambiguous prefix with a gated candidate fails closed
    ("bash -c 'sudo -S id'", None),  # payload variant: checked via keys below
])
def test_sudo_own_privilege_flag(cmd, expected):
    keys = detect_dangerous_keys(cmd)
    if cmd == "sudo -nus":
        assert PRIVILEGE not in keys and COMBINED not in keys
        return
    if expected is None:
        assert PRIVILEGE in keys, cmd
        return
    dangerous, key, _ = detect_dangerous_command(cmd)
    assert dangerous and key == expected, (cmd, key)
    assert expected in keys


def test_descriptions_stay_on_the_human_required_floor():
    floor = " | ".join(DEFAULT_HUMAN_REQUIRED)
    assert PRIVILEGE.startswith("sudo with privilege flag") and "sudo with privilege flag" in floor
    assert COMBINED in floor


@pytest.mark.parametrize("cmd", [
    's""u\\do -S id',      # quotes and escapes inside the word
    "$(echo s)udo -S id",  # literal command substitution
    "${x:-sudo} -i",       # parameter default
    "S\\UDO -l",           # case-insensitive filesystem
])
def test_obfuscated_sudo_words_still_parse(cmd):
    # The no-sudo prefilter keys on s-u-d-o in order, which every deobfuscated spelling keeps.
    assert PRIVILEGE in detect_dangerous_keys(cmd), cmd


@pytest.mark.parametrize("cmd", [
    # Independent review of c885294835 (v0.23 release): each of these yielded no finding.
    "sudo 2>&1 -S id",                # an & redirection split the segment before -S
    "sudo >&2 -S id",
    "sudo &>/dev/null -S id",
    "sudo >& 2 -S id",
    "sudo -C 3 >/dev/null -S id",     # a spaced number is -C's value, not the redirection's fd
    "sudo -u 2 >/dev/null -S id",
    'echo "$(sudo -S id $(true))"',   # nested substitution inside double quotes
    'echo "$(sudo -S id; (true))"',
    "${x:=sudo} -S id",               # parameter defaults beyond :-
    "${x-sudo} -S id",
    "${HOME:+sudo} -S id",
    "sudo{,} -S id",                  # brace or glob: unresolved, fail closed
    "echo -S id | xargs sudo",        # behind xargs, stdin supplies the rest of sudo's argv
    "printf '%s\\n' -S id | xargs sudo",
    "echo x | xargs sudo -n",
    # Round 2: syntax that hides where sudo's own options end. A command that is not plain also gets the
    # coarse rule (sudo at a command position, then any gated-looking option).
    "sudo >|/dev/null -S id",
    'sudo -p ">" -S id',
    "sudo -p '>' -S id",
    "sudo -p \\> -S id",
    "sudo -p 2\\>x -S id",
    "sudo -p x\\&y -S id",
    "sudo -p a\\|b -S id",
    "sudo -p \\; -S id",
    'echo "$( (true); sudo -S id)"',
    'x="$(case a in a) true;; esac; sudo -S id)"',
    "find ./eval -name '#*' -exec sudo -S rm {} +",
    "find . -name eval -o -name '#' -exec sudo -S id \\;",
    "find ./eval -name x -exec sudo -S rm",  # plain: the words after an eval are still scanned
    "echo -S | xargs -I{} sudo {} id",
    "xargs -J % sudo % id",                  # plain: the replacement word may be any option
    "sudo {fd}>/dev/null -S id",
    "sudo \\\n -S id",
])
def test_review_bypasses_are_caught(cmd):
    keys = detect_dangerous_keys(cmd)
    assert PRIVILEGE in keys or COMBINED in keys, cmd


@pytest.mark.parametrize("cmd", [
    "sudo -n ls 2>&1 | tee x",
    "sudo -n cmd 2>&1",
    "sudo 2>/dev/null -n /usr/bin/shasum -a 256 f",
    "find . | xargs sudo rm -f x",
    # No sudo at a command position: a path, a variable or a glob that merely holds the letters.
    "find /Users/$USER/Documents -iname '*.pdf'",
    "du /Users/$USER/Downloads -sh",
    'find "$HOME/Studio" -size +1M',
    "cp $SRC/studio.txt -i dst",
    "find . -name '*.sudo' -size +1k",
    "ls ~/src/{sudo,doas}.md -l",
    "echo $SUDO_USER -l",
    "sudo -n /usr/bin/env -i PATH=/usr/bin /usr/local/bin/x --apply",
    "ls -la; sudo -n true",
])
def test_redirections_and_xargs_with_a_command_stay_quiet(cmd):
    keys = detect_dangerous_keys(cmd)
    assert PRIVILEGE not in keys and COMBINED not in keys, (cmd, keys)


@pytest.mark.parametrize("command", [
    "eval " * 20 + "sudo id",
    "env -S " * 20 + "sudo id",
    "(" * 8000 + "/" * 8000 + "; sudo -n id",
    "echo " + "$sudo " * 4000,
    "sudo -n " + "'x' " * 25000 + "-S",
])
def test_sudo_parse_stays_bounded(command):
    from tools.approval_detection import _sudo_coarse_findings, _sudo_privilege_findings
    start = time.perf_counter()
    list(_sudo_privilege_findings(command))
    list(_sudo_coarse_findings(command))
    assert time.perf_counter() - start < 2


@pytest.mark.parametrize("cmd", [
    # Not plain (quotes): where sudo's options end is not read exactly, so a gated-looking option after a
    # command-position sudo needs a human, as on r23. Quoting can hide an operator among sudo's options
    # (`sudo -p '>' '-S' id`), so quoted text is not trusted as prose.
    "sudo -p 'pw -S' id",
    'git commit -m "use sudo -S"',
    "echo 'sudo -S id'",
    "grep -n 'sudo -s' notes.txt",
    "sudo -p '>' '-S' id",
])
def test_quoted_sudo_with_a_gated_looking_option_fails_closed(cmd):
    assert PRIVILEGE in detect_dangerous_keys(cmd), cmd


def test_xargs_replacement_after_sudos_command_stays_quiet():
    # `cp` is sudo's command, so the replacement word is cp's argument, never one of sudo's options.
    keys = detect_dangerous_keys("xargs -I{} sudo cp {} /x")
    assert PRIVILEGE not in keys and COMBINED not in keys


def test_commands_without_sudo_letters_skip_the_parse():
    from tools.approval_detection import _may_spell_sudo
    assert not _may_spell_sudo("; ssh -x" * 3000)
    assert _may_spell_sudo("s;u;d;o")  # conservative: any in-order spelling is parsed


def test_legacy_regex_key_still_resolves():
    # Approvals stored under the old regex-derived key ("sudo") keep matching.
    assert "sudo" in _approval_key_aliases(PRIVILEGE)
    assert "sudo" in _approval_key_aliases(COMBINED)

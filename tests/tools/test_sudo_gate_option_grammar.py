"""The sudo privilege-flag gate parses sudo's real option grammar.

The old regex read ANY single-letter flag after the word "sudo" as sudo's own
(`sudo -n /usr/bin/shasum -a 256 f` -> `-a`), and matched "sudo" inside path
segments (`agent-sudo-scope --apply` -> `--a...`). Option parsing in sudo stops
at `--` or the first non-option word (the command), so only flags before it count.
"""

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
    "sudo -p 'pw -S' id",
    "sudo -u -S id",  # -u owns the next word, even option-shaped
    "sudo -uS id",  # user "S"
    "sudo -- -S",
    "sudo -a pam id",  # BSD auth type, not --askpass
    "sudo -n /usr/bin/shasum -a 256 f && sudo -n ls -l",
    "/usr/local/bin/agent-sudo-scope --apply -s",
    'git commit -m "use sudo -S"',
    # "sudo" inside a path or argument is not a sudo invocation.
    "/Users/vex/bin/agent-sudo-scope --agent dumx --apply",
    "~/.local/bin/agent-sudo-scope --apply --agent x",
    "pseudosudo -S id",
    # Unquoted `echo sudo -S id` still flags (as before): a bare sudo word is treated as a possible
    # launcher argument (`xargs sudo -S`). Quoted prose is data:
    "echo 'sudo -S id'",
    "grep -n 'sudo -s' notes.txt",
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


def test_legacy_regex_key_still_resolves():
    # Approvals stored under the old regex-derived key ("sudo") keep matching.
    assert "sudo" in _approval_key_aliases(PRIVILEGE)
    assert "sudo" in _approval_key_aliases(COMBINED)

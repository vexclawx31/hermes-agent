"""Regression test: google-workspace SKILL.md must declare required_credential_files.

PR #9931 accidentally removed the required_credential_files header, which broke
credential file mounting in Docker/Modal remote backends (#16452). This test
prevents the regression from silently reappearing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/SKILL.md"
)

_EXPECTED_PATHS = {"google_token.json", "google_client_secret.json"}


def _parse_frontmatter(content: str) -> dict:
    from agent.skill_utils import parse_frontmatter

    fm, _ = parse_frontmatter(content)
    return fm


class TestGoogleWorkspaceCredentialFiles:

    def test_entries_are_registered_when_files_exist_and_authorized(self, tmp_path):
        import hermes_yaml as yaml
        import tools.credential_files as cred_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_token.json").write_text("{}")
        (hermes_home / "google_client_secret.json").write_text("{}")
        # Skills can't authorize their own mounts; the operator lists them in config.
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"terminal": {"credential_files": sorted(_EXPECTED_PATHS)}}))
        cred_mod._config_files = {}

        from tools.credential_files import (
            clear_credential_files,
            get_credential_file_mounts,
            register_credential_files,
        )

        clear_credential_files()
        try:
            content = SKILL_MD.read_text(encoding="utf-8")
            fm = _parse_frontmatter(content)
            entries = fm.get("required_credential_files", [])

            with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
                missing = register_credential_files(entries)

            assert missing == [], f"Unexpected missing files: {missing}"
            mounts = get_credential_file_mounts()
            container_paths = {m["container_path"] for m in mounts}
            assert "/root/.hermes/google_token.json" in container_paths
            assert "/root/.hermes/google_client_secret.json" in container_paths
        finally:
            clear_credential_files()
            cred_mod._config_files = {}

    def test_entries_are_refused_without_config_authorization(self, tmp_path):
        import tools.credential_files as cred_mod
        from tools.credential_files import clear_credential_files, get_credential_file_mounts, register_credential_files

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        for name in _EXPECTED_PATHS:
            (hermes_home / name).write_text("{}")
        clear_credential_files()
        cred_mod._config_files = {}
        try:
            entries = _parse_frontmatter(SKILL_MD.read_text(encoding="utf-8")).get("required_credential_files", [])
            with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home), "TERMINAL_ENV": "docker"}):
                missing = register_credential_files(entries)
                assert set(missing) == _EXPECTED_PATHS
                assert get_credential_file_mounts() == []
        finally:
            clear_credential_files()
            cred_mod._config_files = {}


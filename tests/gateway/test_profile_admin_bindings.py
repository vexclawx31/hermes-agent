from pathlib import Path
import os
import pytest
from tools import sibling_service_policy as policy

@pytest.fixture
def binding_root(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, 'PROFILE_POLICY_ROOT', str(tmp_path))
    return tmp_path

def select(monkeypatch, profile, filename='run-admin.json'):
    monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: profile)
    return policy.profile_policy_path(filename, '/protected/legacy/' + filename)

@pytest.mark.parametrize('filename', ['run-admin.json', 'sibling-services.json'])
def test_default_legacy_compatibility(binding_root, monkeypatch, filename):
    assert select(monkeypatch, 'default', filename) == Path('/protected/legacy') / filename

@pytest.mark.parametrize('profile', ['nox', 'exnox', 'apex'])
@pytest.mark.parametrize('filename', ['run-admin.json', 'sibling-services.json'])
def test_named_missing_never_inherits_default(binding_root, monkeypatch, profile, filename):
    assert select(monkeypatch, profile, filename) == binding_root / profile / filename

@pytest.mark.parametrize('filename', ['run-admin.json', 'sibling-services.json'])
def test_default_explicit_binding_wins(binding_root, monkeypatch, filename):
    path = binding_root / 'default' / filename
    path.parent.mkdir()
    path.write_text('{}')
    assert select(monkeypatch, 'default', filename) == path

@pytest.mark.parametrize('profile', ['', '../default', '/default', 'nox/default', None])
def test_invalid_profile_denied(binding_root, monkeypatch, profile):
    with pytest.raises(ValueError):
        select(monkeypatch, profile)

def test_unknown_filename_denied(binding_root, monkeypatch):
    with pytest.raises(ValueError):
        select(monkeypatch, 'default', '../../config.yaml')

def test_dangling_symlink_never_falls_back(binding_root, monkeypatch):
    path = binding_root / 'default' / 'run-admin.json'
    path.parent.mkdir()
    path.symlink_to(binding_root / 'missing')
    assert select(monkeypatch, 'default') == path
    with pytest.raises((OSError, ValueError)):
        policy.load_policy(path)

def test_two_profile_switching_has_no_cached_identity(binding_root, monkeypatch):
    for profile in ['nox', 'exnox', 'nox']:
        assert select(monkeypatch, profile) == binding_root / profile / 'run-admin.json'

def test_unsafe_selected_file_is_not_legacy_fallback(binding_root, monkeypatch):
    path = binding_root / 'default' / 'run-admin.json'
    path.parent.mkdir()
    path.write_text('{}')
    os.chmod(path, 0o666)
    assert select(monkeypatch, 'default') == path
    with pytest.raises(ValueError):
        policy.load_policy(path)

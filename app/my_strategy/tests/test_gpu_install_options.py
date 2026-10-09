"""Platform installation profiles and verified wheels for a complete workbench."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def installer(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / 'install.py'
    spec = importlib.util.spec_from_file_location('khquant_install', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'APP', tmp_path / 'app')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(module.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(module, 'nvidia_available', lambda: False)
    return module


def test_clean_install_includes_cpu_torch_without_writing_during_dry_run(installer, capsys, tmp_path):
    assert installer.install(['--dry-run']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['compute_device'] == 'auto'
    assert config['torch_profile'] == 'cpu'
    assert config['torch_requirements'] == 'app/requirements-czsc-torch-cpu.lock'
    assert Path(config['venv']).name == '.venv'
    assert config['gpu_requirements'] is None
    assert not (tmp_path / '.env').exists()


def test_gpu_dry_run_selects_optional_cuda_lock(installer, capsys, tmp_path):
    assert installer.install(['--gpu', '--device', 'cuda:1', '--dry-run']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['compute_device'] == 'cuda:1'
    assert config['gpu_requirements'] == 'app/requirements-czsc-torch-cu130.lock'
    assert not (tmp_path / '.env').exists()


def test_install_device_syntax_is_validated(installer):
    with pytest.raises(SystemExit) as error:
        installer.install(['--device', 'cuda:-1', '--dry-run'])
    assert error.value.code == 2


def test_existing_environment_and_explicit_cpu_are_preserved(installer, capsys, tmp_path):
    (tmp_path / '.env').write_text('KHQUANT_VENV_DIR=./.venv-win\nKHQUANT_COMPUTE_DEVICE=cpu\n')
    assert installer.install(['--dry-run']) == 0
    config = json.loads(capsys.readouterr().out)
    assert Path(config['venv']).name == '.venv-win'
    assert config['compute_device'] == 'cpu'


@pytest.mark.parametrize('system,machine,nvidia,device,profile', [
    ('Windows', 'AMD64', True, 'auto', 'cu130'),
    ('Linux', 'x86_64', False, 'auto', 'cpu'),
    ('Linux', 'x86_64', True, 'cpu', 'cpu'),
    ('Darwin', 'arm64', False, 'auto', 'mps'),
    ('Darwin', 'arm64', False, 'mps', 'mps'),
    ('Darwin', 'arm64', False, 'cpu', 'mps'),
])
def test_platform_profile(installer, monkeypatch, system, machine, nvidia, device, profile):
    monkeypatch.setattr(installer.platform, 'system', lambda: system)
    monkeypatch.setattr(installer.platform, 'machine', lambda: machine)
    monkeypatch.setattr(installer, 'nvidia_available', lambda: nvidia)
    assert installer.torch_profile(device) == profile


@pytest.mark.parametrize('system,machine,device', [('Windows', 'AMD64', 'mps'), ('Darwin', 'arm64', 'cuda:0'), ('Darwin', 'x86_64', 'auto')])
def test_unavailable_platform_backend_is_rejected(installer, monkeypatch, system, machine, device):
    monkeypatch.setattr(installer.platform, 'system', lambda: system)
    monkeypatch.setattr(installer.platform, 'machine', lambda: machine)
    with pytest.raises(SystemExit) as error:
        installer.install(['--device', device, '--dry-run'])
    assert error.value.code == 2


def wheel_fixture(installer, monkeypatch, tmp_path, *, indexed=True):
    directory = tmp_path / 'wheels'
    directory.mkdir()
    wheel = directory / 'czsc-1.0.1-cp310-abi3-win_amd64.whl'
    wheel.write_bytes(b'frozen-wheel')
    manifest = {'version': '1.0.1', 'files': [{'path': 'LICENSE', 'sha256': 'source-hash'}],
                'built_wheel_name': wheel.name, 'built_wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
    if indexed:
        index = {'schema_version': 1, 'package': 'czsc', 'version': '1.0.1',
                 'source_tree_sha256': installer.source_tree_hash(manifest),
                 'wheels': [{'filename': wheel.name, 'sha256': manifest['built_wheel_sha256'],
                             'build': {'cargo_locked': True, 'source_algorithm_patches': False}}]}
        (directory / 'wheel-index.json').write_text(json.dumps(index))
    monkeypatch.setattr(installer.subprocess, 'check_output', lambda *a, **kw: json.dumps({
        'target_tags': ['cp312-cp312-win_amd64', 'cp310-abi3-win_amd64'],
        'wheels': {wheel.name: {'package': 'czsc', 'version': '1.0.1', 'tags': ['cp310-abi3-win_amd64']}}}))
    return directory, wheel, manifest


@pytest.mark.parametrize('indexed', [False, True])
def test_wheel_matches_target_abi_and_verified_provenance(installer, monkeypatch, tmp_path, indexed):
    directory, wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path, indexed=indexed)
    assert installer.compatible_wheel(Path('target-python'), directory, manifest) == wheel


def test_foreign_wheel_is_never_selected(installer, monkeypatch, tmp_path):
    directory, _wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path)
    monkeypatch.setattr(installer.subprocess, 'check_output', lambda *a, **kw: json.dumps({
        'target_tags': ['cp310-abi3-macosx_11_0_arm64'],
        'wheels': {_wheel.name: {'package': 'czsc', 'version': '1.0.1', 'tags': ['cp310-abi3-win_amd64']}}}))
    assert installer.compatible_wheel(Path('target-python'), directory, manifest) is None


def test_wheel_corruption_is_rejected(installer, monkeypatch, tmp_path):
    directory, wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path)
    wheel.write_bytes(b'changed')
    with pytest.raises(ValueError, match='校验失败'):
        installer.compatible_wheel(Path('target-python'), directory, manifest)


def test_wheel_from_different_frozen_source_is_rejected(installer, monkeypatch, tmp_path):
    directory, _wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path)
    manifest['files'][0]['sha256'] = 'different-source'
    with pytest.raises(ValueError, match='冻结源码不匹配'):
        installer.compatible_wheel(Path('target-python'), directory, manifest)


def test_source_only_checkout_can_build_without_a_prebuilt_wheel(installer, tmp_path):
    assert installer.compatible_wheel(Path('target-python'), tmp_path / 'wheels', {'version': '1.0.1', 'files': []}) is None


def test_external_selected_wheel_provenance_is_retained_for_runtime_doctor(installer, monkeypatch, tmp_path):
    directory, wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path)
    external_before = (directory / 'wheel-index.json').read_bytes()
    destination = installer.APP / 'vendor/wheels/wheel-index.json'
    destination.parent.mkdir(parents=True)
    local = json.loads(external_before)
    retained = {'filename': 'czsc-1.0.1-cp310-abi3-macosx_11_0_arm64.whl', 'sha256': 'prior-build-hash'}
    local['wheels'] = [retained]
    destination.write_text(json.dumps(local))
    source_manifest = installer.APP / 'czsc-source-manifest.json'
    source_manifest.write_text(json.dumps(manifest))
    manifest_before = source_manifest.read_bytes()
    selected = installer.compatible_wheel(Path('target-python'), directory, manifest)
    installer.record_wheel_provenance(selected, manifest)
    stored = json.loads(destination.read_text())
    assert stored['source_tree_sha256'] == installer.source_tree_hash(manifest)
    assert stored['wheels'] == [retained, json.loads(external_before)['wheels'][0]]
    assert (directory / 'wheel-index.json').read_bytes() == external_before
    assert source_manifest.read_bytes() == manifest_before
    assert not destination.with_suffix('.json.tmp').exists()


def test_external_index_cannot_overwrite_local_provenance_from_another_source(installer, monkeypatch, tmp_path):
    directory, wheel, manifest = wheel_fixture(installer, monkeypatch, tmp_path)
    destination = installer.APP / 'vendor/wheels/wheel-index.json'
    destination.parent.mkdir(parents=True)
    local = json.loads((directory / 'wheel-index.json').read_text())
    local['source_tree_sha256'] = 'different-source'
    destination.write_text(json.dumps(local))
    original = destination.read_bytes()
    with pytest.raises(ValueError, match='冻结源码不匹配'):
        installer.record_wheel_provenance(wheel, manifest)
    assert destination.read_bytes() == original


def test_real_install_always_installs_torch_and_project(installer, monkeypatch, tmp_path):
    installer.APP.mkdir()
    source = installer.APP / 'vendor/czsc/LICENSE'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'source')
    manifest = {'version': '1.0.1', 'files': [{'path': 'LICENSE', 'sha256': hashlib.sha256(b'source').hexdigest()}]}
    (installer.APP / 'czsc-source-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(installer.venv.EnvBuilder, 'create', lambda *a: None)
    monkeypatch.setattr(installer, 'compatible_wheel', lambda *a: tmp_path / 'verified-czsc.whl')
    calls = []
    monkeypatch.setattr(installer.subprocess, 'run', lambda command, **kw: calls.append(command))
    assert installer.install(['--device', 'cpu']) == 0
    assert any('requirements-czsc-torch-cpu.lock' in ' '.join(call) for call in calls)
    assert any('--editable' in call and '--no-deps' in call and '--no-build-isolation' in call for call in calls)
    assert any(call[-2:] == ['pip', 'check'] for call in calls)
    assert 'KHQUANT_VENV_DIR="./.venv"' in (tmp_path / '.env').read_text()

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dependencies():
    spec = importlib.util.spec_from_file_location('server_dependencies', ROOT / 'scripts/server_dependencies.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_prefix_proof_rejects_stale_conda_variables_before_importing_torch(dependencies, monkeypatch, tmp_path):
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'cuda_env')
    monkeypatch.setenv('CONDA_PREFIX', str(tmp_path / 'cuda_env'))
    with pytest.raises(RuntimeError, match='não pertence'):
        dependencies.validate_environment()


@pytest.mark.parametrize('version', ['', '2.4.1', '2.5.0', '2.10.0', '2.15.0'])
def test_installer_never_repairs_missing_or_unsupported_torch(dependencies, version):
    with pytest.raises(RuntimeError, match='não substitui PyTorch'):
        dependencies.validate_torch({'torch': version})


def test_protects_complete_installed_cuda_stack(dependencies):
    installed = {'torch': '2.5.1+cu118', 'torchvision': '0.20.1+cu118', 'torchaudio': '2.5.1+cu118',
                 'triton': '3.1.0', 'nvidia-cuda-runtime-cu11': '11.8.89', 'cuda-bindings': '12.9.4',
                 'numpy': '1.26.4'}
    assert dependencies.validate_torch(installed) == {name: version for name, version in installed.items()
                                                     if name != 'numpy'}


@pytest.fixture
def installer(dependencies, monkeypatch, tmp_path):
    requirement = tmp_path / 'requirements-server.txt'
    constraint = tmp_path / 'constraints-server.txt'
    requirement.write_text('numpy\ntorch>=2.5.1,<2.10\n')
    constraint.write_text('numpy==2.2.6\ntorch>=2.5.1,<2.10\n')
    monkeypatch.setattr(dependencies, 'REQUIREMENTS', requirement)
    monkeypatch.setattr(dependencies, 'CONSTRAINTS', constraint)
    before = {'torch': '2.5.1+cu118', 'nvidia-cuda-runtime-cu11': '11.8.89', 'triton': '3.1.0', 'numpy': '1.26.4'}
    after = {**before, 'numpy': '2.2.6'}
    versions = iter([before, after])
    monkeypatch.setattr(dependencies, 'installed_versions', lambda: next(versions))
    monkeypatch.setattr(dependencies, 'freeze', lambda destination, label: (destination / f'{label}.freeze.txt').write_text(label))
    monkeypatch.setattr(dependencies, 'pip_check', lambda *args: SimpleNamespace(returncode=0, stdout='', stderr=''))
    state = {'commands': [], 'plan': [{'metadata': {'name': 'numpy', 'version': '2.2.6'}}]}
    def run(command, **options):
        state['commands'].append(command)
        if '--dry-run' in command:
            Path(command[command.index('--report') + 1]).write_text(json.dumps({'install': state['plan']}))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'device': 'cuda:0', 'cuda_kernel_test': 'passed'}))
    monkeypatch.setattr(dependencies.subprocess, 'run', run)
    return dependencies, tmp_path / 'audit', before, after, state


def test_normal_resolver_precedes_single_install_and_records_cuda_constraints(installer):
    dependencies, destination, before, after, state = installer
    dependencies.install(destination, {'environment': 'cuda_env'})
    probe, dry_run, install = state['commands']
    assert 'cuda_probe' in probe[-1]
    assert '--dry-run' in dry_run and '--report' in dry_run
    assert '--dry-run' not in install
    assert '--requirement' in install and '--editable' in install
    assert '--no-deps' not in install and 'uninstall' not in install
    exact = (destination / 'existing-cuda-constraints.txt').read_text()
    assert 'torch==2.5.1+cu118\n' in exact and 'nvidia-cuda-runtime-cu11==11.8.89\n' in exact
    assert 'triton==3.1.0\n' in exact and 'numpy' not in exact
    manifest = json.loads((destination / 'environment.json').read_text())
    assert manifest['status'] == 'ready'
    assert manifest['protected_changes'] == {}
    assert manifest['installed_versions'] == after
    assert (destination / 'before.freeze.txt').exists() and (destination / 'after.freeze.txt').exists()


@pytest.mark.parametrize('package', ['torch', 'NVIDIA_CUDA_RUNTIME_CU12', 'triton', 'cuda-bindings'])
def test_resolver_cannot_install_or_replace_cuda_packages(installer, package):
    dependencies, destination, _, _, state = installer
    state['plan'] = [{'metadata': {'name': package, 'version': '9.9.9'}}]
    with pytest.raises(RuntimeError, match='Nenhum pacote foi instalado'):
        dependencies.install(destination, {'environment': 'cuda_env'})
    assert len(state['commands']) == 2 and '--dry-run' in state['commands'][1]


def test_broken_existing_optional_packages_fail_check_without_automatic_removal(installer, monkeypatch):
    dependencies, destination, _, _, state = installer
    monkeypatch.setattr(dependencies, 'pip_check', lambda *args: SimpleNamespace(
        returncode=1, stdout='fastplot requires statsmodels, which is not installed.\n', stderr=''))
    with pytest.raises(RuntimeError, match='Nenhum pacote alheio ao projeto foi removido'):
        dependencies.install(destination, {'environment': 'cuda_env'})
    manifest = json.loads((destination / 'environment.json').read_text())
    assert manifest['status'] == 'failed' and manifest['pip_check_after_returncode'] == 1
    assert all('uninstall' not in command for command in state['commands'])

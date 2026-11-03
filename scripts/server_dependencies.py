#!/usr/bin/env python3
"""Resolve extras in the existing cuda_env without replacing its CUDA PyTorch.

A normal pip resolution is reviewed before the single install. No environment
is created, no package is uninstalled explicitly, and metadata is not bypassed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / 'requirements-server.txt'
CONSTRAINTS = ROOT / 'constraints-server.txt'


def canonical_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def protected_package(name):
    name = canonical_name(name)
    return (name in {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton'}
            or name.startswith(('nvidia-', 'cuda-')))


def installed_versions():
    return {canonical_name(distribution.metadata['Name']): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get('Name')}


def validate_environment():
    active = os.environ.get('CONDA_DEFAULT_ENV', '')
    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix or (active != 'cuda_env' and Path(active).name != 'cuda_env'):
        raise RuntimeError('Ative o cuda_env existente: source activate cuda_env. '
                           'Este script não cria nem repara ambientes Conda.')
    expected = Path(prefix).resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(f'O Python ativo ({sys.executable}; prefixo {sys.prefix}) não pertence '
                           f'ao cuda_env informado ({expected}). Reative o cuda_env e confirme '
                           '`python -c "import sys; print(sys.executable, sys.prefix)"`.')
    if not (expected / 'conda-meta' / 'history').is_file():
        raise RuntimeError(f'O prefixo {expected} não contém conda-meta/history e não é um '
                           'ambiente Conda válido. Ative o cuda_env válido já instalado; '
                           'nenhum diretório será convertido ou sobrescrito.')
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError('O cuda_env precisa de Python 3.12 ou 3.13 para os modelos nativos; '
                           f'Python ativo: {sys.version.split()[0]}. Nenhum pacote foi instalado.')
    return {'python': sys.version, 'executable': sys.executable,
            'prefix': str(expected), 'environment': 'cuda_env'}


def check_coordinators(root=ROOT, proc=Path('/proc')):
    active = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or (entry / 'cwd').resolve(strict=True) != root.resolve():
                continue
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except (OSError, RuntimeError):
            continue
        if not argv:
            continue
        executable = Path(argv[0]).name
        coordinator = (executable == 'energygap' and len(argv) > 1
                       and argv[1] in {'run', 'resume', 'reproduce'})
        if executable.startswith('python'):
            coordinator = any(argv[index:index + 2] == ['-m', 'energygap']
                              and argv[index + 2] in {'run', 'resume', 'reproduce'}
                              for index in range(len(argv) - 2))
        if coordinator:
            active.append({'pid': int(entry.name), 'command': argv})
    if active:
        for process in sorted(active, key=lambda process: process['pid']):
            print(json.dumps(process), flush=True)
        raise RuntimeError('Já existe um coordenador deste projeto. Nenhum processo foi '
                           'interrompido. Aguarde seu encerramento antes de instalar ou executar.')


def validate_torch(versions):
    version = versions.get('torch', '')
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version)
    if not match or not ((2, 5, 1) <= tuple(map(int, match.groups())) < (2, 10, 0)):
        raise RuntimeError('O cuda_env precisa fornecer um PyTorch CUDA compatível com Volta '
                           f'(>=2.5.1,<2.10); encontrado: {version or "ausente"}. '
                           'O instalador não substitui PyTorch nem o runtime CUDA.')
    return {name: version for name, version in versions.items() if protected_package(name)}


def requirements_satisfied(versions):
    from packaging.requirements import Requirement
    constraints = {}
    for line in CONSTRAINTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirement = Requirement(line)
            constraints[canonical_name(requirement.name)] = requirement
    required = dict(constraints)
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '-')):
            continue
        requirement = Requirement(line)
        name = canonical_name(requirement.name)
        required.setdefault(name, requirement)
    missing = []
    for name, requirement in required.items():
        version = versions.get(name)
        if version is None or version not in requirement.specifier:
            missing.append({'package': name, 'installed': version, 'required': str(requirement)})
    return missing


def pip_check(destination, label):
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
    (destination / f'{label}.pip-check.txt').write_text(result.stdout + result.stderr)
    return result


def freeze(destination, label):
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                            text=True, capture_output=True)
    (destination / f'{label}.freeze.txt').write_text(result.stdout)


def install(destination, environment):
    destination.mkdir(parents=True, exist_ok=True)
    before = installed_versions()
    protected = validate_torch(before)
    print('Testando o PyTorch CUDA existente antes de instalar os requisitos extras...', flush=True)
    try:
        result = subprocess.run([sys.executable, '-u', '-c',
            'import json; from energygap.runtime.device import cuda_probe; '
            'print(json.dumps(cuda_probe("cuda:0")))'],
            capture_output=True, text=True, check=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('O teste CUDA excedeu 120 segundos. Nenhum pacote foi instalado.') from exc
    except subprocess.CalledProcessError as exc:
        print((exc.stdout or '') + (exc.stderr or ''), file=sys.stderr, flush=True)
        raise RuntimeError('O PyTorch CUDA existente falhou no teste real. Nenhum pacote foi instalado.') from exc
    probe = json.loads(result.stdout)
    freeze(destination, 'before')
    prior_check = pip_check(destination, 'before')
    constraint = destination / 'existing-cuda-constraints.txt'
    constraint.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(protected.items())))
    command = [sys.executable, '-m', 'pip', 'install', '--constraint', str(constraint),
               '--requirement', str(REQUIREMENTS), '--editable', str(ROOT)]
    manifest = {'status': 'resolving', **environment, 'environment_policy': 'single_cuda_env',
                'protected_packages': protected, 'cuda_before': probe,
                'requirements_sha256': hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
                'constraints_sha256': hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest(),
                'pip_check_before_returncode': prior_check.returncode, 'install_command': command}
    manifest_path = destination / 'environment.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    report = destination / 'resolver.json'
    print('Resolvendo os requisitos sem modificar o ambiente...', flush=True)
    subprocess.run([*command, '--dry-run', '--report', str(report)], check=True)
    plan = json.loads(report.read_text())
    changes = [row['metadata']['name'] for row in plan.get('install', [])]
    forbidden = [name for name in changes if protected_package(name)]
    if forbidden:
        raise RuntimeError('O resolver tentou instalar/substituir componentes CUDA protegidos: '
                           + ', '.join(forbidden) + '. Nenhum pacote foi instalado; '
                           'consulte resolver.json e a instalação existente de PyTorch.')
    # Keep the actual installation on the versions reviewed by the dry-run.
    resolved = destination / 'resolved-constraints.txt'
    resolved.write_text(''.join(f"{row['metadata']['name']}=={row['metadata']['version']}\n"
                                for row in plan.get('install', [])))
    install_command = [*command, '--constraint', str(resolved)]
    manifest.update(status='installing', planned_installations=changes, install_command=install_command)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Instalando os requisitos extras resolvidos; PyTorch e CUDA permanecem fixados...', flush=True)
    subprocess.run(install_command, check=True)
    after = installed_versions()
    protected_after = {name: version for name, version in after.items() if protected_package(name)}
    changed = {name: {'before': protected.get(name), 'after': protected_after.get(name)}
               for name in protected.keys() | protected_after.keys()
               if protected.get(name) != protected_after.get(name)}
    freeze(destination, 'after')
    check = pip_check(destination, 'after')
    missing = requirements_satisfied(after)
    manifest.update(status='ready' if not changed and check.returncode == 0 and not missing else 'failed',
                    protected_changes=changed, pip_check_after_returncode=check.returncode,
                    unsatisfied_requirements=missing, installed_versions=after)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if changed:
        raise RuntimeError(f'Componentes protegidos mudaram inesperadamente: {changed}. Não execute experimentos.')
    if check.returncode:
        print(check.stdout + check.stderr, file=sys.stderr, flush=True)
        raise RuntimeError('`pip check` encontrou dependências inconsistentes no cuda_env. '
                           f'Consulte {destination / "after.pip-check.txt"}. '
                           'Nenhum pacote alheio ao projeto foi removido automaticamente.')
    if missing:
        raise RuntimeError(f'Requisitos do servidor não satisfeitos: {missing}')
    print(json.dumps({'status': 'ready', 'audit_directory': str(destination),
                      'torch_preserved': protected['torch'], 'environment_policy': 'single_cuda_env'}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--environment-only', action='store_true')
    mode.add_argument('--install', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--audit-dir', type=Path)
    args = parser.parse_args(argv)
    environment = validate_environment()
    if args.environment_only:
        check_coordinators()
        print(json.dumps({'status': 'environment_valid', **environment}), flush=True)
        return 0
    if args.install:
        destination = args.audit_dir or (ROOT / 'artifacts/analyses/server_environment' /
                                        datetime.now(timezone.utc).strftime('setup-%Y%m%dT%H%M%SZ'))
        install(destination.resolve(), environment)
        return 0
    versions = installed_versions()
    validate_torch(versions)
    try:
        missing = requirements_satisfied(versions)
    except ImportError as exc:
        raise RuntimeError('Requisitos ausentes. Execute: bash scripts/server.sh setup') from exc
    if missing:
        raise RuntimeError(f'Requisitos do servidor incompatíveis ou ausentes: {missing}. '
                           'Execute: bash scripts/server.sh setup')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    print(json.dumps({'status': 'requirements_ready', 'torch_preserved': versions['torch']}), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'ERRO: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 2)

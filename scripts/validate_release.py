#!/usr/bin/env python3
"""Optional validation levels; never start an experiment or install dependencies.

fast checks source syntax and the selected scientific configuration. core adds a
single regression-suite invocation. native checks actual installed adapters and
hardware. full runs all tests once, without rerunning core or native separately.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
LEVELS = ('fast', 'core', 'native', 'full')
NATIVE_METHODS = ('BRITS', 'SAITS', 'CSDI', 'ImputeFormer', 'T1', 'MOMENT-1-base', 'TS-ICL')
DEFAULT_CONFIG = 'configs/experiments/server_focused.yaml'

# These are the prespecified dimensions, independent of throughput settings.
SCIENTIFIC_CONFIG = {
    'sites': ['H1', 'H2', 'I1'], 'folds': [1, 2, 3],
    'conditions': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'methods': ['median', 'locf', 'linear', 'seasonal', 'kalman_filter',
                'kalman_smoother', 'BRITS', 'SAITS', 'CSDI', 'ImputeFormer',
                'T1', 'MOMENT-1-base', 'TS-ICL', 'FIM', 'TabPFN-TS', 'MoTM', 'CBiT'],
    'require_available': ['BRITS', 'SAITS', 'CSDI', 'ImputeFormer', 'T1',
                          'MOMENT-1-base', 'TS-ICL'],
    'context': 512, 'training_stride': 288,
    'max_training_windows': None, 'max_episodes': None,
    'train_seeds': [11, 23, 47],
    'validation_mask_seeds': [1001, 1002, 1003, 1004, 1005],
    'test_mask_seeds': list(range(2001, 2011)),
    'hpo_trials': 6, 'max_epochs': 100, 'patience': 10, 'batch_size': 32,
    'csdi_sample_batch_size': 32, 'samples': 100,
    'forecast_horizons_seconds': [1800, 3600, 10800, 21600],
    'physical_hypotheses': ['instantaneous', 'interval_mean_start', 'interval_mean_end'],
    'physical_conditions': ['C3', 'C4', 'C5'], 'physical_confirmatory': False,
    'bootstrap_attempts': 10000, 'block_days': 7, 'block_sensitivity_days': [2, 14],
}


def check_scientific_config(config):
    mismatches = [f'{key}: expected {expected!r}, found {config.get(key)!r}'
                  for key, expected in SCIENTIFIC_CONFIG.items()
                  if key not in config or config[key] != expected]
    inference = config.get('inference_batch_size')
    if isinstance(inference, bool) or not isinstance(inference, int) or inference < 1:
        mismatches.append('inference_batch_size must be a positive integer')
    for key, expected in [('hpo_trial_indices', list(range(6))), ('hpo_max_epochs', 100),
                          ('hpo_patience', 10), ('epoch_validation_max_per_stratum', None)]:
        if key in config and config[key] != expected:
            mismatches.append(f'{key}: expected {expected!r}, found {config[key]!r}')
    if mismatches:
        raise ValueError('The full scientific design changed:\n' + '\n'.join(mismatches))


def load_configuration(root=ROOT, config_path=DEFAULT_CONFIG):
    from energygap.conference import load_settings
    path = Path(config_path)
    path = path if path.is_absolute() else root / path
    config = load_settings(root, config_path=path)
    if path.resolve() == (root / 'configs/experiments/server_full.yaml').resolve():
        check_scientific_config(config)
    return config


def selected_native_methods(config):
    return [name for name in config['methods'] if name in NATIVE_METHODS]


def fast_checks(root=ROOT, config_path=DEFAULT_CONFIG):
    count = 0
    for folder in ('src', 'scripts', 'tests'):
        for path in sorted((root / folder).rglob('*.py')):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            count += 1
    for path in sorted((root / 'scripts').glob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], check=True, cwd=root)
    config = load_configuration(root, config_path)
    print(f'Fast: {count} Python sources, shell syntax and selected configuration passed: {config_path}; '
          f'inference_batch_size={config["inference_batch_size"]}.', flush=True)
    return config


def commands_for(args):
    python = sys.executable
    if args.level == 'core':
        return [[python, '-m', 'pytest', '-q', 'tests/unit', 'tests/leakage',
                 'tests/integration', 'tests/recovery', '-m', 'not heavy']]
    if args.level == 'full':
        return [[python, '-m', 'pytest', '-q', 'tests']]
    if args.level == 'native':
        methods = getattr(args, 'methods', list(NATIVE_METHODS))
        options = ['--device', args.device, '--threads', str(args.threads)]
        result = [[python, '-u', '-m', 'energygap', 'doctor', *options]]
        if args.device.startswith('cuda:') and 'SAITS' in methods:
            result.append([python, '-u', '-m', 'energygap', 'gpu-smoke', *options])
        if methods:
            selected = [] if list(methods) == list(NATIVE_METHODS) else ['--methods', *methods]
            result.append([python, '-u', 'scripts/verify_server_models.py', *options, *selected])
        return result
    return []


@contextmanager
def native_lock(root=ROOT):
    """Do not compete with an active coordinator or setup for the GPU."""
    directory = root / 'artifacts/analyses'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'server.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A coordinator/setup holds the project lock; no process was interrupted.') from exc
        try:
            if Path('/proc').is_dir():
                spec = importlib.util.spec_from_file_location('validation_dependencies', root / 'scripts/server_dependencies.py')
                helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helper)
                helper.check_coordinators(root)
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=LEVELS, default='fast')
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0',
                        help='native hardware check device; CPU never certifies CUDA')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='Configuration to validate; canonical server_full retains its complete-design guard')
    export = parser.add_mutually_exclusive_group()
    export.add_argument('--required-models', action='store_true', help='Print selected native model names, one per line, without running checks')
    export.add_argument('--required-sites', action='store_true', help='Print selected dataset IDs, one per line, without running checks')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.required_models or args.required_sites:
        config = load_configuration(config_path=args.config)
        print('\n'.join(selected_native_methods(config) if args.required_models else config['sites']))
        return 0
    config = fast_checks(config_path=args.config)
    args.methods = selected_native_methods(config)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
        MPLBACKEND='Agg', MPL_IGNORE_SYSTEM_FONTS='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
        NUMEXPR_NUM_THREADS=str(args.threads))
    from contextlib import nullcontext
    with native_lock() if args.level in ('native', 'full') else nullcontext():
        for command in commands_for(args):
            print('Validation:', ' '.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Validation levels keep scientific scope and avoid redundant suite launches."""
from copy import deepcopy
import fcntl
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def validator():
    spec = importlib.util.spec_from_file_location('release_validation', ROOT / 'scripts/validate_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delivered_matrix_retains_all_scientific_repetitions(validator):
    config = yaml.safe_load((ROOT / 'configs/experiments/server_full.yaml').read_text())
    validator.check_scientific_config(config)
    assert config['inference_batch_size'] == 256
    for name, replacement in [('hpo_trials', 3), ('max_epochs', 20), ('max_episodes', 5),
                              ('samples', 10), ('csdi_sample_batch_size', 16),
                              ('train_seeds', [11]), ('conditions', ['C3', 'C4', 'C5'])]:
        changed = deepcopy(config)
        changed[name] = replacement
        with pytest.raises(ValueError, match=name):
            validator.check_scientific_config(changed)


def test_full_runs_the_test_suite_once_without_chaining_other_levels(validator):
    args = SimpleNamespace(level='full', device='cuda:0', threads=32)
    commands = validator.commands_for(args)
    assert len(commands) == 1
    assert commands[0][1:] == ['-m', 'pytest', '-q', 'tests']
    args.level = 'core'
    assert len(validator.commands_for(args)) == 1
    args.level = 'fast'
    assert validator.commands_for(args) == []


def test_native_checks_real_adapters_and_gpu_without_running_pytest(validator):
    args = SimpleNamespace(level='native', device='cuda:0', threads=32)
    commands = validator.commands_for(args)
    assert len(commands) == 3
    assert commands[0][4] == 'doctor'
    assert commands[1][4] == 'gpu-smoke'
    assert commands[2][2] == 'scripts/verify_server_models.py'
    assert not any('pytest' in command for command in commands)
    args.device = 'cpu'
    commands = validator.commands_for(args)
    assert len(commands) == 2
    assert not any('gpu-smoke' in command for command in commands)


def test_native_validation_cannot_compete_with_active_coordinator(validator, tmp_path):
    directory = tmp_path / 'artifacts/analyses'
    directory.mkdir(parents=True)
    with (directory / 'server.lock').open('a') as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='holds the project lock'):
            with validator.native_lock(tmp_path):
                pytest.fail('native validation competed with active GPU work')


def test_selected_configuration_drives_native_checks_without_requiring_omitted_models(validator):
    config = validator.load_configuration(config_path='configs/experiments/server_focused.yaml')
    assert validator.selected_native_methods(config) == ['SAITS', 'MOMENT-1-base']
    args = SimpleNamespace(level='native', device='cuda:0', threads=32,
                           methods=validator.selected_native_methods(config))
    commands = validator.commands_for(args)
    assert commands[-1][-3:] == ['--methods', 'SAITS', 'MOMENT-1-base']
    assert not any('BRITS' in command or 'CSDI' in command for command in commands)
    args.methods = []
    assert len(validator.commands_for(args)) == 1  # Real device check only.


def test_canonical_full_guard_preserves_unreduced_hpo_and_epoch_validation(validator):
    config = validator.load_configuration(config_path='configs/experiments/server_full.yaml')
    assert config['hpo_trial_indices'] == list(range(6))
    assert config['hpo_max_epochs'] == 100 and config['hpo_patience'] == 10
    assert config['epoch_validation_max_per_stratum'] is None
    for field, value in [('hpo_trial_indices', [0, 1]), ('hpo_max_epochs', 20),
                         ('hpo_patience', 5), ('epoch_validation_max_per_stratum', 20)]:
        changed = deepcopy(config)
        changed[field] = value
        with pytest.raises(ValueError, match=field):
            validator.check_scientific_config(changed)


def test_custom_configs_are_schema_validated_without_forcing_the_full_catalogue(validator, tmp_path):
    config = yaml.safe_load((ROOT / 'configs/experiments/server_focused.yaml').read_text())
    config['methods'] = ['median', 'SAITS']
    config['require_available'] = ['SAITS']
    config['hpo_trials'] = 1
    config['hpo_trial_indices'] = [0]
    custom = tmp_path / 'my study.yaml'
    custom.write_text(yaml.safe_dump(config))
    loaded = validator.load_configuration(config_path=custom)
    assert loaded['methods'] == ['median', 'SAITS'] and loaded['hpo_trials'] == 1
    config['hpo_trial_indices'] = [0, 1]
    custom.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='hpo_trial_indices'):
        validator.load_configuration(config_path=custom)

import json
import os
from types import SimpleNamespace

import pytest

from energygap.runtime.threads import (
    ENVIRONMENT_VARIABLES, configure_cpu_threads, configure_torch_threads,
    requested_threads,
)


def clean_environment(monkeypatch):
    for name in (*ENVIRONMENT_VARIABLES, 'ENERGYGAP_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC'):
        # Track originally absent keys too, since configure_cpu_threads writes env.
        monkeypatch.setenv(name, '1')
        monkeypatch.delenv(name)


def test_neural_default_preserves_one_thread(monkeypatch):
    clean_environment(monkeypatch)
    assert requested_threads() == 1


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 'wrong'])
def test_invalid_threads(value):
    with pytest.raises(ValueError, match='positive integer'):
        requested_threads(value)


def test_explicit_budget_is_exported_to_every_cpu_pool(monkeypatch):
    clean_environment(monkeypatch)
    # Do not change actual test-process BLAS/Torch pools.
    import sys
    for name in ('numpy', 'scipy', 'torch'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    result = configure_cpu_threads(32)
    assert result['requested'] == 32
    assert requested_threads() == 32
    assert all(os.environ[name] == '32' for name in ENVIRONMENT_VARIABLES)


def test_omitted_flag_preserves_preexisting_blas_environment(monkeypatch):
    clean_environment(monkeypatch)
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    assert configure_cpu_threads()['requested'] is None
    assert os.environ['OMP_NUM_THREADS'] == '8'


def test_torch_obeys_inherited_budget_and_initialized_interop(monkeypatch):
    monkeypatch.setenv('ENERGYGAP_NUM_THREADS', '32')
    calls = []
    fake = SimpleNamespace(
        set_num_threads=calls.append,
        get_num_threads=lambda: calls[-1],
        get_num_interop_threads=lambda: 8,
    )
    def initialized(_):
        raise RuntimeError('parallel work already started')
    fake.set_num_interop_threads = initialized
    metadata = configure_torch_threads(fake)
    assert calls == [32]
    assert metadata['intraop'] == 32
    assert metadata['interop'] == 8
    assert metadata['interop_control'] == 'already_initialized_retained'


def test_cli_configures_threads_before_pipeline_import_work(monkeypatch, capsys):
    from energygap.cli import main
    import energygap.pipeline as pipeline
    import energygap.runtime.threads as module
    events = []
    monkeypatch.setattr(module, 'configure_cpu_threads', lambda value: events.append(('threads', value)) or {})
    monkeypatch.setattr(pipeline, 'run_pipeline', lambda **kwargs: events.append(('run', kwargs['device'])) or {'status': 'completed'})
    assert main(['run', '--profile', 'smoke', '--device', 'cpu', '--threads', '32']) == 0
    assert events == [('threads', 32), ('run', 'cpu')]
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


def test_moment_bridge_inherits_explicit_thread_budget(monkeypatch):
    monkeypatch.setenv('ENERGYGAP_NUM_THREADS', '32')
    from energygap.methods.moment_subprocess import MOMENTSubprocess
    assert MOMENTSubprocess().threads == 32

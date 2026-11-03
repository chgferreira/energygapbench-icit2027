import fcntl
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location('server_inference_benchmark', ROOT / 'scripts/benchmark_server_inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_refuses_active_canonical_project_lock(benchmark, monkeypatch, tmp_path):
    helper = benchmark.environment_helper()
    monkeypatch.setattr(benchmark, 'environment_helper', lambda: helper)
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    directory = tmp_path / 'artifacts/analyses'
    directory.mkdir(parents=True)
    with (directory / 'server.lock').open('a') as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already holds the project lock'):
            with benchmark.exclusive_project('cpu'):
                pytest.fail('benchmark ran concurrently with a coordinator')


def test_benchmark_retains_scalar_reference_and_rejects_changed_observations(benchmark):
    queries = benchmark.synthetic_queries(4, 16)
    outputs = [{'point_estimate': np.nan_to_num(query.values)} for query in queries]
    reference, difference = benchmark.validate_outputs(outputs, queries)
    assert reference.shape == (4, 16) and difference == 0
    changed = [{'point_estimate': row['point_estimate'].copy()} for row in outputs]
    changed[0]['point_estimate'][0] += .001
    with pytest.raises(ValueError, match='admitted observations changed'):
        benchmark.validate_outputs(changed, queries, reference)
    changed = [{'point_estimate': row['point_estimate'].copy()} for row in outputs]
    hidden = np.flatnonzero(~queries[0].available_to_method)[0]
    changed[0]['point_estimate'][hidden] += .1
    with pytest.raises(ValueError, match='differs from scalar reference'):
        benchmark.validate_outputs(changed, queries, reference)


def test_benchmark_never_accepts_in_progress_epoch_as_completed_model(benchmark):
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--checkpoint', 'runs/study/models/model.epoch.pt'])


def test_benchmark_keeps_measurements_outside_paper_and_requires_full_cuda_context(benchmark):
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--output', 'paper/results.json'])
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--device', 'cuda:0', '--context', '16'])
    args = benchmark.parse_args(['--device', 'cpu', '--context', '16', '--batch-sizes', '4', '4'])
    assert args.batch_sizes == [1, 4]
    assert args.output.is_relative_to(ROOT / 'artifacts/analyses/performance')


def test_benchmark_reports_requested_and_effective_sizes_after_oom_halving(benchmark):
    outputs = [{'timing': {'batch_size': 128, 'requested_batch_size': 256, 'oom_retries': 1},
                'model_metadata': {'adaptive_batch_size': 128, 'input_cache_hits': 128,
                                   'preprocess_policy': 'sanitized'}} for _ in range(256)]
    result = benchmark.batch_metadata(outputs, 256, 256)
    assert result['requested_batch_size'] == 256
    assert result['effective_native_batch_sizes'] == [128]
    assert result['oom_fallback_used'] and result['max_oom_retries_per_query'] == 1
    assert result['preprocessing_and_cache_metadata']['input_cache_hits'] == 128
    # The adapter remembers a safe size after warmup, so measured calls may
    # avoid another allocation failure while still using the smaller size.
    reused = [{'timing': {'batch_size': 128, 'adaptive_batch_size': 128, 'oom_retries': 0}}]
    assert benchmark.batch_metadata(reused, 256, 256)['oom_fallback_used']
    with pytest.raises(ValueError, match='valid native batch sizes'):
        benchmark.batch_metadata([{'timing': {'batch_size': 512}}], 256, 256)


def test_benchmark_defaults_exercise_complete_256_query_batch(benchmark):
    args = benchmark.parse_args([])
    assert args.threads == 32 and 256 in args.batch_sizes
    assert args.queries >= 256
    assert args.brits_execution == 'both'


def test_benchmark_supports_frozen_moment_and_explicit_cuda_graph_optin(benchmark):
    args = benchmark.parse_args(['--method', 'MOMENT-1-base'])
    assert args.context == 512 and args.checkpoint is None
    args = benchmark.parse_args(['--brits-execution', 'vectorized_cuda_graph'])
    assert args.brits_execution == 'vectorized_cuda_graph'
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--method', 'MOMENT-1-base', '--checkpoint', 'local.pt'])

"""GPU activity and allocated VRAM are distinct, physical-device diagnostics."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROW = '2026/11/01 12:00:00, 0, GPU-abc-123, Tesla V100-PCIE-16GB, 7, 3, 1392, 16384, 39, 250, P0, 42\n'


@pytest.fixture
def telemetry():
    spec = importlib.util.spec_from_file_location('gpu_telemetry_diagnostic', ROOT / 'scripts/gpu_telemetry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_allocation_cannot_be_reported_as_gpu_utilization(telemetry):
    sample = telemetry.parse_sample(ROW)
    assert sample['gpu_busy_percent'] == 7
    assert sample['memory_controller_busy_percent'] == 3
    assert sample['device_memory_used_mib'] == 1392
    assert sample['device_memory_total_mib'] == 16384
    missing = telemetry.parse_sample(ROW.replace(', 39,', ', [N/A],'))
    assert missing['power_draw_watts'] is None
    with pytest.raises(ValueError, match='exactly one physical GPU'):
        telemetry.parse_sample(ROW + ROW)


def test_poll_has_bounded_subprocess_and_explicit_physical_device(telemetry, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=ROW, stderr='')
    monkeypatch.setattr(telemetry.subprocess, 'run', run)
    sample = telemetry.poll_gpu('GPU-abc-123', timeout_seconds=2.)
    command, options = calls[0]
    assert '--id=GPU-abc-123' in command and command[0] == 'nvidia-smi'
    assert options['timeout'] == 2. and not options.get('shell', False)
    assert sample['uuid'] == 'GPU-abc-123'


def test_telemetry_keeps_available_samples_after_query_timeout(telemetry, monkeypatch, tmp_path):
    clock, calls = [0.], []
    monkeypatch.setattr(telemetry.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(telemetry.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def poll(gpu, timeout_seconds):
        calls.append(timeout_seconds)
        clock[0] += .2
        if len(calls) == 1:
            raise subprocess.TimeoutExpired('nvidia-smi', timeout_seconds)
        return telemetry.parse_sample(ROW)
    monkeypatch.setattr(telemetry, 'poll_gpu', poll)
    args = SimpleNamespace(gpu='0', duration_seconds=.5, interval_seconds=.2,
                           query_timeout_seconds=5., output=tmp_path / 'telemetry.json')
    assert telemetry.collect(args) == 0
    report = json.loads(args.output.read_text())
    assert report['status'] == 'completed_with_errors'
    assert report['summary']['sample_count'] == 2
    assert len(report['errors']) == 1 and max(calls) <= .5
    assert report['diagnostic_only'] and 'not_process_specific' in report['scope']


def test_polling_options_are_bounded_and_do_not_overwrite_reports(telemetry, tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'ROOT', tmp_path)
    for options in [['--gpu', 'cuda:0'], ['--duration-seconds', 'inf'],
                    ['--duration-seconds', '3601'], ['--interval-seconds', '0'],
                    ['--query-timeout-seconds', '31'], ['--output', 'paper/results.json']]:
        with pytest.raises(SystemExit):
            telemetry.parse_args(options)
    destination = tmp_path / 'artifacts/analyses/performance/old.json'
    destination.parent.mkdir(parents=True)
    destination.write_text('preserve')
    with pytest.raises(SystemExit):
        telemetry.parse_args(['--output', str(destination)])
    assert destination.read_text() == 'preserve'

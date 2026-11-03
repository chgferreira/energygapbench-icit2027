#!/usr/bin/env python3
"""Sample physical-GPU activity without running CUDA or touching an experiment.

This is device-wide diagnostic telemetry, separate from scientific job timings.
Memory allocation is not GPU utilization. A physical nvidia-smi index/UUID is
explicit because CUDA_VISIBLE_DEVICES can remap the process's cuda:0.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('timestamp', 'index', 'uuid', 'name', 'utilization.gpu', 'utilization.memory',
          'memory.used', 'memory.total', 'power.draw', 'power.limit', 'pstate', 'temperature.gpu')
NUMERIC_FIELDS = {'utilization.gpu': 'gpu_busy_percent',
    'utilization.memory': 'memory_controller_busy_percent',
    'memory.used': 'device_memory_used_mib', 'memory.total': 'device_memory_total_mib',
    'power.draw': 'power_draw_watts', 'power.limit': 'power_limit_watts',
    'temperature.gpu': 'temperature_celsius'}


def parse_sample(output):
    rows = list(csv.reader(io.StringIO(output.strip())))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise ValueError('Expected exactly one physical GPU row with the requested fields')
    raw = {key: value.strip() for key, value in zip(FIELDS, rows[0])}
    result = {key: raw[key] for key in ('timestamp', 'index', 'uuid', 'name', 'pstate')}
    for source, target in NUMERIC_FIELDS.items():
        try:
            value = float(raw[source])
            result[target] = value if math.isfinite(value) else None
        except ValueError:
            result[target] = None
    return result


def poll_gpu(gpu, timeout_seconds=5.):
    command = ['nvidia-smi', '--id=' + gpu, '--query-gpu=' + ','.join(FIELDS),
               '--format=csv,noheader,nounits']
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=timeout_seconds, check=False)
    if completed.returncode:
        raise RuntimeError(f'nvidia-smi exited {completed.returncode}: {completed.stderr.strip()[:1000]}')
    return parse_sample(completed.stdout)


def summarize(samples):
    result = {'sample_count': len(samples)}
    for name in NUMERIC_FIELDS.values():
        values = [row[name] for row in samples if row.get(name) is not None]
        result[name] = {'mean': statistics.mean(values), 'max': max(values),
                        'supported_samples': len(values)} if values else None
    return result


def publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def collect(args):
    started = time.monotonic()
    report = dict(status='sampling', diagnostic_only=True,
        created_at=datetime.now(timezone.utc).isoformat(), physical_gpu_selector=args.gpu,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='physical_device_wide_not_process_specific_not_scientific_compute_timing',
        interpretation={
            'gpu_busy_percent': 'nvidia-smi device sampling-period GPU activity, not allocated VRAM',
            'memory_controller_busy_percent': 'memory read/write activity, not fraction of VRAM allocated',
            'device_memory_used_mib': 'device-wide memory allocation, separate from GPU activity',
            'sampling_limits': 'periodic samples may miss short kernels; other processes contribute'},
        duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds,
        query_timeout_seconds=args.query_timeout_seconds, samples=[], errors=[])
    interrupted = False
    try:
        while time.monotonic() - started < args.duration_seconds:
            before = time.monotonic()
            remaining = args.duration_seconds - (before - started)
            try:
                sample = poll_gpu(args.gpu, timeout_seconds=min(args.query_timeout_seconds, remaining))
                sample['elapsed_seconds'] = time.monotonic() - started
                report['samples'].append(sample)
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                report['errors'].append(dict(elapsed_seconds=time.monotonic() - started,
                                             error=f'{type(exc).__name__}: {exc}'))
                # A missing executable is not repaired by polling repeatedly.
                if isinstance(exc, FileNotFoundError):
                    break
            report['summary'] = summarize(report['samples'])
            publish(args.output, report)
            delay = min(args.interval_seconds - (time.monotonic() - before),
                        args.duration_seconds - (time.monotonic() - started))
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        interrupted = True
    report.update(status=('interrupted' if interrupted else
                  'completed_with_errors' if report['errors'] and report['samples'] else
                  'completed' if report['samples'] else 'unavailable'),
                  elapsed_seconds=time.monotonic() - started,
                  summary=summarize(report['samples']))
    publish(args.output, report)
    print(json.dumps({'report': str(args.output), 'status': report['status'],
                      'samples': len(report['samples']), 'physical_gpu_selector': args.gpu}))
    return 0 if report['samples'] else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0',
        help='Physical nvidia-smi index or GPU UUID; this is not a remapped cuda:N index')
    parser.add_argument('--duration-seconds', type=float, default=120.)
    parser.add_argument('--interval-seconds', type=float, default=1.)
    parser.add_argument('--query-timeout-seconds', type=float, default=5.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)', args.gpu):
        parser.error('--gpu requires a physical integer index or GPU UUID')
    for name, low, high in [('duration_seconds', .1, 3600.), ('interval_seconds', .2, 60.),
                            ('query_timeout_seconds', .1, 30.)]:
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f'--{name.replace("_", "-")} must be finite and between {low} and {high}')
    directory = (ROOT / 'artifacts/analyses/performance').resolve()
    path = args.output or directory / ('gpu-telemetry-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}.json')
    args.output = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not args.output.is_relative_to(directory) or args.output.exists():
        parser.error('--output must be a new file under artifacts/analyses/performance')
    return args


if __name__ == '__main__':
    raise SystemExit(collect(parse_args()))

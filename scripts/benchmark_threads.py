#!/usr/bin/env python3
"""Compare CPU pool sizes in fresh processes, optionally with a CUDA ridge solve.

Synthetic timing diagnostics do not estimate full neural training or article ETA.
Reports stay under artifacts/analyses and are excluded from source releases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def measure(args):
    from energygap.runtime.threads import configure_cpu_threads, configure_torch_threads
    configuration = configure_cpu_threads(args.threads)
    import numpy as np
    rng = np.random.default_rng(20261031)
    design = rng.normal(size=(args.dimension*8, args.dimension))
    target = rng.normal(size=len(design))
    device = 'cpu'
    runtime = None
    if args.device != 'cpu':
        import torch
        from energygap.runtime.device import resolve_device, torch_device_metadata, synchronize
        device = resolve_device(args.device, allow_missing_torch=False)
        if not device.startswith('cuda:'):
            raise ValueError('Benchmark GPU branch requires explicit CUDA; use --device cpu for CPU')
        configuration['torch'] = configure_torch_threads(torch, args.threads)
        runtime = torch_device_metadata(device)
        gpu_x = torch.as_tensor(design, dtype=torch.float64, device=device)
        gpu_y = torch.as_tensor(target, dtype=torch.float64, device=device)
    timings = []
    # First pass is a warmup, reported separately and not included in medians.
    for index in range(args.repeats+1):
        start = perf_counter()
        gram = design.T@design + np.eye(args.dimension)
        rhs = design.T@target
        cpu_preparation = perf_counter()-start
        start = perf_counter()
        coefficients = np.linalg.solve(gram, rhs)
        cpu_solve = perf_counter()-start
        if not np.isfinite(coefficients).all():
            raise RuntimeError('nonfinite CPU benchmark solve')
        gpu_seconds = None
        if device.startswith('cuda:'):
            synchronize(device)
            start = perf_counter()
            gpu_gram = gpu_x.T@gpu_x + torch.eye(args.dimension, dtype=torch.float64, device=device)
            result = torch.linalg.solve(gpu_gram, gpu_x.T@gpu_y)
            synchronize(device)
            gpu_seconds = perf_counter()-start
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError('nonfinite CUDA benchmark solve')
        timings.append({'warmup': index == 0, 'cpu_gram_seconds': cpu_preparation,
                        'cpu_solve_seconds': cpu_solve, 'cuda_ridge_seconds': gpu_seconds})
    measured = timings[1:]
    return {'threads': args.threads, 'device': device, 'configuration': configuration,
            'numpy': np.__version__, 'runtime': runtime, 'measurements': timings,
            'median_cpu_seconds': statistics.median(r['cpu_gram_seconds']+r['cpu_solve_seconds'] for r in measured),
            'median_cuda_seconds': statistics.median(r['cuda_ridge_seconds'] for r in measured) if device.startswith('cuda:') else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--counts', nargs='+', type=int, default=[1, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dimension', type=int, default=512)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/analyses/thread_benchmark.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--threads', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(count < 1 for count in args.counts) or args.threads < 1 or args.repeats < 1 or args.dimension < 2:
        parser.error('counts, threads and repeats must be positive; dimension must be >=2')
    if args.worker:
        print(json.dumps(measure(args), allow_nan=False))
        return 0
    results = []
    for count in dict.fromkeys(args.counts):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--threads', str(count),
                   '--device', args.device, '--repeats', str(args.repeats), '--dimension', str(args.dimension)]
        execution = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if execution.returncode:
            raise RuntimeError(f'{count}-thread benchmark failed: '+execution.stderr[-3000:]+execution.stdout[-1000:])
        record = json.loads(execution.stdout.strip().splitlines()[-1])
        results.append(record)
        print(json.dumps({'threads': count, 'median_cpu_seconds': record['median_cpu_seconds'],
                          'median_cuda_seconds': record['median_cuda_seconds']}), flush=True)
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'purpose': 'synthetic CPU pool comparison; not full training ETA or scientific results',
              'cpu_count': os.cpu_count(), 'dimension': args.dimension, 'repeats': args.repeats,
              'results': results, 'fastest_cpu_threads_in_this_probe': min(results, key=lambda r: r['median_cpu_seconds'])['threads'],
              'interpretation': '32 threads is a pool limit; more threads may be slower. CUDA experiments remain serial.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(args.output), 'fastest_cpu_threads_in_this_probe': report['fastest_cpu_threads_in_this_probe']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

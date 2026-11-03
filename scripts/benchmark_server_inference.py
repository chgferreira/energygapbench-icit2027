#!/usr/bin/env python3
"""Measure deterministic native inference batches on fixed synthetic queries.

Run only after the experiment has stopped. This diagnostic does not change the
scientific run, its checkpoint, or its training batch size. CUDA is the default;
CPU is available to validate the diagnostic locally without GPU timing claims.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ('BRITS', 'SAITS', 'ImputeFormer', 'T1', 'MOMENT-1-base')


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='BRITS')
    parser.add_argument('--brits-execution', choices=('native', 'vectorized', 'vectorized_cuda_graph', 'both'), default='both',
                        help='BRITS only: compare FP32 paths using identical fitted weights and queries')
    parser.add_argument('--checkpoint', type=Path,
                        help='Optional completed fitted model (.pt plus checksum), not an .epoch.pt training checkpoint')
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--threads', type=positive, default=32)
    parser.add_argument('--batch-sizes', nargs='+', type=positive, default=[1, 32, 64, 128, 256],
                        help='Requested deterministic inference batches; CUDA OOM halving is reported separately')
    parser.add_argument('--queries', type=positive, default=256,
                        help='Use at least 256 queries to measure a complete requested batch of 256')
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--context', type=positive, default=512,
                        help='512 for server measurements; smaller values are CPU diagnostic checks only')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--output', type=Path,
                        help='New JSON file under artifacts/analyses/performance (default: timestamped file)')
    args = parser.parse_args(argv)
    if args.context < 8 or args.device != 'cpu' and args.context != 512:
        parser.error('use context=512 on CUDA; CPU checks require context>=8')
    if not (0 <= args.atol < float('inf') and 0 <= args.rtol < float('inf')):
        parser.error('atol and rtol must be finite and nonnegative')
    args.batch_sizes = list(dict.fromkeys([1, *args.batch_sizes]))
    if args.checkpoint and args.checkpoint.name.endswith('.epoch.pt'):
        parser.error('--checkpoint requires a completed fitted model, not an in-progress epoch checkpoint; omit it for a synthetic fit')
    if args.method == 'MOMENT-1-base' and (args.checkpoint or args.context != 512):
        parser.error('MOMENT uses the pinned local pretrained checkpoint and context=512; omit --checkpoint')
    directory = (ROOT / 'artifacts/analyses/performance').resolve()
    output = args.output or directory / ('inference-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}.json')
    args.output = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    if not args.output.is_relative_to(directory):
        parser.error('--output must stay under artifacts/analyses/performance')
    if args.output.exists():
        parser.error('--output already exists; choose a new diagnostic filename')
    return args


def environment_helper():
    spec = importlib.util.spec_from_file_location('energygap_server_dependencies', ROOT / 'scripts/server_dependencies.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def exclusive_project(device):
    """Share the canonical server lock and reject direct CLI coordinators too."""
    helper = environment_helper()
    if device.startswith('cuda:'):
        helper.validate_environment()
        helper.validate_torch(helper.installed_versions())
    directory = ROOT / 'artifacts/analyses'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'server.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('An experiment/setup already holds the project lock. Stop it safely before benchmarking; no process was interrupted.') from exc
        try:
            if Path('/proc').is_dir():
                helper.check_coordinators(ROOT)
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def synthetic_queries(count, context):
    import numpy as np
    result = []
    positions = np.arange(context)
    for index in range(count):
        values = (np.sin(positions / (9 + index % 5) + index / 11)
                  + .25 * np.cos(positions / 31) + positions / (10 * context))
        length = min((3, 12, 36)[index % 3], context - 2)
        first = 1 + (index * 7) % (context - length - 1)
        observed = np.ones(context, bool)
        observed[first:first + length] = False
        values[~observed] = np.nan
        times = (positions + index * context).astype(float) * 300
        result.append(SimpleNamespace(values=values, available_to_method=observed,
            padding_mask=np.zeros(context, bool), target_gap_mask=~observed, times=times,
            issuance_time=float(times[-1]), task='R'))
    return result


def validate_outputs(outputs, queries, reference=None, *, atol=1e-5, rtol=1e-5):
    import numpy as np
    if len(outputs) != len(queries):
        raise ValueError('one result per query required')
    points = []
    for output, query in zip(outputs, queries):
        point = np.asarray(output['point_estimate'], float)
        if point.shape != query.values.shape or not np.isfinite(point).all():
            raise ValueError('nonfinite output or changed context shape')
        observed = np.asarray(query.available_to_method, bool)
        if not np.array_equal(point[observed], query.values[observed]):
            raise ValueError('admitted observations changed')
        points.append(point)
    points = np.stack(points)
    difference = 0. if reference is None else float(np.max(np.abs(points - reference)))
    if reference is not None and not np.allclose(points, reference, atol=atol, rtol=rtol):
        raise ValueError(f'batch output differs from scalar reference: max_abs_error={difference:g}, atol={atol:g}, rtol={rtol:g}')
    return points, difference


def batch_metadata(outputs, requested, query_count):
    """Distinguish the requested size from actual batches after CUDA OOM retries."""
    effective = sorted({int(output.get('timing', {}).get('batch_size', 0)) for output in outputs})
    if not effective or min(effective) < 1 or max(effective) > min(requested, query_count):
        raise ValueError(f'adapter did not confirm valid native batch sizes: {effective}')
    retries = [int(output.get('timing', {}).get('oom_retries', 0)) for output in outputs]
    adapted = any(int(output.get('timing', {}).get('adaptive_batch_size', requested)) < requested
                  for output in outputs)
    policies = {}
    for output in outputs:
        for key, value in output.get('model_metadata', {}).items():
            if any(term in key for term in ('preprocess', 'cache', 'adaptive', 'oom', 'inference_path', 'brits_execution', 'cuda_graph')):
                policies[key] = value
    return dict(requested_batch_size=requested, effective_native_batch_sizes=effective,
        oom_fallback_used=any(retries) or adapted, max_oom_retries_per_query=max(retries, default=0),
        preprocessing_and_cache_metadata=policies,
        preprocessing_metadata_scope='last_native_batch; input_cache_hits_and_misses_are_per_native_batch')


def clear_preprocessing_cache(method):
    if method == 'BRITS':
        from energygap.methods.prepared_inference import BRITS_INPUT_CACHE
        BRITS_INPUT_CACHE.clear()
        return True
    return False


def load_model(args, report):
    import numpy as np
    from energygap.methods.conference_models import create_adapter, hpo_configurations, load_adapter
    started = perf_counter()
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        try:
            model = load_adapter(checkpoint, root=ROOT, device=args.device, threads=args.threads)
        except KeyError as exc:
            raise ValueError('Checkpoint is not a completed fitted model; omit --checkpoint for the independent synthetic fit') from exc
        if model.name != args.method or model.context != args.context:
            raise ValueError('--method and --context must match the completed fitted checkpoint')
        source = dict(kind='completed_fitted_checkpoint', path=str(checkpoint), sha256=file_sha256(checkpoint))
    elif args.method == 'MOMENT-1-base':
        model = create_adapter(args.method, root=ROOT, device=args.device, threads=args.threads,
                               context=512, inference_batch_size=max(args.batch_sizes))
        # Trigger verified local checkpoint loading outside all inference timings.
        model.infer_query(synthetic_queries(1, 512)[0])
        source = dict(kind='pinned_local_frozen_pretrained_checkpoint', local_training=False)
    else:
        print(f'{args.method}: fitting one diagnostic epoch on 32 synthetic windows...', flush=True)
        model = create_adapter(args.method, root=ROOT, device=args.device, threads=args.threads,
            seed=11, params=hpo_configurations(args.method)[0], epochs=1, patience=1,
            batch_size=32, context=args.context)
        t = np.arange(40 * args.context, dtype=np.float32).reshape(40, args.context, 1)
        windows = np.sin(t / 19) + .2 * np.cos(t / 43)
        windows[0, 2], windows[32, 4] = np.nan, np.nan
        model.fit_windows(windows[:32], windows[32:])
        source = dict(kind='synthetic_diagnostic_fit', completed_epochs=model.completed_epochs,
                      training_windows=32, chronological_validation_windows=8,
                      training_execution=getattr(model, 'brits_execution', None) if args.method == 'BRITS' else 'native',
                      parameters=hpo_configurations(args.method)[0])
    if args.method == 'MOMENT-1-base':
        native = model.adapter.native.model
    else:
        native = model.adapter.model.model if args.method != 'T1' else model.adapter.model
    devices = {str(parameter.device) for parameter in native.parameters()}
    if devices != {args.device}:
        raise RuntimeError(f'model parameters use {sorted(devices)}, expected {args.device}')
    report.update(model_source=source, model_parameter_devices=sorted(devices),
                  model_preparation_seconds=perf_counter() - started,
                  training_batch_size=model.batch_size if args.method != 'MOMENT-1-base' else None)
    return model


def benchmark(args):
    print('Starting independent inference diagnostic; importing scientific libraries...', flush=True)
    from energygap.runtime.threads import configure_cpu_threads, configure_torch_threads
    configure_cpu_threads(args.threads)
    import torch
    configure_torch_threads(torch, args.threads)
    from energygap.runtime.device import cuda_probe, resolve_device, synchronize, torch_device_metadata
    device = resolve_device(args.device, allow_missing_torch=False)
    runtime = cuda_probe(device) if device.startswith('cuda:') else torch_device_metadata(device)
    report = dict(schema_version=1, status='running',
        purpose='synthetic_deterministic_inference_diagnostic_not_article_results_or_full_run_ETA',
        created_at=datetime.now(timezone.utc).isoformat(), executable=sys.executable,
        device=device, cuda_tested=device.startswith('cuda:'), threads=args.threads,
        method=args.method, context=args.context, queries=args.queries, repeats=args.repeats,
        batch_sizes=args.batch_sizes, runtime=runtime, scalar_parity_tolerance=dict(atol=args.atol, rtol=args.rtol),
        sampling_policy='deterministic_query_batching_including_frozen_MOMENT; CSDI_and_TSICL_not_measured',
        precision='float32_native_no_autocast_no_bf16_tf32_or_triton_requirement',
        batch_results=[])
    publish(args.output, report)
    try:
        model = load_model(args, report)
        report['runtime'] = torch_device_metadata(device)
        queries = synthetic_queries(args.queries, args.context)
        seeds = [10001 + index for index in range(args.queries)]
        print(f'Scalar reference: {len(queries)} fixed queries...', flush=True)
        synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            scalar_outputs = [model.infer_query(query, sample_seed=seed) for query, seed in zip(queries, seeds)]
        synchronize(device)
        report['scalar_reference_seconds'] = perf_counter() - started
        reference, _ = validate_outputs(scalar_outputs, queries)
        del scalar_outputs
        variants = (['native', 'vectorized'] if args.brits_execution == 'both'
                    else [args.brits_execution]) if args.method == 'BRITS' else ['default']
        report['execution_variants'] = variants
        for variant, batch_size in ((variant, size) for variant in variants for size in args.batch_sizes):
            if args.method == 'BRITS':
                model.set_brits_execution(variant)
                # Each requested configuration gets an independent allocation
                # attempt rather than inheriting another path's OOM ceiling.
                model.__dict__.pop('_cuda_inference_batch_limit', None)
            print(f'{variant}, batch {batch_size}: warmup + {args.repeats} measured repetitions...', flush=True)
            gc.collect()
            if device.startswith('cuda:'):
                torch.cuda.empty_cache()
            row = dict(batch_size=batch_size, requested_batch_size=batch_size,
                       execution_variant=variant, status='running', measurements=[])
            row['preprocessing_cache_cleared_before_warmup'] = clear_preprocessing_cache(args.method)
            report['batch_results'].append(row)
            try:
                synchronize(device)
                started = perf_counter()
                with torch.inference_mode():
                    outputs = model.infer_queries(queries, sample_seeds=seeds, batch_size=batch_size)
                synchronize(device)
                row['warmup_seconds'] = perf_counter() - started
                _, difference = validate_outputs(outputs, queries, reference, atol=args.atol, rtol=args.rtol)
                row['scalar_max_abs_error'] = difference
                row.update(batch_metadata(outputs, batch_size, len(queries)))
                row['warmup_preprocessing_and_cache_metadata'] = row['preprocessing_and_cache_metadata']
                del outputs
                for repeat in range(args.repeats):
                    if device.startswith('cuda:'):
                        torch.cuda.reset_peak_memory_stats(device)
                    synchronize(device)
                    started = perf_counter()
                    with torch.inference_mode():
                        outputs = model.infer_queries(queries, sample_seeds=seeds, batch_size=batch_size)
                    synchronize(device)
                    elapsed = perf_counter() - started
                    allocated = int(torch.cuda.max_memory_allocated(device)) if device.startswith('cuda:') else None
                    reserved = int(torch.cuda.max_memory_reserved(device)) if device.startswith('cuda:') else None
                    _, difference = validate_outputs(outputs, queries, reference, atol=args.atol, rtol=args.rtol)
                    row['scalar_max_abs_error'] = max(row['scalar_max_abs_error'], difference)
                    row['measurements'].append(dict(repeat=repeat + 1, seconds=elapsed,
                        queries_per_second=args.queries / elapsed,
                        peak_gpu_allocated_bytes=allocated, peak_gpu_reserved_bytes=reserved,
                        **batch_metadata(outputs, batch_size, len(queries))))
                    del outputs
                row.update(status='passed', observed_preserved=True,
                    median_seconds=statistics.median(item['seconds'] for item in row['measurements']),
                    peak_gpu_allocated_bytes=max(item['peak_gpu_allocated_bytes'] for item in row['measurements']) if device.startswith('cuda:') else None,
                    peak_gpu_reserved_bytes=max(item['peak_gpu_reserved_bytes'] for item in row['measurements']) if device.startswith('cuda:') else None)
                row['median_queries_per_second'] = args.queries / row['median_seconds']
                print(f'Requested batch {batch_size}, effective {row["effective_native_batch_sizes"]}: '
                      f'{row["median_queries_per_second"]:.2f} queries/s; scalar parity passed.', flush=True)
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                print(f'Batch {batch_size}: failed: {row["error"]}', flush=True)
                gc.collect()
                if device.startswith('cuda:'):
                    torch.cuda.empty_cache()
            publish(args.output, report)
        passed = [row for row in report['batch_results'] if row['status'] == 'passed']
        for row in passed:
            baseline = next((item for item in passed if item['batch_size'] == 1
                             and item['execution_variant'] == row['execution_variant']), None)
            if baseline:
                row['speedup_vs_batch_1'] = baseline['median_seconds'] / row['median_seconds']
            native = next((item for item in passed if item['execution_variant'] == 'native'
                           and item['batch_size'] == row['batch_size']), None)
            if native:
                row['speedup_vs_native_same_requested_batch'] = native['median_seconds'] / row['median_seconds']
        best = max(passed, key=lambda row: row['median_queries_per_second']) if passed else None
        report.update(status='passed' if len(passed) == len(args.batch_sizes) * len(variants) else 'completed_with_failures',
            fastest_measured_batch_size=best['batch_size'] if best else None,
            fastest_execution_variant=best['execution_variant'] if best else None,
            fastest_effective_native_batch_sizes=best['effective_native_batch_sizes'] if best else None,
            interpretation='Measured inference throughput only; no inference about whole-run ETA, training speed, or GPU utilization percentage.')
        publish(args.output, report)
        print(json.dumps({'report': str(args.output), 'status': report['status'],
                          'fastest_measured_batch_size': report['fastest_measured_batch_size'],
                          'cuda_tested': report['cuda_tested']}), flush=True)
        return 0 if report['status'] == 'passed' else 2
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        publish(args.output, report)
        raise


def main(argv=None):
    args = parse_args(argv)
    with exclusive_project(args.device):
        return benchmark(args)


if __name__ == '__main__':
    raise SystemExit(main())

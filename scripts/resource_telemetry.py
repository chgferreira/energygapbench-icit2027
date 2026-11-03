#!/usr/bin/env python3
"""Read Linux process-tree CPU/RAM/I/O and optional GPU telemetry; no model or DB writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_stat(raw, page_size):
    end = raw.rfind(')')
    values = raw[end + 2:].split()
    return dict(pid=int(raw[:raw.index('(')].strip()), name=raw[raw.index('(') + 1:end],
                state=values[0], ppid=int(values[1]), cpu_ticks=int(values[11]) + int(values[12]),
                threads=int(values[17]), start_ticks=int(values[19]), rss_bytes=int(values[21]) * page_size)


def read_processes(proc=Path('/proc')):
    processes = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            row = parse_stat((directory / 'stat').read_text(), os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            continue
        processes[row['pid']] = row
    return processes


def tree_rows(processes, pid):
    admitted = {pid}
    while True:
        expanded = admitted | {key for key, row in processes.items() if row['ppid'] in admitted}
        if expanded == admitted:
            break
        admitted = expanded
    return [processes[key] for key in sorted(admitted) if key in processes]


def rates(rows, previous, elapsed, ticks_per_second):
    for row in rows:
        old = previous.get((row['pid'], row['start_ticks']))
        delta = row['cpu_ticks'] - old['cpu_ticks'] if old else None
        row['cpu_percent_one_core'] = max(0., delta / ticks_per_second / elapsed * 100.) if delta is not None and elapsed > 0 else None
    measured = [row['cpu_percent_one_core'] for row in rows if row['cpu_percent_one_core'] is not None]
    return sum(measured) / 100. if measured else None


def memory(proc):
    fields = {}
    for line in (proc / 'meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            fields[name + '_bytes'] = int(value.split()[0]) * 1024
    return fields


def process_memory_io(proc, row):
    directory = proc / str(row['pid'])
    try:
        line = next(line for line in (directory / 'smaps_rollup').read_text().splitlines() if line.startswith('Pss:'))
        row['pss_bytes'] = int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        row['pss_bytes'] = None
    try:
        io = dict(line.split(':', 1) for line in (directory / 'io').read_text().splitlines())
        row['read_bytes'] = int(io['read_bytes']);row['write_bytes'] = int(io['write_bytes'])
    except (OSError, ValueError, KeyError):
        row['read_bytes'] = row['write_bytes'] = None


def collect(args):
    from gpu_telemetry import poll_gpu, publish
    proc = Path('/proc')
    started = previous_time = time.monotonic()
    previous = {};root_start = None;samples = [];errors = []
    report = dict(diagnostic_only=True, pid=args.pid, created_at=datetime.now(timezone.utc).isoformat(),
        scope='Linux process tree; GPU device-wide; no experiment/model/database mutation',
        cpu_count=os.cpu_count(), affinity_cpu_count=len(os.sched_getaffinity(0)),
        interpretation=dict(cpu='100% is one CPU core; CPU cores used sums process deltas',
            memory='PSS apportions shared pages; summed RSS can double count shared memory',
            gpu='device-wide periodic samples can miss short kernels and include other processes',
            throughput='resource occupancy is not scientific throughput or a speedup guarantee'), samples=samples, errors=errors)
    while time.monotonic() - started < args.duration_seconds:
        before = time.monotonic()
        processes = read_processes(proc)
        root = processes.get(args.pid)
        if root is None or (root_start is not None and root['start_ticks'] != root_start):
            errors.append('Target process exited or PID was reused');break
        root_start = root['start_ticks']
        rows = tree_rows(processes, args.pid)
        cores = rates(rows, previous, before - previous_time, os.sysconf('SC_CLK_TCK'))
        for row in rows:process_memory_io(proc, row)
        sample = dict(elapsed_seconds=before - started, cpu_cores_used=cores, processes=rows,
                      tree_rss_bytes=sum(row['rss_bytes'] for row in rows),
                      tree_pss_bytes=sum(row['pss_bytes'] for row in rows if row['pss_bytes'] is not None),
                      pss_supported_processes=sum(row['pss_bytes'] is not None for row in rows),
                      system_memory=memory(proc), load_average=list(os.getloadavg()))
        if args.gpu is not None:
            try:sample['gpu'] = poll_gpu(args.gpu, timeout_seconds=min(2., args.duration_seconds))
            except Exception as exc:sample['gpu_error'] = f'{type(exc).__name__}: {exc}'
        samples.append(sample)
        previous = {(row['pid'], row['start_ticks']):row for row in rows};previous_time = before
        publish(args.output, report)
        delay = min(args.interval_seconds - (time.monotonic() - before), args.duration_seconds - (time.monotonic() - started))
        if delay > 0:time.sleep(delay)
    measured = [row['cpu_cores_used'] for row in samples if row['cpu_cores_used'] is not None]
    report.update(status='completed' if samples and not errors else 'completed_with_errors' if samples else 'unavailable',
        summary=dict(sample_count=len(samples), mean_cpu_cores_used=statistics.mean(measured) if measured else None,
                     peak_cpu_cores_used=max(measured) if measured else None,
                     peak_tree_pss_bytes=max((row['tree_pss_bytes'] for row in samples),default=None)))
    publish(args.output, report)
    print(json.dumps(dict(report=str(args.output),status=report['status'],summary=report['summary']),indent=2))
    return 0 if samples else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True,help='Scientific Python coordinator PID, not the tee/shell PID')
    parser.add_argument('--duration-seconds',type=float,default=60.)
    parser.add_argument('--interval-seconds',type=float,default=2.)
    parser.add_argument('--gpu',default=None,help='Optional physical NVIDIA index or UUID')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.pid < 1:parser.error('--pid must be positive')
    for name,low,high in [('duration_seconds',1.,3600.),('interval_seconds',.2,60.)]:
        value=getattr(args,name)
        if not math.isfinite(value) or not low<=value<=high:parser.error(f'{name} is outside the supported bounds')
    directory=(ROOT/'artifacts/analyses/performance').resolve()
    args.output=(args.output or directory/('resource-telemetry-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}.json')).resolve()
    if not args.output.is_relative_to(directory) or args.output.exists():parser.error('--output must be a new analysis file under artifacts/analyses/performance')
    if not sys.platform.startswith('linux') or not Path('/proc/stat').exists():parser.error('Linux /proc is required; no CPU/GPU workload is launched')
    return args


if __name__ == '__main__':
    raise SystemExit(collect(parse_args()))

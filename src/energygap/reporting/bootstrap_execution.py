"""Bounded CPU execution of unchanged, independently seeded bootstrap draws.

Workers receive only small day-reduction arrays. They never read the ledger,
publish results, import a model library, or alter the numerical estimator.
"""
from __future__ import annotations

from collections import deque
import multiprocessing as mp
import os
from contextlib import contextmanager

import numpy as np

from energygap.statistics.paired import stable_seed


def _worker_init():
    # The spawn environment is also restricted before NumPy is imported. Keep
    # an explicit runtime limit for already initialized BLAS implementations.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


@contextmanager
def _spawn_environment():
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def draw_values(arrays, folds, block_days, start, end, seed=20261031):
    """Preserve the original per-ID RNG and NumPy reduction order exactly."""
    result = []
    for index in range(start, end):
        rng = np.random.default_rng(stable_seed(seed, index))
        values = []
        for fold in folds:
            sums, counts = arrays[fold]
            n = len(sums)
            length = min(block_days, n)
            starts = rng.integers(0, n-length+1, size=(n+length-1)//length)
            selected = (starts[:, None]+np.arange(length)).ravel()[:n]
            denominator = counts[selected].sum(axis=0)
            if np.any(denominator == 0):
                values = None
                break
            values.extend((sums[selected].sum(axis=0)/denominator).tolist())
        result.append(float(np.mean(values)) if values is not None else None)
    return result


def _draw_task(task):
    return draw_values(*task)


def worker_environment():
    """Small test/diagnostic probe; no neural dependency is imported here."""
    import sys
    from threadpoolctl import threadpool_info
    return dict(pid=os.getpid(), torch_imported="torch" in sys.modules,
                threadpools=threadpool_info())


class BootstrapExecutor:
    """One lazily spawned pool shared across contrasts, with bounded prefetch."""
    def __init__(self, workers=1, chunk_size=1000):
        for name, value in (("report_workers", workers), ("bootstrap_chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.workers, self.chunk_size, self.pool = workers, chunk_size, None

    def __enter__(self):
        return self

    def _ensure_pool(self):
        if self.pool is None:
            with _spawn_environment():
                self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init)
        return self.pool

    def chunks(self, arrays, folds, block_days, start, end, check=lambda: None):
        """Yield in logical-ID order; at most two tasks per worker are queued."""
        if self.workers == 1 or end-start <= self.chunk_size:
            for first in range(start, end, self.chunk_size):
                check()
                last = min(first+self.chunk_size, end)
                # Bounded subcalls retain a responsive signal/budget boundary.
                values = []
                for offset in range(first, last, 32):
                    check()
                    values.extend(draw_values(arrays, folds, block_days,
                                              offset, min(offset+32, last)))
                yield first, values
            return
        pool = self._ensure_pool()
        pending, next_index = deque(), start
        while next_index < end or pending:
            check()
            while next_index < end and len(pending) < 2*self.workers:
                last = min(next_index+self.chunk_size, end)
                task = (arrays, tuple(folds), block_days, next_index, last)
                pending.append((next_index, pool.apply_async(_draw_task, (task,))))
                next_index = last
            first, future = pending.popleft()
            while True:
                try:
                    values = future.get(timeout=.2)
                    break
                except mp.TimeoutError:
                    check()
            yield first, values

    def __exit__(self, kind, value, traceback):
        if self.pool is not None:
            if kind is None:
                self.pool.close()
            else:
                # Only CPU calculations are in flight; the parent is the sole
                # writer. Uncommitted work is recomputed on the next resume.
                self.pool.terminate()
            self.pool.join()
            self.pool = None

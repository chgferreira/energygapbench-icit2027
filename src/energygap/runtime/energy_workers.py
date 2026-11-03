"""Bounded CPU-only correction tasks; the coordinator alone owns evaluation/state.

One task contains one episode, all requested physical hypotheses and every native
joint draw. Spawn prevents inheriting a CUDA context. Worker initialization limits
both the environment and already-loaded BLAS/OpenMP pools to one thread.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
import multiprocessing
import os
import signal
import sys
from time import perf_counter


@dataclass
class EnergyEpisodeTask:
    base: object
    times: object
    target_mask: object
    seasonal: object
    scale: float
    hypotheses: tuple
    samples: object = None


_cancel_event = None


class EnergyWorkCancelled(RuntimeError):
    pass


def _initialize_worker(cancel_event):
    global _cancel_event
    _cancel_event = cancel_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from energygap.runtime.threads import configure_cpu_threads
    configure_cpu_threads(1)


def _check_cancelled():
    if _cancel_event is not None and _cancel_event.is_set():
        raise EnergyWorkCancelled('CPU energy correction cancelled at a trajectory boundary')


def correct_episode(task, *, check=None):
    """Inference-only payload: no truth, neural models, CUDA, or experiment files."""
    import numpy as np
    from energygap.methods.correction import correct_energy
    from energygap.methods.probabilistic import correct_joint_samples, ProbabilisticCapabilities

    def cancelled():
        _check_cancelled()
        if check is not None:
            check()

    results = []
    for hypothesis, constraint, parameters in task.hypotheses:
        cancelled()
        options = dict(scale_train=task.scale, lambda_energy=parameters['lambda_energy'],
                       gamma=parameters['gamma'], seasonal_start=task.seasonal)
        point = correct_energy(task.base, task.times, task.target_mask, constraint, **options)
        corrected = None
        seconds = None
        if task.samples is not None:
            started = perf_counter()
            draws = np.asarray(task.samples, float)
            if draws.ndim == 3:
                draws = draws[:, :, 0]
            corrected = correct_joint_samples(draws,task.times,task.target_mask,constraint,
                capabilities=ProbabilisticCapabilities(joint_samples=True),cancel_callback=cancelled,**options)
            seconds = perf_counter() - started
        results.append(dict(hypothesis=hypothesis, point=point, samples=corrected,
                            sample_correction_seconds=seconds))
    return results


def _worker_diagnostics():
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_info
    return dict(pid=os.getpid(),torch_imported='torch' in sys.modules,
                pools=threadpool_info(),numpy_blas=numpy.__config__.show(mode='dicts')['Build Dependencies']['blas'],
                environment={key:os.environ.get(key) for key in
                    ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})


class EnergyCorrectionPool:
    """Lazy reusable spawn pool with ordered results and a bounded input queue."""

    def __init__(self, workers=1, max_pending=None):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('energy_cpu_workers must be a positive integer')
        max_pending = 2 * workers if max_pending is None else max_pending
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError('energy_cpu_max_pending must be a positive integer')
        self.workers, self.max_pending = workers, max_pending
        self._executor = self._event = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        if self._executor is None:
            context = multiprocessing.get_context('spawn')
            self._event = context.Event()
            self._executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                initializer=_initialize_worker, initargs=(self._event,))

    @staticmethod
    def _check(stop, check):
        if check is not None:
            check()
        if stop is not None and stop.requested:
            raise EnergyWorkCancelled('CPU energy correction stop requested')

    def map_ordered(self, tasks, *, stop=None, check=None):
        """Consume at most max_pending tasks, in original episode/sample order.

        Iterating input can perform coordinator-owned GPU inference while older
        CPU tasks run. The queue bounds both prepared contexts and pending work.
        """
        tasks = iter(tasks)
        if self.workers == 1:
            for task in tasks:
                self._check(stop, check)
                result = correct_episode(task,check=lambda:self._check(stop,check))
                self._check(stop, check)
                yield result
            return
        self._start()
        pending = deque()
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.max_pending:
                    self._check(stop, check)
                    try:
                        task = next(tasks)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(self._executor.submit(correct_episode, task))
                if not pending:
                    break
                while True:
                    self._check(stop, check)
                    try:
                        result = pending[0].result(timeout=.1)
                        break
                    except TimeoutError:
                        pass
                pending.popleft()
                yield result
        except BaseException:
            self._event.set()
            for future in pending:
                future.cancel()
            # Running workers finish only their current solver, then observe
            # cancellation before the next draw/hypothesis. Nothing is published.
            self.close(cancel=True)
            raise

    def close(self, cancel=False):
        if self._executor is not None:
            if cancel:
                self._event.set()
            self._executor.shutdown(wait=True, cancel_futures=cancel)
            self._executor = self._event = None

    def diagnostics(self):
        """Audit the actual spawned pools; does not load any model/runtime Torch."""
        if self.workers == 1:
            return []
        self._start()
        futures=[self._executor.submit(_worker_diagnostics) for _ in range(self.workers)]
        return [future.result() for future in futures]

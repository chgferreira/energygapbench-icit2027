"""Serial safe-boundary executor for immutable unit shards."""
from __future__ import annotations
import time
from .state import StateStore, BudgetExhausted, stable_seed
from .eta import ETAEstimator
from .progress import Progress
from .signals import StopRequested

def run_units(store: StateStore, job_id: str, compute, *, total_units: int | None = None,
              shard_size: int = 32, progress: str = 'none', stop: StopRequested | None = None,
              signature: str = 'unit', metadata: dict | None = None,
              reservation_seconds: float = 60.0, progress_observer=None,
              publish_eta: bool = True) -> dict:
    """Execute remaining IDs only; compute receives logical index and a stable seed.

    Result keys are supplied by ``compute``. Invalid bootstrap draws should return
    a record with status=invalid, not raise or be redrawn. Each unit is indivisible.
    An observer owns the run-level display; local bars are then suppressed.
    ``publish_eta=False`` also leaves run-level ETA metadata to that observer.
    """
    if shard_size <= 0: raise ValueError('shard_size must be positive')
    if total_units is not None: store.add_job(job_id, total_units, metadata=metadata)
    job = store.job(job_id)
    if job['status'] == 'succeeded': return {'executed_units':0,'initial':job['completed_units'],'completed':True}
    if store.session_id is None: store.start_session()
    estimator = ETAEstimator(store.get_meta('eta_history', {})) if publish_eta else None
    initial = job['completed_units']; done = initial
    executed = 0
    observed_kind = job['metadata'].get('phase', 'unit')
    def notify(units=0, seconds=0.):
        if progress_observer is not None:
            progress_observer(signature, units, seconds, job_id=job_id, kind=observed_kind)
    with store.claim(job_id) as lease, Progress(job['total_units'], initial,
            'none' if progress_observer is not None else progress, job_id) as display:
        notify()
        try:
            while done < job['total_units']:
                store.check_budget()
                if stop and stop.requested:
                    store.interrupt(job_id, lease.token, 'signal requested at safe shard boundary')
                    return {'executed_units':executed,'initial':initial,'completed':False}
                records = []; reservation = store.begin_work(job_id, lease.token, reservation_seconds)
                started = time.monotonic()
                start = done
                for index in range(done, min(done+shard_size, job['total_units'])):
                    seed = stable_seed(store.fingerprint, job_id, index)
                    record = compute(index, seed)
                    if isinstance(record, list): records.extend(record)
                    else: records.append(record)
                    done = index + 1; executed += 1
                    if stop and stop.requested: break
                elapsed = time.monotonic()-started
                store.publish(job_id, lease.token, records, done)
                store.finish_work(reservation, elapsed)
                estimate = None
                if estimator is not None:
                    estimator.observe(signature, done-start, elapsed)
                    store.set_meta('eta_history', estimator.history)
                    estimate = estimator.estimate(job['total_units']-done, signature)
                    store.set_meta('eta', estimate)
                notify(done-start, elapsed)
                store.snapshot()
                display.update(done, estimate)
            store.finish(job_id, lease.token)
            notify()
            return {'executed_units':executed,'initial':initial,'completed':True}
        except BudgetExhausted:
            store.interrupt(job_id, lease.token, 'cumulative budget exhausted at safe boundary')
            raise

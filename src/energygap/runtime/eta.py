"""Active-time ETA, with explicit calibration and resource-aware DAG scheduling."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Callable
from .state import utcnow

class ETAEstimator:
    def __init__(self, history: dict | None = None, now: Callable | None = None):
        self.history = history or {}
        self.now = now or (lambda: datetime.now(timezone.utc))

    def observe(self, signature: str, units: int, active_seconds: float):
        if units <= 0 or active_seconds <= 0: return
        self.history.setdefault(signature, []).append({'units': units, 'active_seconds': active_seconds})

    def rate(self, signature: str):
        rows = self.history.get(signature, [])
        if not rows: return None
        return sum(x['units'] for x in rows) / sum(x['active_seconds'] for x in rows)

    def estimate(self, units_remaining: int, signature: str) -> dict:
        rate = self.rate(signature)
        remaining = units_remaining / rate if rate else (0.0 if units_remaining == 0 else None)
        now = self.now()
        if not isinstance(now, datetime): now = datetime.fromtimestamp(now, timezone.utc)
        rows = self.history.get(signature, [])
        spread = None
        if len(rows) >= 3:
            per_unit = sorted(x['active_seconds']/x['units'] for x in rows)
            spread = [units_remaining * per_unit[0], units_remaining * per_unit[-1]]
        return {'remaining_seconds': remaining, 'estimated_finish': (now + timedelta(seconds=remaining)).isoformat() if remaining is not None else None,
                'throughput_units_per_second': rate, 'signature': signature,
                'calibration': 'unavailable' if not rows else ('pilot_single_observation_uncertain' if len(rows) < 3 else 'empirical_active_time'),
                'heuristic_range_seconds': spread,
                'range_rule': 'observed min/max seconds per unit, not a confidence interval' if spread else None,
                'updated_at': now.isoformat()}

    def schedule(self, jobs: list[dict], capacities: dict[str,int] | None = None) -> dict:
        """Greedy deterministic list schedule; never divide total work by worker count.

        Jobs require one resource class. Uncalibrated durations keep ETA unavailable.
        """
        capacities = capacities or {'cpu': 1, 'gpu': 1}
        slots = {r: [0.0] * n for r, n in capacities.items()}
        end, schedule = {}, {}
        pending = {j['job_id']: j for j in jobs}
        unknown = []
        while pending:
            ready = sorted(k for k, j in pending.items() if all(d in end for d in j.get('dependencies', [])))
            if not ready: raise ValueError('DAG has missing dependencies or a cycle')
            for key in ready:
                job = pending.pop(key)
                if job.get('status') == 'succeeded': end[key] = 0; continue
                resource = job.get('resource', 'cpu')
                if resource not in slots or not slots[resource]:
                    unknown.append(key); end[key] = float('inf'); continue
                signature = job.get('signature', resource)
                rate = self.rate(signature)
                duration = job.get('estimated_seconds')
                if duration is None and rate: duration = max(0, job['total_units']-job.get('completed_units',0))/rate
                if duration is None:
                    unknown.append(key); end[key] = float('inf'); continue
                index = min(range(len(slots[resource])), key=slots[resource].__getitem__)
                start = max(slots[resource][index], max([end[d] for d in job.get('dependencies', [])] or [0]))
                stop = start + duration; slots[resource][index] = stop; end[key] = stop
                schedule[key] = {'start_seconds_from_now': start, 'end_seconds_from_now': stop, 'resource': resource}
        return {'remaining_seconds': None if unknown else max(end.values(), default=0),
                'unknown_jobs': unknown, 'schedule': schedule, 'rule': 'deterministic resource-constrained list schedule'}

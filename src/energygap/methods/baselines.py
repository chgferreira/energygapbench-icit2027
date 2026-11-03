"""Train-only reference methods; no method sees withheld values or future caches."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from .base import UnsupportedTask, check_series, training_only


class MedianImputer:
    name = "median"

    def fit(self, values, times, *, split="train"):
        x, _ = training_only(values, times, split)
        self.median_ = float(np.nanmedian(x))
        return self

    def impute(self, values, times, *, task="R"):
        x, _ = check_series(values, times)
        x[~np.isfinite(x)] = self.median_
        return x


class LOCFImputer(MedianImputer):
    name = "locf"

    def impute(self, values, times, *, task="R"):
        x, _ = check_series(values, times)
        last = self.median_
        for j in range(len(x)):
            if np.isfinite(x[j]):
                last = x[j]
            else:
                x[j] = last
        return x


class LinearImputer(MedianImputer):
    name = "linear"

    def impute(self, values, times, *, task="R"):
        if task == "C":
            raise UnsupportedTask("Linear interpolation requires two admissible borders")
        x, t = check_series(values, times)
        good = np.isfinite(x)
        if good.sum() >= 2:
            inside = (~good) & (t > t[good][0]) & (t < t[good][-1])
            x[inside] = np.interp(t[inside], t[good], x[good])
        # No extrapolation. Operational history filling is a separate common policy.
        return x


class SeasonalImputer(MedianImputer):
    """Median weekday/5-min slot -> slot -> weekday -> global, each group >=20."""
    name = "seasonal"

    def __init__(self, min_count=20, timezone="America/Sao_Paulo"):
        if min_count < 1:
            raise ValueError("min_count must be positive")
        self.min_count = min_count
        self.timezone = ZoneInfo(timezone)

    def _key(self, time):
        d = datetime.fromtimestamp(float(time), self.timezone)
        return d.weekday(), (d.hour * 60 + d.minute) // 5

    def fit(self, values, times, *, split="train"):
        x, t = training_only(values, times, split)
        super().fit(x, t, split=split)
        groups = [defaultdict(list) for _ in range(3)]
        for value, time in zip(x, t):
            if np.isfinite(value):
                dow, slot = self._key(time)
                for group, key in zip(groups, [(dow, slot), slot, dow]):
                    group[key].append(float(value))
        self.counts_ = [{k: len(v) for k, v in g.items()} for g in groups]
        self.profiles_ = [{k: float(np.median(v)) for k, v in g.items()
                           if len(v) >= self.min_count} for g in groups]
        return self

    def impute(self, values, times, *, task="R"):
        x, t = check_series(values, times)
        for j in np.flatnonzero(~np.isfinite(x)):
            dow, slot = self._key(t[j])
            x[j] = next((profile[key] for profile, key in
                         zip(self.profiles_, [(dow, slot), slot, dow])
                         if key in profile), self.median_)
        return x


class DailyKalmanImputer(MedianImputer):
    """Harmonic daily regression plus a local-level residual state-space model.

    The daily sine/cosine coefficients, measurement variance and random-walk
    intensity are fitted only on train. Q is scaled by actual elapsed seconds.
    `smooth=False` is a forward filter; `smooth=True` is an RTS smoother within
    the already admissible query. This is a specified reference, not an upstream
    implementation of a named neural architecture.
    """

    def __init__(self, smooth=False, harmonics=3):
        self.smooth = smooth
        self.harmonics = harmonics
        self.name = "kalman_smoother" if smooth else "kalman_filter"

    def _design(self, times):
        phase = np.asarray(times) * (2 * np.pi / 86400)
        return np.column_stack([np.ones(len(phase))] +
                               [f(k * phase) for k in range(1, self.harmonics + 1)
                                for f in (np.sin, np.cos)])

    def fit(self, values, times, *, split="train"):
        x, t = training_only(values, times, split)
        super().fit(x, t, split=split)
        good = np.isfinite(x)
        self.coef_ = np.linalg.lstsq(self._design(t[good]), x[good], rcond=None)[0]
        residual = x[good] - self._design(t[good]) @ self.coef_
        variance = float(np.var(residual))
        # Numerical regularization only; not a metrological uncertainty claim.
        floor = max(1e-10, variance * 1e-8)
        delta = np.diff(residual)
        median_dt = float(np.median(np.diff(t[good]))) if good.sum() > 1 else 300.
        self.measurement_variance_ = max(floor, min(variance, float(np.var(delta)) / 2
                                                   if len(delta) else variance))
        self.process_per_second_ = max(floor / median_dt,
                                       (variance - self.measurement_variance_) /
                                       (288 * median_dt))
        self.initial_variance_ = max(variance, self.measurement_variance_)
        return self

    def impute(self, values, times, *, task="R"):
        if self.smooth and task == "C":
            raise UnsupportedTask("RTS smoother is not admitted for ongoing-gap task C")
        x, t = check_series(values, times)
        if not len(x):
            return x
        seasonal = self._design(t) @ self.coef_
        m = np.zeros(len(x)); v = np.zeros(len(x)); pred_v = np.zeros(len(x))
        previous_m, previous_v = 0., self.initial_variance_
        for j in range(len(x)):
            dt = t[j] - t[j - 1] if j else 0.
            predicted_v = previous_v + dt * self.process_per_second_
            pred_v[j] = predicted_v
            if np.isfinite(x[j]):
                gain = predicted_v / (predicted_v + self.measurement_variance_)
                previous_m += gain * ((x[j] - seasonal[j]) - previous_m)
                previous_v = (1 - gain) * predicted_v
            else:
                previous_v = predicted_v
            m[j], v[j] = previous_m, previous_v
        if self.smooth:
            for j in range(len(x) - 2, -1, -1):
                gain = v[j] / pred_v[j + 1] if pred_v[j + 1] else 0.
                m[j] += gain * (m[j + 1] - m[j])
        missing = ~np.isfinite(x)
        x[missing] = (seasonal + m)[missing]
        return x


def create_baseline(name: str):
    constructors = {"median": MedianImputer, "locf": LOCFImputer,
                    "linear": LinearImputer, "seasonal": SeasonalImputer,
                    "kalman_filter": DailyKalmanImputer,
                    "kalman_smoother": lambda: DailyKalmanImputer(smooth=True)}
    if name not in constructors:
        raise KeyError(f"Unknown baseline {name!r}; no substitute is allowed")
    return constructors[name]()

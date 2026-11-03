"""Aligned shape and regime diagnostics, using physical elapsed time."""
from __future__ import annotations

import numpy as np


def event_metrics(truth, prediction, times, mask, *, sign_tolerance=0., extreme_threshold=None):
    y, p, t, m = np.asarray(truth, float), np.asarray(prediction, float), np.asarray(times, float), np.asarray(mask, bool)
    if not (y.shape == p.shape == t.shape == m.shape):
        raise ValueError("Event arrays must have matching shapes")
    if not m.any():
        return {"reason": "no_evaluation_support", "n": 0}
    if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
        raise ValueError("Finite originally observed evaluation values required")
    selected = np.flatnonzero(m)
    def sign(z):
        return np.where(z > sign_tolerance, 1, np.where(z < -sign_tolerance, -1, 0))
    result = {"n": int(m.sum()), "sign_error_rate": float(np.mean(sign(y[m]) != sign(p[m])))}
    for direction, factor in (("import", 1), ("export", -1)):
        original, estimated = y[m] * factor, p[m] * factor
        # Stable earliest-timestamp argmax tie rule.
        a, b = int(np.argmax(original)), int(np.argmax(estimated))
        result[direction + "_peak_error_kw"] = float(max(0., estimated[b]) - max(0., original[a]))
        result[direction + "_peak_time_error_seconds"] = float(t[selected[b]] - t[selected[a]]) if original[a] > sign_tolerance else None
    pairs = m[:-1] & m[1:] & np.isfinite(y[:-1]) & np.isfinite(y[1:])
    dt = np.diff(t)
    pairs &= dt > 0
    result["ramp_mae_kw_per_minute"] = float(np.mean(np.abs((np.diff(p)[pairs] - np.diff(y)[pairs]) / (dt[pairs]/60)))) if pairs.any() else None
    if extreme_threshold is not None:
        extreme = m & (np.abs(y) >= extreme_threshold)
        result["extreme_n"] = int(extreme.sum())
        result["extreme_mae_kw"] = float(np.mean(np.abs(y[extreme]-p[extreme]))) if extreme.any() else None
    return result

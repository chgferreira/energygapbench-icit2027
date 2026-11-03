"""Empirical marginal scores; joint trajectory capability is declared separately."""
from __future__ import annotations

import numpy as np


QUANTILES = (.05, .1, .25, .5, .75, .9, .95)


def sample_crps(samples, truth, mask=None):
    """Return mean empirical CRPS using an O(S log S), O(S*N) equivalent formula."""
    s, y = np.asarray(samples, float), np.asarray(truth, float)
    if s.ndim != 2 or s.shape[1:] != y.shape:
        raise ValueError("Expected samples shape (S,T) and truth shape (T,)")
    m = np.isfinite(y) if mask is None else np.asarray(mask, bool)
    if m.shape != y.shape:
        raise ValueError("Scoring mask shape mismatch")
    if not m.any():
        return None
    if s.shape[0] < 1 or not np.isfinite(s[:, m]).all() or not np.isfinite(y[m]).all():
        raise ValueError("Valid sample distribution and observed truth required")
    selected = s[:, m]
    count = len(s)
    order = np.sort(selected, axis=0)
    coefficients = (2 * np.arange(1, count + 1) - count - 1)[:, None]
    pair_term = np.sum(coefficients * order, axis=0) / count ** 2
    return float(np.mean(np.mean(np.abs(selected - y[m]), axis=0) - pair_term))


def pinball(quantile_values, truth, quantile, mask=None):
    if not 0 < quantile < 1:
        raise ValueError("Quantile level must lie strictly inside (0,1)")
    q, y = np.asarray(quantile_values, float), np.asarray(truth, float)
    if q.shape != y.shape:
        raise ValueError("Quantile and truth shapes differ")
    m = np.isfinite(y) if mask is None else np.asarray(mask, bool)
    if not m.any():
        return None
    if not np.isfinite(y[m]).all() or not np.isfinite(q[m]).all():
        raise ValueError("Scored values must be finite")
    error = y[m] - q[m]
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_metrics(lower, upper, truth, *, coverage=.9, scale_train=None, mask=None):
    lo, hi, y = np.asarray(lower, float), np.asarray(upper, float), np.asarray(truth, float)
    if lo.shape != hi.shape or lo.shape != y.shape or not 0 < coverage < 1:
        raise ValueError("Invalid interval shapes or coverage")
    m = np.isfinite(y) if mask is None else np.asarray(mask, bool)
    if not m.any():
        return {"coverage": None, "width": None, "normalized_width": None,
                "interval_score": None, "n": 0, "reason": "no_evaluation_support"}
    lo, hi, y = lo[m], hi[m], y[m]
    if not np.isfinite([lo, hi, y]).all() or (lo > hi).any():
        raise ValueError("Intervals must be finite and ordered")
    width = hi - lo; alpha = 1 - coverage
    score = width + 2 / alpha * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    valid_scale = scale_train is not None and scale_train > 0
    return {"coverage": float(np.mean((lo <= y) & (y <= hi))), "width": float(np.mean(width)),
            "normalized_width": float(np.mean(width)) / scale_train if valid_scale else None,
            "interval_score": float(np.mean(score)), "n": len(y), "reason": None}


def summarize_samples(samples, truth=None, mask=None, *, scale_train=None):
    s = np.asarray(samples, float)
    if s.ndim != 2 or len(s) < 1:
        raise ValueError("Expected valid (sample,time) distribution")
    output = {"point": np.median(s, axis=0), "point_estimator": "sample_median",
              "sample_count": len(s),
              "quantiles": {str(q): np.quantile(s, q, axis=0) for q in QUANTILES}}
    if truth is not None:
        output["crps"] = sample_crps(s, truth, mask)
        output["pinball"] = {str(q): pinball(output["quantiles"][str(q)], truth, q, mask) for q in QUANTILES}
        output["intervals"] = {str(c): interval_metrics(np.quantile(s, (1-c)/2, axis=0),
                                                      np.quantile(s, (1+c)/2, axis=0),
                                                      truth, coverage=c, scale_train=scale_train, mask=mask)
                               for c in (.5, .8, .9, .95)}
    return output


def brier_exceedance(samples, truth, threshold, *, mask=None, min_events=5):
    s, y = np.asarray(samples, float), np.asarray(truth, float)
    m = np.isfinite(y) if mask is None else np.asarray(mask, bool)
    events = y[m] > threshold
    if not len(events) or events.sum() < min_events:
        return None, "insufficient_exceedance_events"
    probability = np.mean(s[:, m] > threshold, axis=0)
    return float(np.mean((probability - events) ** 2)), None

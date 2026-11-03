"""Physical-time signed energy operators (kW, seconds -> kWh)."""
from __future__ import annotations

import numpy as np
from energygap.methods.base import check_series


def integrate_signed(values, times, *, semantics="instantaneous") -> dict[str, float]:
    """Integrate both directions separately, including zero crossings exactly.

    Instantaneous samples define piecewise-linear segments. For `interval_mean`
    values[j] describes (times[j-1], times[j]] and gross import/export assumes no
    within-interval reversal; that assumption is external and must be validated.
    """
    x, t = check_series(values, times)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Energy needs at least two finite boundary-inclusive samples")
    dt = np.diff(t) / 3600.
    if semantics in ("interval_mean", "interval_mean_end", "interval_mean_start"):
        means=x[:-1] if semantics=="interval_mean_start" else x[1:]
        imp = float(np.sum(np.maximum(means, 0) * dt))
        exp = float(np.sum(np.maximum(-means, 0) * dt))
    elif semantics == "instantaneous":
        a, b = x[:-1], x[1:]
        imp_parts = (np.maximum(a, 0) + np.maximum(b, 0)) * dt / 2
        exp_parts = (np.maximum(-a, 0) + np.maximum(-b, 0)) * dt / 2
        cross = a * b < 0
        # When crossing, the positive and negative triangle widths differ.
        denominator = np.abs(a[cross]) + np.abs(b[cross])
        imp_parts[cross] = (np.maximum(a[cross], 0) ** 2 +
                            np.maximum(b[cross], 0) ** 2) * dt[cross] / (2 * denominator)
        exp_parts[cross] = (np.maximum(-a[cross], 0) ** 2 +
                            np.maximum(-b[cross], 0) ** 2) * dt[cross] / (2 * denominator)
        imp, exp = float(imp_parts.sum()), float(exp_parts.sum())
    else:
        raise ValueError("Unknown power timestamp semantics")
    return {"import_kwh": imp, "export_kwh": exp, "net_kwh": imp - exp}


def counter_increment(left, right, *, reset=False, rollover=None, rollover_confirmed=False):
    """Return increment + reason; negative differences never become abs(delta)."""
    if reset:
        return None, "reset"
    if left is None or right is None or not np.isfinite([left, right]).all():
        return None, "missing_endpoint"
    delta = float(right - left)
    if delta < 0:
        if rollover_confirmed and rollover is not None and 0 <= left < rollover and 0 <= right < rollover:
            return float(rollover - left + right), "known_rollover"
        return None, "negative_increment_reset_or_unknown_rollover"
    return delta, "valid"


def signed_energy_gradient(values, times, *, semantics="instantaneous"):
    """Exact piecewise analytic derivative of the boundary-inclusive operator.

    At an all-zero segment the symmetric subgradient is used. This is a solver
    convention at a nonsmooth point, not a change to the physical objective.
    """
    x, t = check_series(values, times)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Energy gradient needs finite boundary-inclusive samples")
    dt = np.diff(t)/3600
    if semantics in ("interval_mean", "interval_mean_end", "interval_mean_start"):
        positive = np.zeros(len(x)); net = np.zeros(len(x))
        positions=slice(None,-1) if semantics=="interval_mean_start" else slice(1,None)
        means=x[positions]
        positive[positions] = dt*np.where(means > 0, 1., np.where(means < 0, 0., .5))
        net[positions] = dt
    elif semantics == "instantaneous":
        a, b = x[:-1], x[1:]
        da = np.zeros(len(a)); db = np.zeros(len(a))
        above = (a >= 0) & (b >= 0)
        da[above] = dt[above]/2; db[above] = dt[above]/2
        zeros = (a == 0) & (b == 0)
        da[zeros] = dt[zeros]/4; db[zeros] = dt[zeros]/4
        falling = (a > 0) & (b < 0)
        aa, bb, dd = a[falling], b[falling], dt[falling]
        da[falling] = dd*aa*(aa-2*bb)/(2*(aa-bb)**2)
        db[falling] = dd*aa**2/(2*(aa-bb)**2)
        rising = (a < 0) & (b > 0)
        aa, bb, dd = a[rising], b[rising], dt[rising]
        da[rising] = dd*bb**2/(2*(bb-aa)**2)
        db[rising] = dd*bb*(bb-2*aa)/(2*(bb-aa)**2)
        positive = np.zeros(len(x)); net = np.zeros(len(x))
        positive[:-1] += da; positive[1:] += db
        net[:-1] += dt/2; net[1:] += dt/2
    else:
        raise ValueError("Unknown power semantics")
    return {"import_kwh": positive, "export_kwh": positive-net, "net_kwh": net}


def energy_episode_metrics(truth, prediction, times, *, counter_import=None,
                           counter_export=None, semantics="instantaneous"):
    """Keep truth-integral error separate from input-counter consistency."""
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        return {"energy_metrics_reason": "incomplete_original_power_integral_support"}
    reference = integrate_signed(y, times, semantics=semantics)
    estimated = integrate_signed(p, times, semantics=semantics)
    result = {"energy_metrics_reason": None,
              "energy_reference_kind": "integral_of_original_power_not_independent_measurement"}
    for direction, counter in (("import", counter_import), ("export", counter_export)):
        result[direction+"_power_integral_error_kwh"] = estimated[direction+"_kwh"]-reference[direction+"_kwh"]
        result[direction+"_input_counter_residual_kwh"] = estimated[direction+"_kwh"]-counter if counter is not None else None
    return result

"""Train-only energy uncertainty calibration and validation-only correction tuning."""
from __future__ import annotations

import itertools
from time import perf_counter
import numpy as np

from .correction import EnergyConstraint, calibrate_tolerance, correct_energy
from energygap.evaluation.energy import integrate_signed, counter_increment
from energygap.evaluation.reconstruction import reconstruction_metrics


CORRECTION_GRID = tuple(dict(lambda_energy=l, gamma=g, tau_multiplier=m)
                        for l, g, m in itertools.product((0., .1, 1., 10.), (0., .01), (1., 2.)))


def fit_energy_tolerances(series, train_indices, *, durations=(15, 60, 180), split="train",cutoff_time=None):
    """Nonoverlapping, fully observed training intervals for each duration.

    Duration names denote hidden positions; boundary distance is one interval
    longer. Interior counter resets invalidate an interval. The callee receives
    explicit train indices and cannot access other records during integration.
    """
    if split != "train":
        raise ValueError("Energy uncertainty calibration is train-only")
    rows = [series.records[i] for i in train_indices]
    cutoff=(cutoff_time.timestamp() if hasattr(cutoff_time,'timestamp') else float(cutoff_time)) if cutoff_time is not None else (rows[-1].grid_time.timestamp()+series.nominal_seconds if rows else None)
    tolerance = {}; counts = {}
    semantics = series.metadata.get("power_semantics", "unconfirmed")
    if semantics not in ("instantaneous", "interval_mean", "interval_mean_start", "interval_mean_end"):
        return {"status": "blocked", "reason": "unconfirmed_power_semantics", "directions": {}}
    for direction in ("import", "export"):
        residuals = {duration: [] for duration in durations}
        for duration in durations:
            hidden_count = int(round(duration*60/series.nominal_seconds))
            width = hidden_count+1
            for left in range(0, max(0, len(rows)-width), width):
                interval = rows[left:left+width+1]
                values = np.array([r.value_canonical for r in interval])
                times = np.array([r.grid_time.timestamp() for r in interval])
                if (not all(r.observed_original and not r.natural_missing for r in interval)
                        or not all(r.available_to_method and r.availability_time.timestamp()<cutoff and
                                   (r.counter_availability_time or r.availability_time).timestamp()<cutoff and
                                   r.measurement_time.timestamp()<cutoff for r in interval)
                        or not np.isfinite(values).all()
                        or (np.diff(times) != series.nominal_seconds).any()):
                    continue
                counters = [getattr(r, direction+"_counter") for r in interval]
                increments = [counter_increment(a, b)[0] for a, b in zip(counters[:-1], counters[1:])]
                if any(v is None for v in increments):
                    continue
                delta = float(counters[-1]-counters[0])
                energy = integrate_signed(values, times, semantics=semantics)[direction+"_kwh"]
                residuals[duration].append(abs(energy-delta))
        calibrated = calibrate_tolerance(residuals, resolution_kwh=series.metadata.get("counter_resolution"), split="train")
        tolerance[direction] = calibrated
        counts[direction] = {str(duration): len(v) for duration, v in residuals.items()}
    return {"status": "calibrated", "split": "train", "directions": tolerance,
            "counts": counts, "operator": semantics,
            "counter_agreement_is_independent_calibration": False,
            "interval_sampling": "nonoverlapping_within_duration"}


def constraint_from_query(query, calibration, duration, *, tau_multiplier=1., metadata=None):
    metadata = metadata or {}
    if not query.energy_intervals:
        return EnergyConstraint(reliable=False, reason="no_admissible_energy_intervals")
    interval = query.energy_intervals[0]
    tolerances = calibration.get("directions", {})
    duration_key = format(float(duration), "g")
    tau = {d: tolerances.get(d, {}).get(duration_key, {}).get("tau_kwh") for d in ("import", "export")}
    directions = [d for d in ("import", "export") if interval.get(d+"_kwh") is not None and tau[d] is not None]
    if not directions:
        return EnergyConstraint(reliable=False, reason="no_train_calibrated_admissible_counter_direction")
    return EnergyConstraint(**{d+"_kwh": interval[d+"_kwh"] if d in directions else None for d in ("import", "export")},
                            **{"tau_"+d: tau[d]*tau_multiplier if d in directions else None for d in ("import", "export")},
                            semantics=metadata.get("power_semantics", "instantaneous"),
                            no_within_interval_reversal=metadata.get("no_within_interval_reversal", False))


def select_correction_parameters(models, cases, calibration, scale_train, *, metadata=None,
                                 grid=CORRECTION_GRID, partition="validation"):
    """cases contain (query, separate truth, duration); fixed equal duration weights.

    Truth is consumed only here to select the correction hyperparameters on
    validation. It is never an argument to the imputer or physical solver.
    """
    if partition != "validation":
        raise ValueError("Correction parameters are selected only on validation")
    expected = {15, 60, 180}
    if {duration for _, _, duration in cases} != expected:
        return {name: {"selected": dict(lambda_energy=0., gamma=0., tau_multiplier=1.),
                       "status": "not_estimable", "reason": "missing_validation_duration_support", "trials": []}
                for name in models}
    cache = {}
    for name, model in models.items():
        for query, truth, duration in cases:
            try:
                values = model.impute(query.values, query.times, task="R")
                if not np.isfinite(values[np.asarray(truth.evaluation_mask)]).all():
                    raise ValueError("nonfinite_validation_target")
                status = "success"
            except Exception as error:
                values = models["seasonal"].impute(query.values, query.times, task="R")
                status = "operational_fallback:"+type(error).__name__
            cache[(name, query.query_id)] = (values, status)
    selections = {}
    for name in models:
        trials = []
        for parameters in grid:
            losses = {duration: [] for duration in expected}
            tic = perf_counter()
            failed = 0
            for query, truth, duration in cases:
                base, _ = cache[(name, query.query_id)]
                constraint = constraint_from_query(query, calibration, duration,
                    tau_multiplier=parameters["tau_multiplier"], metadata=metadata)
                result = correct_energy(base, query.times, query.target_gap_mask, constraint,
                    scale_train=scale_train, lambda_energy=parameters["lambda_energy"], gamma=parameters["gamma"],
                    seasonal_start=cache[("seasonal", query.query_id)][0])
                if result.status in ("failed", "unavailable", "unsupported"):
                    failed += 1
                score = reconstruction_metrics(truth.values, result.values, truth.evaluation_mask, scale_train=scale_train)["nmae"]
                if score is not None:
                    losses[duration].append(score)
            loss = float(np.mean([np.mean(v) for v in losses.values()])) if all(losses.values()) else None
            trials.append({**parameters, "validation_gap_macro_nmae": loss,
                           "elapsed_seconds": perf_counter()-tic, "correction_failures": failed})
        valid = [trial for trial in trials if trial["validation_gap_macro_nmae"] is not None]
        if valid:
            best = min(valid, key=lambda trial: (trial["validation_gap_macro_nmae"], trial["lambda_energy"],
                       trial["gamma"], trial["elapsed_seconds"], str(trial["tau_multiplier"])))
            selected = {key: best[key] for key in ("lambda_energy", "gamma", "tau_multiplier")}
            selections[name] = {"selected": selected, "status": "selected_validation_only", "trials": trials,
                                "selection_endpoint": "equal_duration_gap_macro_nmae"}
        else:
            selections[name] = {"selected": dict(lambda_energy=0., gamma=0., tau_multiplier=1.),
                                "status": "not_estimable", "reason": "normalization_or_validation_unsupported", "trials": trials}
    return selections

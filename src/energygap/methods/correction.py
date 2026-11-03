"""Uncertainty-tolerant correction of one signed power trajectory and target gap."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from time import perf_counter
from typing import Any
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.optimize import minimize

from .base import check_series
from energygap.evaluation.energy import integrate_signed, signed_energy_gradient


@dataclass(frozen=True)
class EnergyConstraint:
    import_kwh: float | None = None
    export_kwh: float | None = None
    tau_import: float | None = None
    tau_export: float | None = None
    net_kwh: float | None = None
    tau_net: float | None = None
    reliable: bool = True
    reason: str = "admissible_counter_endpoints"
    semantics: str = "instantaneous"
    no_within_interval_reversal: bool = False


@dataclass
class CorrectionResult:
    values: np.ndarray
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=128)
def _interval_mean_quadratic_factor(target_count, gamma):
    """Factor Q0 = I/n + gamma*L/(n+1), with zero boundary displacements.

    L has diagonal 2 and adjacent entries -1. The data term makes Q0 positive
    definite even when gamma=0. Scale and physical interval lengths remain
    outside this reusable factor; cache eviction changes no numerical policy.
    """
    smooth = gamma / (target_count + 1)
    band = np.zeros((2, target_count))
    band[0] = 1. / target_count + 2 * smooth
    band[1, :-1] = -smooth
    factor = cholesky_banded(band, lower=True, check_finite=True)
    factor.setflags(write=False)
    return factor


def _exact_net_interval_mean(segment, local_t, *, semantics, scale_train,
                             lambda_energy, gamma, energy, tau):
    """Unique minimizer of the existing single-net interval-mean objective.

    For raw-power displacement d, J=d'Qd+lambda*(max(0,|r+a'd|/tau-1))^2,
    where Q=Q0/scale^2 and r=E(base)-counter. Inside the deadzone d=0. Outside,
    d=-lambda*(r-sign(r)*tau)/(tau^2+lambda*a'Q^-1*a)*Q^-1*a.
    Its residual stays on the original side of the deadzone, so this stationary
    point minimizes the globally strictly convex objective, not just one branch.
    Return None for unrepresentable arithmetic, retaining the general solver.
    """
    if not math.isfinite(lambda_energy) or not math.isfinite(gamma):
        return None
    initial_energy = integrate_signed(segment, local_t, semantics=semantics)
    residual = initial_energy['net_kwh'] - energy
    if not math.isfinite(residual):
        return None
    if abs(residual) <= tau:
        return segment[1:-1].copy(), initial_energy
    gradient = signed_energy_gradient(segment, local_t, semantics=semantics)['net_kwh'][1:-1]
    try:
        factor = _interval_mean_quadratic_factor(len(gradient), float(gamma))
        direction = cho_solve_banded((factor, True), gradient, check_finite=True)
        leverage = float(gradient @ direction)
        if not math.isfinite(leverage) or leverage <= 0 or not np.isfinite(direction).all():
            return None
        # lambda*scale^2*leverage/(tau^2+lambda*scale^2*leverage), evaluated
        # without squaring very large/small positive measurement scales.
        log_strength = (math.log(lambda_energy) + 2 * math.log(scale_train)
                        + math.log(leverage) - 2 * math.log(tau))
        if log_strength >= 0:
            fraction = 1. / (1. + math.exp(-log_strength))
        else:
            strength = math.exp(log_strength)
            fraction = strength / (1. + strength)
        excess = residual - math.copysign(tau, residual)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            corrected = segment[1:-1] - (excess * fraction) * (direction / leverage)
        if not np.isfinite(corrected).all():
            return None
        return corrected, initial_energy
    except (ValueError, np.linalg.LinAlgError, OverflowError):
        return None


def calibrate_tolerance(residuals_by_duration, *, resolution_kwh: float | None,
                        split="train", min_count=30, quantile=.95):
    """Fit train-only tau per duration; fallback pooled -> known resolution.

    Caller supplies absolute *training* integral-minus-counter residuals for one
    direction. Low-count resolution-only fallback is explicit and uncalibrated.
    Unknown resolution with insufficient support leaves that duration blocked.
    """
    if split != "train":
        raise ValueError("Tolerance calibration may only use training data")
    if resolution_kwh is not None and (not np.isfinite(resolution_kwh) or resolution_kwh <= 0):
        raise ValueError("Known counter resolution must be positive")
    groups = {str(k): np.abs(np.asarray(v, dtype=float)) for k, v in residuals_by_duration.items()}
    groups = {k: v[np.isfinite(v)] for k, v in groups.items()}
    pooled = np.concatenate(list(groups.values())) if groups else np.array([])
    out = {}
    for duration, residual in groups.items():
        if len(residual) >= min_count:
            sample, source = residual, "duration_train_q95"
        elif len(pooled) >= min_count:
            sample, source = pooled, "pooled_train_q95_fallback"
        else:
            sample, source = np.array([]), "resolution_only_uncalibrated_fallback"
        empirical = float(np.quantile(sample, quantile)) if len(sample) else None
        candidates = [v for v in [empirical, resolution_kwh] if v is not None and v > 0]
        tau = max(candidates) if candidates else None
        out[duration] = {"tau_kwh": tau, "n_duration": len(residual),
                         "n_used": len(sample), "source": source if tau else "insufficient_support_unknown_resolution",
                         "quantile": quantile, "split": "train"}
    return out


def correct_energy(base, times, target_mask, constraint: EnergyConstraint, *,
                   scale_train: float, lambda_energy=1., gamma=.01,
                   seasonal_start=None, max_iterations=300) -> CorrectionResult:
    """Minimize the exact objective, with a closed form for one convex net mean.

    Only target-mask interior values are optimization variables. No truth,
    quality targets or hidden counter increments are accepted. Fixed boundary
    samples are integrated. Parameters must have been selected on validation.
    """
    x, t = check_series(base, times)
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != x.shape:
        raise ValueError("Target gap mask shape mismatch")
    started = perf_counter()

    def unchanged(status, reason, **extra):
        return CorrectionResult(x.copy(), status, {"reason": reason,
                                "elapsed_seconds": perf_counter() - started, **extra})

    if not constraint.reliable:
        return unchanged("unavailable", constraint.reason)
    if lambda_energy < 0 or gamma < 0:
        raise ValueError("Regularization coefficients must be nonnegative")
    if lambda_energy == 0:
        return unchanged("disabled", "lambda_zero_identity")
    if scale_train is None or not np.isfinite(scale_train) or scale_train <= 0:
        return unchanged("unavailable", "nonpositive_training_scale")
    positions = np.flatnonzero(mask)
    if not len(positions):
        return unchanged("unavailable", "empty_target_gap")
    if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("Core correction accepts one contiguous target episode only")
    left, right = positions[0] - 1, positions[-1] + 1
    if left < 0 or right >= len(x):
        return unchanged("unavailable", "two_admissible_boundaries_required")
    segment = x[left:right + 1]
    if not np.isfinite(segment).all():
        return unchanged("unavailable", "nonfinite_base_or_boundary")
    if constraint.semantics.startswith("interval_mean") and not constraint.no_within_interval_reversal and (
            constraint.import_kwh is not None or constraint.export_kwh is not None):
        return unchanged("unsupported", "interval_net_mean_does_not_identify_gross_flows")
    pairs = [("import", constraint.import_kwh, constraint.tau_import),
             ("export", constraint.export_kwh, constraint.tau_export),
             ("net", constraint.net_kwh, constraint.tau_net)]
    active = []
    for direction, energy, tau in pairs:
        if energy is not None:
            if not np.isfinite(energy) or (direction != "net" and energy < 0):
                return unchanged("unavailable", "invalid_or_negative_counter_increment")
            if tau is None or not np.isfinite(tau) or tau <= 0:
                return unchanged("unavailable", "missing_positive_metrological_tolerance")
            active.append((direction, float(energy), float(tau)))
    if not active:
        return unchanged("unavailable", "no_reliable_counter_constraints")
    local_t = t[left:right + 1]
    seasonal = None
    if seasonal_start is not None:
        seasonal = np.asarray(seasonal_start, dtype=float)
        if seasonal.shape != x.shape:
            raise ValueError("Seasonal multistart shape mismatch")

    if (constraint.semantics in ('interval_mean_start', 'interval_mean_end')
            and len(active) == 1 and active[0][0] == 'net'):
        _, energy, tau = active[0]
        exact = _exact_net_interval_mean(segment, local_t, semantics=constraint.semantics,
            scale_train=scale_train, lambda_energy=lambda_energy, gamma=gamma, energy=energy, tau=tau)
        if exact is not None:
            z, initial_energy = exact
            corrected_segment = segment.copy()
            corrected_segment[1:-1] = z
            energies = integrate_signed(corrected_segment, local_t, semantics=constraint.semantics)
            residual = energies['net_kwh'] - energy
            delta = (corrected_segment - segment) / scale_train
            with np.errstate(over='ignore', invalid='ignore'):
                best = float(np.mean(delta[1:-1] ** 2) + gamma * np.mean(np.diff(delta) ** 2)
                    + lambda_energy * max(0., abs(residual) / tau - 1.) ** 2)
                before = float(lambda_energy * max(0., abs(initial_energy['net_kwh'] - energy) / tau - 1.) ** 2)
            if math.isfinite(best) and math.isfinite(before) and best <= before:
                output = x.copy()
                output[positions] = z
                label = 'base_identity' if np.array_equal(z, segment[1:-1]) else 'exact_convex_net_interval_mean'
                return CorrectionResult(output, 'success', {
                    'reason': 'exact_objective_convex_net_interval_mean',
                    'solver_variant': 'exact_convex_net_interval_mean',
                    'objective': best, 'objective_before': before, 'selected_start': label,
                    'energies_kwh': energies, 'residuals_kwh': {'net': residual},
                    'trials': [dict(start='exact_convex_net_interval_mean', converged=True,
                        iterations=0, objective=best, message='closed-form unique global minimizer')],
                    'iterations': 0, 'elapsed_seconds': perf_counter() - started,
                    'global_optimum_guaranteed': True,
                    'boundary_indices': [int(left), int(right)], 'target_count': len(positions)})

    def curve(z):
        y = segment.copy(); y[1:-1] = z
        return y

    def objective(z):
        delta = (curve(z) - segment) / scale_train
        energies = integrate_signed(curve(z), local_t, semantics=constraint.semantics)
        value = float(np.mean(delta[1:-1] ** 2))
        value += lambda_energy * sum(max(0., abs(energies[d + "_kwh"] - e) / tau - 1.) ** 2
                                     for d, e, tau in active)
        value += gamma * float(np.mean(np.diff(delta) ** 2))
        return value

    def objective_with_gradient(z):
        y = curve(z)
        delta = (y-segment)/scale_train
        energies = integrate_signed(y, local_t, semantics=constraint.semantics)
        derivative = signed_energy_gradient(y, local_t, semantics=constraint.semantics)
        value = float(np.mean(delta[1:-1]**2))
        gradient = 2*(z-segment[1:-1])/(scale_train**2*len(z))
        for direction, energy, tau in active:
            residual = energies[direction+"_kwh"]-energy
            excess = max(0., abs(residual)/tau-1.)
            value += lambda_energy*excess**2
            gradient += (2*lambda_energy*excess*np.sign(residual)/tau)*derivative[direction+"_kwh"][1:-1]
        differences = np.diff(delta)
        value += gamma*float(np.mean(differences**2))
        adjacent_gradient = np.zeros(len(y))
        adjacent_gradient[:-1] -= 2*gamma*differences/(len(differences)*scale_train)
        adjacent_gradient[1:] += 2*gamma*differences/(len(differences)*scale_train)
        gradient += adjacent_gradient[1:-1]
        return value, gradient

    starts = [("base", segment[1:-1].copy())]
    if seasonal is not None:
        if np.isfinite(seasonal[positions]).all():
            starts.append(("seasonal", seasonal[positions].copy()))
    # A signed energy-scale initialization; all starts obey fixed observations.
    initial = segment[1:-1].copy()
    initial_energy = integrate_signed(segment, local_t, semantics=constraint.semantics)
    for direction, energy, _ in active:
        if direction == "import":
            if (initial > 0).any() and initial_energy["import_kwh"] > 0:
                initial[initial > 0] *= energy / initial_energy["import_kwh"]
        elif direction == "export":
            if (initial < 0).any() and initial_energy["export_kwh"] > 0:
                initial[initial < 0] *= energy / initial_energy["export_kwh"]
        else:
            initial += (energy - initial_energy["net_kwh"]) / ((local_t[-1] - local_t[0]) / 3600.)
    starts.append(("energy_scaled", initial))
    trials = []
    candidates = []
    for label, start in starts:
        try:
            result = minimize(objective_with_gradient, start, jac=True, method="L-BFGS-B",
                              options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-7})
            valid = bool(result.success and np.isfinite(result.x).all() and np.isfinite(result.fun))
            trials.append({"start": label, "converged": valid, "iterations": int(result.nit),
                           "objective": float(result.fun), "message": str(result.message)})
            if valid:
                candidates.append((float(result.fun), label, result.x))
        except (ValueError, FloatingPointError) as error:
            trials.append({"start": label, "converged": False, "message": str(error)})
    if not candidates:
        return unchanged("failed", "all_multistart_solvers_failed", trials=trials)
    # Base itself is always feasible; numerical optimization cannot worsen J.
    candidates.append((objective(segment[1:-1]), "base_identity", segment[1:-1]))
    best, label, z = min(candidates, key=lambda item: (item[0], item[1]))
    output = x.copy(); output[positions] = z
    energies = integrate_signed(output[left:right + 1], local_t, semantics=constraint.semantics)
    return CorrectionResult(output, "success", {"reason": "exact_objective_multistart",
        "objective": best, "objective_before": objective(segment[1:-1]), "selected_start": label,
        "energies_kwh": energies, "residuals_kwh": {d: energies[d + "_kwh"] - e for d, e, _ in active},
        "trials": trials, "iterations": sum(trial.get("iterations", 0) for trial in trials),
        "elapsed_seconds": perf_counter() - started, "global_optimum_guaranteed": False,
        "boundary_indices": [int(left), int(right)], "target_count": len(positions)})

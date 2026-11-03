"""Capability-safe trajectory correction; marginal quantiles are not trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .correction import EnergyConstraint, correct_energy
from energygap.evaluation.energy import integrate_signed
from energygap.evaluation.probabilistic import summarize_samples


@dataclass(frozen=True)
class ProbabilisticCapabilities:
    joint_samples: bool = False
    marginal_samples: bool = False
    marginal_quantiles: bool = False


def correct_joint_samples(samples, times, target_mask, constraint: EnergyConstraint, *,
                          capabilities: ProbabilisticCapabilities, scale_train,
                          cancel_callback=None,
                          **solver_options):
    if 'truth' in solver_options or 'evaluation_mask' in solver_options:
        raise ValueError('Evaluator truth and supervision masks must remain outside the correction inference API')
    if not capabilities.joint_samples:
        raise ValueError("Energy trajectory correction requires declared joint samples; marginal quantiles are insufficient")
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2 or not len(s):
        raise ValueError("Expected (sample,time) trajectories")
    results = []
    for sample in s:
        if cancel_callback is not None:
            cancel_callback()
        results.append(correct_energy(sample, times, target_mask, constraint,
                                      scale_train=scale_train, **solver_options))
    corrected = np.stack([r.values for r in results])
    # All distribution summaries and scores are rebuilt from transformed samples.
    summary = summarize_samples(corrected, scale_train=scale_train)
    positions = np.flatnonzero(target_mask)
    energies = None
    if len(positions) and positions[0] > 0 and positions[-1]+1 < s.shape[1]:
        sl = slice(positions[0]-1, positions[-1]+2)
        energies = integrate_signed(summary["point"][sl], np.asarray(times)[sl], semantics=constraint.semantics)
    return {"samples": corrected, "summary": summary, "sample_diagnostics": [r.diagnostics for r in results],
            "sample_statuses": [r.status for r in results], "summary_curve_energy_kwh": energies,
            "calibrated_coverage_guaranteed": False}

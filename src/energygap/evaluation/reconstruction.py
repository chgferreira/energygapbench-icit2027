"""Reconstruction scores restricted to explicitly held-out original observations."""
from __future__ import annotations

import numpy as np


def training_scale(values, *, split="train"):
    if split != "train":
        raise ValueError("Normalization scale must be fitted on train")
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None, "no_training_support"
    scale = float(np.quantile(x, .95) - np.quantile(x, .05))
    if scale > 0:
        return scale, "train_q95_minus_q05"
    scale = float(np.std(x))
    return (scale, "train_std_fallback") if scale > 0 else (None, "constant_train_normalization_undefined")


def evaluation_mask(observed_original, artificial_hidden, padding_mask=None):
    observed = np.asarray(observed_original, dtype=bool)
    hidden = np.asarray(artificial_hidden, dtype=bool)
    if observed.shape != hidden.shape:
        raise ValueError("Observation and hiding masks must match")
    mask = observed & hidden
    if padding_mask is not None:
        padding = np.asarray(padding_mask, dtype=bool)
        if padding.shape != mask.shape:
            raise ValueError("Padding mask must match")
        mask &= ~padding
    return mask


def reconstruction_metrics(truth, prediction, mask, *, scale_train=None):
    y, p, m = np.asarray(truth, float), np.asarray(prediction, float), np.asarray(mask, bool)
    if y.shape != p.shape or m.shape != y.shape:
        raise ValueError("Truth, predictions and scoring mask must have identical shapes")
    if (m & ~np.isfinite(y)).any():
        raise ValueError("Natural missing/padding cannot serve as evaluation truth")
    if not m.any():
        return {"n": 0, "mae_kw": None, "rmse_kw": None, "bias_kw": None,
                "nmae": None, "nrmse": None, "reason": "no_evaluation_support"}
    if (m & ~np.isfinite(p)).any():
        return {"n": int(m.sum()), "mae_kw": None, "rmse_kw": None, "bias_kw": None,
                "nmae": None, "nrmse": None, "reason": "prediction_failure_requires_operational_fallback"}
    error = p[m] - y[m]
    mae, rmse = float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))
    valid_scale = scale_train is not None and np.isfinite(scale_train) and scale_train > 0
    return {"n": int(m.sum()), "mae_kw": mae, "rmse_kw": rmse,
            "bias_kw": float(np.mean(error)), "nmae": mae / scale_train if valid_scale else None,
            "nrmse": rmse / scale_train if valid_scale else None,
            "reason": None if valid_scale else "normalization_undefined"}


def skill(loss, reference_loss):
    if loss is None or reference_loss is None:
        return None, "missing_loss"
    if not np.isfinite([loss, reference_loss]).all() or reference_loss <= 0:
        return None, "zero_or_invalid_reference_loss"
    return 1 - float(loss) / float(reference_loss), None

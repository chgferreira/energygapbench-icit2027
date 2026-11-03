"""Separate episode-macro and point-micro loss; never hide absent strata."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def gap_macro(errors_by_gap):
    groups = [np.asarray(g, dtype=float) for g in errors_by_gap]
    if not groups or any(not len(g) or not np.isfinite(g).all() for g in groups):
        return {"gap_macro_mae": None, "point_micro_mae": None,
                "n_gaps": len(groups), "reason": "empty_or_invalid_gap_support"}
    return {"gap_macro_mae": float(np.mean([np.mean(np.abs(g)) for g in groups])),
            "point_micro_mae": float(np.mean(np.abs(np.concatenate(groups)))),
            "n_gaps": len(groups), "reason": None}


def equal_weight_estimand(rows, *, folds=(1, 2, 3), durations=(15, 60, 180), metric="nmae"):
    """Rows contain fold,duration,episode_id,seed,metric; average seeds then gaps."""
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouped[(row["fold"], row["duration"], row["episode_id"])].append(float(value))
    cells = {}
    for fold in folds:
        for duration in durations:
            episode_values = [np.mean(v) for (f, d, _), v in grouped.items() if (f, d) == (fold, duration)]
            cells[(fold, duration)] = float(np.mean(episode_values)) if episode_values else None
    missing = [f"{f}/{d}" for (f, d), value in cells.items() if value is None]
    return {"estimate": None if missing else float(np.mean(list(cells.values()))),
            "cells": {f"{f}/{d}": value for (f, d), value in cells.items()},
            "reason": "missing_required_fold_duration:" + ",".join(missing) if missing else None}

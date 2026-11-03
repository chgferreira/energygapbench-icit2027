"""Bounded downstream forecast evaluation for already admitted targets.

Target eligibility belongs to the evaluator.  This queue only batches prediction
for the history, issuance and target it is given; it neither finds targets nor
fits, fills, rescales or modifies histories.
"""
from __future__ import annotations

import numpy as np


HISTORY_POINTS = 288


class ForecastQueue:
    """Evaluate one bounded group per horizon and fill its existing result rows.

    Ordinary inference shards need one ``predict_many`` call per horizon.
    Larger callers are flushed in groups of at most 1024 rows per horizon, so
    retained history storage stays bounded independently of experiment size.
    Measured pipeline time remains the caller's responsibility.
    """

    def __init__(self, forecasts, *, max_batch_size=1024):
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError('max_batch_size must be a positive integer')
        self.forecasts = {int(horizon): model for horizon, model in forecasts.items()}
        if len(self.forecasts) != len(forecasts):
            raise ValueError('duplicate forecast horizon')
        self.max_batch_size = max_batch_size
        self._pending = {horizon: [] for horizon in self.forecasts}

    def add(self, horizon, history, issuance, target, scale, row):
        """Queue an admitted target; unavailable targets perform no prediction.

        ``target`` can be the evaluator's ``(value, offset_seconds)`` tuple or
        its scalar value.  The offset/exclusion fields already in the row are
        retained.  The history is copied because later correction variants may
        reuse the caller's original array.
        """
        if target is None:
            return
        horizon = int(horizon)
        if horizon not in self.forecasts:
            raise KeyError(f'no forecaster for horizon {horizon}')
        values = np.asarray(history, dtype=float)
        if values.ndim != 1:
            raise ValueError('forecast history must be one-dimensional')
        target_value = target[0] if isinstance(target, tuple) else target
        self._pending[horizon].append((values[-HISTORY_POINTS:].copy(), float(issuance),
                                       float(target_value), scale, row))
        if len(self._pending[horizon]) >= self.max_batch_size:
            self._flush_horizon(horizon)

    def _flush_horizon(self, horizon):
        pending = self._pending[horizon]
        if not pending:
            return
        histories = [item[0] for item in pending]
        issuances = [item[1] for item in pending]
        predictions = np.asarray(self.forecasts[horizon].predict_many(histories, issuances), dtype=float).reshape(-1)
        if len(predictions) != len(pending):
            raise ValueError('forecast batch output count differs from queued targets')
        if not np.isfinite(predictions).all():
            raise ValueError('forecast batch returned a nonfinite prediction')
        for prediction, (_, _, target, scale, row) in zip(predictions, pending):
            error = abs(float(prediction) - target)
            normalized = float(error / scale) if scale else None
            cell = row.setdefault('forecasts', {}).setdefault(str(horizon), {})
            cell.update(mae_kw=error, nmae=normalized)
            if horizon == 3600:
                row.update(forecast_mae_kw=error, forecast_nmae=normalized)
        pending.clear()

    def flush(self):
        for horizon in self.forecasts:
            self._flush_horizon(horizon)

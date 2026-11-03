"""Explicit CPI-inspired adaptation for cumulative energy, not instantaneous power.

Original: KIT-IAI/CopyPasteImputation, c1ec6facaf0a762d7e3c22ad988b26d0361085f9,
LGPL-3.0. This file is independently implemented integration/protocol code. It
does not redistribute upstream. It intentionally does not claim output parity:
bank restricted to train; matching by known *gap* total rather than Prophet's
estimated daily total. Output derivatives are interval means in a separate track.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

from .base import check_series, training_only
from energygap.evaluation.energy import counter_increment


@dataclass
class CPIEnergyResult:
    cumulative_kwh: np.ndarray
    interval_mean_kw: np.ndarray
    status: str
    diagnostics: dict


class CPIEnergyDonorAdapted:
    """Training-day copy/paste of cumulative-energy increments with gap scaling."""
    name = "cpi_energy_donor_adapted"

    def __init__(self, timezone="America/Sao_Paulo", frequency_seconds=300,
                 w_weekday=1., w_season=5., w_energy=10.):
        if frequency_seconds <= 0 or 86400 % frequency_seconds:
            raise ValueError("Frequency must divide a nominal day")
        self.timezone = ZoneInfo(timezone)
        self.frequency_seconds = frequency_seconds
        self.values_per_day = 86400 // frequency_seconds
        self.weights = w_weekday, w_season, w_energy

    def _slot(self, timestamp):
        d = datetime.fromtimestamp(float(timestamp), self.timezone)
        seconds = d.hour*3600 + d.minute*60 + d.second
        if d.microsecond or seconds % self.frequency_seconds:
            raise ValueError("CPI adaptation requires exact regular grid")
        # Interval ending at midnight belongs to the previous day's last slot.
        date = (d - timedelta(seconds=1)).date()
        return date, int((seconds // self.frequency_seconds - 1) % self.values_per_day)

    def fit(self, cumulative_kwh, times, *, split="train"):
        values, t = training_only(cumulative_kwh, times, split)
        days = defaultdict(dict)
        excluded_intervals = 0
        for j in range(1, len(values)):
            date, slot = self._slot(t[j])
            if t[j]-t[j-1] != self.frequency_seconds:
                excluded_intervals += 1
                continue
            delta, reason = counter_increment(values[j-1], values[j])
            if delta is None or slot in days[date]:
                days[date][slot] = np.nan
                excluded_intervals += 1
            else:
                days[date][slot] = delta
        self.bank_ = []
        for date in sorted(days):
            increments = days[date]
            if len(increments) == self.values_per_day and np.isfinite(list(increments.values())).all():
                self.bank_.append((date, np.array([increments[j] for j in range(self.values_per_day)])))
        self.training_end_ = float(t[-1])
        self.fit_diagnostics_ = {"complete_train_days": len(self.bank_), "candidate_days": len(days),
                                 "excluded_intervals": excluded_intervals, "split": "train",
                                 "bank_restricted_to_training": True}
        return self

    def impute_cumulative(self, cumulative_kwh, times, target_gap_mask, *,
                          issuance_time, availability_times=None):
        x, t = check_series(cumulative_kwh, times)
        mask = np.asarray(target_gap_mask, bool)
        if mask.shape != x.shape:
            raise ValueError("Gap mask shape mismatch")
        arrival = t if availability_times is None else np.asarray(availability_times, float)
        if arrival.shape != t.shape:
            raise ValueError("Availability shape mismatch")
        # Defensive cut even if caller supplied a later-valued record.
        x[(arrival > issuance_time) | (t > issuance_time)] = np.nan

        def finish(status, reason, **diagnostics):
            mean = np.full(len(x), np.nan)
            adjacent = np.isfinite(x[:-1]) & np.isfinite(x[1:])
            mean[1:][adjacent] = np.diff(x)[adjacent] / (np.diff(t)[adjacent]/3600)
            return CPIEnergyResult(x.copy(), mean, status, {"reason": reason,
                "output_semantics": "cumulative_energy_and_interval_mean_not_instantaneous",
                "official_output_parity_claimed": False, **diagnostics})

        positions = np.flatnonzero(mask)
        if not len(positions):
            return finish("unavailable", "empty_target_gap")
        if not np.array_equal(positions, np.arange(positions[0], positions[-1]+1)):
            raise ValueError("One contiguous cumulative-energy episode is required")
        left, right = positions[0]-1, positions[-1]+1
        if left < 0 or right >= len(x):
            return finish("unavailable", "missing_counter_boundaries")
        delta, reason = counter_increment(x[left], x[right])
        if delta is None:
            return finish("unavailable", reason)
        if (np.diff(t[left:right+1]) != self.frequency_seconds).any():
            return finish("unsupported", "irregular_grid")
        if self.training_end_ >= t[left]:
            return finish("unsupported", "donor_training_not_strictly_before_query")
        if not self.bank_:
            return finish("unavailable", "no_complete_training_donor_day")
        dates_slots = [self._slot(time) for time in t[left+1:right+1]]
        date = dates_slots[0][0]
        slots = np.array([slot for _, slot in dates_slots])
        candidates = [(d, increments[slots]) for d, increments in self.bank_]
        totals = np.array([pattern.sum() for _, pattern in candidates])
        spread = float(totals.max()-totals.min())
        scores = []
        for (donor_date, pattern), total in zip(candidates, totals):
            weekday = 0. if date.weekday() == donor_date.weekday() else (
                1. if (date.weekday() < 5) != (donor_date.weekday() < 5) else .5)
            year_distance = abs(date.timetuple().tm_yday-donor_date.timetuple().tm_yday)
            season = min(year_distance, 365-year_distance) / 182.
            energy = abs(delta-total)/spread if spread > 0 else 0.
            score = self.weights[0]*weekday + self.weights[1]*season + self.weights[2]*energy
            if total > 0 or delta == 0:
                scores.append((score, donor_date, pattern, float(total)))
        if not scores:
            return finish("unavailable", "all_donor_segments_zero_but_positive_gap_energy")
        score, donor_date, pattern, total = min(scores, key=lambda item: (item[0], item[1]))
        scaled = pattern * (delta/total) if total > 0 else np.zeros_like(pattern)
        restored = x[left] + np.cumsum(scaled)
        # Preserve both boundary measurements exactly; fill internal counters only.
        x[positions] = restored[:-1]
        return finish("success", "train_donor_matched_and_scaled_gap_energy",
                      donor_date=donor_date.isoformat(), matching_score=float(score),
                      delta_energy_kwh=delta, donor_energy_kwh=total,
                      donor_count=len(self.bank_), weights=list(self.weights),
                      cumulative_gap_points=len(positions), interval_count=len(pattern))

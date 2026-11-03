"""Minimal numerical contract: inference receives admissible values, never truth."""
from __future__ import annotations

from typing import Protocol
import numpy as np


class Imputer(Protocol):
    name: str

    def fit(self, values, times, *, split: str = "train"): ...

    def impute(self, values, times, *, task: str = "R") -> np.ndarray: ...


def check_series(values, times) -> tuple[np.ndarray, np.ndarray]:
    x, t = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    if x.ndim != 1 or t.shape != x.shape:
        raise ValueError("Expected matching one-dimensional values/times")
    if len(t) and (not np.isfinite(t).all() or (np.diff(t) <= 0).any()):
        raise ValueError("Times must be finite and strictly increasing physical seconds")
    if np.isinf(x).any():
        raise ValueError("Infinite measurements are invalid; missing values use NaN")
    return x.copy(), t.copy()


def training_only(values, times, split: str):
    if split != "train":
        raise ValueError("Fitting is permitted only on the training partition")
    x, t = check_series(values, times)
    if not np.isfinite(x).any():
        raise ValueError("No observed training support")
    return x, t


class UnsupportedTask(ValueError):
    """The adapter cannot satisfy this task's information budget."""

"""Bounded CPU preparation of already-sanitized BRITS inference inputs.

The native assembly and model forward remain unchanged. Only the deterministic
DatasetForBRITS transform is prepared here: missing values become zero, backward
inputs are reversed, and deltas satisfy the exact upstream recurrence. No truth,
unmasked source arrays, model states or predictions are accepted or retained.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np


def brits_deltas(mask):
    """Vectorize d[0]=0; d[t]=1+(1-mask[t-1])*d[t-1] exactly.

    A delta is the current index minus the last observed index strictly before
    it (or zero when no earlier observation exists). Integer indices avoid any
    floating-point reduction; conversion to native float32 happens at the end.
    The time axis is the penultimate dimension, as in the native dataset.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim not in (2, 3):
        raise ValueError("BRITS masks require [L,D] or [N,L,D]")
    length = mask.shape[-2]
    indices = np.arange(length, dtype=np.int64).reshape((1,) * (mask.ndim - 2) + (length, 1))
    previous = np.maximum.accumulate(np.where(mask, indices, 0), axis=-2)
    shifted = np.zeros_like(previous)
    shifted[..., 1:, :] = previous[..., :-1, :]
    return (indices - shifted).astype(np.float32)


class BritsInputCache:
    """LRU of input transforms with a strict retained-array byte ceiling.

    Keys include shape and canonical normalized values/missingness. Therefore a
    changed scaler, visible value, mask or fold cannot reuse an incompatible row.
    Entries contain four private CPU arrays; backward values/masks are views of
    the same observations. Batched tensors are newly allocated, so a native
    consumer cannot mutate the cache. Cache presence never changes output.
    """
    def __init__(self, max_bytes=128 * 1024 * 1024):
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("Cache byte limit must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._rows = OrderedDict()

    def clear(self):
        self._rows.clear()
        self.bytes = self.hits = self.misses = 0

    def prepare(self, normalized):
        """Return the seven tensors expected by native testing assembly."""
        import torch

        values = np.asarray(normalized, dtype=np.float32)
        if values.ndim != 3 or not len(values) or np.isinf(values).any():
            raise ValueError("Finite/NaN normalized [N,L,D] BRITS inputs required")
        rows = []
        hits_before, misses_before = self.hits, self.misses
        for row in values:
            # Canonicalize NaN payloads while retaining the mask in the key.
            observed = np.isfinite(row)
            filled = np.where(observed, row, np.float32(0)).astype(np.float32)
            key = (row.shape, hashlib.sha256(filled.tobytes() + observed.tobytes()).digest())
            prepared = self._rows.pop(key, None)
            if prepared is None:
                self.misses += 1
                mask = observed.astype(np.float32)
                prepared = (filled, mask, brits_deltas(observed), brits_deltas(observed[::-1]))
                retained = sum(array.nbytes for array in prepared)
                if retained <= self.max_bytes:
                    while self._rows and self.bytes + retained > self.max_bytes:
                        _, evicted = self._rows.popitem(last=False)
                        self.bytes -= sum(array.nbytes for array in evicted)
                    self._rows[key] = prepared
                    self.bytes += retained
            else:
                self.hits += 1
                self._rows[key] = prepared
            rows.append(prepared)
        filled, mask, forward_delta, backward_delta = (
            np.stack([row[column] for row in rows]) for column in range(4))
        tensors = [torch.arange(len(values), dtype=torch.int64),
                   torch.from_numpy(filled), torch.from_numpy(mask), torch.from_numpy(forward_delta),
                   torch.from_numpy(filled[:, ::-1].copy()), torch.from_numpy(mask[:, ::-1].copy()),
                   torch.from_numpy(backward_delta)]
        return tensors, dict(preprocess_policy="sanitized_brits_exact_vectorized_delta_lru_v1",
            input_cache_limit_bytes=self.max_bytes, input_cache_bytes=self.bytes,
            input_cache_hits=self.hits - hits_before, input_cache_misses=self.misses - misses_before)


# Shared across epoch/trial/model reloads: entries depend only on sanitized,
# normalized inputs, never on fitted weights. CPU-only and deliberately bounded.
BRITS_INPUT_CACHE = BritsInputCache()

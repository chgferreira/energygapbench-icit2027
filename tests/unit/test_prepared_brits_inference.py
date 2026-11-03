"""Prepared BRITS inputs match the official transform, with bounded reuse."""
from types import MethodType

import numpy as np
import pytest

from energygap.methods.prepared_inference import BritsInputCache, brits_deltas


@pytest.mark.parametrize("length,features", [(1, 1), (16, 3), (512, 1)])
def test_vectorized_delta_and_all_prepared_tensors_match_upstream(length, features):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypots")
    from pypots.imputation.brits.data import DatasetForBRITS
    from pypots.data.utils import _parse_delta_torch

    rng = np.random.default_rng(381)
    values = rng.normal(size=(7, length, features)).astype(np.float32)
    values[rng.random(values.shape) < .4] = np.nan
    values[0] = np.nan
    values[1] = 0.
    observed = np.isfinite(values)
    np.testing.assert_array_equal(brits_deltas(observed),
        _parse_delta_torch(torch.as_tensor(observed, dtype=torch.float32)).numpy())
    cache = BritsInputCache()
    actual, stats = cache.prepare(values)
    dataset = DatasetForBRITS({"X": values}, return_X_ori=False, return_y=False)
    expected = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(values), shuffle=False)))
    for received, native in zip(actual, expected):
        assert torch.equal(received, native)
    assert stats["input_cache_misses"] >= 2
    assert stats["input_cache_misses"] + stats["input_cache_hits"] == 7
    repeated, stats = cache.prepare(values.copy())
    assert stats["input_cache_hits"] == 7 and stats["input_cache_misses"] == 0
    for received, before in zip(repeated, actual):
        assert torch.equal(received, before)


def test_input_cache_is_bounded_scaler_sensitive_and_consumer_mutation_cannot_poison_it():
    pytest.importorskip("torch")
    values = np.arange(3 * 16, dtype=np.float32).reshape(3, 16, 1)
    values[:, 3:8] = np.nan
    row_bytes = 4 * 16 * np.dtype(np.float32).itemsize
    cache = BritsInputCache(max_bytes=2 * row_bytes)
    first, stats = cache.prepare(values[:2])
    assert stats["input_cache_bytes"] == 2 * row_bytes
    expected = [tensor.clone() for tensor in first]
    first[1].fill_(1234.)
    repeated, stats = cache.prepare(values[:2])
    assert stats["input_cache_hits"] == 2
    for actual, reference in zip(repeated, expected):
        assert actual.equal(reference)
    _, stats = cache.prepare(values[2:])
    assert stats["input_cache_bytes"] <= cache.max_bytes
    _, stats = cache.prepare(values[:1])
    assert stats["input_cache_misses"] == 1  # Oldest row was evicted.
    changed = (values[:1] - 1) / 2  # A changed train scaler has another key.
    transformed, stats = cache.prepare(changed)
    assert stats["input_cache_misses"] == 1
    np.testing.assert_array_equal(transformed[1].numpy(), np.nan_to_num(changed))
    disabled = BritsInputCache(max_bytes=0)
    disabled.prepare(values)
    assert disabled.bytes == 0 and not disabled._rows


@pytest.mark.parametrize("name", ["BRITS", "MOMENT-1-base"])
def test_cuda_oom_halves_deterministic_batch_preserving_order_and_reuses_safe_limit(monkeypatch, name):
    torch = pytest.importorskip("torch")
    from energygap.methods.conference_models import ConferenceModel
    model = ConferenceModel.__new__(ConferenceModel)
    model.name, model.device, model.inference_batch_size = name, "cuda:0", 8
    calls = []

    def infer(self, queries):
        calls.append(list(queries))
        if len(queries) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        return [dict(point_estimate=q, model_metadata={},
                     timing={"batch_size": len(queries)}) for q in queries]

    model._infer_deterministic_batch = MethodType(infer, model)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    outputs = model.infer_queries(list(range(9)))
    assert [r["point_estimate"] for r in outputs] == list(range(9))
    assert [len(batch) for batch in calls] == [8, 4, 2, 2, 2, 2, 1]
    assert all(r["timing"]["requested_batch_size"] == 8 for r in outputs)
    assert all(r["timing"]["oom_retries"] == 2 for r in outputs)
    assert all(r["model_metadata"]["adaptive_batch_size"] == 2 for r in outputs)
    assert all(r["model_metadata"]["configured_inference_batch_size"] == 8 for r in outputs)
    calls.clear()
    model.infer_queries([10, 11])
    assert [len(batch) for batch in calls] == [2]


def test_cuda_oom_at_one_and_nonallocation_errors_are_not_hidden(monkeypatch):
    torch = pytest.importorskip("torch")
    from energygap.methods.conference_models import ConferenceModel
    model = ConferenceModel.__new__(ConferenceModel)
    model.name, model.device, model.inference_batch_size = "SAITS", "cuda:0", 1
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def oom(self, queries):
        raise torch.cuda.OutOfMemoryError("synthetic allocation failure")

    model._infer_deterministic_batch = MethodType(oom, model)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        model.infer_queries([0])

    def invalid(self, queries):
        raise RuntimeError("invalid native output")

    model._infer_deterministic_batch = MethodType(invalid, model)
    with pytest.raises(RuntimeError, match="invalid native output"):
        model.infer_queries([0, 1], batch_size=8)

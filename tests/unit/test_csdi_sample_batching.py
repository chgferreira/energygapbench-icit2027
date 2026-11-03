"""Native CSDI condition-row batching changes RNG order, not diffusion paths."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.methods.conference_models import create_adapter, load_adapter


def _make_model(tmp_path, sample_batch_size, checkpoint=None):
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    params = dict(n_layers=1, n_heads=2, n_channels=8, d_time_embedding=8,
        d_feature_embedding=4, d_diffusion_embedding=8, n_diffusion_steps=3)
    model = create_adapter("CSDI", root=tmp_path, context=16, batch_size=2,
        epochs=1, patience=1, seed=11, threads=1, params={"architecture": params},
        csdi_sample_batch_size=sample_batch_size)
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train, val = np.sin(t / 5) + t / 100, np.cos(t[:3] / 4) + 1
    model.fit_windows(train, val, checkpoint_path=checkpoint)
    return model


def _query():
    values = np.sin(np.arange(16) / 5)
    observed, padding = np.ones(16, bool), np.zeros(16, bool)
    observed[12:15], padding[:2] = False, True
    values[~observed | padding] = 1e20
    return SimpleNamespace(values=values, available_to_method=observed,
        padding_mask=padding, task="R")


class _ControlledNoise:
    """Serve each trajectory the same fixed Gaussian sequence in either order."""
    def __init__(self, bank):
        self.bank, self.draw, self.step = bank, 0, 0
        self.calls = []

    def __call__(self, tensor, *args, **kwargs):
        import torch
        batch = tensor.shape[0]
        assert tensor.shape[1:] == self.bank.shape[2:]
        values = self.bank[self.draw:self.draw + batch, self.step]
        assert len(values) == batch
        self.calls.append(batch)
        self.step += 1
        if self.step == self.bank.shape[1]:
            self.step = 0
            self.draw += batch
        return torch.as_tensor(values.copy(), dtype=tensor.dtype, device=tensor.device)


def test_native_outer_samples_and_condition_rows_match_under_the_same_gaussian_noise(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model, query, count = _make_model(tmp_path, 3), _query(), 7
    backbone = model.adapter.model.model.backbone
    assert not backbone.is_unconditional
    bank = np.random.default_rng(47).normal(size=(count, backbone.n_diffusion_steps, 1, 16)).astype(np.float32)
    safe = np.where(query.available_to_method & ~query.padding_mask, query.values, np.nan)
    old_noise, new_noise = _ControlledNoise(bank), _ControlledNoise(bank)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "randn_like", old_noise)
        old = model.adapter.infer(safe, samples=count, sample_seed=23)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "randn_like", new_noise)
        new = model.infer_query(query, samples=count, sample_seed=23)
    assert old_noise.draw == new_noise.draw == count
    assert old_noise.step == new_noise.step == 0
    assert old_noise.calls == [1] * (count * backbone.n_diffusion_steps)
    assert new_noise.calls == [3] * 6 + [1] * 3
    # Only native float32 batch-kernel reduction order can differ here.
    np.testing.assert_allclose(new["samples"], old["samples"].reshape(count, 16), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(new["point_estimate"], old["point_estimate"].reshape(16), rtol=1e-5, atol=1e-6)
    assert new["timing"]["sample_batch_sizes"] == [3, 3, 1]
    assert model.adapter.model.batch_size == 2


def test_100_joint_draws_are_complete_reproducible_poison_safe_and_reloadable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "epoch.pt"
    model, query = _make_model(tmp_path, 32, checkpoint), _query()
    first = model.infer_query(query, samples=100, sample_seed=23)
    again = model.infer_query(query, samples=100, sample_seed=23)
    assert first["samples"].shape == (100, 16)
    assert first["timing"]["sample_batch_sizes"] == [32, 32, 32, 4]
    assert first["timing"]["n_joint_samples"] == 100
    assert first["timing"]["timing_mode"] == "query_wall_mc_batched"
    assert first["model_metadata"]["sampling_policy"] == "native_replicated_condition_rows_v1"
    assert first["model_metadata"]["csdi_sample_batch_size"] == 32
    assert first["model_metadata"]["inference_batch_size"] == 1
    assert np.isfinite(first["samples"]).all()
    assert np.array_equal(first["samples"], again["samples"])
    assert np.array_equal(first["point_estimate"], np.median(first["samples"], axis=0))
    observed = query.available_to_method & ~query.padding_mask
    assert np.array_equal(first["samples"][:, observed], np.repeat(query.values[None, observed], 100, axis=0))
    assert len(np.unique(first["samples"][:, ~observed], axis=0)) == 100
    poisoned = deepcopy(query)
    poisoned.values[~observed] = -1e30
    assert np.array_equal(first["samples"], model.infer_query(poisoned, samples=100, sample_seed=23)["samples"])
    saved = tmp_path / "fitted.pt"
    model.save_fitted(saved)
    restored = load_adapter(saved, root=tmp_path, device="cpu", threads=1)
    assert restored.csdi_sample_batch_size == 32 and restored.batch_size == 2
    assert np.array_equal(first["samples"], restored.infer_query(query, samples=100, sample_seed=23)["samples"])
    with pytest.raises(ValueError, match="differ"):
        _make_model(tmp_path, 16, checkpoint)

    native = model.adapter.model
    def fail(*args, **kwargs):
        assert native.batch_size == 32
        raise RuntimeError("native sampler failed")
    monkeypatch.setattr(native, "predict", fail)
    with pytest.raises(RuntimeError, match="native sampler failed"):
        model.infer_query(query, samples=100, sample_seed=23)
    assert native.batch_size == 2


def test_csdi_sample_batch_default_retains_old_seeded_native_sampler(tmp_path):
    model, query = _make_model(tmp_path, 1), _query()
    safe = np.where(query.available_to_method & ~query.padding_mask, query.values, np.nan)
    expected = model.adapter.infer(safe, samples=6, sample_seed=23)
    actual = model.infer_query(query, samples=6, sample_seed=23)
    assert np.array_equal(actual["samples"], expected["samples"].reshape(6, 16))
    assert actual["model_metadata"]["sampling_policy"] == "native_outer_sample_loop"
    assert model._configuration()["csdi_sample_batch_size"] == 1
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            create_adapter("CSDI", root=tmp_path, csdi_sample_batch_size=invalid)

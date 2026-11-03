from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.data import make_fixture
from energygap.methods.conference_models import (
    ALL_METHODS, LOCAL_METHODS, PYPOTS_METHODS, create_adapter,
    fixed_validation_set, hpo_configurations, load_adapter,
    model_availability, training_windows,
)


def test_partition_windows_keep_natural_and_late_points_absent():
    series = make_fixture(days=1, natural_missing=False)
    series.records[4] = replace(series.records[4], natural_missing=True, observed_original=False)
    series.records[5] = replace(series.records[5], availability_time=series.records[19].grid_time + timedelta(days=1))
    first = training_windows(series, list(range(20)), context=8, stride=4)
    assert first.shape == (4, 8, 1)
    assert np.isnan(first[0, 4:6]).all()
    # Values outside the declared training support cannot change any window.
    for index in range(20, len(series.records)):
        series.records[index] = replace(series.records[index], value_canonical=1e20)
    assert np.array_equal(first, training_windows(series, list(range(20)), context=8, stride=4), equal_nan=True)
    with pytest.raises(ValueError):
        training_windows(series, [0, 1], context=8)


def test_validation_scaler_is_train_owned_and_no_natural_missing_supervision():
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    x = np.arange(32, dtype=np.float32).reshape(2, 16, 1)
    x[0, 4] = np.nan
    result, hidden = fixed_validation_set(x, np.array([2.]), np.array([4.]))
    assert not hidden[0, 4, 0]
    assert np.isnan(result["X_ori"][0, 4, 0])
    assert np.allclose(result["X_ori"][1, :, 0], (x[1, :, 0] - 2) / 4)
    assert np.isnan(result["X"][hidden]).all()
    assert np.isfinite(result["X_ori"][hidden]).all()


def test_fixed_six_trial_native_grids_and_explicit_unavailable(tmp_path):
    for name in LOCAL_METHODS:
        trials = hpo_configurations(name)
        assert len(trials) == 6
        assert len({str(trial) for trial in trials}) == 6
        assert all("learning_rate" in trial and "architecture" in trial for trial in trials)
    assert hpo_configurations("MOMENT-1-base") == []
    availability = model_availability(tmp_path)
    assert set(availability) == set(ALL_METHODS)
    for name in ("FIM", "TabPFN-TS", "MoTM", "CBiT", "MOMENT-1-base", "TS-ICL"):
        assert availability[name]["status"] == "unavailable"
        assert availability[name]["reason"]


def _params(name):
    if name == "SAITS":
        return dict(n_layers=1, d_model=8, n_heads=2, d_k=4, d_v=4, d_ffn=8)
    if name == "BRITS":
        return dict(rnn_hidden_size=8)
    if name == "CSDI":
        return dict(n_layers=1, n_heads=2, n_channels=8, d_time_embedding=8,
            d_feature_embedding=4, d_diffusion_embedding=8, n_diffusion_steps=2)
    if name == "ImputeFormer":
        return dict(n_layers=1, d_input_embed=8, d_learnable_embed=8, d_proj=4, d_ffn=8, n_temporal_heads=2)
    return dict(n_heads=8, n_blocks=[1, 1], kernel_size_large=[7, 3])


def _data():
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train = np.sin(t / 5) + t / 100
    val = np.cos(t[:3] / 4) + 1
    train[0, 2] = np.nan
    val[0, 4] = np.nan
    return train, val


def _query():
    x = np.sin(np.arange(16) / 5)
    x[12:15] = np.nan
    return SimpleNamespace(values=x, available_to_method=np.isfinite(x), padding_mask=np.zeros(16, bool), task="R")


@pytest.mark.parametrize("name", PYPOTS_METHODS)
def test_native_epoch_resume_and_completed_model(name, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypots")
    train, val = _data()
    kwargs = dict(name=name, root=tmp_path, device="cpu", threads=1, seed=11,
        params={"architecture": _params(name)}, epochs=2, patience=2, batch_size=2, context=16)
    fresh = create_adapter(**kwargs)
    fresh.fit_windows(train, val)
    expected = fresh.infer_query(_query(), samples=2, sample_seed=5)["point_estimate"]
    interrupted = create_adapter(**kwargs)
    checkpoint = tmp_path / "epoch.pt"

    class Pause(RuntimeError):
        pass

    def boundary(record):
        assert checkpoint.is_file()
        if record["epoch"] == 1:
            raise Pause()

    with pytest.raises(Pause):
        interrupted.fit_windows(train, val, checkpoint_path=checkpoint, boundary_callback=boundary)
    resumed = create_adapter(**kwargs)
    resumed.fit_windows(train, val, checkpoint_path=checkpoint)
    actual = resumed.infer_query(_query(), samples=2, sample_seed=5)["point_estimate"]
    assert np.array_equal(expected, actual)
    poisoned = _query()
    poisoned.values[~poisoned.available_to_method] = 1e20
    assert np.array_equal(resumed.infer_query(poisoned, samples=2, sample_seed=5)["point_estimate"], actual)
    assert resumed.history == fresh.history
    assert resumed.completed_epochs == 2
    assert np.array_equal(resumed.mean, np.nanmean(train, axis=(0, 1)))
    saved = tmp_path / "fitted.pt"
    metadata = resumed.save_fitted(saved)
    assert metadata["sha256"]
    restored = load_adapter(saved, root=tmp_path, threads=1)
    assert np.array_equal(restored.infer_query(_query(), samples=2, sample_seed=5)["point_estimate"], actual)
    changed = train.copy()
    changed[1, 1] += 1
    with pytest.raises(ValueError, match="differ"):
        create_adapter(**kwargs).fit_windows(changed, val, checkpoint_path=checkpoint)
    if name == "SAITS":
        # A torn/corrupt newest publication can recover the prior durable epoch.
        checkpoint.write_bytes(b"interrupted model write")
        recovered = create_adapter(**kwargs).fit_windows(train, val, checkpoint_path=checkpoint)
        assert np.array_equal(recovered.infer_query(_query())["point_estimate"], fresh.infer_query(_query())["point_estimate"])


def test_t1_native_epoch_resume_if_local_source(tmp_path):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    if not (root / "artifacts/upstream/T1/models/T1.py").is_file():
        pytest.skip("pinned official T1 source is an optional local artifact")
    train, val = _data()
    kwargs = dict(name="T1", root=root, device="cpu", threads=1, seed=11,
        params={"architecture": _params("T1")}, epochs=2, patience=2, batch_size=2, context=16)
    fresh = create_adapter(**kwargs)
    fresh.fit_windows(train, val)
    checkpoint = tmp_path / "epoch.pt"

    def boundary(record):
        if record["epoch"] == 1:
            raise InterruptedError()

    with pytest.raises(InterruptedError):
        create_adapter(**kwargs).fit_windows(train, val, checkpoint_path=checkpoint, boundary_callback=boundary)
    resumed = create_adapter(**kwargs).fit_windows(train, val, checkpoint_path=checkpoint)
    assert resumed.history == fresh.history
    assert np.array_equal(resumed.infer_query(_query())["point_estimate"], fresh.infer_query(_query())["point_estimate"])


def test_no_epoch_selection_without_validation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    model = create_adapter("SAITS", root=tmp_path, params={"architecture": _params("SAITS")}, context=16, epochs=1, patience=1)
    with pytest.raises(ValueError, match="validation"):
        model.fit_windows(_data()[0])


def test_external_validation_endpoint_selects_checkpoint_and_preserves_training_rng(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypots")
    import random
    train, val = _data()
    kwargs = dict(name="SAITS", root=tmp_path, context=16, threads=1, seed=11,
        params={"architecture": _params("SAITS")}, batch_size=2)

    def selection(model, epoch):
        # Mimic a probabilistic validation evaluator consuming all RNG families.
        model.infer_query(_query(), sample_seed=47)
        random.random()
        np.random.random(100)
        torch.rand(100)
        return float(epoch - 1)

    first = create_adapter(**kwargs, epochs=1, patience=1).fit_windows(train, val, selection_callback=selection)
    trained = create_adapter(**kwargs, epochs=3, patience=1).fit_windows(train, val, selection_callback=selection)
    assert trained.best_epoch == 1
    assert trained.completed_epochs == 2
    assert [row["validation_metric"] for row in trained.history] == [0., 1.]
    assert all("native_validation_metric" in row for row in trained.history)
    assert np.array_equal(first.infer_query(_query())["point_estimate"], trained.infer_query(_query())["point_estimate"])
    assert trained.metadata["epoch_selection"] == "validation_equal_duration_gap_nmae"
    assert trained.metadata["validation_metric"] == "validation_equal_duration_gap_nmae"
    assert trained.metadata["native_validation_metric"] == "native_validation_criterion"
    assert trained.metadata["native_validation_mask_seed"] == 1001
    assert trained.metadata["validation_mask_scope"] == "native_diagnostic_only"
    assert trained.metadata["validation_selection_mask_policy"] == "caller_prespecified_masks"

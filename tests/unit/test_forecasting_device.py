"""Device dispatch preserves the direct ridge experiment and portable model state."""
import pickle

import numpy as np
import pytest

from energygap.forecasting.ridge import DirectRidge, select_alpha


def _training_data(size=140):
    times = np.arange(size)*300.
    values = np.sin(np.arange(size)/7.)+np.arange(size)*0.002
    values[0:3] = np.nan
    values[30:33] = np.nan
    values[50] = 0.
    return values, times


def test_torch_ridge_preserves_scipy_fit_and_prediction():
    pytest.importorskip("torch")
    values, times = _training_data()
    options = dict(history=20, alpha=1.)
    bounds = (times[40], times[120])
    scipy_model = DirectRidge(**options).fit(values, times, target_bounds=bounds)
    torch_model = DirectRidge(**options, backend="torch").fit(values, times, target_bounds=bounds)
    assert scipy_model.execution_backend == "scipy"
    assert torch_model.execution_backend == "torch"
    assert torch_model.execution_device == "cpu"
    assert scipy_model.n_training_targets == torch_model.n_training_targets
    np.testing.assert_allclose(torch_model.mean, scipy_model.mean, atol=1e-13)
    np.testing.assert_allclose(torch_model.scale, scipy_model.scale, atol=1e-13)
    np.testing.assert_allclose(torch_model.coef, scipy_model.coef, atol=1e-11)
    for stop in (15, 40, 100):
        assert torch_model.predict(values[max(0, stop-20):stop], times[stop]) == pytest.approx(
            scipy_model.predict(values[max(0, stop-20):stop], times[stop]), abs=1e-11
        )


def test_torch_ridge_pickle_stores_arrays_and_rebuilds_device_cache():
    torch = pytest.importorskip("torch")
    values, times = _training_data()
    model = DirectRidge(history=20, backend="torch").fit(values, times)
    expected = model.predict(values[-20:], times[-1])
    restored = pickle.loads(pickle.dumps(model))
    assert "_torch_state" not in restored.__dict__
    assert not any(isinstance(value, torch.Tensor) for value in restored.__getstate__().values())
    assert restored.predict(values[-20:], times[-1]) == pytest.approx(expected, abs=1e-13)
    assert all(tensor.dtype == torch.float64 for tensor in restored._torch_state.values())
    assert all(tensor.device.type == "cpu" for tensor in restored._torch_state.values())


def test_ridge_validation_forwards_torch_backend_without_changing_selection():
    pytest.importorskip("torch")
    values, times = _training_data(700)
    options = dict(candidates=(0.1, 1.))
    scipy_alpha, scipy_scores = select_alpha(values[:600], times[:600], values[600:], times[600:], **options)
    torch_alpha, torch_scores = select_alpha(
        values[:600], times[:600], values[600:], times[600:], backend="torch", device="cpu", **options
    )
    assert torch_alpha == scipy_alpha
    np.testing.assert_allclose(torch_scores, scipy_scores, atol=1e-10)


def test_ridge_rejects_unknown_backend():
    values, times = _training_data()
    with pytest.raises(ValueError, match="backend"):
        DirectRidge(history=20, backend="unknown").fit(values, times)


def test_ridge_explicit_cuda_does_not_fall_back_to_scipy(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    values, times = _training_data()
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        DirectRidge(history=20, device="cuda:0").fit(values, times)


def test_legacy_scipy_model_state_remains_predictable():
    values, times = _training_data()
    model = DirectRidge(history=20).fit(values, times)
    expected = model.predict(values[-20:], times[-1])
    # Saved models from before device dispatch have no execution metadata.
    for name in ("device", "backend", "execution_device", "execution_backend"):
        model.__dict__.pop(name, None)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(values[-20:], times[-1]) == expected


def test_cuda_ridge_uses_float64_on_requested_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; CPU torch parity is tested separately")
    values, times = _training_data()
    reference = DirectRidge(history=20).fit(values, times)
    model = DirectRidge(history=20, device="cuda:0").fit(values, times)
    assert model.execution_device == "cuda:0"
    assert model.execution_backend == "torch"
    assert all(tensor.device == torch.device("cuda:0") for tensor in model._torch_state.values())
    assert all(tensor.dtype == torch.float64 for tensor in model._torch_state.values())
    assert model.predict(values[-20:], times[-1]) == pytest.approx(
        reference.predict(values[-20:], times[-1]), abs=1e-10
    )

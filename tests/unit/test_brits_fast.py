"""The accelerated BRITS path preserves native outputs, losses and gradients."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.methods.brits_fast import forward
from energygap.methods.prepared_inference import BritsInputCache


def _cores_and_inputs(length, features, hidden, batch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pypots")
    from pypots.imputation.brits.core import _BRITS
    from pypots.nn.modules.loss import MAE, MSE
    torch.set_num_threads(1)
    torch.manual_seed(815)
    native = _BRITS(length, features, hidden, MAE(), MSE())
    fast = deepcopy(native)
    rng = np.random.default_rng(817)
    original = rng.normal(size=(batch, length, features)).astype(np.float32)
    missing = rng.random(original.shape) < .3
    missing[0, :min(5, length)] = True
    values = np.where(missing, np.nan, original)
    data, _ = BritsInputCache(max_bytes=0).prepare(values)
    inputs = dict(indices=data[0],
        forward=dict(X=data[1], missing_mask=data[2], deltas=data[3]),
        backward=dict(X=data[4], missing_mask=data[5], deltas=data[6]),
        X_ori=torch.from_numpy(original), indicating_mask=torch.from_numpy(missing.astype(np.float32)))
    return torch, native, fast, inputs


@pytest.mark.parametrize("length,features,hidden,batch", [(16, 1, 8, 3), (16, 3, 8, 3), (512, 1, 64, 2)])
def test_native_training_output_loss_gradient_and_adam_step_parity(length, features, hidden, batch):
    torch, native, fast, inputs = _cores_and_inputs(length, features, hidden, batch)
    native.train();fast.train()
    expected = native(inputs, calc_criterion=True)
    rng_before = torch.get_rng_state().clone()
    actual = forward(fast, inputs, calc_criterion=True)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)
    expected["loss"].backward();actual["loss"].backward()
    assert list(native.state_dict()) == list(fast.state_dict())
    gradients = dict(native.named_parameters())
    for name, parameter in fast.named_parameters():
        reference = gradients[name].grad
        assert (parameter.grad is None) == (reference is None), name
        if reference is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference, rtol=4e-4, atol=2e-6, msg=name)
    # Adam is sensitive to small gradient sign differences on the first step;
    # comparing parameters catches errors which a loose gradient norm could miss.
    torch.optim.Adam(native.parameters(), lr=.001).step()
    torch.optim.Adam(fast.parameters(), lr=.001).step()
    for name, parameter in fast.named_parameters():
        torch.testing.assert_close(parameter, dict(native.named_parameters())[name],
                                   rtol=4e-5, atol=4e-6, msg=name)


def test_eval_criterion_is_native_but_inference_never_calls_unused_losses(monkeypatch):
    torch, native, fast, inputs = _cores_and_inputs(32, 1, 8, 3)
    native.eval();fast.eval()
    calls = []
    criterion = fast.model.rits_f.training_loss
    original = criterion.forward

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(criterion, "forward", counted)
    with torch.no_grad():
        expected = native(inputs, calc_criterion=True)
        actual = forward(fast, inputs, calc_criterion=True)
        assert len(calls) == 6 * 32
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)
        calls.clear()
        inference = forward(fast, inputs)
        assert calls == []
        assert set(inference) == {"imputation", "reconstruction", "f_reconstruction", "b_reconstruction"}
        for key in inference:
            torch.testing.assert_close(inference[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)
        observed = inputs["forward"]["missing_mask"].bool()
        assert torch.equal(inference["imputation"][observed], inputs["forward"]["X"][observed])


def test_execution_mode_is_explicit_saved_and_recovery_identity_protected(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    from energygap.methods.conference_models import create_adapter, load_adapter
    kwargs = dict(name="BRITS", root=tmp_path, device="cpu", threads=1, seed=11,
        params={"architecture": {"rnn_hidden_size": 8}}, epochs=1, patience=1,
        batch_size=2, context=16, inference_batch_size=3)
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train, val = np.sin(t / 5), np.cos(t[:3] / 4)
    epoch = tmp_path / "epoch.pt"
    model = create_adapter(**kwargs, brits_execution="vectorized")
    model.fit_windows(train, val, checkpoint_path=epoch)
    assert model._configuration()["brits_execution"] == "vectorized"
    fitted = tmp_path / "fitted.pt"
    model.save_fitted(fitted)
    restored = load_adapter(fitted, root=tmp_path, device="cpu", threads=1)
    assert restored.brits_execution == "vectorized"
    assert restored.metadata["brits_execution"] == "vectorized"
    query = SimpleNamespace(values=val[0, :, 0].copy(), available_to_method=np.ones(16, bool),
                            padding_mask=np.zeros(16, bool), task="R")
    query.available_to_method[3:8] = False
    query.values[3:8] = 1e20
    before = model.infer_queries([query])[0]["point_estimate"]
    after = restored.infer_queries([query])[0]["point_estimate"]
    assert np.array_equal(before, after)
    with pytest.raises(ValueError, match="differ"):
        create_adapter(**kwargs, brits_execution="native").fit_windows(train, val, checkpoint_path=epoch)
    with pytest.raises(ValueError, match="native or vectorized"):
        create_adapter(**kwargs, brits_execution="unverified_compiler")


def test_native_and_vectorized_training_use_the_same_masks_and_optimizer_schedule(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    from energygap.methods.conference_models import create_adapter
    kwargs = dict(name="BRITS", root=tmp_path, device="cpu", threads=1, seed=13,
        params={"architecture": {"rnn_hidden_size": 8}}, epochs=2, patience=2,
        batch_size=2, context=16, inference_batch_size=3)
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train, val = np.sin(t / 5), np.cos(t[:3] / 4)
    train[0, 2] = np.nan
    native = create_adapter(**kwargs, brits_execution="native").fit_windows(train, val)
    fast = create_adapter(**kwargs, brits_execution="vectorized").fit_windows(train, val)
    assert native.completed_epochs == fast.completed_epochs == 2
    assert native.best_epoch == fast.best_epoch
    assert [row["training_mask_sha256"] for row in native.history] == [row["training_mask_sha256"] for row in fast.history]
    assert [row["artificially_hidden_original"] for row in native.history] == [row["artificially_hidden_original"] for row in fast.history]
    for left, right in zip(native.history, fast.history):
        assert left["training_loss"] == pytest.approx(right["training_loss"], rel=2e-5, abs=2e-6)
        assert left["validation_metric"] == pytest.approx(right["validation_metric"], rel=2e-5, abs=2e-6)

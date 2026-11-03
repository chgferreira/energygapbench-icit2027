"""Native deterministic batching retains the scalar scientific input contract."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.methods.conference_models import create_adapter, load_adapter


def _architecture(name):
    if name == "BRITS":
        return dict(rnn_hidden_size=8)
    if name == "SAITS":
        return dict(n_layers=1, d_model=8, n_heads=2, d_k=4, d_v=4, d_ffn=8)
    if name == "ImputeFormer":
        return dict(n_layers=1, d_input_embed=8, d_learnable_embed=8, d_proj=4,
                    d_ffn=8, n_temporal_heads=2)
    if name == "CSDI":
        return dict(n_layers=1, n_heads=2, n_channels=8, d_time_embedding=8,
                    d_feature_embedding=4, d_diffusion_embedding=8, n_diffusion_steps=2)
    return dict(n_heads=8, n_blocks=[1, 1], kernel_size_large=[7, 3])


def _model(name, tmp_path):
    pytest.importorskip("torch")
    if name != "T1":
        pytest.importorskip("pypots")
    root = Path(__file__).resolve().parents[2] if name == "T1" else tmp_path
    if name == "T1" and not (root / "artifacts/upstream/T1/models/T1.py").is_file():
        pytest.skip("verified official T1 source is a separately prepared artifact")
    model = create_adapter(name, root=root, device="cpu", threads=1, seed=11,
        params={"architecture": _architecture(name)}, epochs=1, patience=1,
        batch_size=2, context=16)
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train, validation = np.sin(t / 5) + t / 100, np.cos(t[:3] / 4) + 1
    train[0, 2], validation[0, 4] = np.nan, np.nan
    return model.fit_windows(train, validation)


def _queries(count=5):
    queries = []
    for index in range(count):
        values = np.sin(np.arange(16) / (4 + index)) + index * .3
        available = np.ones(16, bool)
        available[(2 if index % 2 else 12):(5 if index % 2 else 15)] = False
        padding = np.zeros(16, bool)
        padding[:index % 3] = True
        # Deliberately finite evaluator-owned placeholders must never be input.
        values[~available | padding] = (-1 if index % 2 else 1) * 1e20
        queries.append(SimpleNamespace(values=values, available_to_method=available,
            padding_mask=padding, task="R", query_id=str(index)))
    return queries


@pytest.mark.parametrize("name", ["BRITS", "SAITS", "ImputeFormer", "T1"])
def test_native_batch_scalar_parity_order_padding_poisoning_and_reload(name, tmp_path, monkeypatch):
    model, queries = _model(name, tmp_path), _queries()
    scalar_results = [model.infer_query(query) for query in queries]
    assert all(result["model_metadata"]["inference_batch_size"] == 1 for result in scalar_results)
    scalar = [result["point_estimate"] for result in scalar_results]
    calls = []
    if name == "BRITS":
        native = model.adapter.model
        original_assemble = native._assemble_input_for_testing

        def assemble(data):
            calls.append((len(data[0]), native.batch_size,
                          np.where(data[2].numpy().astype(bool), data[1].numpy(), np.nan)))
            return original_assemble(data)

        monkeypatch.setattr(native, "_assemble_input_for_testing", assemble)
        monkeypatch.setattr(native, "predict", lambda *a, **k: pytest.fail("prepared BRITS must avoid repeated Dataset construction"))
    elif name != "T1":
        native = model.adapter.model
        original_predict = native.predict

        def predict(data, *args, **kwargs):
            calls.append((len(data["X"]), native.batch_size, data["X"].copy()))
            return original_predict(data, *args, **kwargs)

        monkeypatch.setattr(native, "predict", predict)
    outputs = model.infer_queries(queries, sample_seeds=[101, 103, 107, 109, 113])
    assert len(outputs) == 5
    if name != "T1":
        assert [(n, b) for n, b, _ in calls] == ([(2, 2), (2, 2), (1, 2)] if name == "BRITS"
                                               else [(2, 2), (2, 2), (1, 1)])
        assert native.batch_size == 2
        sent = np.concatenate([x for _, _, x in calls], axis=0)
        for query, x in zip(queries, sent):
            assert np.isnan(x[~query.available_to_method | query.padding_mask]).all()
    for index, (query, output) in enumerate(zip(queries, outputs)):
        # Native float32 matrix kernels may choose another reduction order for B>1.
        # A 1e-6 absolute / 1e-5 relative tolerance covers that roundoff; observed
        # input preservation below remains exact, not approximate.
        np.testing.assert_allclose(output["point_estimate"], scalar[index], rtol=1e-5, atol=1e-6)
        received = query.available_to_method & ~query.padding_mask
        assert np.array_equal(output["point_estimate"][received], query.values[received])
        assert output["samples"] is None and output["quantiles"] is None
        timing = output["timing"]
        assert timing["timing_mode"] == "batch_amortized"
        assert timing["batch_size"] == (2 if index < 4 else 1)
        assert timing["inference_seconds"] * timing["batch_size"] == pytest.approx(timing["batch_wall_seconds"])
        assert output["model_metadata"]["configured_inference_batch_size"] == 2
        assert output["model_metadata"]["inference_batching"] == "native_deterministic_batch_v1"

    # Other batch composition and ordering must not let one query normalize another.
    order = [4, 1, 3, 0, 2]
    permuted = model.infer_queries([queries[i] for i in order], batch_size=3)
    for output, index in zip(permuted, order):
        np.testing.assert_allclose(output["point_estimate"], scalar[index], rtol=1e-5, atol=1e-6)
    poisoned = deepcopy(queries)
    for query in poisoned:
        query.values[~query.available_to_method | query.padding_mask] *= -1e10
    for before, after in zip(outputs, model.infer_queries(poisoned)):
        assert np.array_equal(before["point_estimate"], after["point_estimate"])

    checkpoint = tmp_path / (name + ".pt")
    model.save_fitted(checkpoint)
    restored = load_adapter(checkpoint, root=model.root, device="cpu", threads=1)
    for before, after in zip(outputs, restored.infer_queries(queries)):
        assert np.array_equal(before["point_estimate"], after["point_estimate"])
    assert model.batch_size == 2 and restored.batch_size == 2
    assert model.inference_batch_size == 2 and restored.inference_batch_size == 2
    assert model.infer_queries([]) == []
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            model.infer_queries(queries, batch_size=invalid)

    if name == "BRITS":
        def fail(data):
            raise RuntimeError("native prediction failure")

        monkeypatch.setattr(native, "_assemble_input_for_testing", fail)
        with pytest.raises(RuntimeError, match="native prediction failure"):
            model.infer_queries(queries, batch_size=4)
        assert native.batch_size == 2
    elif name != "T1":
        def fail(data, *args, **kwargs):
            assert native.batch_size == 4
            raise RuntimeError("native prediction failure")

        monkeypatch.setattr(native, "predict", fail)
        with pytest.raises(RuntimeError, match="native prediction failure"):
            model.infer_queries(queries, batch_size=4)
        assert native.batch_size == 2


def test_csdi_retains_exact_per_query_samples_and_seeds_across_requested_batch_sizes(tmp_path):
    model, queries, seeds = _model("CSDI", tmp_path), _queries(3), [101, 211, 307]
    scalar = [model.infer_query(query, samples=2, sample_seed=seed)
              for query, seed in zip(queries, seeds)]
    for batch_size in (1, 2, 8):
        outputs = model.infer_queries(queries, samples=2, sample_seeds=seeds, batch_size=batch_size)
        for expected, actual in zip(scalar, outputs):
            assert np.array_equal(actual["samples"], expected["samples"])
            assert np.array_equal(actual["point_estimate"], expected["point_estimate"])
            assert "inference_batching" not in actual["model_metadata"]
            assert actual["model_metadata"]["inference_batch_size"] == 1


def test_inference_batch_size_is_separate_saved_and_part_of_epoch_identity(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pypots")
    kwargs = dict(name="SAITS", root=tmp_path, device="cpu", threads=1, seed=11,
        params={"architecture": _architecture("SAITS")}, epochs=1, patience=1,
        batch_size=2, context=16)
    t = np.arange(6 * 16, dtype=np.float32).reshape(6, 16, 1)
    train, validation = np.sin(t / 5) + t / 100, np.cos(t[:3] / 4) + 1
    checkpoint = tmp_path / "epoch.pt"
    model = create_adapter(**kwargs, inference_batch_size=3)
    model.fit_windows(train, validation, checkpoint_path=checkpoint)
    outputs = model.infer_queries(_queries())
    assert model.batch_size == 2 and model.inference_batch_size == 3
    assert [r["timing"]["batch_size"] for r in outputs] == [3, 3, 3, 2, 2]
    assert model._configuration()["inference_batch_size"] == 3
    fitted = tmp_path / "fitted.pt"
    model.save_fitted(fitted)
    restored = load_adapter(fitted, root=tmp_path, device="cpu", threads=1)
    assert restored.batch_size == 2 and restored.inference_batch_size == 3
    assert restored.metadata["inference_policy"] == "det4_and_MOMENT_batches_CSDI_TSICL_serial"
    for before, after in zip(outputs, restored.infer_queries(_queries())):
        assert np.array_equal(before["point_estimate"], after["point_estimate"])
    with pytest.raises(ValueError, match="differ"):
        create_adapter(**kwargs, inference_batch_size=4).fit_windows(train, validation, checkpoint_path=checkpoint)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            create_adapter(**kwargs, inference_batch_size=invalid)

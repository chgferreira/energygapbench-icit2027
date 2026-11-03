"""Actual frozen weights: independent normalization, masking and grouping parity."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.methods.conference_models import create_adapter, load_adapter, model_availability


@pytest.mark.heavy
def test_native_moment_batch_preserves_scalar_contract_and_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not model_availability(root)['MOMENT-1-base']['available']:
        pytest.skip('compatible MOMENT packages and verified local checkpoint required')
    from energygap.methods.frozen_active import ActiveFrozenModel
    native = ActiveFrozenModel('MOMENT-1-base', root, device='cpu', threads=1)
    queries = []
    for i, (scale, offset) in enumerate(((1., 0.), (30., -100.), (.02, 3.))):
        ticks = np.arange(512, dtype=float) * 300
        values = scale * np.sin(np.arange(512) / (19 + i)) + offset
        observed = np.ones(512, bool)
        observed[497 - i:511] = False  # includes a partially masked native patch
        observed[20:23] = False
        padding = np.zeros(512, bool)
        padding[:5 * i] = True
        # Poison hidden and padding positions, including padding marked available.
        values[~observed | padding] = 1e20
        queries.append(SimpleNamespace(task='R', values=values, times=ticks,
            available_to_method=observed, padding_mask=padding, issuance_time=ticks[-1]))
    scalar = [native.infer(query)['point_estimate'] for query in queries]
    batched = native.infer_batch(queries)
    for query, reference, output in zip(queries, scalar, batched):
        admitted = query.available_to_method & ~query.padding_mask
        np.testing.assert_array_equal(output['point_estimate'][admitted], query.values[admitted])
        np.testing.assert_allclose(output['point_estimate'], reference, rtol=1e-5, atol=1e-5, equal_nan=True)
        assert output['timing']['batch_size'] == 3
        assert output['timing']['timing_mode'] == 'batch_amortized'
    # Different peers and batch ordering cannot alter the same row's statistics.
    permuted = native.infer_batch([queries[2], queries[0]])
    for index, output in zip((2, 0), permuted):
        np.testing.assert_allclose(output['point_estimate'], scalar[index], rtol=1e-5, atol=1e-5, equal_nan=True)
    model = create_adapter('MOMENT-1-base', root, device='cpu', threads=1, inference_batch_size=2)
    model.adapter = native
    grouped = model.infer_queries(queries)
    assert [row['timing']['batch_size'] for row in grouped] == [2, 2, 1]
    for reference, output in zip(scalar, grouped):
        np.testing.assert_allclose(output['point_estimate'], reference, rtol=1e-5, atol=1e-5, equal_nan=True)
    path = tmp_path / 'moment.json'
    model.save_fitted(path)
    restored = load_adapter(path, root=root, device='cpu', threads=1)
    assert restored.inference_batch_size == 2
    # Reuse already verified weights to avoid a redundant 385MB model load.
    restored.adapter = native
    np.testing.assert_allclose(restored.infer_queries(queries)[0]['point_estimate'],
                               grouped[0]['point_estimate'], rtol=0, atol=0, equal_nan=True)


@pytest.mark.heavy
def test_moment_batch_validates_all_queries_before_loading(monkeypatch):
    from energygap.methods.frozen_active import ActiveFrozenModel
    model = ActiveFrozenModel.__new__(ActiveFrozenModel)
    model.name = 'MOMENT-1-base'
    monkeypatch.setattr(model, '_load', lambda: pytest.fail('invalid input reached native loader'))
    q = SimpleNamespace(task='R', values=np.ones(512), times=np.arange(512) * 300,
        padding_mask=np.zeros(512, bool), available_to_method=np.ones(512, bool), issuance_time=0.)
    with pytest.raises(ValueError, match='reconnection availability'):
        model.infer_batch([q])

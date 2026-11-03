"""Native frozen models in the active environment; no downloads or CUDA claim."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import os
import sys

import numpy as np
import pytest

from energygap.methods.conference_models import create_adapter, model_availability


@pytest.mark.heavy
@pytest.mark.parametrize("name", ["MOMENT-1-base", "TS-ICL"])
def test_active_frozen_shard_preserves_inputs_and_ignores_hidden_placeholders(name, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    if not model_availability(root)[name]["available"]:
        pytest.skip("compatible active native packages/local checkpoint unavailable")
    # Historical interpreter overrides have no effect on the server pipeline.
    monkeypatch.setenv("ENERGYGAP_MOMENT_PYTHON", "/missing/legacy/python")
    monkeypatch.setenv("ENERGYGAP_TSICL_PYTHON", "/missing/legacy/python")
    t = np.arange(512, dtype=float) * 300
    values = np.sin(np.arange(512) / 20)
    observed = np.ones(512, bool)
    observed[500:511] = False
    safe = np.where(observed, values, np.nan)
    poisoned = np.where(observed, values, 1e20)
    common = dict(times=t, available_to_method=observed, padding_mask=np.zeros(512, bool),
        task="R", issuance_time=datetime.fromtimestamp(t[-1], timezone.utc))
    queries = [SimpleNamespace(query_id="safe", values=safe, **common),
        SimpleNamespace(query_id="poisoned", values=poisoned, **common)]
    model = create_adapter(name, root=root, device="cpu", threads=1)
    outputs = model.infer_queries(queries)
    assert len(outputs) == 2
    assert np.array_equal(outputs[0]["point_estimate"], outputs[1]["point_estimate"])
    assert np.array_equal(outputs[0]["point_estimate"][observed], values[observed])
    assert np.isfinite(outputs[0]["point_estimate"][~observed]).all()
    assert outputs[0]["model_metadata"]["local_training"] == "not_applicable"
    assert outputs[0]["model_metadata"]["threads"] == 1
    assert outputs[0]["model_metadata"]["environment_policy"] == "single_cuda_env"
    assert outputs[0]["model_metadata"]["training_variant"] == "frozen_native_cuda_env"
    assert outputs[0]["model_metadata"]["executable"] == sys.executable
    assert outputs[0]["model_metadata"]["process_id"] == os.getpid()
    assert outputs[0]["model_metadata"]["dependency_override"] is False
    native = model.adapter.native
    repeated = model.infer_queries([queries[0]])
    assert model.adapter.native is native
    assert np.array_equal(repeated[0]["point_estimate"], outputs[0]["point_estimate"])
    if name == "TS-ICL":
        assert outputs[0]["samples"] is None
        assert outputs[0]["quantiles"].shape == (512, 7)
        assert np.array_equal(outputs[0]["point_estimate"], outputs[0]["quantiles"][:, 3])


@pytest.mark.heavy
@pytest.mark.parametrize("name", ["MOMENT-1-base", "TS-ICL"])
def test_active_frozen_padding_is_never_observation_and_future_is_rejected(name):
    root = Path(__file__).resolve().parents[2]
    if not model_availability(root)[name]["available"]:
        pytest.skip("compatible active native packages/local checkpoint unavailable")
    times = np.arange(512, dtype=float) * 300
    observed = np.ones(512, bool)
    observed[500:511] = False
    padding = np.zeros(512, bool)
    padding[:5] = True
    # A padding position marked available still cannot become model input.
    values = np.sin(np.arange(512) / 20)
    first = values.copy()
    first[padding] = 1e20
    second = values.copy()
    second[padding] = -1e20
    common = dict(times=times, available_to_method=observed, padding_mask=padding,
                  task="R", issuance_time=datetime.fromtimestamp(times[-1], timezone.utc))
    model = create_adapter(name, root=root, device="cpu", threads=1)
    queries = [SimpleNamespace(query_id="padding1", values=first, **common),
               SimpleNamespace(query_id="padding2", values=second, **common)]
    outputs = model.infer_queries(queries)
    assert np.array_equal(outputs[0]["point_estimate"], outputs[1]["point_estimate"], equal_nan=True)
    assert np.isnan(outputs[0]["point_estimate"][padding]).all()
    assert np.isfinite(outputs[0]["point_estimate"][~padding]).all()
    future = SimpleNamespace(query_id="future", values=values, **{**common,
                             "issuance_time": datetime.fromtimestamp(times[-2], timezone.utc)})
    with pytest.raises(ValueError, match="reconnection availability"):
        model.infer_query(future)

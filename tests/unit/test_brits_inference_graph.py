"""Graph lifecycle tests and an opt-in real CUDA parity probe, never CPU CUDA claims."""
import numpy as np
import pytest

from energygap.methods.brits_fast import InferenceGraph, forward
from test_brits_fast import _cores_and_inputs


def _slice(inputs, count):
    return {direction: {name: value[:count].clone() for name, value in inputs[direction].items()}
            for direction in ("forward", "backward")}


def test_graph_mode_on_cpu_is_explicit_eager_fallback_and_preserves_outputs(tmp_path):
    from energygap.methods.conference_models import create_adapter
    from types import SimpleNamespace
    torch, native, fast, inputs = _cores_and_inputs(16, 1, 8, 3)
    fast.eval()
    runner = InferenceGraph()
    with torch.no_grad():
        result, metadata = runner.run(fast, inputs)
        expected = forward(fast, inputs)
    for name in expected:
        assert torch.equal(result[name], expected[name])
    assert metadata["cuda_graph_status"] == "eager_fallback"
    assert "CPU_execution_is_not_GPU_validation" in metadata["cuda_graph_fallback_reason"]
    assert metadata["cuda_graph_cache_shapes"] == 0
    assert metadata["cuda_graph_capture_count"] == 0
    assert metadata["cuda_graph_experimental"] is True
    model = create_adapter("BRITS", root=tmp_path, device="cpu", threads=1, context=16,
        params={"architecture": {"rnn_hidden_size": 8}}, brits_execution="vectorized_cuda_graph")
    model.adapter.mean, model.adapter.scale = np.array([0.]), np.array([1.])
    query = SimpleNamespace(values=np.arange(16, dtype=float), available_to_method=np.ones(16, bool),
                            padding_mask=np.zeros(16, bool), task="R")
    query.available_to_method[4:8] = False
    output = model.infer_queries([query])[0]
    assert output["model_metadata"]["cuda_graph_status"] == "eager_fallback"
    assert output["model_metadata"]["brits_cuda_graph_requested"] is True
    assert output["model_metadata"]["brits_training_cuda_graph"] is False
    assert output["model_metadata"]["inference_path"] == "brits_vectorized_decay_gates_loss_free"
    model.set_brits_execution("native")
    assert not hasattr(model, "_brits_inference_graph")
    assert model.metadata["brits_cuda_graph_requested"] is False


def test_simulated_graph_copies_inputs_outputs_and_keeps_one_dominant_shape(monkeypatch):
    torch, native, fast, inputs = _cores_and_inputs(16, 1, 8, 3)
    fast.eval()
    runner = InferenceGraph()
    monkeypatch.setattr(runner, "_eligibility", lambda *args: None)
    captures = []

    class SimulatedGraph:
        def replay(self):
            fresh = forward(fast, runner.inputs)
            for name in fresh:
                runner.outputs[name].copy_(fresh[name])

        def reset(self):
            pass

    def capture(core, received):
        captures.append(received["forward"]["X"].shape)
        runner.inputs = _slice(received, len(received["forward"]["X"]))
        runner.outputs = forward(core, runner.inputs)
        runner.graph = SimulatedGraph()
        runner.capture_count += 1

    monkeypatch.setattr(runner, "_capture", capture)
    with torch.no_grad():
        first, metadata = runner.run(fast, inputs)
        held = {name: value.clone() for name, value in first.items()}
        assert metadata["cuda_graph_status"] == "captured_and_replayed"
        changed = _slice(inputs, 3)
        changed["forward"]["X"] += .2
        changed["backward"]["X"] += .2
        second, metadata = runner.run(fast, changed)
        for name, expected in forward(fast, changed).items():
            assert torch.equal(second[name], expected)
            assert torch.equal(first[name], held[name])
        assert metadata["cuda_graph_status"] == "replayed"
        _, metadata = runner.run(fast, _slice(changed, 1))
        assert metadata["cuda_graph_fallback_reason"] == "non_dominant_batch_shape_uses_eager"
        assert len(captures) == 1 and metadata["cuda_graph_cache_shapes"] == 1
        # Updated parameter values use the same captured storage; model weights
        # are read again, rather than caching outputs from an earlier epoch.
        fast.model.rits_f.hist_reg.bias.add_(.1)
        updated, _ = runner.run(fast, changed)
        assert torch.equal(updated["imputation"], forward(fast, changed)["imputation"])
        assert len(captures) == 1


def test_capture_failure_is_explicit_eager_and_does_not_retry_forever(monkeypatch):
    torch, native, fast, inputs = _cores_and_inputs(16, 1, 8, 3)
    fast.eval()
    runner = InferenceGraph()
    monkeypatch.setattr(runner, "_eligibility", lambda *args: None)
    attempts = []

    def fail(*args):
        attempts.append(1)
        raise RuntimeError("unsupported capture operation")

    monkeypatch.setattr(runner, "_capture", fail)
    with torch.no_grad():
        for _ in range(2):
            result, metadata = runner.run(fast, inputs)
            assert torch.equal(result["imputation"], forward(fast, inputs)["imputation"])
            assert metadata["cuda_graph_status"] == "eager_fallback"
            assert "unsupported capture operation" in metadata["cuda_graph_fallback_reason"]
            assert metadata["cuda_graph_cache_shapes"] == 0
    assert len(attempts) == 1


def test_replay_failure_releases_graph_and_uses_explicit_eager_fallback(monkeypatch):
    torch, native, fast, inputs = _cores_and_inputs(16, 1, 8, 3)
    fast.eval()
    runner = InferenceGraph()
    monkeypatch.setattr(runner, "_eligibility", lambda *args: None)
    resets = []

    class FailedGraph:
        def replay(self):
            raise RuntimeError("simulated replay failure")

        def reset(self):
            resets.append(1)

    def capture(core, received):
        runner.inputs = _slice(received, 3)
        runner.outputs = forward(core, runner.inputs)
        runner.graph = FailedGraph()

    monkeypatch.setattr(runner, "_capture", capture)
    with torch.no_grad():
        result, metadata = runner.run(fast, inputs)
        assert torch.equal(result["imputation"], forward(fast, inputs)["imputation"])
        assert metadata["cuda_graph_status"] == "eager_fallback"
        assert "cuda_graph_replay_failed" in metadata["cuda_graph_fallback_reason"]
        assert runner.graph is runner.inputs is runner.outputs is None
        assert resets == [1]


@pytest.mark.heavy
def test_real_cuda_graph_replay_matches_native_and_updated_weights():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A real CUDA GPU is required; CPU tests do not establish V100 parity")
    torch, native, fast, inputs = _cores_and_inputs(512, 1, 64, 2)
    native = native.to("cuda:0").eval()
    fast = fast.to("cuda:0").eval()
    received = {direction: {name: value.to("cuda:0") for name, value in inputs[direction].items()}
                for direction in ("forward", "backward")}
    runner = InferenceGraph()
    with torch.no_grad():
        rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        first, metadata = runner.run(fast, received)
        assert metadata["cuda_graph_status"] == "captured_and_replayed", metadata
        assert torch.equal(rng_cpu, torch.get_rng_state())
        assert torch.equal(rng_cuda, torch.cuda.get_rng_state())
        expected = native(received)
        torch.testing.assert_close(first["imputation"], expected["imputation"], rtol=2e-5, atol=2e-6)
        snapshot = first["imputation"].clone()
        fast.model.rits_f.hist_reg.bias.add_(.1)
        second, metadata = runner.run(fast, received)
        assert metadata["cuda_graph_status"] == "replayed"
        torch.testing.assert_close(second["imputation"], forward(fast, received)["imputation"], rtol=2e-5, atol=2e-6)
        assert torch.equal(first["imputation"], snapshot)
    runner.clear()

"""FP32 BRITS execution using the existing native modules and parameters.

Adapted from PyPOTS 1.5 imputation/brits/core.py and nn/modules/brits/backbone.py.
Copyright (c) 2023-present, Wenjie Du. BSD-3-Clause; the upstream notice and
disclaimer are retained in docs/licenses/pypots-bsd-3-clause.txt.

Only time-independent decay/gate work moves outside the recurrent loop. The
LSTM, feature/history regressions, per-step native losses, masks and optimizer
steps are unchanged. Evaluation without a criterion skips losses which cannot
affect reconstruction. The ordinary PyPOTS forward remains the scalar reference.
No module is replaced or registered, so state_dict and optimizer keys are native.
"""
from __future__ import annotations

import logging


EXECUTION_MODES = ("native", "vectorized", "vectorized_cuda_graph")
POLICY = "native_brits_vectorized_decay_gates_fp32_v1"
LOG = logging.getLogger(__name__)


def _rits(module, values, mask, deltas, *, compute_losses):
    import torch

    if values.ndim != 3 or values.shape != mask.shape or values.shape != deltas.shape:
        raise ValueError("BRITS requires matching [N,L,D] values/masks/deltas")
    if values.shape[1:] != (module.n_steps, module.n_features):
        raise ValueError("BRITS input shape differs from the native architecture")
    # TemporalDecay and Linear operate on the final dimension. These quantities
    # depend on the received masks/deltas and current weights, never on an LSTM
    # state, so this is the same computation over a larger leading dimension.
    # Keep autograd attached: all trainable decay/gate parameters still receive
    # gradients from the full native loss on every optimizer step.
    gamma_h = module.temp_decay_h(deltas)
    gamma_x = module.temp_decay_x(deltas)
    alpha = torch.sigmoid(module.combining_weight(torch.cat((gamma_x, mask), dim=-1)))
    hidden = values.new_zeros((values.shape[0], module.rnn_hidden_size))
    cell = values.new_zeros((values.shape[0], module.rnn_hidden_size))
    estimates = []
    loss = values.new_zeros(()) if compute_losses else None
    for step in range(module.n_steps):
        x, observed = values[:, step, :], mask[:, step, :]
        hidden = hidden * gamma_h[:, step, :]
        history = module.hist_reg(hidden)
        if compute_losses:
            loss = loss + module.training_loss(history, x, observed)
        received = observed * x + (1 - observed) * history
        features = module.feat_reg(received)
        if compute_losses:
            loss = loss + module.training_loss(features, x, observed)
        weight = alpha[:, step, :]
        estimate = weight * features + (1 - weight) * history
        if compute_losses:
            loss = loss + module.training_loss(estimate, x, observed)
        completed = observed * x + (1 - observed) * estimate
        estimates.append(estimate.unsqueeze(1))
        hidden, cell = module.rnn_cell(torch.cat((completed, observed), dim=1), (hidden, cell))
    reconstruction = torch.cat(estimates, dim=1)
    imputed = mask * values + (1 - mask) * reconstruction
    if compute_losses:
        loss = loss / (module.n_steps * 3)
    return imputed, reconstruction, hidden, loss


def forward(core, inputs, *, calc_criterion=False):
    """Evaluate the native _BRITS weights with precomputed temporal gates.

    Training and criterion-bearing validation preserve all native result fields.
    The inference-only result deliberately omits loss fields rather than exposing
    zeros as measured losses. It retains imputation and both reconstructions.
    No randomness, model mutation, dtype conversion or hidden-state cache occurs.
    """
    import torch

    compute_losses = core.training or calc_criterion
    backbone = core.model
    left, right = inputs["forward"], inputs["backward"]
    f_imputed, f_reconstruction, _, f_loss = _rits(backbone.rits_f,
        left["X"], left["missing_mask"], left["deltas"], compute_losses=compute_losses)
    b_imputed, b_reconstruction, _, b_loss = _rits(backbone.rits_b,
        right["X"], right["missing_mask"], right["deltas"], compute_losses=compute_losses)
    # This is the native time-index reversal of backward reconstructions. The
    # hidden-state reversal in BackboneBRITS is unused by the _BRITS output API.
    b_imputed = torch.flip(b_imputed, dims=(1,))
    b_reconstruction = torch.flip(b_reconstruction, dims=(1,))
    result = dict(imputation=(f_imputed + b_imputed) / 2,
        reconstruction=(f_reconstruction + b_reconstruction) / 2,
        f_reconstruction=f_reconstruction, b_reconstruction=b_reconstruction)
    if compute_losses:
        consistency = backbone._get_consistency_loss(f_imputed, b_imputed)
        reconstruction_loss = f_loss + b_loss
        result.update(consistency_loss=consistency, reconstruction_loss=reconstruction_loss)
        if calc_criterion:
            if core.training:
                result["loss"] = consistency + reconstruction_loss
            else:
                result["metric"] = core.validation_metric(result["reconstruction"],
                    inputs["X_ori"], inputs["indicating_mask"])
    return result


class InferenceGraph:
    """Experimental, opt-in CUDA graph for one dominant inference shape.

    No training, optimizer, stochastic operation or hidden state is captured.
    Tail shapes use eager execution without evicting the dominant graph. A larger
    batch or changed parameter storage replaces the single cached graph. Failed
    capture disables graphs for this instance and exposes its eager fallback.
    CUDA/V100 acceptance must be established on the server, never inferred from
    CPU tests. Only sanitized input buffers are retained; returned outputs are
    cloned so a later replay cannot overwrite an earlier query's result.
    """
    def __init__(self):
        self.graph = None
        self.inputs = None
        self.outputs = None
        self.key = None
        self.disabled_reason = None
        self.capture_count = 0

    def clear(self):
        graph = self.graph
        device = self.inputs["forward"]["X"].device if self.inputs is not None else None
        if device is not None and device.type == "cuda":
            import torch
            # Shape/storage replacement is rare. Complete outstanding replay and
            # output clones before releasing that graph's private memory pool.
            torch.cuda.current_stream(device).synchronize()
        self.graph = self.inputs = self.outputs = self.key = None
        if graph is not None:
            try:
                graph.reset()
            except RuntimeError:
                # A failed capture can leave no instantiated graph to reset.
                pass

    @staticmethod
    def _eligibility(core, inputs):
        import torch
        tensors = [inputs[direction][name] for direction in ("forward", "backward")
                   for name in ("X", "missing_mask", "deltas")]
        if any(tensor.device.type != "cuda" for tensor in tensors):
            return "cuda_graph_requires_cuda_inputs;CPU_execution_is_not_GPU_validation"
        if core.training or torch.is_grad_enabled():
            return "cuda_graph_is_inference_only_requires_eval_and_no_grad"
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            return "cuda_graph_requires_native_float32_inputs"
        if len({tensor.device for tensor in tensors}) != 1:
            return "cuda_graph_requires_one_CUDA_device"
        return None

    @staticmethod
    def _key(core, inputs):
        values = inputs["forward"]["X"]
        storage = tuple((name, parameter.data_ptr(), str(parameter.device), str(parameter.dtype))
                        for name, parameter in core.named_parameters())
        return (tuple(values.shape), str(values.device), str(values.dtype), storage)

    def _capture(self, core, inputs):
        import torch
        device = inputs["forward"]["X"].device
        self.inputs = {direction: {name: inputs[direction][name].detach().clone()
                                  for name in ("X", "missing_mask", "deltas")}
                       for direction in ("forward", "backward")}
        # Warm up on a side stream before capture. Keep RNG state unchanged even
        # if an installed backend performs incidental initialization work.
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        current = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current)
        try:
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    forward(core, self.inputs)
            current.wait_stream(capture_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream):
                self.outputs = forward(core, self.inputs)
            current.wait_stream(capture_stream)
        finally:
            try:
                current.wait_stream(capture_stream)
            finally:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)
        self.capture_count += 1

    def _metadata(self, status, reason=None):
        return dict(cuda_graph_status=status, cuda_graph_fallback_reason=reason,
            cuda_graph_cache_shapes=int(self.graph is not None),
            cuda_graph_capture_count=self.capture_count, cuda_graph_experimental=True,
            cuda_graph_policy="single_dominant_shape_eager_tails_fp32",
            inference_path=("brits_vectorized_cuda_graph_replay" if status in ("replayed", "captured_and_replayed")
                            else "brits_vectorized_decay_gates_loss_free"))

    def run(self, core, inputs):
        import torch
        reason = self._eligibility(core, inputs) or self.disabled_reason
        if reason is not None:
            return forward(core, inputs), self._metadata("eager_fallback", reason)
        key = self._key(core, inputs)
        if self.graph is not None and key != self.key:
            same_storage = key[1:] == self.key[1:]
            if same_storage and key[0][0] <= self.key[0][0]:
                return forward(core, inputs), self._metadata("eager_fallback", "non_dominant_batch_shape_uses_eager")
            self.clear()
        captured = False
        failure = None
        if self.graph is None:
            try:
                self._capture(core, inputs)
                self.key = key
                captured = True
            except RuntimeError as error:
                failure = type(error).__name__ + ": " + str(error)[:500]
                self.clear()
                self.disabled_reason = "cuda_graph_capture_failed; " + failure
            if failure is not None:
                # The exception's traceback must be released before reclaiming
                # partial graph allocations. Eager OOM still reaches the common
                # adaptive batch handler instead of becoming a fabricated output.
                import gc
                gc.collect()
                if inputs["forward"]["X"].device.type == "cuda":
                    torch.cuda.empty_cache()
                LOG.warning("BRITS CUDA graph unavailable; keeping FP32 eager inference: %s", failure)
                return forward(core, inputs), self._metadata("eager_fallback", self.disabled_reason)
        replay_failure = None
        try:
            for direction in ("forward", "backward"):
                for name in ("X", "missing_mask", "deltas"):
                    self.inputs[direction][name].copy_(inputs[direction][name])
            self.graph.replay()
            # Do not expose reusable graph-owned output storage to the caller.
            outputs = {name: value.clone() for name, value in self.outputs.items()}
        except RuntimeError as error:
            replay_failure = type(error).__name__ + ": " + str(error)[:500]
            self.clear()
            self.disabled_reason = "cuda_graph_replay_failed; " + replay_failure
        if replay_failure is not None:
            import gc
            gc.collect()
            if inputs["forward"]["X"].device.type == "cuda":
                torch.cuda.empty_cache()
            LOG.warning("BRITS CUDA graph replay unavailable; keeping FP32 eager inference: %s", replay_failure)
            return forward(core, inputs), self._metadata("eager_fallback", self.disabled_reason)
        return outputs, self._metadata("captured_and_replayed" if captured else "replayed")

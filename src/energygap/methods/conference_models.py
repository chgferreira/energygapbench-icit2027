"""Scientific-run backends with native architectures and durable epoch boundaries.

The PyPOTS architecture, model-specific dataset transforms and criterion remain
upstream. Mask corruption and epoch orchestration follow the declared protocol
to make selection and interruption explicit. This is named an epoch runner rather
than an assertion
that every upstream training script was reproduced. Frozen models receive only
sanitized observations in the active interpreter; evaluator truth is never passed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import io
import json
import logging
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from types import MethodType

import numpy as np

from energygap.runtime.device import resolve_device, synchronize, torch_device_metadata
from energygap.runtime.state import atomic_bytes, atomic_json, fingerprint

PYPOTS_METHODS = ("BRITS", "SAITS", "CSDI", "ImputeFormer")
LOCAL_METHODS = (*PYPOTS_METHODS, "T1")
FROZEN_METHODS = ("MOMENT-1-base", "TS-ICL")
ALL_METHODS = (*LOCAL_METHODS, "MOMENT-1-base", "FIM", "TabPFN-TS", "MoTM", "TS-ICL", "CBiT")
TSICL_REVISION = "19c94031439fb31f36ce395088ee50a6762d3774"
TSICL_SHA256 = "a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431"
LOG = logging.getLogger(__name__)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def model_availability(root, device="cpu", threads=1):
    """Audit local prerequisites without downloading, training or loading weights.

    ``available`` means local prerequisites exist, not convergence or CUDA parity.
    The active native adapter repeats checksum, package and device checks before use.
    """
    from .foundation import MOMENT_REVISION, MOMENT_SHA256
    from .t1 import MODEL_SHA256, COMMIT
    root = Path(root).resolve()
    registry = {}
    registry_path = root / "model_registry.yaml"
    if registry_path.exists():
        import yaml
        registry = {row["name"]: row for row in yaml.safe_load(registry_path.read_text())["methods"]}
    results = {}
    for name in ALL_METHODS:
        results[name] = dict(name=name, status="unavailable", available=False,
            local_training="required" if name in LOCAL_METHODS else "not_applicable" if name in FROZEN_METHODS else "unknown_not_reproduced",
            reason=registry.get(name, {}).get("availability_reason", "upstream integration not verified"),
            environment="cuda_env_active" if name in (*LOCAL_METHODS, *FROZEN_METHODS) else None,
            executable=sys.executable if name in (*LOCAL_METHODS, *FROZEN_METHODS) else None,
            checkpoint=None, device=str(device), threads=int(threads))
    pypots_ready = importlib.util.find_spec("pypots") is not None and importlib.util.find_spec("torch") is not None
    for name in PYPOTS_METHODS:
        results[name].update(available=pypots_ready, status="available" if pypots_ready else "unavailable",
            reason="native PyPOTS architecture/loss; conference epoch runner" if pypots_ready else "install optional PyPOTS and compatible PyTorch in cuda_env",
            environment_versions={"pypots": _version("pypots"), "torch": _version("torch")})
    source = root / "artifacts/upstream/T1/models/T1.py"
    license_path = source.parent.parent / "LICENSE"
    t1_ready = importlib.util.find_spec("torch") is not None and source.is_file() and _sha(source) == MODEL_SHA256 and license_path.is_file() and "MIT License" in license_path.read_text()
    results["T1"].update(available=t1_ready, status="available" if t1_ready else "unavailable",
        reason="pinned official source; validation-selected epoch schedule adaptation" if t1_ready else "pinned MIT T1 source or main Torch prerequisites absent; run scripts/fetch_t1.py explicitly",
        checkpoint={"source_commit": COMMIT, "source_sha256": MODEL_SHA256})
    moment = root / "artifacts/checkpoints/hf/models--AutonLab--MOMENT-1-base/snapshots" / MOMENT_REVISION / "model.safetensors"
    tsicl = root / "artifacts/checkpoints/tsicl/tsicl-v1.ckpt"
    for name, path, sha, revision in (("MOMENT-1-base", moment, MOMENT_SHA256, MOMENT_REVISION), ("TS-ICL", tsicl, TSICL_SHA256, TSICL_REVISION)):
        from .frozen_active import active_environment
        environment = active_environment(name)
        ready = environment["available"] and path.is_file()
        if ready and _sha(path) != sha:
            ready = False
            reason = "local checkpoint SHA-256 mismatch"
        else:
            reason = ("verified local checkpoint and native active cuda_env" if ready else
                      environment["reason"] if not environment["available"] else
                      "verified local checkpoint required; run scripts/setup_server.sh; no implicit download")
        if name == "TS-ICL" and ready:
            license_file = path.parent / "LICENSE"
            if not license_file.is_file() or "TS-ICL NON-COMMERCIAL LICENSE" not in license_file.read_text():
                ready = False
                reason = "original TS-ICL noncommercial model license must accompany checkpoint"
        results[name].update(available=ready, status="available" if ready else "unavailable", reason=reason,
            environment_versions=environment["packages"], environment_policy="single_cuda_env",
            checkpoint={"path": str(path.relative_to(root)), "sha256": sha, "revision": revision})
    # These entries remain unavailable even if an unrelated package is installed.
    reasons = {"FIM": "temporal and pointwise checkpoint weight licenses unresolved",
        "TabPFN-TS": "audited R24 wrapper license and exact licensed checkpoint unresolved",
        "MoTM": "R24 INR mixture weights/config bundles and weight license unresolved",
        "CBiT": "usable author-linked implementation not verified; not reproduced"}
    for name, reason in reasons.items():
        results[name]["reason"] = reason
    return results


def training_windows(series, train_indices, context=512, stride=288, max_windows=None,cutoff_time=None):
    """Chronological windows wholly inside the supplied partition, no padding.

    The caller passes training indices for fitting and validation indices for
    early stopping. No naturally absent or late-arriving point becomes truth.
    A cap is optional, explicit and deterministically spread over the partition.
    """
    if context < 2 or stride < 1 or max_windows is not None and max_windows < 1:
        raise ValueError("positive context, stride and optional cap required")
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < context or len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("insufficient or unordered partition indices")
    domain = set(indices.tolist())
    cutoff = ((cutoff_time.timestamp() if hasattr(cutoff_time,'timestamp') else float(cutoff_time)) if cutoff_time is not None
              else series.records[int(indices[-1])].grid_time.timestamp() + series.nominal_seconds)
    admitted = np.asarray([r.value_canonical if r.observed_original and not r.natural_missing and r.available_to_method and r.availability_time.timestamp() < cutoff and r.measurement_time.timestamp()<cutoff else np.nan for r in series.records], np.float32)
    windows = []
    for start in range(int(indices[0]), int(indices[-1]) - context + 2, stride):
        end = start + context
        if not all(index in domain for index in range(start, end)):
            continue
        window = admitted[start:end]
        if np.isfinite(window).sum() >= 2:
            windows.append(window)
    if not windows:
        raise ValueError("no original-support windows inside the supplied partition")
    if max_windows is not None and len(windows) > max_windows:
        selection = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        windows = [windows[index] for index in selection]
    return np.asarray(windows, np.float32)[..., None]


train_windows = training_windows


def hpo_configurations(name):
    """Fixed six-trial grids declared before test evaluation (three lr × two sizes)."""
    if name not in LOCAL_METHODS:
        return []
    trials = []
    for width in (64, 128):
        for lr in (1e-3, 3e-4, 1e-4):
            if name == "SAITS":
                architecture = dict(n_layers=2, d_model=width, n_heads=4, d_k=width // 4, d_v=width // 4, d_ffn=width * 2, dropout=.1)
            elif name == "BRITS":
                architecture = dict(rnn_hidden_size=width)
            elif name == "CSDI":
                architecture = dict(n_layers=4, n_heads=4, n_channels=width, d_time_embedding=64, d_feature_embedding=16, d_diffusion_embedding=64, n_diffusion_steps=50, target_strategy="mix", schedule="quad")
            elif name == "ImputeFormer":
                architecture = dict(n_layers=2, d_input_embed=width // 2, d_learnable_embed=width // 2, d_proj=16, d_ffn=width * 2, n_temporal_heads=4, dropout=.1)
            else:
                # T1's official attention head count is preserved. Regularization,
                # rather than an invented d_model parameter, is the second axis.
                architecture = dict(drop_attn=0. if width == 64 else .1)
            trials.append(dict(learning_rate=lr, architecture=architecture))
    return trials


def _validate_windows(windows, context, *, label):
    x = np.asarray(windows, np.float32)
    if x.ndim != 3 or x.shape[1:] != (context, 1) or not len(x) or np.isinf(x).any() or (np.isfinite(x).sum(axis=(1, 2)) < 2).any():
        raise ValueError(label + " must have original support in every [N,L,1] window")
    return x


def fixed_validation_set(windows, mean, scale, seed=1001):
    """Fixed balanced point/block masks on original validation values only."""
    from .training import _mixed_mask
    x = np.asarray(windows, np.float32)
    normalized = (x - mean) / scale
    observed = np.isfinite(normalized)
    hidden = _mixed_mask(observed, np.random.default_rng(seed))
    return {"X": np.where(hidden, np.nan, normalized), "X_ori": normalized}, hidden


def _rng_state(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA epoch checkpoint cannot resume in a CPU runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(path, payload):
    import torch
    path = Path(path)
    # A torn publication of the new blob/sidecar retains the last valid epoch.
    prior_sidecar = Path(str(path) + ".sha256")
    if path.is_file() and prior_sidecar.is_file() and prior_sidecar.read_text().strip() == _sha(path):
        atomic_bytes(Path(str(path) + ".previous"), path.read_bytes())
        atomic_bytes(Path(str(path) + ".previous.sha256"), prior_sidecar.read_bytes())
    output = io.BytesIO()
    torch.save(payload, output)
    digest = atomic_bytes(Path(path), output.getvalue())
    atomic_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode())
    return digest


def _load_checkpoint(path):
    import torch
    path = Path(path)
    digest = Path(str(path) + ".sha256")
    if not digest.is_file() or digest.read_text().strip() != _sha(path):
        previous = Path(str(path) + ".previous")
        previous_digest = Path(str(path) + ".previous.sha256")
        if not previous.is_file() or not previous_digest.is_file() or previous_digest.read_text().strip() != _sha(previous):
            raise ValueError("completed/epoch model checksum missing or different")
        path = previous
    # Local, self-produced, checksum-verified optimizer/RNG artifacts only.
    return torch.load(path, map_location="cpu", weights_only=False)


class ConferenceModel:
    """Native trainable/frozen models in the active interpreter, with one query API."""

    def __init__(self, name, root, device="cpu", threads=1, seed=11, params=None, epochs=100, patience=10, batch_size=16, context=512, inference_batch_size=None, csdi_sample_batch_size=1, brits_execution="vectorized"):
        if name not in (*LOCAL_METHODS, *FROZEN_METHODS):
            raise ValueError("no verified scientific backend for " + name)
        if epochs < 1 or patience < 1 or batch_size < 1 or threads < 1:
            raise ValueError("positive training limits and thread count required")
        self.name, self.root, self.threads = name, Path(root).resolve(), int(threads)
        self.device = resolve_device(device, allow_missing_torch=name in FROZEN_METHODS)
        self.seed, self.context, self.epochs, self.patience, self.batch_size = int(seed), int(context), int(epochs), min(int(patience), int(epochs)), int(batch_size)
        inference_batch_size = self.batch_size if inference_batch_size is None else inference_batch_size
        if (isinstance(inference_batch_size, (bool, np.bool_))
                or not isinstance(inference_batch_size, (int, np.integer)) or inference_batch_size < 1):
            raise ValueError("inference_batch_size must be a positive integer")
        self.inference_batch_size = int(inference_batch_size)
        if (isinstance(csdi_sample_batch_size, (bool, np.bool_))
                or not isinstance(csdi_sample_batch_size, (int, np.integer)) or csdi_sample_batch_size < 1):
            raise ValueError("csdi_sample_batch_size must be a positive integer")
        self.csdi_sample_batch_size = int(csdi_sample_batch_size)
        from .brits_fast import EXECUTION_MODES
        if brits_execution not in EXECUTION_MODES:
            raise ValueError("BRITS execution must be native or vectorized (optionally vectorized_cuda_graph)")
        self.brits_execution = brits_execution
        self.params = deepcopy(params or {})
        self.history = []
        self.best_epoch, self.best_metric, self.completed_epochs = None, None, 0
        self.mean, self.scale = np.zeros(1, np.float32), np.ones(1, np.float32)
        self.adapter = None
        self.metadata = dict(name=name, device=self.device, threads=self.threads, precision="float32", local_training="required" if name in LOCAL_METHODS else "not_applicable",
            inference_policy="det4_and_MOMENT_batches_CSDI_TSICL_serial", configured_inference_batch_size=self.inference_batch_size,
            training_variant="native_model_epoch_runner_balanced_masks" if name in PYPOTS_METHODS else "official_T1_validation_epoch_schedule_adapted" if name == "T1" else "frozen_native_cuda_env",
            training_masks="per_epoch_dynamic_equal_probability_point20_or_block1_3_12_36;half_blocks_end_before_reconnection" if name in LOCAL_METHODS else "not_applicable",
            checkpoint_granularity="completed_epoch_optimizer_rng_best_state" if name in LOCAL_METHODS else "inference_shard")
        if name == "BRITS":
            self.set_brits_execution(self.brits_execution)
        if name == "CSDI":
            self.metadata.update(csdi_sample_batch_size=self.csdi_sample_batch_size,
                sampling_policy="native_replicated_condition_rows_v1" if self.csdi_sample_batch_size > 1
                else "native_outer_sample_loop")
        if name in LOCAL_METHODS:
            self._construct()

    def _construct(self):
        import torch
        from energygap.runtime.threads import configure_torch_threads
        configure_torch_threads(torch, self.threads)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.startswith("cuda:"):
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        architecture = self.params.get("architecture", {k: v for k, v in self.params.items() if k != "learning_rate"})
        if self.name in PYPOTS_METHODS:
            from .neural import PyPOTSAdapter, PARAMETERS
            from pypots.optim import Adam
            module = importlib.import_module("pypots.imputation")
            native_parameters = {**PARAMETERS[self.name], **architecture}
            native = getattr(module, self.name)(n_steps=self.context, n_features=1,
                epochs=self.epochs, patience=self.patience, batch_size=self.batch_size,
                optimizer=Adam(lr=self.params.get("learning_rate", .001)),
                num_workers=0, device=self.device, verbose=False, model_saving_strategy=None, **native_parameters)
            # Reuse audited native inference without the diagnostic constructor's
            # fixed batch size, default epochs or thread settings.
            self.adapter = PyPOTSAdapter.__new__(PyPOTSAdapter)
            for key, value in dict(name=self.name, context=self.context, n_features=1, seed=self.seed,
                device=self.device, model=native, parameters=native_parameters, timings={},
                runtime=torch_device_metadata(self.device)).items():
                setattr(self.adapter, key, value)
        else:
            from .t1 import T1Adapter
            self.adapter = T1Adapter(context=self.context, n_features=1, seed=self.seed,
                source_root=self.root / "artifacts/upstream/T1", device=self.device, **architecture)
            configure_torch_threads(torch, self.threads)

    def _configuration(self):
        return dict(name=self.name, seed=self.seed, context=self.context, epochs=self.epochs,
            patience=self.patience, batch_size=self.batch_size, params=self.params,
            inference_batch_size=self.inference_batch_size,
            csdi_sample_batch_size=self.csdi_sample_batch_size,
            brits_execution=self.brits_execution,
            device=self.device, threads=self.threads, torch_version=_version("torch"),
            pypots_version=_version("pypots"), training_variant=self.metadata["training_variant"])

    def fit_windows(self, windows, validation_windows=None, *, split="train", checkpoint_path=None, boundary_callback=None, selection_callback=None, progress_callback=None):
        if self.name in FROZEN_METHODS:
            return self
        if split != "train":
            raise ValueError("fit accepts training windows only")
        x = _validate_windows(windows, self.context, label="training")
        if validation_windows is None:
            raise ValueError("chronological validation windows required; never select epochs on test")
        val = _validate_windows(validation_windows, self.context, label="validation")
        self.mean = np.nanmean(x, axis=(0, 1))
        self.scale = np.nanstd(x, axis=(0, 1))
        self.scale[self.scale < 1e-8] = 1
        self.adapter.mean, self.adapter.scale = self.mean, self.scale
        selection_name = "validation_equal_duration_gap_nmae" if selection_callback else "native_validation_criterion_diagnostic"
        panel_policy = self.metadata.get("epoch_validation_policy") if selection_callback else None
        if selection_callback and panel_policy == "fixed_hash_stratified_epoch_validation_panel":
            selection_name = "fixed_panel_validation_equal_duration_gap_nmae"
        identity = fingerprint(dict(configuration=self._configuration(), train_sha256=hashlib.sha256(x.tobytes()).hexdigest(), validation_sha256=hashlib.sha256(val.tobytes()).hexdigest(), validation_mask_seed=1001, epoch_selection=selection_name,
            epoch_validation_policy=panel_policy,
            epoch_validation_panel_sha256=self.metadata.get("epoch_validation_panel_sha256") if selection_callback else None))
        resumed = None
        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            resumed = _load_checkpoint(checkpoint_path)
            if resumed.get("training_identity") != identity:
                raise ValueError("epoch checkpoint scientific configuration or data differ")
        validation, hidden = fixed_validation_set(val, self.mean, self.scale)
        started = perf_counter()
        if self.name in PYPOTS_METHODS:
            self._fit_pypots((x - self.mean) / self.scale, validation, identity, resumed, checkpoint_path, boundary_callback, selection_callback, progress_callback)
        else:
            self._fit_t1((x - self.mean) / self.scale, validation, hidden, identity, resumed, checkpoint_path, boundary_callback, selection_callback, progress_callback)
        synchronize(self.device)
        native_metric_name = "native_validation_criterion" if self.name in PYPOTS_METHODS else "masked_validation_MSE"
        self.metadata.update(fit_seconds=perf_counter() - started, best_epoch=self.best_epoch,
            best_validation_metric=self.best_metric, completed_epochs=self.completed_epochs,
            validation_mask_seed=1001, native_validation_mask_seed=1001,
            validation_mask_scope="native_diagnostic_only",
            validation_metric=selection_name if selection_callback else native_metric_name,
            native_validation_metric=native_metric_name,
            validation_selection_mask_policy="caller_prespecified_masks" if selection_callback else "fixed_native_diagnostic_seed_1001",
            epoch_selection=selection_name,
            train_windows=len(x), validation_windows=len(val), train_scale_only=True,
            exact_recovery_validated_device="CPU tests; CUDA requires server validation")
        return self

    def _fit_pypots(self, train, validation, identity, resumed, checkpoint_path, callback, selection_callback, progress_callback):
        import torch
        from .training import _mixed_mask
        owner = self
        native = self.adapter.model
        mask_rng = np.random.default_rng(self.seed)

        def epoch_loop(model, train_loader, val_loader):
            # Native fit constructs the correct model-specific datasets and then
            # calls this loop. Dataset corruption and gradient losses are native.
            if val_loader is None:
                raise ValueError("validation loader is required")
            start_epoch, bad_epochs = 0, 0
            best_state, best_metric, best_epoch, history = None, math.inf, None, []
            if resumed:
                model.model.load_state_dict(resumed["model_state"])
                model.optimizer.load_state_dict(resumed["optimizer_state"])
                start_epoch, bad_epochs = resumed["epoch"], resumed["bad_epochs"]
                best_state, best_metric, best_epoch, history = resumed["best_state"], resumed["best_metric"], resumed["best_epoch"], resumed["history"]
                mask_rng.bit_generator.state = resumed["mask_rng"]
                _restore_rng(torch, resumed["rng"])
            for epoch in range(start_epoch + 1, owner.epochs + 1):
                if bad_epochs >= owner.patience:
                    break
                epoch_started = perf_counter()
                if progress_callback:
                    progress_callback(dict(stage="epoch_start", epoch=epoch, max_epochs=owner.epochs,
                        patience=owner.patience, bad_epochs=bad_epochs))
                model.model.train()
                training_losses = []
                # Use the native model-specific dataset transforms (including
                # BRITS deltas and CSDI side-information), with an explicit
                # protocol mask. SAITS/ImputeFormer receive original targets;
                # BRITS retains its original received-value reconstruction loss.
                hidden = _mixed_mask(np.isfinite(train), mask_rng)
                epoch_data = {"X": np.where(hidden, np.nan, train), "X_ori": train}
                if owner.name == "BRITS":
                    if owner.brits_execution == "native":
                        from pypots.imputation.brits.data import DatasetForBRITS
                        dataset = DatasetForBRITS({"X": epoch_data["X"]}, return_X_ori=False, return_y=False)
                    else:
                        from .prepared_inference import BritsInputCache
                        # Same seven tensors and same shuffled DataLoader/RNG;
                        # only deterministic delta preparation is vectorized.
                        # Dynamic training masks are never cached across epochs.
                        prepared, _ = BritsInputCache(max_bytes=0).prepare(epoch_data["X"])
                        dataset = torch.utils.data.TensorDataset(*prepared)
                elif owner.name == "CSDI":
                    from pypots.imputation.csdi.data import DatasetForCSDI
                    dataset = DatasetForCSDI(epoch_data, model.target_strategy, return_X_ori=True)
                else:
                    from pypots.imputation.saits.data import DatasetForSAITS
                    dataset = DatasetForSAITS(epoch_data, return_X_ori=True, return_y=False)
                epoch_loader = torch.utils.data.DataLoader(dataset, batch_size=owner.batch_size,
                    shuffle=True, num_workers=0)
                last_progress = perf_counter()
                for batch_index, data in enumerate(epoch_loader, 1):
                    inputs = model._assemble_input_for_training(data)
                    model.optimizer.zero_grad()
                    loss = owner._model_forward(inputs, calc_criterion=True)["loss"].sum()
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite native training loss")
                    loss.backward()
                    model.optimizer.step()
                    training_losses.append(float(loss.detach()))
                    if progress_callback and (perf_counter() - last_progress >= 1. or batch_index == len(epoch_loader)):
                        progress_callback(dict(stage="training", epoch=epoch, max_epochs=owner.epochs,
                            batches_done=batch_index, batches_total=len(epoch_loader)))
                        last_progress = perf_counter()
                model.model.eval()
                validation_losses = []
                with torch.no_grad():
                    for data in val_loader:
                        inputs = model._assemble_input_for_validating(data)
                        metric = owner._model_forward(inputs, calc_criterion=True)["metric"].sum()
                        validation_losses.append(float(metric.detach()))
                native_metric = float(np.mean(validation_losses))
                if not math.isfinite(native_metric):
                    raise RuntimeError("nonfinite native validation criterion")
                metric = owner._selection_value(selection_callback, epoch, native_metric)
                improved = metric < best_metric
                if improved:
                    best_metric, best_epoch, bad_epochs = metric, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
                else:
                    bad_epochs += 1
                history.append(dict(epoch=epoch, training_loss=float(np.mean(training_losses)), native_validation_metric=native_metric,
                    validation_metric=metric, training_mask_sha256=hashlib.sha256(hidden.tobytes()).hexdigest(), artificially_hidden_original=int(hidden.sum())))
                synchronize(owner.device)
                payload = dict(schema_version=1, training_identity=identity, epoch=epoch, bad_epochs=bad_epochs,
                    best_state=best_state, best_metric=best_metric, best_epoch=best_epoch, history=history,
                    model_state=model.model.state_dict(), optimizer_state=model.optimizer.state_dict(), rng=_rng_state(torch), mask_rng=mask_rng.bit_generator.state)
                if checkpoint_path is not None:
                    _save_checkpoint(checkpoint_path, payload)
                owner.history, owner.best_epoch, owner.best_metric, owner.completed_epochs = history, best_epoch, best_metric, epoch
                if callback:
                    callback(dict(stage="epoch_complete", epoch=epoch, max_epochs=owner.epochs,
                        patience=owner.patience, bad_epochs=bad_epochs, epoch_seconds=perf_counter()-epoch_started,
                        best_epoch=best_epoch, best_metric=best_metric, history=history,
                        checkpoint_path=str(checkpoint_path) if checkpoint_path else None))
            if best_state is None:
                raise RuntimeError("no validated native epoch completed")
            model.best_model_dict, model.best_loss, model.best_epoch = best_state, best_metric, best_epoch
            owner.history, owner.best_epoch, owner.best_metric = history, best_epoch, best_metric
            owner.completed_epochs = history[-1]["epoch"]

        native._train_model = MethodType(epoch_loop, native)
        try:
            native.fit({"X": train}, val_set=validation)
        finally:
            del native._train_model

    def _selection_value(self, selection_callback, epoch, native_metric):
        """Validation sampling never perturbs subsequent training RNG streams."""
        if selection_callback is None:
            return native_metric
        import torch
        before = _rng_state(torch)
        try:
            selected = float(selection_callback(self, epoch))
        finally:
            _restore_rng(torch, before)
        if not math.isfinite(selected):
            raise RuntimeError("nonfinite gap-macro validation selection metric")
        return selected

    def _model_forward(self, inputs, *, calc_criterion=False):
        """Execute the same native weights; the scalar API remains upstream."""
        core = self.adapter.model.model
        if self.name == "BRITS" and self.brits_execution != "native":
            from .brits_fast import forward
            if self.brits_execution == "vectorized_cuda_graph" and not core.training and not calc_criterion:
                from .brits_fast import InferenceGraph
                if not hasattr(self, "_brits_inference_graph"):
                    self._brits_inference_graph = InferenceGraph()
                output, self._last_brits_graph_metadata = self._brits_inference_graph.run(core, inputs)
                return output
            return forward(core, inputs, calc_criterion=calc_criterion)
        return core(inputs, calc_criterion=calc_criterion)

    def set_brits_execution(self, mode):
        """Select an explicit implementation for diagnostics or a new run.

        Scientific runs save this mode in their immutable configuration. Changing
        it is never a way to resume an epoch checkpoint under another policy.
        """
        from .brits_fast import EXECUTION_MODES, POLICY
        if mode not in EXECUTION_MODES:
            raise ValueError("BRITS execution must be native or vectorized (optionally vectorized_cuda_graph)")
        if hasattr(self, "_brits_inference_graph"):
            self._brits_inference_graph.clear()
            del self._brits_inference_graph
        self.__dict__.pop("_last_brits_graph_metadata", None)
        self.brits_execution = mode
        if self.name == "BRITS":
            self.metadata.update(brits_execution=mode,
                brits_execution_policy=POLICY if mode != "native" else "native_pypots_forward",
                brits_loss_policy=("all_native_losses_during_training_and_criterion_validation;loss_free_inference_only"
                                   if mode != "native" else "native_all_losses"),
                brits_cuda_graph_requested=mode == "vectorized_cuda_graph",
                brits_training_cuda_graph=False)
        return self

    def _fit_t1(self, train, validation, hidden, identity, resumed, checkpoint_path, callback, selection_callback, progress_callback):
        import torch
        from .training import _mixed_mask
        model = self.adapter.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.get("learning_rate", .001))
        rng = np.random.default_rng(self.seed)
        original = np.isfinite(train)
        start_epoch, bad_epochs, best_metric, best_state, best_epoch, history = 0, 0, math.inf, None, None, []
        if resumed:
            model.load_state_dict(resumed["model_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            start_epoch, bad_epochs = resumed["epoch"], resumed["bad_epochs"]
            best_metric, best_state, best_epoch, history = resumed["best_metric"], resumed["best_state"], resumed["best_epoch"], resumed["history"]
            rng.bit_generator.state = resumed["sampler_rng"]
            _restore_rng(torch, resumed["rng"])
        for epoch in range(start_epoch + 1, self.epochs + 1):
            if bad_epochs >= self.patience:
                break
            epoch_started = perf_counter()
            if progress_callback:
                progress_callback(dict(stage="epoch_start", epoch=epoch, max_epochs=self.epochs,
                    patience=self.patience, bad_epochs=bad_epochs))
            model.train()
            losses = []
            last_progress = perf_counter()
            for offset in range(0, len(train), self.batch_size):
                if offset == 0:
                    order = rng.permutation(len(train))
                indices = order[offset:offset + self.batch_size]
                observed = original[indices]
                masked = _mixed_mask(observed, rng)
                visible = observed & ~masked
                truth = torch.as_tensor(np.where(observed, train[indices], 0), device=self.device)
                inputs = torch.as_tensor(np.where(visible, train[indices], 0), device=self.device)
                mask = torch.as_tensor(visible.astype(np.float32), device=self.device)
                targets = torch.as_tensor(masked, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs, None, None, None, mask)
                loss = torch.nn.functional.mse_loss(output[targets], truth[targets])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite official T1 MIT loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if progress_callback and (perf_counter() - last_progress >= 1. or offset + self.batch_size >= len(train)):
                    progress_callback(dict(stage="training", epoch=epoch, max_epochs=self.epochs,
                        batches_done=math.ceil((offset+self.batch_size)/self.batch_size),
                        batches_total=math.ceil(len(train)/self.batch_size)))
                    last_progress = perf_counter()
            model.eval()
            squared, count = 0., 0
            with torch.no_grad():
                for offset in range(0, len(validation["X"]), self.batch_size):
                    x = validation["X"][offset:offset + self.batch_size]
                    truth = validation["X_ori"][offset:offset + self.batch_size]
                    target = hidden[offset:offset + self.batch_size]
                    visible = np.isfinite(x)
                    pred = model(torch.as_tensor(np.where(visible, x, 0), device=self.device), None, None, None,
                        torch.as_tensor(visible.astype(np.float32), device=self.device)).cpu().numpy()
                    squared += float(np.sum((pred[target] - truth[target]) ** 2))
                    count += int(target.sum())
            native_metric = squared / count if count else math.nan
            if not math.isfinite(native_metric):
                raise RuntimeError("nonfinite T1 validation MSE")
            metric = self._selection_value(selection_callback, epoch, native_metric)
            if metric < best_metric:
                best_metric, best_epoch, bad_epochs = metric, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), native_validation_metric=native_metric, validation_metric=metric))
            synchronize(self.device)
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path, dict(schema_version=1, training_identity=identity, epoch=epoch, bad_epochs=bad_epochs,
                    best_state=best_state, best_metric=best_metric, best_epoch=best_epoch, history=history,
                    model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), rng=_rng_state(torch), sampler_rng=rng.bit_generator.state))
            self.history, self.best_epoch, self.best_metric, self.completed_epochs = history, best_epoch, best_metric, epoch
            if callback:
                callback(dict(stage="epoch_complete", epoch=epoch, max_epochs=self.epochs,
                    patience=self.patience, bad_epochs=bad_epochs, epoch_seconds=perf_counter()-epoch_started,
                    best_epoch=best_epoch, best_metric=best_metric, history=history,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None))
        if best_state is None:
            raise RuntimeError("no validated T1 epoch completed")
        model.load_state_dict(best_state)
        model.eval()
        self.history, self.best_epoch, self.best_metric = history, best_epoch, best_metric
        self.completed_epochs = history[-1]["epoch"]
        self.adapter.metadata.update(training_schedule="validation_selected_full_epochs_adapted", loss="official_T1_MIT_MSE",
            upstream_deviations=["chronological validation epoch early stopping", "balanced point/block training masks", "train-only channel scaling"],
            best_epoch=best_epoch, completed_epochs=self.completed_epochs)

    def _infer_csdi_sample_batches(self, safe, *, task, samples, sample_seed):
        """Use native independent condition rows to parallelize diffusion draws.

        Every row is one full native diffusion trajectory under the same context.
        The query seed is set once. Grouping changes RNG traversal relative to
        the native outer sample loop, and is part of the saved configuration.
        """
        import torch
        if task not in ("R", "O"):
            raise ValueError("bidirectional CSDI unsupported for causal task C")
        if (isinstance(samples, (bool, np.bool_))
                or not isinstance(samples, (int, np.integer)) or samples < 1):
            raise ValueError("CSDI samples must be a positive integer")
        x = np.asarray(safe, float)
        if x.shape != (self.context,) or np.isinf(x).any():
            raise ValueError("CSDI requires one sanitized uncompressed univariate context")
        x = x[:, None]
        normalized = ((x - self.adapter.mean) / self.adapter.scale).astype(np.float32)
        torch.manual_seed(sample_seed)
        if self.device.startswith("cuda:"):
            torch.cuda.manual_seed_all(sample_seed)
        native = self.adapter.model
        native.model.eval()
        previous_batch_size = native.batch_size
        synchronize(self.device)
        started = perf_counter()
        chunks, sizes = [], []
        try:
            for offset in range(0, int(samples), self.csdi_sample_batch_size):
                count = min(self.csdi_sample_batch_size, int(samples) - offset)
                native.batch_size = count
                repeated = np.repeat(normalized[None], count, axis=0)
                with torch.no_grad():
                    raw = native.predict({"X": repeated}, n_sampling_times=1)["imputation"]
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                raw = np.asarray(raw)
                if raw.shape != (count, 1, self.context, 1):
                    raise ValueError("unexpected native CSDI condition-row sample shape: " + str(raw.shape))
                chunks.append(raw[:, 0])
                sizes.append(count)
        finally:
            native.batch_size = previous_batch_size
        draws = np.concatenate(chunks, axis=0) * self.adapter.scale + self.adapter.mean
        draws = np.where(np.isfinite(x)[None], x[None], draws)
        if draws.shape != (samples, self.context, 1) or not np.isfinite(draws).all():
            raise ValueError("native CSDI returned nonfinite/incomplete joint trajectories")
        point = np.median(draws, axis=0)
        synchronize(self.device)
        elapsed = perf_counter() - started
        return dict(point_estimate=point, samples=draws, quantiles=None, status="success",
            timing=dict(inference_seconds=elapsed, timing_mode="query_wall_mc_batched",
                sample_batch_sizes=sizes, sample_batch_size=self.csdi_sample_batch_size,
                n_joint_samples=int(samples)),
            model_metadata=dict(implementation="pypots", name=self.name, parameters=self.adapter.parameters,
                device=self.device, precision="float32", runtime=self.adapter.runtime,
                sampling_policy="native_replicated_condition_rows_v1",
                csdi_sample_batch_size=self.csdi_sample_batch_size, n_joint_samples=int(samples),
                diffusion_steps=native.model.backbone.n_diffusion_steps,
                sampling_rng_policy="query_seed_once_native_rng_over_fixed_condition_row_chunks",
                sampling_compatibility="same_native_conditional_distribution; not_bitwise_old_outer_loop",
                joint_samples=True))

    def infer_query(self, query, samples=100, sample_seed=0):
        if self.name in FROZEN_METHODS:
            return self.infer_queries([query], samples=samples, sample_seed=sample_seed)[0]
        visible = np.asarray(query.available_to_method, bool) & ~np.asarray(query.padding_mask, bool)
        safe = np.where(visible, np.asarray(query.values, float), np.nan)
        if self.name == "T1":
            result = self.adapter.infer(safe, task=query.task)
        elif self.name == "CSDI" and self.csdi_sample_batch_size > 1:
            result = self._infer_csdi_sample_batches(safe, task=query.task, samples=samples, sample_seed=sample_seed)
        else:
            result = self.adapter.infer(safe, task=query.task, samples=samples, sample_seed=sample_seed)
        result["point_estimate"] = np.asarray(result["point_estimate"]).reshape(self.context)
        if result.get("samples") is not None:
            result["samples"] = np.asarray(result["samples"]).reshape(samples, self.context)
        result["model_metadata"] = {**result.get("model_metadata", {}), **self.metadata,
            "inference_batch_size": 1}
        return result

    def _infer_deterministic_batch(self, queries):
        """Native batch forward with unchanged per-window masks/scaling.

        ``inference_seconds`` is the group's synchronized native wall time
        divided by its query count. It measures amortized throughput cost, not
        the response latency of an independently submitted individual query.
        """
        if self.name == "MOMENT-1-base":
            from .frozen_active import ActiveFrozenModel
            if self.adapter is None:
                self.adapter = ActiveFrozenModel(self.name, self.root, device=self.device, threads=self.threads)
            outputs = self.adapter.infer_batch(queries)
            for result in outputs:
                result["model_metadata"] = {**self.metadata, **result.get("model_metadata", {})}
            return outputs
        import torch
        values, masks = [], []
        for query in queries:
            if query.task not in ("R", "O"):
                raise ValueError("bidirectional native batch unsupported for causal task C")
            x = np.asarray(query.values, float)
            available = np.asarray(query.available_to_method, bool)
            padding = np.asarray(query.padding_mask, bool)
            if any(array.shape != (self.context,) for array in (x, available, padding)):
                raise ValueError("matching uncompressed univariate query shapes required")
            admitted = available & ~padding
            safe = np.where(admitted, x, np.nan)
            if np.isinf(safe).any():
                raise ValueError("invalid infinite admitted input")
            values.append(safe)
            masks.append(np.isfinite(safe))
        safe = np.asarray(values, float)[..., None]
        observed = np.asarray(masks, bool)[..., None]
        normalized = ((safe - self.adapter.mean) / self.adapter.scale).astype(np.float32)
        synchronize(self.device)
        started = perf_counter()
        if self.name == "T1":
            self.adapter.model.eval()
            with torch.no_grad():
                raw = self.adapter.model(torch.as_tensor(np.where(observed, normalized, 0), device=self.device),
                    None, None, None, torch.as_tensor(observed.astype(np.float32), device=self.device))
                raw = raw.detach().cpu().numpy()
            metadata = dict(self.adapter.metadata)
        else:
            native = self.adapter.model
            native.model.eval()
            prepared_metadata = {}
            if self.name == "BRITS":
                from .prepared_inference import BRITS_INPUT_CACHE
                data, prepared_metadata = BRITS_INPUT_CACHE.prepare(normalized)
                # Keep native placement, input assembly and architecture. The
                # selected BRITS policy either calls the native core or uses its
                # same weights with vectorized gates and loss-free inference.
                # infer_query remains the independent upstream scalar reference.
                with torch.no_grad():
                    inputs = native._assemble_input_for_testing(data)
                    raw = self._model_forward(inputs)["imputation"]
            else:
                # Other models retain upstream datasets and channel transforms.
                previous_batch_size = native.batch_size
                native.batch_size = len(queries)
                try:
                    with torch.no_grad():
                        raw = native.predict({"X": normalized})["imputation"]
                finally:
                    native.batch_size = previous_batch_size
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            metadata = dict(implementation="pypots", name=self.name, parameters=self.adapter.parameters,
                training_resume=False, device=self.device, precision="float32", runtime=self.adapter.runtime,
                **prepared_metadata)
            if self.name == "BRITS":
                metadata["inference_path"] = ("brits_vectorized_decay_gates_loss_free"
                    if self.brits_execution != "native" else "brits_native_forward")
                if self.brits_execution == "vectorized_cuda_graph":
                    metadata.update(getattr(self, "_last_brits_graph_metadata", {}))
        raw = np.asarray(raw)
        if raw.shape != (len(queries), self.context, 1):
            raise ValueError("unexpected native deterministic batch output shape: " + str(raw.shape))
        points = raw * self.adapter.scale + self.adapter.mean
        points = np.where(observed, safe, points)
        if not np.isfinite(points).all():
            raise ValueError("native deterministic batch returned nonfinite output")
        synchronize(self.device)
        elapsed = perf_counter() - started
        policy = dict(inference_batching="native_deterministic_batch_v1",
            inference_batch_size=len(queries), sampling="not_applicable_deterministic",
            batch_independence="per_window_native_mask_and_train_only_channel_scaler",
            batching_precision="float32_native_no_autocast")
        metadata = {**metadata, **self.metadata, **policy}
        return [dict(point_estimate=point.reshape(self.context), samples=None, quantiles=None,
            status="success", timing=dict(inference_seconds=elapsed / len(queries),
                timing_mode="batch_amortized", batch_size=len(queries), batch_wall_seconds=elapsed),
            model_metadata=dict(metadata)) for point in points]

    def infer_queries(self, queries, samples=100, sample_seed=0, sample_seeds=None, *, batch_size=None):
        if sample_seeds is None:
            sample_seeds = [sample_seed + index for index in range(len(queries))]
        if len(sample_seeds) != len(queries):
            raise ValueError("one deterministic sampling seed per query required")
        effective_batch_size = self.inference_batch_size if batch_size is None else batch_size
        if (isinstance(effective_batch_size, (bool, np.bool_))
                or not isinstance(effective_batch_size, (int, np.integer)) or effective_batch_size < 1):
            raise ValueError("inference batch_size must be a positive integer")
        if self.name in ("BRITS", "SAITS", "ImputeFormer", "T1", "MOMENT-1-base"):
            import torch
            results = []
            offset = 0
            adaptive_limit = min(int(effective_batch_size),
                                 getattr(self, "_cuda_inference_batch_limit", int(effective_batch_size)))
            retries, retry_seconds = 0, 0.
            while offset < len(queries):
                count = min(adaptive_limit, len(queries) - offset)
                started = perf_counter()
                oom = False
                try:
                    batch = self._infer_deterministic_batch(queries[offset:offset + count])
                except RuntimeError as error:
                    # Adapt only CUDA allocation failures. Malformed inputs and
                    # model errors retain the ordinary explicit failure path.
                    cuda_oom = (str(self.device).startswith("cuda:") and
                        (isinstance(error, torch.cuda.OutOfMemoryError)
                         or "cuda out of memory" in str(error).lower()))
                    if not cuda_oom or count == 1:
                        raise
                    oom = True
                    retries += 1
                    retry_seconds += perf_counter() - started
                if oom:
                    # Leave the exception block first so its traceback releases
                    # tensors before reclaiming CUDA's unused cached blocks.
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    adaptive_limit = max(1, count // 2)
                    self._cuda_inference_batch_limit = adaptive_limit
                    LOG.warning("%s CUDA OOM for inference batch %d; retrying the same queries in batches of %d (requested %d)",
                                self.name, count, adaptive_limit, effective_batch_size)
                    continue
                for result in batch:
                    result["model_metadata"].update(configured_inference_batch_size=int(effective_batch_size),
                        adaptive_batch_size=adaptive_limit, oom_retries=retries)
                    result["timing"].update(requested_batch_size=int(effective_batch_size),
                        adaptive_batch_size=adaptive_limit, oom_retries=retries,
                        oom_retry_seconds=retry_seconds,
                        oom_retry_timing_policy="cumulative_failed_attempts_per_infer_queries_call;not_additive_across_rows")
                results.extend(batch)
                offset += count
            return results
        if self.name not in FROZEN_METHODS:
            # CSDI retains one native stochastic query at a time. Its explicit
            # per-query seeds and sample trajectories do not depend on grouping.
            return [self.infer_query(query, samples=samples, sample_seed=int(seed)) for seed, query in zip(sample_seeds, queries)]
        if not queries:
            return []
        from .frozen_active import ActiveFrozenModel
        if self.adapter is None:
            self.adapter = ActiveFrozenModel(self.name, self.root, device=self.device, threads=self.threads)
        started = perf_counter()
        results = []
        for query in queries:
            result = self.adapter.infer(query)
            result["model_metadata"] = {**self.metadata, **result.get("model_metadata", {})}
            result["model_metadata"]["inference_batch_size"] = 1
            results.append(result)
        shard_seconds = perf_counter() - started
        for result in results:
            result["timing"]["shard_wall_seconds"] = shard_seconds
        return results

    def save_fitted(self, path):
        if self.name in FROZEN_METHODS:
            atomic_json(path, dict(schema_version=1, frozen=True, configuration=self._configuration(), metadata=self.metadata))
            return dict(path=str(path), sha256=_sha(path), local_training="not_applicable")
        model = self.adapter.model.model if self.name in PYPOTS_METHODS else self.adapter.model
        digest = _save_checkpoint(path, dict(schema_version=1, frozen=False, configuration=self._configuration(),
            model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}, mean=self.mean, scale=self.scale,
            history=self.history, best_epoch=self.best_epoch, best_metric=self.best_metric,
            completed_epochs=self.completed_epochs, metadata=self.metadata))
        return dict(path=str(path), sha256=digest, training_resume="epoch_checkpoint_separate", state_scope="completed_fit_best_validation_epoch")


def create_adapter(name, root=Path("."), device="cpu", threads=1, seed=11, params=None, epochs=100, patience=10, batch_size=16, context=512, inference_batch_size=None, csdi_sample_batch_size=1, brits_execution="vectorized"):
    return ConferenceModel(name, root, device=device, threads=threads, seed=seed, params=params,
        epochs=epochs, patience=patience, batch_size=batch_size, context=context,
        inference_batch_size=inference_batch_size, csdi_sample_batch_size=csdi_sample_batch_size,
        brits_execution=brits_execution)


def load_adapter(path, root=Path("."), device="cpu", threads=1):
    path = Path(path)
    with path.open("rb") as stream:
        is_descriptor = stream.read(1) == b"{"
    if is_descriptor:
        state = json.loads(path.read_text())
    else:
        state = _load_checkpoint(path)
    config = dict(state["configuration"])
    for key in ("torch_version", "pypots_version", "training_variant"):
        expected = config.pop(key, None)
        if not state["frozen"] and key.endswith("version") and expected != _version(key.removesuffix("_version")):
            raise ValueError("completed fitted model package version differs: " + key)
    config.update(root=root, device=device, threads=threads)
    model = ConferenceModel(**config)
    current_metadata = model.metadata
    model.metadata = {**current_metadata, **state["metadata"], "device": model.device, "threads": threads,
                      "inference_policy": current_metadata["inference_policy"]}
    if model.name == "BRITS":
        model.set_brits_execution(model.brits_execution)
    if not state["frozen"]:
        target = model.adapter.model.model if model.name in PYPOTS_METHODS else model.adapter.model
        target.load_state_dict(state["model_state"])
        target.eval()
        model.mean, model.scale = state["mean"], state["scale"]
        model.adapter.mean, model.adapter.scale = model.mean, model.scale
        model.history, model.best_epoch, model.best_metric, model.completed_epochs = state["history"], state["best_epoch"], state["best_metric"], state["completed_epochs"]
    return model



"""Native frozen models in the active interpreter, with local verified weights.

Only sanitized observations and their availability contract reach upstream code.
The returned TS-ICL quantiles are marginal; they are not joint sample paths.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from energygap.runtime.device import resolve_device, synchronize, torch_device_metadata
from energygap.runtime.threads import configure_torch_threads

TSICL_REVISION = "19c94031439fb31f36ce395088ee50a6762d3774"
TSICL_SHA256 = "a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431"
TSICL_SOURCE_COMMIT = "6d4861e9eb29d59f05a0c253432afb2ae936e02a"
TSICL_PIPELINE_BLOB = "acd8c53f627e765429ba812d4ce14791c407c14d"
MOMENT_SOURCE_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
LEVELS = np.asarray([.05, .1, .25, .5, .75, .9, .95])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_environment(name):
    """Cheap metadata check; no Torch/model import, network, or environment switch."""
    from packaging.specifiers import SpecifierSet
    required = {"torch": ">=2.5.1,<2.10", "numpy": "==2.2.6"}
    if name == "MOMENT-1-base":
        required.update(momentfm="==0.1.5", transformers="==4.57.6",
                        **{"huggingface-hub": "==0.34.4"})
    elif name == "TS-ICL":
        required.update(tsicl="==0.2.0", **{"huggingface-hub": "==0.34.4"})
    else:
        raise ValueError("unsupported native frozen model: " + str(name))
    versions, problems = {}, []
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        problems.append("Python 3.12 or 3.13 required")
    for package, spec in required.items():
        try:
            actual = versions[package] = version(package)
        except PackageNotFoundError:
            problems.append(package + spec + " missing")
        else:
            if actual not in SpecifierSet(spec):
                problems.append(package + "==" + actual + " does not satisfy " + spec)
    return dict(available=not problems, reason="; ".join(problems) if problems else
                "native packages in active cuda_env", packages=versions,
                executable=sys.executable, environment_policy="single_cuda_env")


class ActiveFrozenModel:
    """Load one upstream frozen model lazily and reuse it across query shards."""

    def __init__(self, name, root, device="cpu", threads=1):
        self.name, self.root, self.threads = name, Path(root).resolve(), int(threads)
        self.environment = active_environment(name)
        if not self.environment["available"]:
            raise RuntimeError("incompatible active cuda_env: " + self.environment["reason"])
        self.device = resolve_device(device, allow_missing_torch=False)
        self.native, self.load_seconds = None, 0.
        self.checkpoint_metadata = {}

    def _load(self):
        if self.native is not None:
            return
        import torch
        configure_torch_threads(torch, self.threads)
        torch.use_deterministic_algorithms(True)
        started = perf_counter()
        if self.name == "MOMENT-1-base":
            from .foundation import MOMENTFrozen, MOMENT_REVISION, MOMENT_SHA256
            checkpoint = (self.root / "artifacts/checkpoints/hf/models--AutonLab--MOMENT-1-base"
                          / "snapshots" / MOMENT_REVISION / "model.safetensors")
            self._verify_checkpoint(checkpoint, MOMENT_SHA256)
            self.checkpoint_metadata = dict(revision=MOMENT_REVISION, sha256=MOMENT_SHA256)
            native = MOMENTFrozen(self.root / "artifacts/checkpoints/hf", device=self.device,
                                  allow_network=False)
            native._load()
        else:
            from tsicl import TSICL
            import inspect
            source = Path(inspect.getfile(TSICL)).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
            if blob != TSICL_PIPELINE_BLOB:
                raise RuntimeError("native TS-ICL pipeline differs from the audited official source")
            checkpoint = self.root / "artifacts/checkpoints/tsicl/tsicl-v1.ckpt"
            self._verify_checkpoint(checkpoint, TSICL_SHA256)
            self.checkpoint_metadata = dict(revision=TSICL_REVISION, sha256=TSICL_SHA256)
            license_file = checkpoint.parent / "LICENSE"
            if not license_file.is_file() or "TS-ICL NON-COMMERCIAL LICENSE" not in license_file.read_text():
                raise ValueError("original TS-ICL noncommercial license must accompany local checkpoint")
            # Upstream permits random initialization when a component is absent.
            # Require the actual pretrained imputer before constructing that API.
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "imputer" not in state or "config" not in state:
                raise ValueError("TS-ICL checkpoint lacks pretrained imputer/config")
            del state
            native = TSICL(model_path=checkpoint, allow_auto_download=False)
        synchronize(self.device)
        self.load_seconds = perf_counter() - started
        self.native = native

    @staticmethod
    def _verify_checkpoint(path, expected):
        if not path.is_file():
            raise FileNotFoundError("verified local checkpoint required; automatic download disabled: " + str(path))
        if sha256_file(path) != expected:
            raise ValueError("frozen checkpoint SHA-256 mismatch: " + str(path))

    @staticmethod
    def _prepare_query(query):
        """Validate the availability contract before any native model call."""
        if query.task != "R":
            raise ValueError("scientific frozen adapters support audited task R only")
        values, times = np.asarray(query.values, float), np.asarray(query.times, float)
        padding = np.asarray(query.padding_mask, bool)
        observed = np.asarray(query.available_to_method, bool) & ~padding
        if any(a.shape != (512,) for a in (values, times, padding, observed)):
            raise ValueError("native frozen adapters require matching uncompressed L512 inputs")
        issuance = query.issuance_time
        issue = float(issuance.timestamp() if hasattr(issuance, "timestamp") else issuance)
        if (not np.isfinite(issue) or not np.isfinite(times).all()
                or not np.allclose(np.diff(times), 300, atol=1e-6, rtol=0)
                or np.any(observed & (times > issue))):
            raise ValueError("invalid five-minute grid or reconnection availability")
        if observed.sum() < 2 or not np.isfinite(values[observed]).all():
            raise ValueError("finite admitted observations required")
        # This is the entire upstream input boundary: hidden/padding placeholders
        # and evaluator-owned truth are never forwarded to the native model.
        safe = np.where(observed, values, np.nan)
        return values, times, padding, observed, safe

    def infer(self, query):
        values, times, padding, observed, safe = self._prepare_query(query)
        self._load()
        import torch
        synchronize(self.device)
        started = perf_counter()
        if self.name == "MOMENT-1-base":
            point = self.native.impute(safe, times, task="R", observed_mask=observed,
                                       padding_mask=padding)
            quantiles, levels = None, None
        else:
            with torch.inference_mode():
                point, quantiles = self.native.impute(inputs=safe.copy(), covars=None,
                    batch_size=1, quantile_levels=LEVELS.tolist(), device=torch.device(self.device),
                    denormalize=True, point_estimator="median", allow_auto_complete=False,
                    replace_by_gt=False, squeeze_output=True)
            point = point.detach().cpu().numpy().astype(float)
            quantiles = quantiles.detach().cpu().numpy().astype(float)
            if point.shape != (512,) or quantiles.shape != (512, len(LEVELS)):
                raise ValueError("native TS-ICL output shape mismatch")
            point = np.where(observed, safe, point)
            quantiles = np.where(observed[:, None], safe[:, None], quantiles)
            point[padding], quantiles[padding] = np.nan, np.nan
            levels = LEVELS.copy()
        synchronize(self.device)
        elapsed = perf_counter() - started
        if (not np.isfinite(point[~padding]).all()
                or quantiles is not None and not np.isfinite(quantiles[~padding]).all()):
            raise ValueError("native frozen model returned nonfinite nonpadding output")
        if not np.array_equal(point[observed], values[observed]):
            raise ValueError("native frozen model changed admitted observations")
        return dict(point_estimate=point, samples=None, quantiles=quantiles, quantile_levels=levels,
            status="success", timing=dict(inference_seconds=elapsed, shared_load_seconds=self.load_seconds),
            model_metadata=self._metadata(quantiles is not None))

    def _metadata(self, has_quantiles=False):
        return dict(name=self.name, device=self.device, runtime=torch_device_metadata(self.device),
            threads=self.threads, local_training="not_applicable", joint_samples=False,
            environment_policy="single_cuda_env", executable=sys.executable,
            process_id=os.getpid(),
            package_versions=self.environment["packages"], dependency_override=False,
            checkpoint=self.checkpoint_metadata,
            source_commit=MOMENT_SOURCE_COMMIT if self.name == "MOMENT-1-base" else TSICL_SOURCE_COMMIT,
            native_pipeline_git_blob=TSICL_PIPELINE_BLOB if self.name == "TS-ICL" else None,
            capability="marginal_quantiles" if has_quantiles else "point_estimate")

    def infer_batch(self, queries):
        """MOMENT native batching; TS-ICL keeps its separately audited scalar API.

        CUDA allocation retries are owned by ConferenceModel so every failed
        batch is retried with the same sanitized windows in their original order.
        """
        if self.name != "MOMENT-1-base":
            raise ValueError("native frozen batching is audited for MOMENT only")
        if not queries:
            return []
        rows = [self._prepare_query(query) for query in queries]
        self._load()
        synchronize(self.device)
        started = perf_counter()
        values, times, padding, observed, safe = (np.stack([row[i] for row in rows]) for i in range(5))
        points = self.native.impute_batch(safe, times, task="R", observed_mask=observed,
                                         padding_mask=padding)
        synchronize(self.device)
        elapsed = perf_counter() - started
        if not np.array_equal(points[observed], values[observed]):
            raise ValueError("native frozen model changed admitted observations")
        metadata = dict(self._metadata(), inference_batching="native_MOMENT_independent_masked_RevIN_v1",
            inference_batch_size=len(queries), normalization="per_query_per_channel_official_masked_RevIN")
        timing = dict(inference_seconds=elapsed / len(queries), batch_wall_seconds=elapsed,
                      batch_size=len(queries), timing_mode="batch_amortized",
                      shared_load_seconds=self.load_seconds)
        return [dict(point_estimate=point.copy(), samples=None, quantiles=None, quantile_levels=None,
                     status="success", timing=dict(timing), model_metadata=dict(metadata))
                for point in points]

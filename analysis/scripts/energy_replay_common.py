"""Read-only inputs shared by supplemental energy diagnostics and replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from energygap.data import DatasetConfig
from energygap.data.ingest import ingest
from energygap.masking.generators import Episode, MaskManifest
from energygap.protocols.splits import rolling_origin_splits
from energygap.evaluation.reconstruction import training_scale
from energygap.runtime.state import fingerprint

KEYS = ["site", "fold", "partition", "condition", "mask_seed", "gap_id"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_digest():
    result = hashlib.sha256()
    root = REPOSITORY / "src" / "energygap"
    for path in sorted(root.rglob("*.py")):
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")


def output_directory(path, run_dir):
    path, run_dir = Path(path).resolve(), Path(run_dir).resolve()
    if path == run_dir or run_dir in path.parents:
        raise ValueError("Supplemental output must be outside the immutable run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_lock(run_dir):
    path = Path(run_dir) / "protocol.lock.json"
    lock = json.loads(path.read_text())
    declared = path.with_suffix(".sha256")
    # This sidecar stores the scientific configuration fingerprint, not the
    # byte digest of the JSON serialization.
    if declared.exists() and declared.read_text().strip().split()[0] != fingerprint(lock):
        raise ValueError("Frozen protocol checksum mismatch")
    return lock


def load_series(lock, site, data_root):
    """Use the embedded frozen data contract, never current dataset YAML."""
    config = lock["datasets"][site]["config"]
    inputs = {item["path"]: digest(Path(data_root) / item["path"])
              for item in config["files"]}
    for item in config["files"]:
        if inputs[item["path"]] != item["sha256"]:
            raise ValueError(f"Original input hash mismatch for {site}")
    return ingest(DatasetConfig(config), root=data_root, purpose="statistical"), inputs


def fold_scale(series, fold):
    indices = fold.indices(series, "train")
    values = np.asarray([
        row.value_canonical if row.observed_original and not row.natural_missing
        and row.available_to_method and row.measurement_time < fold.train_end
        and row.availability_time < fold.train_end else np.nan
        for row in (series.records[i] for i in indices)], float)
    scale, reason = training_scale(values)
    if scale is None:
        raise ValueError(f"Undefined training scale for {series.site_id}/{fold.fold_id}")
    return indices, values, float(scale), reason


def load_losses(run_dir, site=None):
    path = Path(run_dir) / "reports" / "episode_losses.parquet"
    filters = [("site", "==", site)] if site else None
    return pd.read_parquet(path, filters=filters)


def load_manifest(path):
    content = json.loads(Path(path).read_text())
    episodes = tuple(Episode(**{**item, "indices": tuple(item["indices"])})
                     for item in content["episodes"])
    result = MaskManifest(**{**content, "episodes": episodes,
                             "hidden_indices": tuple(content["hidden_indices"])})
    if set(result.hidden_indices) != {i for episode in episodes for i in episode.indices}:
        raise ValueError("Replica hidden-index union does not match its episodes")
    if len({e.gap_id for e in episodes}) != len(episodes):
        raise ValueError("Duplicate gap IDs in manifest")
    return result


def checked_manifests(manifest_dir, site, fold, partition, lock, archived):
    """Match every realized gap ID and duration to original archived P losses."""
    selected = archived[(archived.method == "linear") & (archived.variant == "P")
                        & (archived.site == site) & (archived.fold == fold.fold_id)
                        & (archived.partition == partition)]
    if selected.duplicated(KEYS).any():
        raise ValueError("Duplicate archived linear episode keys")
    seen, manifests, hashes = set(), [], {}
    for condition in lock["conditions"]:
        for seed in lock[partition + "_mask_seeds"]:
            path = Path(manifest_dir) / f"{site}:fold{fold.fold_id}-{partition}-{condition}-{seed}.json"
            manifest = load_manifest(path)
            if (manifest.site_id, manifest.fold_id, manifest.partition,
                manifest.condition, manifest.seed) != (site, fold.fold_id, partition, condition, seed):
                raise ValueError("Manifest metadata mismatch")
            actual = selected[(selected.condition == condition) & (selected.mask_seed == seed)]
            expected = {(e.gap_id, float(e.duration_minutes)) for e in manifest.episodes}
            observed = set(zip(actual.gap_id, actual.duration))
            if expected != observed:
                raise ValueError(f"Manifest/archive episode mismatch: {path.name}")
            seen.update(actual.index)
            manifests.append(manifest)
            hashes[path.name] = digest(path)
    if seen != set(selected.index):
        raise ValueError("Incomplete archived linear coverage")
    split_path = Path(manifest_dir) / f"{site}:fold{fold.fold_id}-split.json"
    if json.loads(split_path.read_text()) != fold.manifest():
        raise ValueError("Manifest elapsed-time fold differs from frozen data replay")
    hashes[split_path.name] = digest(split_path)
    return manifests, hashes

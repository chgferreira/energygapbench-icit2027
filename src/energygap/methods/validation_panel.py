"""Deterministic epoch-selection panels chosen from episode metadata only.

This helper never accepts a series, Query, EvaluationTruth, value array or score.
The full validation/test manifests remain the evaluation populations. The panel
only controls the repeated validation endpoint used for epoch early stopping.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_epoch_validation_panel(candidates, *, site, fold, max_per_stratum=None,
                                  seed=1729, durations=(15., 60., 180.), mask_seeds=()):
    """Select at most K episodes per (duration, mask seed), without reading data.

    ``candidates`` contains only duration, mask_seed and gap_id dictionaries.
    Hash ranking includes site/fold/seed and is independent of input ordering,
    method, training seed, trial, observed/hidden values and validation scores.
    The returned manifest is suitable for hashing and durable protocol records.
    """
    if max_per_stratum is not None and (isinstance(max_per_stratum, bool)
            or not isinstance(max_per_stratum, int) or max_per_stratum <= 0):
        raise ValueError("Epoch validation cap must be a positive integer or null")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Epoch validation seed must be an integer")
    rows = []
    identities = set()
    grouped = defaultdict(list)
    for candidate in candidates:
        if set(candidate)!={"duration", "mask_seed", "gap_id"}:
            raise ValueError("Panel candidates accept only duration, mask_seed and gap_id metadata")
        duration = float(candidate["duration"])
        mask_seed = candidate["mask_seed"]
        gap_id = candidate["gap_id"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Panel candidates require positive finite durations")
        if isinstance(mask_seed, bool) or not isinstance(mask_seed, int) or not isinstance(gap_id, str) or not gap_id:
            raise ValueError("Panel candidates require integer mask seeds and nonempty gap IDs")
        identity = (mask_seed, gap_id)
        if identity in identities:
            raise ValueError("Duplicate epoch-validation episode identity")
        identities.add(identity)
        row = dict(duration=duration, mask_seed=mask_seed, gap_id=gap_id)
        rows.append(row)
        grouped[(duration, mask_seed)].append(row)
    ordering = lambda row: (row["duration"], row["mask_seed"], row["gap_id"])
    rows.sort(key=ordering)
    selected = []
    for (duration, mask_seed), group in sorted(grouped.items()):
        if max_per_stratum is None or len(group) <= max_per_stratum:
            chosen = group
        else:
            chosen = sorted(group, key=lambda row: (_hash(dict(seed=seed, site=site, fold=fold,
                duration=duration, mask_seed=mask_seed, gap_id=row["gap_id"])), row["gap_id"]))[:max_per_stratum]
        selected.extend(chosen)
    selected.sort(key=ordering)
    selected_counts = defaultdict(int)
    for row in selected:
        selected_counts[(row["duration"], row["mask_seed"])] += 1
    required_seeds = tuple(mask_seeds) or tuple(sorted({row["mask_seed"] for row in rows}))
    expected = {(float(duration), int(mask_seed)) for duration in durations for mask_seed in required_seeds}
    strata = [dict(duration=duration, mask_seed=mask_seed,
                   candidate_count=len(grouped[(duration, mask_seed)]),
                   selected_count=selected_counts[(duration, mask_seed)])
              for duration, mask_seed in sorted(set(grouped) | expected)]
    manifest = dict(schema_version=1, site=site, fold=fold, partition="validation",
        policy=("all_prespecified_validation_episodes" if max_per_stratum is None
                else "fixed_hash_stratified_epoch_validation_panel"),
        scope="epoch_checkpoint_selection_only;HPO_final_scoring_and_final_validation_test_use_full_manifests",
        selection_seed=seed, max_per_stratum=max_per_stratum,
        stratification=["duration", "mask_seed"],
        ranking="sha256_seed_site_fold_duration_mask_seed_gap_id",
        selection_values_used=False, selection_metrics_used=False,
        candidate_count=len(rows), selected_count=len(selected),
        covers_all_candidates=len(selected) == len(rows),
        candidate_identity_sha256=_hash(dict(site=site, fold=fold, candidates=rows)),
        selected_identity_sha256=_hash(dict(site=site, fold=fold, selected=selected)),
        missing_required_strata=[dict(duration=duration, mask_seed=mask_seed)
                                 for duration, mask_seed in sorted(expected) if not grouped[(duration, mask_seed)]],
        strata=strata, selected=selected)
    manifest["coverage_status"] = "missing_required_strata" if manifest["missing_required_strata"] else "complete"
    manifest["panel_sha256"] = _hash(manifest)
    return manifest

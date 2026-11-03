#!/usr/bin/env python3
"""Reproduce manuscript summaries from the frozen ICIT 2027 result export.

No model fitting, selection on test data, or new inferential tests are performed.
The original unavailable forecast endpoint is retained. A separately labelled
post hoc analysis conditions on common observed one-hour forecast targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "energygap-icit-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow

SITES = ["H1", "H2", "I1"]
METHODS = ["median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother", "SAITS", "MOMENT-1-base"]
LABELS = ["Median", "LOCF", "Linear", "Seasonal", "Kalman filter", "Kalman smoother", "SAITS", "MOMENT"]
HYPOTHESES = ["instantaneous", "interval_mean_start", "interval_mean_end"]
HYP_LABELS = ["Instantaneous", "Interval start", "Interval end"]
BASELINES = METHODS[:6]
METRICS = ["nmae", "mae_kw", "forecast_nmae", "forecast_mae_kw"]
EPISODE = ["site", "fold", "condition", "duration", "mask_seed", "gap_id", "day"]
CELL = ["site", "fold", "duration"]
COLORS = ["#007A87", "#C76B24", "#6C5B8F"]
# This mirrors runtime/state.py's frozen scientific fingerprint contract.
# protocol.lock.sha256 is this fingerprint, not a raw-byte checksum of the JSON.
PRESENTATION_KEYS = {"progress", "output_dir", "state_dir", "checkpoint_every_steps", "checkpoint_every_seconds", "checkpoint_optimizer_steps", "checkpoint_active_seconds", "progress_refresh_seconds", "progress_persist_seconds", "status_snapshot_seconds", "heartbeat_seconds", "resume", "budget_total_hours", "budget_seconds", "timezone", "retry_failed"}


def scientific_config(config):
    return {k: scientific_config(v) if isinstance(v, dict) else v for k, v in config.items() if k not in PRESENTATION_KEYS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def complete_means(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Do not silently discard a missing outcome from the frozen estimand."""
    grouped = frame.groupby(keys, sort=True, dropna=False, observed=True)
    means = grouped[metrics].mean()
    counts = grouped[metrics].count()
    sizes = grouped.size()
    return means.where(counts.eq(sizes, axis=0)).reset_index()


def compare_tables(actual, expected, keys, metrics, label):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate aggregate keys")
    joined = actual.merge(expected, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError(f"{label}: mismatched aggregate coverage")
    errors = {}
    for metric in metrics:
        a = joined[f"{metric}_new"].to_numpy(float)
        b = joined[f"{metric}_old"].to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(a, b, rtol=1e-11, atol=1e-12, equal_nan=True):
            raise ValueError(f"{label}: {metric} does not reproduce the frozen report")
        good = np.isfinite(a) & np.isfinite(b)
        errors[metric] = float(np.max(np.abs(a[good] - b[good]))) if good.any() else None
    return {"rows": len(joined), "maximum_absolute_errors": errors}


def select_baselines(episodes, expected):
    validation = episodes.loc[episodes.partition.eq("validation") & episodes.variant.eq("P") & episodes.method.isin(BASELINES)]
    cells = complete_means(validation, CELL + ["method"], ["nmae"])
    scores = complete_means(cells, ["site", "fold", "method"], ["nmae"]).rename(columns={"nmae": "validation_equal_duration_nmae"})
    latency = validation.groupby(["site", "fold", "method"], observed=True).latency_seconds.median().rename("latency_p50_seconds").reset_index()
    scores = scores.merge(latency, on=["site", "fold", "method"], validate="one_to_one")
    compare_tables(scores, expected, ["site", "fold", "method"], ["validation_equal_duration_nmae", "latency_p50_seconds"], "baseline validation")
    scores = scores.sort_values(["site", "fold", "validation_equal_duration_nmae", "latency_p50_seconds", "method"], na_position="last")
    selected = scores.groupby(["site", "fold"], sort=False, observed=True).head(1).reset_index(drop=True)
    if len(selected) != 9:
        raise ValueError("Expected a baseline selection for all nine site/fold pairs")
    return selected


def policy_rows(test, selected, policy, hypothesis="instantaneous"):
    if policy.startswith("B"):
        rows = test.merge(selected[["site", "fold", "method"]].rename(columns={"method": "chosen"}), on=["site", "fold"], validate="many_to_one")
        expected_method = rows.chosen + ("+energy_consistent" if policy.endswith("E") else "")
        mask = rows.method.eq(expected_method)
    else:
        rows = test
        mask = rows.method.eq("linear+energy_consistent" if policy.endswith("E") else "linear")
    mask &= rows.variant.eq("E:" + hypothesis if policy.endswith("E") else "P")
    return rows.loc[mask, EPISODE + METRICS].copy()


def paired_analysis(test, selected, frozen):
    estimates, posthoc, support, pair_check = [], [], [], []
    for hypothesis in HYPOTHESES:
        panels = {}
        for policy in ("B_P", "B_E", "linear_E"):
            rows = policy_rows(test, selected, policy, hypothesis)
            if rows.duplicated(EPISODE).any():
                raise ValueError("Duplicate episode keys in paired policy analysis")
            panels[policy] = rows.rename(columns={m: f"{policy}_{m}" for m in METRICS})
        joined = panels["B_E"].merge(panels["B_P"], on=EPISODE, how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Reconstruction policies do not share episode identities")
        joined = joined.drop(columns="_merge").merge(panels["linear_E"], on=EPISODE, how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Linear and selected-baseline policies have different episode identities")
        joined = joined.drop(columns="_merge")
        for metric in ("nmae", "forecast_nmae"):
            joined[f"energy_{metric}"] = joined[f"B_E_{metric}"] - joined[f"B_P_{metric}"]
            joined[f"complexity_{metric}"] = joined[f"B_E_{metric}"] - joined[f"linear_E_{metric}"]
        metrics = [f"{c}_{m}" for c in ("energy", "complexity") for m in ("nmae", "forecast_nmae")]
        cells = complete_means(joined, CELL, metrics)
        for site, group in cells.groupby("site", observed=True):
            if len(group) != 9:
                raise ValueError("A required fold/duration stratum is missing")
            for c in ("energy", "complexity"):
                for metric in ("nmae", "forecast_nmae"):
                    values = group[f"{c}_{metric}"].to_numpy(float)
                    estimate = float(values.mean()) if np.isfinite(values).all() else np.nan
                    estimates.append({"site": site, "hypothesis": hypothesis, "contrast": f"contrast_{c}_gain", "endpoint": metric, "estimate": estimate})
        # This is explicitly supplementary and conditions on targets observed for
        # all three compared policies. It does not replace frozen unavailable NA.
        finite = np.isfinite(joined[[f"{p}_forecast_nmae" for p in panels]]).all(axis=1)
        joined["common_observed_target"] = finite
        for keys, group in joined.groupby(CELL, observed=True, sort=True):
            available = group.loc[group.common_observed_target]
            row = dict(zip(CELL, keys))
            row.update(hypothesis=hypothesis, eligible_episodes=len(group), common_observed_target_episodes=len(available), unavailable_episodes=len(group) - len(available), analysis="post_hoc_available_target_1h")
            for policy in panels:
                row[policy + "_forecast_nmae"] = float(available[policy + "_forecast_nmae"].mean()) if len(available) else np.nan
            row["energy_difference"] = float(available.energy_forecast_nmae.mean()) if len(available) else np.nan
            row["complexity_difference"] = float(available.complexity_forecast_nmae.mean()) if len(available) else np.nan
            support.append(row)
        for site, group in joined.groupby("site", observed=True):
            bad = int((~group.common_observed_target).sum())
            pair_check.append({"site": site, "hypothesis": hypothesis, "eligible_episodes": len(group), "unavailable_1h_episodes": bad})
    estimates = pd.DataFrame(estimates)
    check = compare_tables(estimates, frozen, ["site", "hypothesis", "contrast", "endpoint"], ["estimate"], "paired contrast estimates")
    support = pd.DataFrame(support)
    for (site, hypothesis), group in support.groupby(["site", "hypothesis"], observed=True):
        if len(group) != 9 or group.common_observed_target_episodes.eq(0).any():
            raise ValueError("Post hoc analysis must retain all nine fixed strata")
        row = {"site": site, "hypothesis": hypothesis, "analysis": "post_hoc_available_target_1h", "eligible_episodes": int(group.eligible_episodes.sum()), "common_observed_target_episodes": int(group.common_observed_target_episodes.sum()), "unavailable_episodes": int(group.unavailable_episodes.sum()), "fixed_strata": 9}
        for metric in ["B_P_forecast_nmae", "B_E_forecast_nmae", "linear_E_forecast_nmae", "energy_difference", "complexity_difference"]:
            row[metric] = float(group[metric].mean())
        row["relative_energy_change_percent"] = 100 * (row["B_E_forecast_nmae"] / row["B_P_forecast_nmae"] - 1)
        posthoc.append(row)
    return estimates, pd.DataFrame(posthoc), support, check, pair_check


def figure_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .6, "lines.linewidth": 1.25, "lines.markersize": 4, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "energygap-icit2027", "savefig.dpi": 300})


def export_figure(fig, directory, name):
    for suffix in ("pdf", "svg", "png"):
        metadata = {"Creator": "EnergyGapBench ICIT 2027 reproducible analysis"} if suffix == "pdf" else None
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif suffix == "svg":
            metadata = {"Date": None, "Creator": "EnergyGapBench ICIT 2027 reproducible analysis"}
        fig.savefig(directory / f"{name}.{suffix}", metadata=metadata)
    plt.close(fig)


def make_figures(core, duration, contrasts, out):
    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), sharey=True)
    for i, (site, ax) in enumerate(zip(SITES, axes)):
        sub = core.loc[core.site.eq(site)].copy()
        for j, method in enumerate(METHODS):
            a = sub.loc[sub.method.eq(method) & sub.variant.eq("P"), "nmae"].iloc[0]
            b = sub.loc[sub.method.eq(method + "+energy_consistent") & sub.variant.eq("E:instantaneous"), "nmae"].iloc[0]
            ax.plot([a, b], [j, j], color="#BCC5C9", linewidth=1.4, zorder=1)
            ax.scatter(a, j, color="#6E7378", s=16, marker="o", zorder=2)
            ax.scatter(b, j, color=COLORS[0], s=21, marker="s", zorder=3)
        ax.set_title(site)
        ax.set_xlabel("Episode nMAE")
        ax.set_xlim(0, .4)
        ax.set_xticks([0, .1, .2, .3, .4])
        ax.set_yticks(range(8), LABELS)
        ax.grid(axis="x", color="#E6E9EB", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", color="#6E7378", linestyle="none", label="Power only (P)"), Line2D([], [], marker="s", color=COLORS[0], linestyle="none", label="Energy correction (instantaneous)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.54, .005))
    fig.subplots_adjust(left=.15, right=.98, top=.88, bottom=.23, wspace=.13)
    export_figure(fig, out, "reconstruction_methods")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), sharex=True)
    styles = {"B_P": ("#474B50", "o", "--", "$B$ (P)"), "B_E": (COLORS[0], "o", "-", "$B$ (E)"), "linear_P": ("#979B9F", "s", "--", "Linear (P)"), "linear_E": (COLORS[1], "s", "-", "Linear (E)")}
    for site, ax in zip(SITES, axes):
        for policy, (color, marker, ls, label) in styles.items():
            sub = duration.loc[duration.site.eq(site) & duration.policy.eq(policy)].sort_values("duration")
            ax.plot(range(3), sub.nmae, color=color, marker=marker, linestyle=ls, label=label)
        ax.set_title(site)
        ax.set_xticks(range(3), ["15", "60", "180"])
        ax.set_xlabel("Gap duration (min)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E6E9EB", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Episode nMAE")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.53, .0))
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.32, wspace=.32)
    export_figure(fig, out, "reconstruction_duration")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for c, ax, title in zip(["contrast_energy_gain", "contrast_complexity_gain"], axes, ["Energy contrast", "Comparator contrast"]):
        for site_i, site in enumerate(SITES):
            for h_i, (hypothesis, color, marker) in enumerate(zip(HYPOTHESES, COLORS, ["o", "s", "D"])):
                row = contrasts.loc[contrasts.site.eq(site) & contrasts.hypothesis.eq(hypothesis) & contrasts.contrast.eq(c) & contrasts.endpoint.eq("nmae")].iloc[0]
                lo, hi = row.ci95
                y = site_i + (h_i - 1) * .21
                ax.plot([lo, hi], [y, y], color=color, linewidth=1)
                ax.plot([lo, lo], [y-.045, y+.045], color=color, linewidth=.8)
                ax.plot([hi, hi], [y-.045, y+.045], color=color, linewidth=.8)
                ax.plot(row.estimate, y, marker=marker, markeredgecolor=color, markerfacecolor=color if row.temporal_support_sufficient else "white", linestyle="none", markersize=4)
        ax.axvline(0, color="#777777", linewidth=.7, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel((r"$B(E)-B(P)$" if c == "contrast_energy_gain" else r"$B(E)-\mathrm{Linear}(E)$") + " (episode nMAE)")
        ax.set_yticks(range(3), ["H1*", "H2", "I1"])
        ax.grid(axis="x", color="#E6E9EB", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker=m, color=c, label=l, linewidth=1) for m, c, l in zip(["o", "s", "D"], COLORS, HYP_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.29, wspace=.16)
    export_figure(fig, out, "reconstruction_contrasts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Frozen compact run directory containing protocol.lock.json and reports/")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/outputs"))
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    reports = run / "reports"
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    source_names = ["protocol.lock.json", "protocol.lock.sha256", "reports/episode_losses.parquet", "reports/core_aggregate.csv", "reports/main_table.csv", "reports/selected_baseline_validation.csv", "reports/contrasts.json", "reports/contrasts_block_sensitivity.json", "reports/summary.json", "reports/report_complete.json"]
    source_hashes = {name: sha256(run / name) for name in source_names}
    declared = (run / "protocol.lock.sha256").read_text().strip().split()[0]
    config = json.loads((run / "protocol.lock.json").read_text())
    scientific_fingerprint = hashlib.sha256(json.dumps(scientific_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if scientific_fingerprint != declared:
        raise ValueError("Frozen scientific protocol fingerprint mismatch")
    completion = json.loads((reports / "report_complete.json").read_text())
    if completion["fingerprint"] != scientific_fingerprint:
        raise ValueError("Report completion and protocol fingerprints differ")
    # Raw byte hashes are recorded separately below. Neither these nor the
    # completion marker audit the missing ledger, checkpoints or trajectories.
    episodes = pd.read_parquet(reports / "episode_losses.parquet")
    keys = EPISODE + ["partition", "method", "variant"]
    if episodes.duplicated(keys).any():
        raise ValueError("Duplicate episode-policy records")
    if not np.isfinite(episodes.nmae).all():
        raise ValueError("Unexpected unavailable reconstruction loss")
    expected_seeds = np.where(episodes.method.str.startswith("SAITS"), 3, 1)
    if not np.array_equal(episodes.seed_evaluations.to_numpy(), expected_seeds):
        raise ValueError("Unexpected seed-loss aggregation multiplicity")
    test = episodes.loc[episodes.partition.eq("test")].copy()
    selected = select_baselines(episodes, pd.read_csv(reports / "selected_baseline_validation.csv"))
    selected.to_csv(out / "selected_baselines.csv", index=False, float_format="%.12g")
    cells = complete_means(test, CELL + ["method", "variant"], METRICS)
    count = cells.groupby(["site", "method", "variant"], observed=True).size()
    if not count.eq(9).all():
        raise ValueError("All policies must retain the nine required fold/duration strata")
    core = complete_means(cells, ["site", "method", "variant"], METRICS)
    core_check = compare_tables(core, pd.read_csv(reports / "core_aggregate.csv"), ["site", "method", "variant"], METRICS, "core aggregate")
    main_table = complete_means(test, CELL + ["condition", "method", "variant"], METRICS)
    main_check = compare_tables(main_table, pd.read_csv(reports / "main_table.csv"), CELL + ["condition", "method", "variant"], METRICS, "main table")
    core.to_csv(out / "reconstruction_all_methods_all_hypotheses.csv", index=False, float_format="%.12g")
    cells.to_csv(out / "reconstruction_cells.csv", index=False, float_format="%.12g")
    frozen = pd.DataFrame(json.loads((reports / "contrasts.json").read_text()))
    estimates, posthoc, support, contrast_check, pair_check = paired_analysis(test, selected, frozen)
    estimates.to_csv(out / "reproduced_primary_estimates.csv", index=False, float_format="%.12g")
    posthoc.to_csv(out / "posthoc_available_target_1h.csv", index=False, float_format="%.12g")
    support.to_csv(out / "posthoc_available_target_1h_support.csv", index=False, float_format="%.12g")
    exported = frozen.copy()
    exported["ci_lower"] = exported.ci95.apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    exported["ci_upper"] = exported.ci95.apply(lambda x: x[1] if isinstance(x, list) else np.nan)
    exported.drop(columns=["ci95"]).to_csv(out / "frozen_contrasts.csv", index=False, float_format="%.12g")
    sensitivities = pd.DataFrame(json.loads((reports / "contrasts_block_sensitivity.json").read_text()))
    sensitivities["ci_lower"] = sensitivities.ci95.apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    sensitivities["ci_upper"] = sensitivities.ci95.apply(lambda x: x[1] if isinstance(x, list) else np.nan)
    sensitivities.drop(columns=["ci95"]).to_csv(out / "frozen_block_sensitivity.csv", index=False, float_format="%.12g")
    # Each policy is selected independently of test outcomes. B may differ by fold.
    duration = []
    for policy in ("B_P", "B_E", "linear_P", "linear_E"):
        frame = policy_rows(test, selected, policy)
        fold_cells = complete_means(frame, CELL, ["nmae"])
        site_duration = complete_means(fold_cells, ["site", "duration"], ["nmae"])
        site_duration["policy"] = policy
        duration.append(site_duration)
    duration = pd.concat(duration, ignore_index=True)
    duration.to_csv(out / "selected_policy_by_duration.csv", index=False, float_format="%.12g")
    policy_core = complete_means(duration, ["site", "policy"], ["nmae"]).pivot(index="site", columns="policy", values="nmae").reset_index()
    policy_core["relative_energy_change_percent"] = 100 * (policy_core.B_E / policy_core.B_P - 1)
    policy_core.to_csv(out / "selected_policy_overall.csv", index=False, float_format="%.12g")
    # The same missing target support must hold for all methods before presenting
    # the available-target supplement as a common comparison population.
    support_group = test.groupby(EPISODE, observed=True).forecast_nmae.agg(["count", "size"])
    if not support_group["size"].eq(32).all() or not (support_group["count"].eq(0) | support_group["count"].eq(32)).all():
        raise ValueError("Available one-hour targets differ across policies")
    episode_counts = test.loc[test.method.eq("linear") & test.variant.eq("P")].groupby(CELL, observed=True).agg(episodes=("gap_id", "size"), originally_observed_1h_targets=("forecast_nmae", "count"), days=("day", "nunique")).reset_index()
    episode_counts.to_csv(out / "episode_support.csv", index=False)
    # Compact manuscript table. Bold denotes only the lowest descriptive score
    # within the displayed column, never statistical significance.
    table = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"& \multicolumn{2}{c}{H1} & \multicolumn{2}{c}{H2} & \multicolumn{2}{c}{I1} \\", r"Method & P & E & P & E & P & E \\", r"\midrule"]
    value_columns = []
    for site in SITES:
        for variant in ("P", "E:instantaneous"):
            value_columns.append([float(core.loc[core.site.eq(site) & core.method.eq(m if variant == "P" else m + "+energy_consistent") & core.variant.eq(variant), "nmae"].iloc[0]) for m in METHODS])
    for j, label in enumerate(LABELS):
        vals = [(r"\textbf{" + f"{v[j]:.4f}" + "}") if v[j] == min(v) else f"{v[j]:.4f}" for v in value_columns]
        table.append(label + " & " + " & ".join(vals) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "methods_nmae.tex").write_text("\n".join(table) + "\n")
    make_figures(core, duration, frozen, figures)
    validation = {"run_id": run.name, "scientific_protocol_fingerprint_verified": True, "episode_policy_rows": len(episodes), "test_episode_policy_rows": len(test), "test_episode_instances": len(support_group), "reconstruction_missing_rows": int(test.nmae.isna().sum()), "all_methods_share_1h_target_support": True, "core_aggregate_reproduction": core_check, "main_table_reproduction": main_check, "contrast_estimate_reproduction": contrast_check, "paired_support": pair_check, "frozen_primary_forecast_estimates_available": int(frozen.loc[frozen.endpoint.eq("forecast_nmae"), "estimate"].notna().sum()), "limitations": ["Inference CIs are read from the frozen export, not recomputed without the full manifest calendars.", "No ledger, per-record failure reasons, selected correction parameters or individual trajectories are included in the compact input.", "The available-target forecast summary is post hoc and conditional on observed future targets; frozen primary forecast estimates remain unavailable.", "No site pooling, inferential test additions or test-set parameter selection is performed."]}
    save_json(out / "analysis_validation.json", validation)
    # Hash only files authored by this analysis. Separate descriptive supplements
    # maintain their own provenance, so clean and populated output dirs agree.
    owned_names = ["selected_baselines.csv", "reconstruction_cells.csv",
                   "reconstruction_all_methods_all_hypotheses.csv",
                   "reproduced_primary_estimates.csv", "posthoc_available_target_1h.csv",
                   "posthoc_available_target_1h_support.csv", "frozen_contrasts.csv",
                   "frozen_block_sensitivity.csv", "selected_policy_by_duration.csv",
                   "selected_policy_overall.csv", "episode_support.csv",
                   "analysis_validation.json", "tables/methods_nmae.tex"]
    owned_names += [f"figures/{stem}.{suffix}" for stem in
                    ["reconstruction_methods", "reconstruction_duration", "reconstruction_contrasts"]
                    for suffix in ["pdf", "svg", "png"]]
    generated = {name: sha256(out / name) for name in sorted(owned_names)}
    save_json(out / "analysis_manifest.json", {"run_id": run.name, "sources_sha256": source_hashes, "script_sha256": sha256(Path(__file__)), "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__}, "outputs_sha256": generated, "aggregation": "Mean seed losses already supplied per episode; equal episode means within duration/fold; equal weights over the 3 durations and 3 folds; sites separate.", "figure_intervals": "Frozen 7-day moving-block 95% percentile intervals; exploratory and conditional. Hollow H1 markers flag insufficient declared temporal support."})
    print(f"Verified frozen aggregates and paired estimates. Outputs: {out}")
    print(policy_core.to_string(index=False))
    print(posthoc.to_string(index=False))


if __name__ == "__main__":
    main()

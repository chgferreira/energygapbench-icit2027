"""Export actual conference-run measurements to PNG/SVG/PDF, outside the paper.

Compact reports supply primary plots. Optional horizon/energy/uncertainty inputs
are streamed into a temporary scalar-only SQLite reduction: training-seed losses
are averaged within episodes before equal duration/fold means. Missing support
remains NA; no points or confidence intervals are invented for unavailable data.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from energygap.runtime.state import atomic_bytes, atomic_json

BASELINES = ("median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother")
FROZEN = ("MOMENT-1-base", "TS-ICL")
DURATIONS = (15., 60., 180.)
CORE = ("C3", "C4", "C5")
SCALAR_METRICS = ("crps", "crps_truncated_quantile_approximation", "coverage_0.5", "coverage_0.8",
                  "coverage_0.9", "coverage_0.95", "net_integral_absolute_error_kwh",
                  "net_input_constraint_absolute_residual_kwh")


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "energygap-mpl"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "energygap-conference-figures"})
    return plt


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compact(path):
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _method_group(method):
    base = str(method).split("+", 1)[0]
    return "baselines" if base in BASELINES else "frozen" if base in FROZEN else "learned"


def _save(fig, out, name, paths):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    summary_file=out.parent/'summary.json'
    summary=json.loads(summary_file.read_text()) if summary_file.is_file() else {}
    if summary.get('study_profile')=='focused_24h':
        fig.text(.5,.005,'Focused 24h profile: declared subset; 24h is a target, not a completion guarantee',
                 ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1))
    else:
        fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        buffer = BytesIO()
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {}
        fig.savefig(buffer, format=suffix, dpi=180, bbox_inches="tight", metadata=metadata)
        destination = out / f"{name}.{suffix}"
        atomic_bytes(destination, buffer.getvalue())
        paths.append(str(destination))


def _axes(plt, sites, *, height=4.2):
    return plt.subplots(1, len(sites), squeeze=False, figsize=(max(5.4, 5.4 * len(sites)), height))[0:2]


def _empty(ax, reason="NA: no complete measured support"):
    ax.text(.5, .5, reason, ha="center", va="center", transform=ax.transAxes, wrap=True)


def _reconstruction_figures(plt, main, sites, out, paths, manifest):
    required = {"site", "fold", "duration", "method", "variant", "nmae"}
    if main.empty or not required <= set(main):
        return
    rows = main[(main.variant == "P") & main.duration.isin(DURATIONS)].copy()
    if "condition" in rows:
        rows = rows[rows.condition.isin(CORE)]
    rows["group"] = rows.method.map(_method_group)
    for category in ("baselines", "learned", "frozen"):
        selected = rows[rows.group == category]
        if selected.empty:
            continue
        fig, axes = _axes(plt, sites)
        points = []
        for site, ax in zip(sites, axes[0]):
            present = False
            for method, group in selected[selected.site == site].groupby("method", sort=True):
                values = []
                expected_folds = sorted(set(main.loc[main.site == site, "fold"].dropna().astype(int)))
                for duration in DURATIONS:
                    cells = group[group.duration == duration].set_index("fold").nmae.reindex(expected_folds).to_numpy(dtype=float)
                    value = float(cells.mean()) if len(cells) and np.isfinite(cells).all() else None
                    values.append(np.nan if value is None else value)
                    points.append(dict(site=str(site), method=str(method), duration=duration, nmae=value))
                if np.isfinite(values).any():
                    ax.plot(DURATIONS, values, marker="o", label=method)
                    present = True
            if present:
                ax.legend(fontsize=7)
            else:
                _empty(ax)
            ax.set(title=str(site), xlabel="Gap duration (min)", ylabel="Reconstruction gap-macro nMAE (1)", xticks=DURATIONS)
            ax.grid(alpha=.2)
        fig.suptitle(f"R/P reconstruction — {category}; mean over folds", y=1.04)
        name = "reconstruction_duration_" + category
        _save(fig, out, name, paths)
        manifest.append(dict(name=name, source="main_table.csv", aggregation="equal fold means within duration; missing cells remain NA", points=points))
        plt.close(fig)


def _pareto_and_utility(plt, main, core, sites, out, paths, manifest):
    if core.empty or not {"site", "method", "variant", "nmae", "forecast_nmae"} <= set(core):
        return
    base = core[core.variant == "P"]
    for kind in ("pareto", "forecast_utility_1h"):
        fig, axes = _axes(plt, sites)
        points = []
        for site, ax in zip(sites, axes[0]):
            present = False
            for _, row in base[base.site == site].sort_values("method").iterrows():
                y = _number(row.nmae if kind == "pareto" else row.forecast_nmae)
                if kind == "pareto":
                    subset = main[(main.site == site) & (main.method == row.method) & (main.variant == "P") & main.duration.isin(DURATIONS)] if {"site", "method", "variant", "duration", "latency_p50_seconds"} <= set(main) else pd.DataFrame()
                    if not subset.empty and "condition" in subset:
                        subset = subset[subset.condition.isin(CORE)]
                    raw = subset.latency_p50_seconds.to_numpy(dtype=float) if not subset.empty else np.asarray([])
                    x = float(np.median(raw)) if len(raw) and np.isfinite(raw).all() else None
                else:
                    x = _number(row.nmae)
                points.append(dict(site=str(site), method=str(row.method), x=x, y=y))
                if x is None or y is None:
                    continue
                ax.scatter([x], [y], s=28)
                ax.annotate(str(row.method), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6)
                present = True
            if not present:
                _empty(ax)
            ax.set(title=str(site), xlabel="Median of fold-duration latency medians (s)" if kind == "pareto" else "Reconstruction gap-macro nMAE (1)",
                   ylabel="Reconstruction gap-macro nMAE (1)" if kind == "pareto" else "Fixed-ridge forecast nMAE at 1 h (1)")
            ax.grid(alpha=.2)
        fig.suptitle("R/P measured inference cost and reconstruction" if kind == "pareto" else "R/P reconstruction and downstream errors; descriptive association", y=1.04)
        _save(fig, out, kind, paths)
        manifest.append(dict(name=kind, source=["main_table.csv", "core_aggregate.csv"], aggregation="core equal duration/fold losses; Pareto x is median of cell medians, excludes fit/load", points=points))
        plt.close(fig)


def contrast_figure(rows, hypothesis):
    """Return a forest figure, preserving NA/support labels without zero markers."""
    plt = _matplotlib()
    endpoints = ("nmae", "forecast_nmae")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.6, .56 * max(1, len(rows) // 2))), squeeze=False)
    for endpoint, ax in zip(endpoints, axes[0]):
        selected = sorted([r for r in rows if r.get("endpoint") == endpoint], key=lambda r: (str(r.get("site")), str(r.get("contrast"))))
        labels = []
        for index, row in enumerate(selected):
            labels.append(f"{row.get('site')} · {str(row.get('contrast', '')).removeprefix('contrast_')}")
            value = _number(row.get("estimate"))
            ci = row.get("ci95")
            finite_ci = isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_number(v) is not None for v in ci)
            supported = row.get("temporal_support_sufficient")
            if value is None:
                ax.text(.02, index, "NA: " + str(row.get("reason") or "missing matched support")[:95],
                        transform=ax.get_yaxis_transform(), va="center", fontsize=7)
                continue
            if finite_ci:
                ax.hlines(index, float(ci[0]), float(ci[1]), color="C0", linewidth=1.5)
            ax.scatter([value], [index], color="C0", s=30, zorder=3)
            status = "temporal support sufficient" if supported else "support insufficient/unverified"
            if not finite_ci:
                status += "; CI unavailable"
            ax.annotate(status, (value, index), xytext=(4, 5), textcoords="offset points", fontsize=6)
        ax.axvline(0, color=".5", linestyle="--", linewidth=.8, label="Zero difference reference")
        ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Paired difference in " + endpoint + " (1)",
               title="Reconstruction" if endpoint == "nmae" else "Fixed-ridge forecast at 1 h")
        ax.set_ylim(-.5,max(.5,len(labels)-.5))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        if not labels:
            _empty(ax)
    fig.suptitle("Exploratory conditional physics: " + str(hypothesis) + "\nPointwise conditional intervals; no confirmatory physical claims", y=1.06)
    return fig


def _scalar_reduction(reports, sites, folds):
    """Read compact CSV rows once, storing only scalar evaluation values on disk."""
    scalar = []
    horizons = []
    with tempfile.TemporaryDirectory(prefix=".figure-reduction-", dir=reports) as directory:
        db = sqlite3.connect(Path(directory) / "scalars.sqlite")
        db.execute("PRAGMA temp_store=FILE")
        metric_sql = ",".join(f'"{metric}" REAL' for metric in SCALAR_METRICS)
        db.execute("CREATE TABLE scalar(site TEXT,fold INTEGER,duration REAL,method TEXT,variant TEXT,gap_id TEXT,mask_seed TEXT," + metric_sql + ")")
        placeholders = ",".join("?" for _ in range(7 + len(SCALAR_METRICS)))
        batch = []
        for filename, energy_track in (("episodes.csv", False), ("energy_track.csv", True)):
            path = reports / filename
            if not path.is_file():
                continue
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("partition") != "test" or row.get("condition") not in CORE:
                        continue
                    variant = "E:" + row.get("physical_hypothesis", "unknown") if energy_track else row.get("variant", "P")
                    probability = row.get("probabilistic")
                    scores = {}
                    if probability and probability not in ("nan", "None"):
                        try:
                            scores = json.loads(probability)
                        except (ValueError, TypeError):
                            scores = {}
                    if not isinstance(scores, dict):
                        scores = {}
                    base_method = row.get("method", "").split("+", 1)[0]
                    if variant == "P" and not scores and base_method not in ("CSDI", "TS-ICL"):
                        continue
                    values = [_number(scores.get("crps")), _number(scores.get("crps_truncated_quantile_approximation"))]
                    for level in (.5, .8, .9, .95):
                        values.append(_number(scores.get("intervals", {}).get(str(level), {}).get("coverage")))
                    for metric in ("net_power_integral_error_kwh", "net_input_counter_residual_kwh"):
                        number = _number(row.get(metric))
                        values.append(abs(number) if number is not None else None)
                    batch.append((row.get("site"), int(float(row["fold"])), float(row["duration"]), row.get("method"), variant,
                                  row.get("gap_id"), row.get("mask_seed"), *values))
                    if len(batch) == 10000:
                        db.executemany("INSERT INTO scalar VALUES(" + placeholders + ")", batch)
                        batch.clear()
        if batch:
            db.executemany("INSERT INTO scalar VALUES(" + placeholders + ")", batch)
        db.commit()
        means = ",".join(f'CASE WHEN COUNT("{metric}")=COUNT(*) THEN AVG("{metric}") END AS "{metric}"' for metric in SCALAR_METRICS)
        # Sort/group the large scalar input once, not once per plotted score.
        db.execute('CREATE TABLE scalar_episode AS SELECT site,fold,duration,method,variant,gap_id,mask_seed,' + means +
                   ' FROM scalar GROUP BY site,fold,duration,method,variant,gap_id,mask_seed')
        db.execute('CREATE TABLE scalar_cells AS SELECT site,fold,duration,method,variant,' + means +
                   ',COUNT(*) AS episodes FROM scalar_episode GROUP BY site,fold,duration,method,variant')
        for metric in SCALAR_METRICS:
            # A missing seed score propagates NA rather than improving the curve
            # by silently dropping a failed probabilistic/physical evaluation.
            query = f'SELECT site,fold,duration,method,variant,"{metric}",episodes FROM scalar_cells'
            scalar.extend(_equal_cells(db.execute(query).fetchall(), folds, metric))
        forecasts = reports / "forecasts_long.csv"
        if forecasts.is_file():
            db.execute("CREATE TABLE forecast(site TEXT,fold INTEGER,duration REAL,method TEXT,variant TEXT,gap_id TEXT,mask_seed TEXT,horizon INTEGER,value REAL)")
            batch = []
            with forecasts.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("partition") != "test" or row.get("condition") not in CORE or row.get("variant") != "P":
                        continue
                    batch.append((row.get("site"), int(float(row["fold"])), float(row["duration"]), row.get("method"), "P",
                                  row.get("gap_id"), row.get("mask_seed"), int(float(row["horizon_seconds"])), _number(row.get("nmae"))))
                    if len(batch) == 10000:
                        db.executemany("INSERT INTO forecast VALUES(?,?,?,?,?,?,?,?,?)", batch)
                        batch.clear()
            if batch:
                db.executemany("INSERT INTO forecast VALUES(?,?,?,?,?,?,?,?,?)", batch)
            db.commit()
            db.execute('CREATE TABLE forecast_episode AS SELECT site,fold,duration,method,variant,gap_id,mask_seed,horizon,'
                       'CASE WHEN COUNT(value)=COUNT(*) THEN AVG(value) END AS value FROM forecast '
                       'GROUP BY site,fold,duration,method,variant,gap_id,mask_seed,horizon')
            db.execute('CREATE TABLE forecast_cells AS SELECT site,fold,duration,method,variant,horizon,'
                       'CASE WHEN COUNT(value)=COUNT(*) THEN AVG(value) END AS value,COUNT(*) AS episodes '
                       'FROM forecast_episode GROUP BY site,fold,duration,method,variant,horizon')
            levels = [row[0] for row in db.execute("SELECT DISTINCT horizon FROM forecast_cells ORDER BY horizon")]
            for horizon in levels:
                query = 'SELECT site,fold,duration,method,variant,value,episodes FROM forecast_cells WHERE horizon=?'
                entries = _equal_cells(db.execute(query, (horizon,)).fetchall(), folds, "forecast_nmae")
                horizons.extend([{**entry, "horizon_seconds": horizon} for entry in entries])
        db.close()
    return scalar, horizons


def _equal_cells(rows, folds, metric):
    grouped = defaultdict(dict)
    counts = defaultdict(int)
    for site, fold, duration, method, variant, value, count in rows:
        grouped[(site, method, variant)][(int(fold), float(duration))] = _number(value)
        counts[(site, method, variant)] += int(count)
    result = []
    for (site, method, variant), cells in sorted(grouped.items()):
        required = [(fold, duration) for fold in folds.get(site, ()) for duration in DURATIONS]
        values = [cells.get(key) for key in required]
        supported = bool(required) and all(value is not None for value in values)
        result.append(dict(site=site, method=method, variant=variant, metric=metric,
            value=float(np.mean(values)) if supported else None, episodes=counts[(site, method, variant)],
            complete_fold_duration_support=supported, required_cells=len(required),
            missing_cells=sum(value is None for value in values)))
    return result


def _horizon_figures(plt, rows, sites, out, paths, manifest):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    for group in ("baselines", "learned", "frozen"):
        selected = frame[frame.method.map(_method_group) == group]
        if selected.empty:
            continue
        fig, axes = _axes(plt, sites)
        for site, ax in zip(sites, axes[0]):
            present = False
            for method, values in selected[selected.site == site].groupby("method", sort=True):
                values = values.sort_values("horizon_seconds")
                y = [np.nan if value is None else value for value in values.value]
                if np.isfinite(y).any():
                    ax.plot(values.horizon_seconds / 3600, y, marker="o", label=method)
                    present = True
            if present:
                ax.legend(fontsize=7)
            else:
                _empty(ax)
            ax.set(title=str(site), xlabel="Physical forecast horizon (h)", ylabel="Fixed-ridge forecast nMAE (1)")
            ax.grid(alpha=.2)
        name = "forecast_horizons_" + group
        fig.suptitle("R/P downstream errors — " + group + "; original targets only", y=1.04)
        _save(fig, out, name, paths)
        manifest.append(dict(name=name, source="forecasts_long.csv", aggregation="training-seed losses within episodes, equal durations and folds; undefined strata remain NA", points=selected.where(pd.notna(selected), None).to_dict("records")))
        plt.close(fig)


def _uncertainty_figures(plt, rows, sites, out, paths, manifest):
    relevant = [row for row in rows if row["metric"].startswith("coverage_")]
    capable = {(row['site'],row['method'],row['variant']) for row in relevant if row['value'] is not None or row['method'] in ('CSDI','TS-ICL') or row['method'].endswith('+energy_samples')}
    relevant = [row for row in relevant if (row['site'],row['method'],row['variant']) in capable]
    variants = sorted({row["variant"] for row in relevant})
    for variant in variants:
        selected = [row for row in relevant if row["variant"] == variant]
        fig, axes = _axes(plt, sites)
        for site, ax in zip(sites, axes[0]):
            present = False
            for method in sorted({row["method"] for row in selected if row["site"] == site}):
                values = [row for row in selected if row["site"] == site and row["method"] == method]
                lookup = {float(row["metric"].split("_", 1)[1]): row["value"] for row in values}
                levels = (.5, .8, .9, .95)
                y = [np.nan if lookup.get(level) is None else lookup[level] for level in levels]
                if np.isfinite(y).any():
                    ax.plot(levels, y, marker="o", label=method)
                    present = True
                unavailable = [str(level) for level in levels if lookup.get(level) is None]
                if unavailable:
                    ax.text(.02, .02 + .05 * len(ax.texts), str(method) + " NA: " + ", ".join(unavailable), transform=ax.transAxes, fontsize=6)
            ax.plot([.5, .95], [.5, .95], color=".5", linestyle="--", linewidth=.8, label="Nominal reference")
            if present:
                ax.legend(fontsize=7)
            else:
                _empty(ax)
            ax.set(title=str(site), xlabel="Nominal marginal coverage", ylabel="Empirical marginal coverage", ylim=(0, 1.02), xlim=(.48, .97))
            ax.grid(alpha=.2)
        name = "uncertainty_coverage_" + variant.replace(":", "_")
        fig.suptitle("Marginal interval diagnostics; no joint coverage/calibration guarantee" + ("\nExploratory conditional physics: " + variant[2:] if variant.startswith("E:") else " — R/P"), y=1.08)
        _save(fig, out, name, paths)
        manifest.append(dict(name=name, source="episodes.csv", aggregation="training-seed score means within episodes; equal duration/fold means; incomplete scores remain NA", points=selected))
        plt.close(fig)


def _energy_figures(plt, rows, sites, out, paths, manifest):
    metrics = ("net_integral_absolute_error_kwh", "net_input_constraint_absolute_residual_kwh")
    relevant = [row for row in rows if row["metric"] in metrics]
    variants = sorted({row["variant"] for row in relevant if row["variant"].startswith("E:") and row["value"] is not None})
    for variant in variants:
        selected = [row for row in relevant if row["variant"] == variant]
        n_methods = max([len({row["method"] for row in selected if row["site"] == site}) for site in sites] or [1])
        fig, axes = plt.subplots(2, len(sites), squeeze=False, figsize=(max(5.8, 5.8 * len(sites)), max(7., n_methods * .43 * 2)))
        for column, site in enumerate(sites):
            methods = sorted({row["method"] for row in selected if row["site"] == site})
            for index, metric in enumerate(metrics):
                ax = axes[index, column]
                lookup = {row["method"]: row["value"] for row in selected if row["site"] == site and row["metric"] == metric}
                for position, method in enumerate(methods):
                    value = lookup.get(method)
                    if value is None:
                        ax.text(.02, position, "NA", transform=ax.get_yaxis_transform(), va="center", fontsize=7)
                    else:
                        ax.barh([position], [value], color="C0", height=.65)
                        if value == 0:
                            ax.annotate("0",(0,position),xytext=(3,0),textcoords="offset points",va="center",fontsize=7)
                ax.set(yticks=range(len(methods)), yticklabels=methods, title=str(site),
                    xlabel="Mean |net integral error| (kWh)\nReference: original power under hypothesis" if index == 0 else "Mean |input counter residual| (kWh)\nConstraint agreement; not independent accuracy")
                ax.invert_yaxis()
                ax.set_xlim(left=0)
                ax.grid(axis="x", alpha=.2)
                ax.tick_params(axis="y", labelsize=7)
                if not methods:
                    _empty(ax)
        name = "energy_" + variant[2:]
        fig.suptitle("Exploratory conditional physics: " + variant[2:] + "\nNo independent meter calibration is established", y=1.03)
        _save(fig, out, name, paths)
        manifest.append(dict(name=name, source=["episodes.csv", "energy_track.csv"], aggregation="mean absolute energy error per seed, episode, duration, fold; CPI kept as interval-mean adapter", points=selected))
        plt.close(fig)


def generate_conference_figures(run_dir):
    """Return exported paths; inputs/results remain under this private run only."""
    run = Path(run_dir).resolve()
    reports = run / "reports"
    if not reports.is_dir():
        raise FileNotFoundError("generate the committed conference reports before figures")
    out = reports / "figures"
    out.mkdir(exist_ok=True)
    main, core = _compact(reports / "main_table.csv"), _compact(reports / "core_aggregate.csv")
    sites = sorted(set(main.site.dropna().astype(str))) if "site" in main else sorted(set(core.site.dropna().astype(str))) if "site" in core else []
    folds = {site: sorted(set(main.loc[main.site == site, "fold"].dropna().astype(int))) for site in sites} if "fold" in main else {site: [1, 2, 3] for site in sites}
    paths, manifest = [], []
    plt = _matplotlib()
    if sites:
        _reconstruction_figures(plt, main, sites, out, paths, manifest)
        _pareto_and_utility(plt, main, core, sites, out, paths, manifest)
        scalar, horizons = _scalar_reduction(reports, sites, folds)
        _horizon_figures(plt, horizons, sites, out, paths, manifest)
        _uncertainty_figures(plt, scalar, sites, out, paths, manifest)
        _energy_figures(plt, scalar, sites, out, paths, manifest)
    contrast_path = reports / "contrasts.json"
    contrasts = json.loads(contrast_path.read_text()) if contrast_path.is_file() else []
    if isinstance(contrasts, list):
        for hypothesis in sorted({str(row.get("hypothesis", "unspecified")) for row in contrasts}):
            rows = [row for row in contrasts if str(row.get("hypothesis", "unspecified")) == hypothesis]
            fig = contrast_figure(rows, hypothesis)
            name = "contrasts_" + hypothesis.replace(":", "_")
            _save(fig, out, name, paths)
            manifest.append(dict(name=name, source="contrasts.json", aggregation="paired estimates/intervals copied unchanged; unavailable entries have text labels only", points=rows))
            plt.close(fig)
    summary_path = reports / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    atomic_json(out / "manifest.json", _jsonable(dict(schema_version=1, output_formats=["png", "svg", "pdf"],
        physical_confirmatory=False, figures=manifest, paths=paths,
        study_profile=summary.get('study_profile'),scope_reduced_by_design=summary.get('scope_reduced_by_design'),
        full_catalogue_reproduced=False,omitted_methods_by_design=summary.get('omitted_methods_by_design',[]),
        omitted_conditions_by_design=summary.get('omitted_conditions_by_design',[]),
        blocked_methods=summary.get("blocked_methods", {}),
        exclusions="NA values are not replaced with zero; no statistics are inferred from missing methods; primary curves use C3-C5",
        source_directory=str(reports), manuscript_files_modified=False)))
    return paths

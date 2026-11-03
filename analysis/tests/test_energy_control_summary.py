"""Portable aggregate-only control reproduction and coverage checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_energy_controls.py"
spec = importlib.util.spec_from_file_location("control_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture_bundle(path):
    path.mkdir()
    rows = []
    for site in ("H1", "H2", "I1"):
        for fold in (1, 2, 3):
            for condition, duration in (("C3", 15.), ("C4", 60.), ("C5", 180.)):
                for policy in summary.POLICIES:
                    rows.append(dict(site=site, fold=fold, partition="test", condition=condition,
                        duration=duration, mask_seed=2001, gap_id=f"{site}-{fold}-{condition}", policy=policy,
                        nmae=.2, mae_kw=.4, nmae_original_E=.15, mae_kw_original_E=.3,
                        nmae_P=.25, mae_kw_P=.5, difference_vs_original_E=.05,
                        status="success", reason="synthetic"))
    paired = path / "control_paired_losses.parquet"
    pd.DataFrame(rows).to_parquet(paired, index=False)
    protocol = dict(sites=["H1", "H2", "I1"], folds=[1, 2, 3])
    status = dict(status="completed_descriptive_posthoc_replay", expected_test_rows=len(rows),
        outputs_sha256={paired.name: hashlib.sha256(paired.read_bytes()).hexdigest()},
        linear_P_reproduction=[dict(episodes=27, max_abs_mae_or_nmae_error=0.)])
    (path / "control_protocol.json").write_text(json.dumps(protocol))
    (path / "control_validation.json").write_text(json.dumps(status))
    return status


def test_summary_runs_as_standalone_file_without_experiment_source(tmp_path):
    controls = tmp_path / "inputs"
    fixture_bundle(controls)
    standalone = tmp_path / "summary.py"
    shutil.copyfile(SCRIPT, standalone)
    subprocess.run([sys.executable, str(standalone), "--controls-dir", str(controls),
                    "--output-dir", str(tmp_path / "out")], check=True, capture_output=True)
    table = (tmp_path / "out" / "tables" / "energy_controls.tex").read_text()
    assert "H1 & 0.2500 & 0.2000 & 0.2000 & 0.1500" in table
    validation = json.loads((tmp_path / "out" / "summary_validation.json").read_text())
    assert validation["unique_test_episodes"] == 27
    assert validation["site_fold_duration_policy_cells"] == 54


def test_summary_refuses_missing_cell_even_when_checksum_updated(tmp_path):
    controls = tmp_path / "inputs"
    status = fixture_bundle(controls)
    paired = controls / "control_paired_losses.parquet"
    frame = pd.read_parquet(paired)
    frame = frame[~((frame.site == "I1") & (frame.fold == 3) & (frame.duration == 180.))]
    frame.to_parquet(paired, index=False)
    status["expected_test_rows"] = len(frame)
    status["outputs_sha256"][paired.name] = hashlib.sha256(paired.read_bytes()).hexdigest()
    (controls / "control_validation.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Incomplete site/fold/duration"):
        summary.summarize(controls, tmp_path / "out")
